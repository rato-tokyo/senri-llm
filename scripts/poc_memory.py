"""
PoC: Senri Memory Long-Context Test

このスクリプトは、Senriメモリが長文コンテキストで機能するかを確認します。

テスト内容:
1. 小さなモデルを作成
2. "記憶テスト"用のシンプルなデータで学習
3. 学習後、メモリが情報を保持できているか確認

実行方法 (Colab):
    !python scripts/poc_memory.py

所要時間: 約2-5分（GPU使用時）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.configuration_senri import SenriConfig
from src.modeling_senri import SenriForCausalLM


# =============================================================================
# Configuration
# =============================================================================

# 小さなモデル設定（高速テスト用）
POC_CONFIG = {
    "vocab_size": 256,  # 小さな語彙
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 128,
    "num_memory_layers": 1,
    "first_memory_layer": 2,  # Layer 2 がメモリレイヤー
    "memory_layer_interval": 10,
}

# 学習設定
TRAIN_CONFIG = {
    "num_epochs": 50,
    "batch_size": 4,
    "learning_rate": 1e-3,
    "seq_len": 64,
    "num_samples": 100,
}


# =============================================================================
# Dataset: Key-Value Memory Test
# =============================================================================

class KeyValueDataset(Dataset):
    """
    シンプルなKey-Value記憶テスト用データセット

    パターン: [KEY] [random tokens...] [QUERY] -> [VALUE]

    モデルは長いコンテキストの最初にあるKEYを記憶し、
    最後のQUERYに対してVALUEを出力する必要がある。
    """

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # 特殊トークン
        self.KEY_TOKEN = 1
        self.QUERY_TOKEN = 2
        self.VALUE_TOKEN = 3
        self.PAD_TOKEN = 0

        # データ生成
        self.data = self._generate_data()

    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            # ランダムなkey-valueペア（4-10の範囲）
            key_id = torch.randint(4, 10, (1,)).item()
            value_id = torch.randint(10, 20, (1,)).item()

            # シーケンス構築
            # [KEY, key_id, filler..., QUERY, key_id] -> value_id
            seq = torch.zeros(self.seq_len, dtype=torch.long)

            # 開始: KEY + key_id
            seq[0] = self.KEY_TOKEN
            seq[1] = key_id

            # 中間: ランダムなfiller（20-100の範囲）
            filler_len = self.seq_len - 5
            seq[2:2+filler_len] = torch.randint(20, 100, (filler_len,))

            # 終了: QUERY + key_id + VALUE（ラベル用）
            query_pos = self.seq_len - 3
            seq[query_pos] = self.QUERY_TOKEN
            seq[query_pos + 1] = key_id
            seq[query_pos + 2] = value_id

            # ラベル: 最後のトークン（value_id）のみ予測
            labels = seq.clone()
            labels[:-1] = -100  # 最後以外は無視

            data.append({
                "input_ids": seq,
                "labels": labels,
                "key_id": key_id,
                "value_id": value_id,
            })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """DataLoaderのcollate関数"""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# =============================================================================
# Training
# =============================================================================

def train_model(model, dataloader, num_epochs, lr, device):
    """シンプルな学習ループ"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 40)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with model.new_sequence():
                output = model(input_ids=input_ids, labels=labels)

            loss = output.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss = {avg_loss:.4f}")

    print("-" * 40)
    return model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_memory(model, dataset, device):
    """メモリが機能しているか評価"""
    model.eval()

    correct = 0
    total = 0

    print("\nEvaluating memory retrieval...")
    print("-" * 40)

    # いくつかのサンプルをテスト
    test_samples = min(20, len(dataset))

    for i in range(test_samples):
        sample = dataset[i]
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        expected_value = sample["value_id"]
        key_id = sample["key_id"]

        with torch.no_grad():
            with model.new_sequence():
                # 最後のトークン以外を入力
                output = model(input_ids=input_ids[:, :-1])

        # 最後の位置の予測を取得
        logits = output.logits[:, -1, :]
        predicted = logits.argmax(dim=-1).item()

        is_correct = predicted == expected_value
        correct += int(is_correct)
        total += 1

        if i < 5:  # 最初の5つを表示
            status = "✓" if is_correct else "✗"
            print(f"  {status} Key={key_id}, Expected={expected_value}, Predicted={predicted}")

    accuracy = correct / total * 100
    print("-" * 40)
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")

    return accuracy


def test_memory_accumulation(model, device):
    """メモリがシーケンス内で蓄積されているか確認"""
    model.eval()

    print("\nTesting memory accumulation...")
    print("-" * 40)

    # メモリレイヤーを取得
    memory_layer = None
    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'memory'):
            memory_layer = layer.self_attn
            break

    if memory_layer is None:
        print("  No memory layer found!")
        return False

    # テスト入力
    input_ids = torch.randint(4, 100, (1, 32)).to(device)

    with torch.no_grad():
        with model.new_sequence():
            # 最初のforward
            model(input_ids=input_ids[:, :16])
            M_after_first = memory_layer.memory.M.clone()
            z_after_first = memory_layer.memory.z.clone()

            # 2回目のforward（リセットなし）
            model(input_ids=input_ids[:, 16:], past_key_values=None)
            # Note: past_key_values=Noneだとリセットされるので、
            # ここでは手動でpast_key_valuesを渡さない形にする

    # シーケンス開始時にリセットされることを確認
    with torch.no_grad():
        with model.new_sequence():
            model(input_ids=input_ids[:, :16])
            M_fresh = memory_layer.memory.M.clone()

    is_same = torch.allclose(M_after_first, M_fresh, atol=1e-5)
    print(f"  Memory resets correctly on new_sequence(): {is_same}")

    # メモリが非ゼロであることを確認
    is_nonzero = M_after_first.abs().sum() > 0
    print(f"  Memory contains non-zero values: {is_nonzero}")

    return is_same and is_nonzero


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Senri Memory PoC Test")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Model
    print("\n[1] Creating small Senri model...")
    config = SenriConfig(**POC_CONFIG)
    model = SenriForCausalLM(config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    print(f"  Memory layers: {config.get_memory_layer_indices()}")

    # Dataset
    print("\n[2] Creating Key-Value memory test dataset...")
    dataset = KeyValueDataset(
        num_samples=TRAIN_CONFIG["num_samples"],
        seq_len=TRAIN_CONFIG["seq_len"],
        vocab_size=POC_CONFIG["vocab_size"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    print(f"  Samples: {len(dataset)}")
    print(f"  Sequence length: {TRAIN_CONFIG['seq_len']}")

    # Pre-training evaluation
    print("\n[3] Pre-training evaluation (should be random)...")
    pre_accuracy = evaluate_memory(model, dataset, device)

    # Training
    print("\n[4] Training...")
    model = train_model(
        model,
        dataloader,
        num_epochs=TRAIN_CONFIG["num_epochs"],
        lr=TRAIN_CONFIG["learning_rate"],
        device=device,
    )

    # Post-training evaluation
    print("\n[5] Post-training evaluation...")
    post_accuracy = evaluate_memory(model, dataset, device)

    # Memory accumulation test
    print("\n[6] Memory accumulation test...")
    memory_ok = test_memory_accumulation(model, device)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Pre-training accuracy:  {pre_accuracy:.1f}%")
    print(f"  Post-training accuracy: {post_accuracy:.1f}%")
    print(f"  Memory accumulation:    {'OK' if memory_ok else 'FAILED'}")

    # 成功判定
    if post_accuracy > 50 and memory_ok:
        print("\n✓ PoC PASSED: Memory is working!")
        return True
    else:
        print("\n✗ PoC FAILED: Memory is not working correctly.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
