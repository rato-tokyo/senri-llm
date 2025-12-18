# Senri-LLM Development Guidelines

## Project Overview

Senri-LLMは、Qwen2.5-0.5Bをベースに、直交基底ルーティングによるInfini Attentionを実装するプロジェクトです。

## Architecture Specification

### Core Concept

```
学習時: 通常のInfini Attention（単一テンソル積メモリ）
推論時: 直交基底ベースの動的テンソル積選択
```

### Layer Structure (24 layers total)

```
Layer 0-11:  SWA only (Lower Decoder)
Layer 12:    SWA + Senri Memory (Group 1)
Layer 13-15: SWA only
Layer 16:    SWA + Senri Memory (Group 2)
Layer 17-19: SWA only
Layer 20:    SWA + Senri Memory (Group 3)
Layer 21-23: SWA only
```

### Memory Layer Configuration

- `num_memory_layers`: 3 (Senri Memoryを持つレイヤー数)
- `first_memory_layer`: 12 (最初のメモリレイヤー)
- `memory_layer_interval`: 4 (メモリレイヤー間隔)
- メモリレイヤーのインデックス: [12, 16, 20]

### Memory Sharing Policy

**現状: 各層が独立したメモリを持つ（Independent Memory）**

```
Layer 12: SenriMemory_1 (独立)
Layer 16: SenriMemory_2 (独立)
Layer 20: SenriMemory_3 (独立)
```

- 各層が独自の `SenriMemory` インスタンスを保持
- 層間でメモリ状態は共有されない
- メリット: 各層が異なる抽象度の情報を保持可能

**将来検討: 単一共有メモリ（Shared Memory）**

Infini-Attention論文では単一メモリを複数層で共有するバリエーションも議論されている。
最大コンテキストウィンドウ達成のため、将来的に共有メモリ方式も検討予定。

```
Layer 12, 16, 20: SharedSenriMemory (共有)
```

### Positional Encoding

- **SWA (Local Attention)**: RoPE使用
- **Senri Memory (Global Attention)**: NoPE (No Positional Encoding)

## Training vs Inference Mode - 重要

**Senriは `model.train()` と `model.eval()` で挙動が大きく異なる。**

### 比較表

| 項目 | 学習時 (`model.train()`) | 推論時 (`model.eval()`) |
|------|-------------------------|------------------------|
| **メモリ構造** | 単一テンソル積メモリ (`TensorMemory`) | 複数メモリ + 直交基底ルーティング (`OrthogonalBasisMemory`) |
| **メモリ形状** | `[batch, heads, head_dim, head_dim]` | `[batch, heads, hidden_size, head_dim, head_dim]` |
| **メモリ更新** | 単純累積: `M = M + v ⊗ k` | Delta rule: `M = M + (v - retrieve(k)) ⊗ k` |
| **Key割り当て** | なし（全てのKVが同一メモリへ） | 基底ルーティング: `argmax(|k|)` で分散 |
| **Query検索** | 全メモリから一括検索 | Top-k メモリ選択 + 重み付き統合 |
| **勾配計算** | あり | なし (`torch.no_grad()`) |
| **目的** | 重要度の学習、パラメータ更新 | 効率的な長文処理、重複除去 |

### 詳細説明

#### 1. メモリ構造の違い

```python
# 学習時: 単一メモリ
M = torch.zeros(batch, heads, head_dim, head_dim)

# 推論時: hidden_size 個の独立メモリ
M = torch.zeros(batch, heads, hidden_size, head_dim, head_dim)
```

**理由**: 推論時は超長文を扱うため、単一メモリでは情報が混在しすぎる。直交基底でメモリを分割し、関連情報のみを検索。

#### 2. メモリ更新の違い

```python
# 学習時: 単純累積（勾配を流すため）
M = M + outer(v, k)

# 推論時: Delta rule（重複除去）
v_existing = retrieve(k)      # 既存値を取得
v_delta = v - v_existing      # 差分を計算
M = M + outer(v_delta, k)     # 差分のみ追加
```

**理由**:
- 学習時は勾配が必要なため、シンプルな累積
- 推論時は同じ情報の重複蓄積を防ぎ、メモリ効率を向上

#### 3. Key-Valueの割り当て

```python
# 学習時: 全KVが同一メモリへ
memory.update(keys, values)  # 単一Mへ追加

# 推論時: 直交基底で分散
basis_idx = keys.abs().argmax(dim=-1)  # 最大絶対値の次元
# 各KVペアは対応する基底のメモリへ
```

**理由**: 推論時は情報を意味的に分離し、検索時に関連メモリのみを参照

#### 4. Query検索の違い

```python
# 学習時: 全メモリから検索
output = (M @ q) / (z.T @ q + eps)

# 推論時: Top-k選択 + 重み付き統合
scores = queries.abs()
top_k_indices = scores.topk(k=top_k_memories).indices
output = weighted_sum(retrieve_from_each(top_k_indices))
```

**理由**: 推論時は関連性の高いメモリのみを使用し、計算効率と精度を両立

### モード切り替えの注意点

```python
# 正しい使用方法
model.train()   # 学習モード: TensorMemory使用
model.eval()    # 推論モード: OrthogonalBasisMemory + Delta rule

# メモリの状態は内部で自動的に切り替わる
# SenriMemory.training フラグで制御
```

### なぜ学習/推論で異なる戦略を使うのか？

1. **学習時の要件**:
   - 勾配を計算可能な形式が必要
   - シンプルな構造で安定した学習
   - 短いコンテキストで十分（学習データは通常512〜4K tokens）

2. **推論時の要件**:
   - 超長文（16M+ tokens）を効率的に処理
   - 重複情報の蓄積を防止
   - 関連情報のみを高速に検索

## Implementation Details

### Tensor Memory

```python
# 学習時（単一メモリ）
M = torch.zeros(batch, heads, head_dim, head_dim)  # テンソル積
z = torch.zeros(batch, heads, head_dim)            # 正規化係数

# 更新
M = M + torch.einsum('bhd,bhe->bhde', v, k)
z = z + k.sum(dim=seq)

# 検索
output = torch.einsum('bhde,bhe->bhd', M, q) / (z @ q + eps)
```

```python
# 推論時（複数メモリ + Delta Rule）
M = torch.zeros(batch, heads, hidden_dim, head_dim, head_dim)  # 基底ごと
z = torch.zeros(batch, heads, hidden_dim, head_dim)

# keyの割り当て（単位行列基底なので、最大絶対値の次元）
basis_idx = k.abs().argmax(dim=-1)  # [batch, seq]

# Delta Rule による更新（推論時のみ）
# 既存の情報を差し引いてから新しい情報を追加
v_existing = (M @ k) / (z^T @ k + eps)  # メモリから取得
v_delta = v - v_existing                 # 差分を計算
M = M + v_delta ⊗ k                      # 差分のみ追加

# top-k選択
scores = q.abs()  # [batch, heads, seq, head_dim]
top_k_indices = scores.topk(k=top_k_memories, dim=-1).indices
```

### Delta Rule（推論時のみ）

**目的**: 重複情報の蓄積を防ぎ、メモリ効率と検索精度を向上

**学習時 vs 推論時**:
- **学習時**: 単純累積（勾配が流れ、重要度を学習）
- **推論時**: Delta rule（重複除去、効率的なメモリ利用）

```python
# Delta rule の数式
delta = v - retrieve(k)    # 新しい値 - 既存の値
M = M + outer(delta, k)    # 差分のみをメモリに追加
```

**利点**:
1. 同じ情報の重複蓄積を防止
2. メモリ容量の効率的な利用
3. 検索時のノイズ低減

### SVD-based Memory Cleaning (Noise Removal)

テンソル積メモリに蓄積されるノイズを除去するため、周期的SVDクリーニング機能を実装。

**原理**: SVDによる低ランク近似（Eckart-Young定理に基づく最適近似）

```python
# 基本的な使用方法
from src.memory import SenriMemory, SVDCleaningStats

memory = SenriMemory(num_heads=14, head_dim=64, hidden_size=896)
memory.reset(batch_size=1, device=device, dtype=dtype)

# メモリ更新後、ノイズ除去を実行
stats = memory.svd_cleaning(
    energy_threshold=0.95,  # 95%のエネルギーを保持
    max_rank=None,          # Noneの場合、energy_thresholdで決定
)

print(f"Original rank: {stats.original_rank}")
print(f"Retained rank: {stats.retained_rank}")
print(f"Energy retained: {stats.energy_retained:.2%}")
```

**パラメータ**:

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `energy_threshold` | float | 0.95 | 保持するエネルギーの割合（特異値の二乗和） |
| `max_rank` | int | None | 明示的なランク上限。Noneの場合はenergy_thresholdで決定 |
| `basis_indices` | List[int] | None | (推論時のみ) クリーニングする基底インデックス |

**実行タイミング（未定、将来実装予定）**:
- ユーザー入力待機中（アイドル時）
- 一定のメモリ更新回数ごと
- メモリ使用量が閾値を超えた時

**統計情報の活用**:

```python
# 詳細な統計を取得
stats = memory.svd_cleaning(energy_threshold=0.90)

# 特異値の分布を確認（デバッグ用）
print(f"Top singular values: {stats.singular_values_before[0, :5]}")

# OrthogonalBasisMemory（推論時）の場合
# stats.per_basis_stats で各基底の詳細統計を取得可能
```

**注意事項**:
- SVD計算は O(n³) の計算量。頻繁な実行は避ける
- `torch.no_grad()` 内で実行されるため、勾配は計算されない
- メモリ初期化前に呼び出すと `RuntimeError` が発生

### HuggingFace Compatibility

- `SenriConfig`: `Qwen2Config`を継承
- `SenriForCausalLM`: `Qwen2ForCausalLM`の構造を踏襲
- `from_pretrained`/`save_pretrained`完全対応
- `generate()`メソッドでの推論対応

### Key Classes

```python
# src/configuration_senri.py
class SenriConfig(Qwen2Config):
    model_type = "senri"

    # Senri specific
    sliding_window_size: int = 4096
    chunk_size: int = 64
    top_k_memories: int = 64
    num_memory_layers: int = 3
    first_memory_layer: int = 12
    memory_layer_interval: int = 4

# src/memory/tensor_memory.py
class TensorMemory:
    """単一テンソル積メモリ（学習用）"""

class OrthogonalBasisMemory:
    """直交基底ベースの複数テンソル積メモリ（推論用）"""

# src/attention/senri_attention.py
class SenriAttention(nn.Module):
    """SWA + Senri Memory Attention"""

    def forward(self, hidden_states, ...):
        if self.training:
            return self._forward_training(...)  # 単一メモリ
        else:
            return self._forward_inference(...)  # 直交基底ルーティング
```

## Configuration Management Policy

### 設定ファイルベースの管理（重要）

**全ての実験パラメータは `config/` ディレクトリのYAMLファイルで管理する。**

スクリプトにコマンドラインオプションを追加することは禁止。設定変更は必ずconfigファイルを編集して行う。

### Config Directory Structure

```
config/
├── model.yaml       # モデルアーキテクチャ設定
├── training.yaml    # 学習ハイパーパラメータ
└── experiment.yaml  # 実験全体の設定
```

### 設定ファイルの役割

| ファイル | 内容 |
|---------|------|
| `model.yaml` | vocab_size, hidden_size, num_layers, memory layer設定など |
| `training.yaml` | epochs, batch_size, learning_rate, optimizer設定など |
| `experiment.yaml` | output_dir, benchmark設定, Colab設定など |

### 使用方法

```python
from src.config import ConfigManager

# 全設定を読み込み
config = ConfigManager()

# 個別アクセス
model_name = config.base_model_name
batch_size = config.batch_size

# TrainingConfigに変換
training_config = config.to_training_config()

# SenriConfigに変換
senri_config = config.to_senri_config()
```

### スクリプト実行

```bash
# 引数なしで実行（config/*.yamlから設定を読み込む）
python scripts/colab.py train
python scripts/colab.py test
python scripts/colab.py eval
```

### 禁止事項

- ⛔ スクリプトへのargparse引数追加
- ⛔ コード内でのハードコーディング
- ⛔ 環境変数による設定（特殊な場合を除く）

### 設定変更の手順

1. `config/*.yaml` を編集
2. 変更をコミット
3. スクリプトを実行

## Coding Standards

### File Naming
- スネークケース: `tensor_memory.py`, `senri_attention.py`
- クラス名: パスカルケース `SenriAttention`, `TensorMemory`

### Import Order
1. 標準ライブラリ
2. サードパーティ (torch, transformers)
3. ローカルモジュール

### Type Hints
- 全ての関数に型ヒントを付与
- Tensorの形状はdocstringでコメント

```python
def forward(
    self,
    hidden_states: torch.Tensor,  # [batch, seq, hidden]
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Args:
        hidden_states: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, 1, seq_len, seq_len]

    Returns:
        output: [batch_size, seq_len, hidden_size]
    """
```

## Testing Requirements

### Unit Tests
- 各モジュールに対応するテストファイル
- `pytest`使用

### Test Cases
1. `TensorMemory`: 更新と検索の正確性
2. `OrthogonalBasisMemory`: 基底割り当ての正確性
3. `SenriAttention`: 学習/推論モードの切り替え
4. `SenriForCausalLM`: Qwen2.5重みのロード

### Shape Tests
```python
def test_tensor_memory_shapes():
    memory = TensorMemory(hidden_size=896, num_heads=14)
    q = torch.randn(2, 14, 100, 64)
    k = torch.randn(2, 14, 100, 64)
    v = torch.randn(2, 14, 100, 64)

    memory.update(k, v)
    output = memory.retrieve(q)

    assert output.shape == (2, 14, 100, 64)
```

## Git Workflow

### Branch Strategy
- `main`: 安定版
- `dev`: 開発版
- `feature/*`: 機能開発

### Commit Message Format
```
<type>: <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

## Performance Considerations

### Memory Efficiency
- 推論時のメモリ使用量を最小化
- 不要なテンソルの即座解放
- `torch.no_grad()`の適切な使用

### Computational Efficiency
- 直交基底が単位行列なので、射影計算が単純なインデックス参照に簡略化
- Top-k選択は効率的な`torch.topk`使用

## Dependencies

```
torch>=2.0.0
transformers>=4.36.0
accelerate>=0.25.0
datasets>=2.14.0
```

## Common Pitfalls

### 1. メモリリーク
```python
# Bad
self.memory_history.append(memory.clone())

# Good
self.memory_history.append(memory.detach().clone())
```

### 2. 学習/推論モード混同
```python
# 必ずモードを明示的に確認
if self.training:
    # 単一メモリ
else:
    # 直交基底ルーティング
```

### 3. 位置エンコーディングの混在
```python
# SWA: RoPE適用
# Senri Memory: RoPE適用しない（NoPE）
```

## Debugging Tips

### メモリ状態の確認
```python
print(f"Memory M norm: {memory.M.norm()}")
print(f"Memory z norm: {memory.z.norm()}")
```

### Attention重みの可視化
```python
import matplotlib.pyplot as plt
plt.imshow(attention_weights[0, 0].detach().cpu())
```

## Experiment Environment

### ローカル環境
- **用途**: 簡単な動作確認のみ
- **テスト**: `pytest tests/`
- **動作確認**: `python scripts/local_test.py`

### Google Colab（本実験環境）
- **用途**: 本格的な学習・評価実験
- **GPU**: T4 / A100 を想定
- **スクリプト**: `scripts/colab.py`

### Colabでの実行手順

```python
# 1. リポジトリのクローン
!git clone https://github.com/YOUR_USERNAME/senri-llm.git
%cd senri-llm

# 2. 依存関係のインストール
!pip install -e .

# 3. 設定の確認・編集（必要に応じて）
# config/training.yaml を編集して学習パラメータを調整

# 4. 実験の実行
!python scripts/colab.py train   # 学習
!python scripts/colab.py test    # 動作確認
!python scripts/colab.py eval    # 評価
```

### Colab Notebook構成

```
notebooks/
├── 01_model_test.ipynb      # モデル動作確認
├── 02_training.ipynb        # 学習実験
├── 03_evaluation.ipynb      # 評価実験（RULER, NIAH等）
└── 04_analysis.ipynb        # 結果分析・可視化
```

### GPU メモリ管理（Colab）

```python
# メモリクリア
import gc
import torch
gc.collect()
torch.cuda.empty_cache()

# メモリ使用量確認
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### Checkpointing（Colab）

```python
# Google Driveへの保存
from google.colab import drive
drive.mount('/content/drive')

# チェックポイント保存
model.save_pretrained('/content/drive/MyDrive/senri-checkpoints/epoch_1')
```
