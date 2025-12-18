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

### Positional Encoding

- **SWA (Local Attention)**: RoPE使用
- **Senri Memory (Global Attention)**: NoPE (No Positional Encoding)

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
# 推論時（複数メモリ）
M = torch.zeros(batch, heads, hidden_dim, head_dim, head_dim)  # 基底ごと
z = torch.zeros(batch, heads, hidden_dim, head_dim)

# keyの割り当て（単位行列基底なので、最大絶対値の次元）
basis_idx = k.abs().argmax(dim=-1)  # [batch, seq]

# top-k選択
scores = q.abs()  # [batch, heads, seq, head_dim]
top_k_indices = scores.topk(k=top_k_memories, dim=-1).indices
```

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

# 3. 実験の実行
!python scripts/colab.py --experiment train --epochs 3
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
