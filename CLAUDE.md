# Senri-LLM Development Guidelines

## Project Overview

Senri-LLMは、**SmolLM-135M**をベースに、シンプルなテンソル積メモリを実装するプロジェクトです。

**現在のステータス**: 最シンプル化版（new-llm準拠、メモリのみ、ローカルAttentionなし）

## 2024-12-19 シンプル化リファクタリング

**NaN問題解決のため、new-llm準拠の最シンプル設計に変更しました。**

### 変更前 vs 変更後

| 項目 | 変更前 | 変更後 |
|------|--------|--------|
| **メモリ形状** | `[batch, heads, d, d]` | `[d, d]`（バッチ共有） |
| **ローカルAttention** | Softmax + RoPE | **なし** |
| **セグメント処理** | forループ | **なし** |
| **メモリゲート** | あり | **なし** |
| **勾配** | 部分detach | **完全detach** |
| **正規化** | `/seq_len` | **`/(batch*seq)`** |

### ロールバック方法

```bash
git log --oneline  # コミットハッシュを確認
git checkout <hash-before-simplification> -- src/
```

## Architecture Specification

### Core Concept（最シンプル版）

```
入力 → QKV投影 → メモリ検索 → メモリ更新 → 出力投影
```

**特徴**:
- ローカルAttentionなし（メモリのみ）
- 位置エンコーディングなし（NoPE）
- バッチ共有メモリ `[d, d]`
- 完全detach（安定性優先）

### Base Model: SmolLM-135M

| 項目 | 値 |
|------|-----|
| パラメータ数 | 135M |
| hidden_size | 576 |
| num_layers | 30 |
| num_attention_heads | 9 |
| num_key_value_heads | 3 (GQA) |
| head_dim | 64 |
| vocab_size | 49,152 |

### Layer Structure (30 layers total)

```
Layer 0-9:   Standard Attention (RoPE)
Layer 10:    Memory-only Attention (NoPE)
Layer 11-19: Standard Attention (RoPE)
Layer 20:    Memory-only Attention (NoPE)
Layer 21-29: Standard Attention (RoPE)
```

### Memory Layer Configuration

- `num_memory_layers`: 2
- `first_memory_layer`: 10
- `memory_layer_interval`: 10
- メモリレイヤーのインデックス: [10, 20]

## Implementation Details

### TensorMemory（バッチ共有、シングル）

```python
# メモリ構造
M = torch.zeros(memory_dim, memory_dim)  # バッチ共有
z = torch.zeros(memory_dim)              # 正規化係数

# 更新: M = M + σ(K)^T @ V / (batch * seq)
sigma_keys = elu_plus_one(keys)
delta_M = torch.einsum("bsd,bse->de", sigma_keys, values)
delta_M = delta_M / (batch_size * seq_len)
M = (M + delta_M).detach()  # 完全detach

# 検索: output = (σ(Q) @ M) / (σ(Q) @ z)
sigma_queries = elu_plus_one(queries)
numerator = torch.matmul(sigma_queries, M)
denominator = torch.matmul(sigma_queries, z).clamp(min=eps)
output = numerator / denominator.unsqueeze(-1)
```

### SenriAttention（メモリのみ）

```python
class SenriAttention(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, layer_idx=0):
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.memory = SenriMemory(memory_dim=hidden_size, eps=eps)

    def forward(self, hidden_states, ...):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if self.training:
            self.memory.reset(device, dtype)

        output = self.memory.retrieve(q)
        self.memory.update(k, v)

        return self.o_proj(output), None, None
```

### Key Classes

```python
# src/memory/base_memory.py
class TensorMemory:
    """バッチ共有テンソル積メモリ [d, d]"""

# src/memory/senri_memory.py
class SenriMemory:
    """TensorMemoryのラッパー"""

# src/attention/senri_attention.py
class SenriAttention:
    """メモリのみのAttention（ローカルなし）"""

# src/configuration_senri.py
class SenriConfig(LlamaConfig):
    num_memory_layers: int = 2
    first_memory_layer: int = 10
    memory_layer_interval: int = 10
    memory_eps: float = 1e-6
```

## new-llmとの対応

本シンプル化はnew-llmプロジェクトの設計を参考にしています。

| new-llm | senri-llm (現在) |
|---------|------------------|
| `CompressiveMemory` | `TensorMemory` |
| `SenriAttention` (layers/senri.py) | `SenriAttention` |
| `causal_linear_attention` | **なし**（メモリのみ） |
| バッチ共有 `[d, d]` | バッチ共有 `[d, d]` |
| 完全detach | 完全detach |
| `/(batch*seq)` | `/(batch*seq)` |

## Training vs Inference

| 項目 | 学習時 | 推論時 |
|------|--------|--------|
| **メモリリセット** | 毎forward() | 手動 |
| **勾配** | なし（detach） | なし |

```python
# 学習時: forward()内で自動リセット
model.train()

# 推論時: 手動リセット
model.eval()
model.reset_memory(device, dtype)
outputs = model.generate(**inputs)
```

## Configuration Management

設定は `config/*.yaml` で管理。スクリプトへの引数追加は禁止。

```bash
python scripts/colab.py train
python scripts/colab.py test
python scripts/colab.py eval
```

## Dependencies

```
torch>=2.0.0
transformers>=4.36.0
accelerate>=0.25.0
datasets>=2.14.0
```

## Common Pitfalls

### 1. 推論時のメモリリセット忘れ
```python
# Bad
model.eval()
outputs = model.generate(...)  # メモリが前回の状態のまま

# Good
model.eval()
model.reset_memory(device, dtype)  # 明示的にリセット
outputs = model.generate(...)
```

### 2. 学習時のretrieve→update順序
```python
# 現在の実装（正しい）
output = self.memory.retrieve(q)  # まず検索
self.memory.update(k, v)          # その後更新
```

学習時は毎回メモリがリセットされるため、retrieve→update順序では
最初のretrieveがゼロを返す。これは意図した動作（学習時はメモリなしで学習）。

## Experiment Environment

### Google Colab（推奨）
- GPU: T4 / A100
- スクリプト: `scripts/colab.py`

### ローカル
- 簡単な動作確認のみ
- テスト: `pytest tests/`
