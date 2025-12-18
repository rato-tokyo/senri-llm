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

### Why Memory-Only (No Local Attention)

**線形AttentionとテンソルメモリはNoPEにおいて数学的に等価です。**

```
# 線形Attention
output = φ(Q) @ (φ(K)^T @ V)

# テンソル積メモリ
M = M + φ(K)^T @ V    # 更新
output = φ(Q) @ M      # 検索
```

両者とも `φ(K)^T @ V`（外積の累積）という同一の操作を行います。

**参考文献**:
- [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/pdf/2102.11174) - 線形Attentionと外積連想記憶の数学的等価性を証明
- [Infini-attention](https://arxiv.org/abs/2404.07143) - Google DeepMindによる線形Attention＋圧縮メモリ

**NoPE（位置エンコーディングなし）の場合**:
- ローカルAttentionとメモリの区別が不要
- 同じ外積累積操作を2回行う意味がない
- メモリのみで十分（シンプルかつ効率的）

**RoPEを使う場合のみ分離が意味を持つ**:
- ローカル: RoPE適用（位置情報あり）
- メモリ: RoPEなし（位置に依存しない長期記憶）

現在のsenri-llmはNoPE設計のため、メモリのみの実装が理論的に正しい選択です。

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

### SenriAttention（メモリのみ、GQA対応）

```python
class SenriAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, eps=1e-6, layer_idx=0):
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        # GQA projections (SmolLMと同じ構造)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)
        self.memory = TensorMemory(memory_dim=hidden_size, eps=eps)

    def forward(self, hidden_states, ...):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # KVをQに合わせて展開
        k = self._repeat_kv(k, self.num_key_value_groups)
        v = self._repeat_kv(v, self.num_key_value_groups)

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

# src/attention/senri_attention.py
class SenriAttention:
    """メモリのみのAttention（GQA対応）"""

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
