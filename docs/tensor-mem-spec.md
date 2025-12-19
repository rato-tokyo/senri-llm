# tensor-mem 仕様書

## 概要

**tensor-mem**は、線形Attentionとテンソル積メモリを組み合わせた軽量ライブラリです。

### 特徴

- **線形計算量**: O(n²) → O(n) でメモリと計算量を削減
- **無限コンテキスト**: テンソル積メモリによる長期記憶
- **HuggingFace互換**: `transformers`ライブラリと互換性のあるインターフェース
- **シンプル設計**: ローカルAttentionなし、純粋な線形Attention + メモリ

---

## 理論的背景

### 線形AttentionとFast Weight Programmersの等価性

線形Attentionとテンソル積メモリは数学的に等価です。

```
# 標準的な線形Attention
output = φ(Q) @ (φ(K)^T @ V)

# テンソル積メモリとして表現
M = M + φ(K)^T @ V    # 更新（外積の累積）
output = φ(Q) @ M      # 検索
```

両者とも `φ(K)^T @ V`（外積の累積）という同一の操作を行います。

**参考文献**:
- [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174)
- [Infini-attention](https://arxiv.org/abs/2404.07143)

### 活性化関数: ELU + 1

線形Attentionでは、softmaxの代わりに特徴写像 φ(x) を使用します。tensor-memでは `ELU(x) + 1` を採用:

```python
φ(x) = ELU(x) + 1 = max(0, x) + min(0, exp(x) - 1) + 1
```

**利点**:
- 全ての値が正（>= 1 for x >= 0, > 0 for x < 0）
- 正規化項の安定性を保証
- 計算効率が良い

---

## アーキテクチャ

### コンポーネント構成

```
tensor-mem/
├── tensor_mem/
│   ├── __init__.py
│   ├── memory.py          # TensorMemory クラス
│   ├── attention.py       # LinearMemoryAttention クラス
│   └── utils.py           # ヘルパー関数
├── tests/
├── examples/
├── pyproject.toml
└── README.md
```

---

## API仕様

### 1. TensorMemory

テンソル積メモリの核となるクラス。

```python
class TensorMemory(nn.Module):
    """
    テンソル積メモリ（バッチ共有）。

    メモリ形状: [memory_dim, memory_dim]
    正規化係数: [memory_dim]
    """

    def __init__(
        self,
        memory_dim: int,
        eps: float = 1e-6,
    ):
        """
        Args:
            memory_dim: メモリの次元（通常はhidden_size）
            eps: 数値安定性のためのイプシロン
        """
```

#### プロパティ

| プロパティ | 型 | 説明 |
|-----------|-----|------|
| `is_initialized` | `bool` | メモリが初期化済みかどうか |
| `is_empty` | `bool` | メモリが空かどうか |
| `memory_dim` | `int` | メモリの次元 |

#### メソッド

##### `reset(device, dtype)`

メモリを初期化（ゼロクリア）。

```python
def reset(
    self,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    """
    メモリをリセット。

    Args:
        device: テンソルのデバイス
        dtype: テンソルのデータ型
    """
```

##### `update(keys, values)`

メモリにキー・バリューペアを追加。

```python
def update(
    self,
    keys: torch.Tensor,    # [batch, seq, memory_dim]
    values: torch.Tensor,  # [batch, seq, memory_dim]
) -> None:
    """
    メモリを更新。

    数式: M = M + σ(K)^T @ V / (batch * seq)
          z = z + Σσ(K) / batch

    Args:
        keys: キーテンソル [batch, seq, memory_dim]
        values: バリューテンソル [batch, seq, memory_dim]
    """
```

##### `retrieve(queries)`

メモリからクエリに基づいて検索。

```python
def retrieve(
    self,
    queries: torch.Tensor,  # [batch, seq, memory_dim]
) -> torch.Tensor:          # [batch, seq, memory_dim]
    """
    メモリから検索。

    数式: output = (σ(Q) @ M) / (σ(Q) @ z)

    Args:
        queries: クエリテンソル [batch, seq, memory_dim]

    Returns:
        output: 検索結果 [batch, seq, memory_dim]
    """
```

#### 内部状態

| バッファ | 形状 | 説明 |
|---------|------|------|
| `M` | `[memory_dim, memory_dim]` | メモリ行列（外積の累積） |
| `z` | `[memory_dim]` | 正規化係数 |

#### 使用例

```python
from tensor_mem import TensorMemory

# 初期化
memory = TensorMemory(memory_dim=768, eps=1e-6)
memory.reset(device="cuda", dtype=torch.float16)

# 更新
keys = torch.randn(4, 128, 768, device="cuda", dtype=torch.float16)
values = torch.randn(4, 128, 768, device="cuda", dtype=torch.float16)
memory.update(keys, values)

# 検索
queries = torch.randn(4, 32, 768, device="cuda", dtype=torch.float16)
output = memory.retrieve(queries)  # [4, 32, 768]
```

---

### 2. LinearMemoryAttention

Attentionレイヤーの完全な実装。HuggingFaceモデルのAttentionレイヤーと置換可能。

```python
class LinearMemoryAttention(nn.Module):
    """
    線形Attention + テンソル積メモリ。

    特徴:
    - GQA（Grouped Query Attention）対応
    - 位置エンコーディングなし（NoPE）
    - HuggingFace互換のインターフェース
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        eps: float = 1e-6,
        bias: bool = True,
        output_bias: bool = False,
    ):
        """
        Args:
            hidden_size: 隠れ層の次元
            num_attention_heads: クエリヘッドの数
            num_key_value_heads: KVヘッドの数（GQA用、Noneならnum_attention_headsと同じ）
            head_dim: ヘッドあたりの次元（Noneならhidden_size // num_attention_heads）
            eps: 数値安定性のためのイプシロン
            bias: Q/K/V投影にバイアスを使用するか
            output_bias: 出力投影にバイアスを使用するか
        """
```

#### メソッド

##### `forward(hidden_states, **kwargs)`

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:
    """
    Forward pass。

    Args:
        hidden_states: 入力テンソル [batch, seq, hidden_size]
        attention_mask: (未使用) API互換性のため
        position_ids: (未使用) API互換性のため
        past_key_value: (未使用) API互換性のため
        output_attentions: (未使用) API互換性のため
        use_cache: (未使用) API互換性のため

    Returns:
        output: [batch, seq, hidden_size]
        None: attention weights (線形Attentionでは計算しない)
        None: cache (メモリで代替)
    """
```

##### `reset_memory(device, dtype)`

```python
def reset_memory(
    self,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    """新しいシーケンスのためにメモリをリセット。"""
```

#### 使用例

```python
from tensor_mem import LinearMemoryAttention

# 初期化（LLaMA-7B相当の設定）
attn = LinearMemoryAttention(
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=32,  # MHA
    eps=1e-6,
)

# forward
hidden_states = torch.randn(2, 512, 4096)
output, _, _ = attn(hidden_states)

# 新しいシーケンス開始時
attn.reset_memory()
```

---

### 3. ユーティリティ関数

#### `elu_plus_one(x)`

```python
def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """
    ELU + 1 活性化関数。

    φ(x) = ELU(x) + 1

    Args:
        x: 入力テンソル

    Returns:
        活性化後のテンソル（全要素が正）
    """
```

#### `repeat_kv(hidden_states, n_rep, head_dim)`

```python
def repeat_kv(
    hidden_states: torch.Tensor,
    n_rep: int,
    head_dim: int,
) -> torch.Tensor:
    """
    GQA用にKVヘッドを繰り返す。

    Args:
        hidden_states: [batch, seq, num_kv_heads * head_dim]
        n_rep: 繰り返し回数
        head_dim: ヘッドあたりの次元

    Returns:
        [batch, seq, num_kv_heads * n_rep * head_dim]
    """
```

---

## 設計仕様

### NoPE（位置エンコーディングなし）

メモリレイヤーでは位置エンコーディングを使用しません。

**理由**:
- 線形Attentionでは位置情報がメモリに混入すると長期記憶が劣化
- 純粋な内容ベースの検索が可能

---

## 使用例

### TensorMemory（低レベルAPI）

```python
import torch
from tensor_mem import TensorMemory

memory = TensorMemory(memory_dim=768)
memory.reset(device="cuda", dtype=torch.float16)

keys = torch.randn(2, 100, 768, device="cuda", dtype=torch.float16)
values = torch.randn(2, 100, 768, device="cuda", dtype=torch.float16)
queries = torch.randn(2, 10, 768, device="cuda", dtype=torch.float16)

memory.update(keys, values)
output = memory.retrieve(queries)  # [2, 10, 768]
```

### LinearMemoryAttention（高レベルAPI）

```python
import torch
from tensor_mem import LinearMemoryAttention

attn = LinearMemoryAttention(
    hidden_size=768,
    num_attention_heads=12,
    num_key_value_heads=12,
)

hidden = torch.randn(2, 100, 768)
output, _, _ = attn(hidden)  # [2, 100, 768]

# 新しいシーケンス開始時
attn.reset_memory()
```

### カスタムモデルへの統合

```python
import torch.nn as nn
from tensor_mem import LinearMemoryAttention

class CustomTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = LinearMemoryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
        )
        self.mlp = MLP(hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
```

---

## 制限事項

1. **位置情報**: 位置エンコーディングなし（NoPE）
2. **Attention重み**: 線形Attentionのため、attention weightsは出力されない

---

## インストール

```bash
pip install tensor-mem
```

開発版:
```bash
pip install git+https://github.com/your-org/tensor-mem.git
```

---

## 依存関係

```
torch>=2.0.0
```

---

## ライセンス

MIT License

---

## 引用

```bibtex
@software{tensor_mem,
  title = {tensor-mem: Linear Attention with Tensor Product Memory},
  year = {2024},
  url = {https://github.com/your-org/tensor-mem}
}
```

関連論文:
```bibtex
@article{schlag2021linear,
  title={Linear Transformers Are Secretly Fast Weight Programmers},
  author={Schlag, Imanol and Irie, Kazuki and Schmidhuber, J{\"u}rgen},
  journal={ICML},
  year={2021}
}

@article{munkhdalai2024infini,
  title={Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention},
  author={Munkhdalai, Tsendsuren and others},
  journal={arXiv preprint arXiv:2404.07143},
  year={2024}
}
```
