# Senri-LLM

**Senri**: Orthogonal Basis Routed Infinite Attention for Ultra-Long Context

Senriは、Infini AttentionとHSA (Hierarchical Sparse Attention) の概念を組み合わせ、直交基底ベースの動的テンソル選択機構を導入した新しいアテンション手法です。

## Core Idea

従来のInfini Attentionは単一のテンソル積メモリ（M = Σ v⊗k）に全ての過去情報を圧縮します。Senriは複数のテンソル積メモリを管理し、推論時に直交基底を用いて動的にメモリを選択することで、効率的な長文脈処理を実現します。

### Key Innovation

| 項目 | 学習時 | 推論時 |
|------|--------|--------|
| メモリ構造 | 単一テンソル積（通常のInfini Attention） | 複数テンソル積（直交基底でルーティング） |
| Attention | Global Attention (NoPE) | Top-k メモリ選択 + Sparse Attention |
| 位置エンコーディング | SWA: RoPE, Global: NoPE | 同左 |

## Architecture

### Base Model
- **Qwen2.5-0.5B** (Apache 2.0 License)
  - Hidden size: 896
  - Layers: 24
  - Attention heads: 14
  - KV heads: 2 (GQA)

### Layer Configuration

HSA論文に倣い、以下の構成を採用：

```
Lower Decoder (Layer 0-11): SWA only
Upper Decoder (Layer 12-23):
  - Layer 12: SWA + Senri Memory (Group 1)
  - Layer 13-15: SWA only
  - Layer 16: SWA + Senri Memory (Group 2)
  - Layer 17-19: SWA only
  - Layer 20: SWA + Senri Memory (Group 3)
  - Layer 21-23: SWA only
```

- **Lower Decoder**: 全レイヤーがSliding Window Attention (SWA) のみ
- **Upper Decoder**: Gグループに分割、各グループの最初の1層のみSWA + Senri Memory

### Senri Memory Mechanism

#### 1. 直交基底ベクトル（固定）
```
B = I ∈ R^(d×d)  (単位行列 = 標準直交基底)
d = hidden_size = 896
```

#### 2. テンソル積メモリ
```
学習時: M = Σ v ⊗ k  (単一メモリ、通常のInfini Attention)
推論時: M_i for i = 1..d  (基底ごとに分離されたメモリ)
```

#### 3. Key→基底への割り当て（推論時のみ）
```python
# keyを最も類似した基底に割り当て
assignment(k) = argmax_i |<k, b_i>|
# 単位行列の場合、最大絶対値を持つ次元のインデックス
assignment(k) = argmax_i |k_i|
```

#### 4. メモリ更新タイミング
- SWAウィンドウが満杯になった時
- シーケンス終了時

#### 5. Query→メモリ選択（推論時のみ）
```python
# queryとの類似度でtop-kメモリを選択
scores = [|<q, b_i>| for i in 1..d]
selected = top_k(scores, k=top_k_memories)
```

#### 6. 出力計算
```python
# 学習時
output = (M @ q) / (z^T @ q + eps) + local_attention_output

# 推論時
memory_output = Σ_{i ∈ selected} softmax(scores[i]) * (M_i @ q) / (z_i^T @ q + eps)
output = memory_output + local_attention_output
```

## Hyperparameters

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `sliding_window_size` | 4096 | SWAのウィンドウサイズ |
| `chunk_size` | 64 | メモリ更新のチャンクサイズ |
| `top_k_memories` | 64 | 推論時に選択するメモリ数 |
| `num_memory_layers` | 3 | Senri Memoryを持つレイヤー数 |
| `memory_layer_interval` | 4 | メモリレイヤー間のインターバル |

## Project Structure

```
senri-llm/
├── src/
│   ├── __init__.py
│   ├── configuration_senri.py      # SenriConfig (extends Qwen2Config)
│   ├── modeling_senri.py           # SenriForCausalLM
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── senri_attention.py      # Senri Memory Attention
│   │   └── sliding_window.py       # SWA (Qwen2から流用)
│   └── memory/
│       ├── __init__.py
│       └── tensor_memory.py        # 直交基底ベースのテンソル積メモリ
├── scripts/
│   ├── convert_qwen_to_senri.py    # Qwen2.5→Senri変換スクリプト
│   └── train.py                    # 学習スクリプト
├── tests/
│   └── test_senri.py
├── CLAUDE.md                       # AI開発ガイドライン
├── README.md
├── setup.py
└── requirements.txt
```

## Installation

```bash
pip install -e .
```

## Usage

### Model Conversion
```python
from src.convert import convert_qwen_to_senri

model = convert_qwen_to_senri("Qwen/Qwen2.5-0.5B")
```

### Inference
```python
from src import SenriForCausalLM, SenriConfig

config = SenriConfig.from_pretrained("path/to/senri-model")
model = SenriForCausalLM.from_pretrained("path/to/senri-model")

# 推論時は自動的に直交基底ルーティングが有効化
model.eval()
output = model.generate(input_ids, max_length=100000)
```

### Training
```python
# 学習時は通常のInfini Attention（単一メモリ）として動作
model.train()
output = model(input_ids, labels=labels)
loss = output.loss
```

## Comparison with Related Work

| 手法 | メモリ構造 | チャンク選択 | 学習 |
|------|----------|-------------|------|
| Infini Attention | 単一テンソル積 | なし（全参照） | End-to-end |
| HSA | チャンク単位KVキャッシュ | 学習されたretrieval score | End-to-end |
| **Senri** | 複数テンソル積 | 直交基底への射影（学習不要） | End-to-end (単一メモリとして) |

## References

- [Infini-attention: Infinite Context Transformers with Infinite Attention](https://arxiv.org/abs/2404.07143)
- [HSA-UltraLong: Every Token Counts: Generalizing 16M Ultra-Long Context](https://arxiv.org/abs/2511.23319)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)

## License

Apache 2.0 (following Qwen2.5)
