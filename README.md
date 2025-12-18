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
- **SmolLM-135M** (Apache 2.0 License)
  - Hidden size: 576
  - Layers: 30
  - Attention heads: 9
  - KV heads: 3 (GQA)
  - Head dim: 64

### Small Model Philosophy

Senriアーキテクチャは**コンテキスト長に関係なく固定サイズのメモリ**を使用します：

```python
M = torch.zeros(batch, heads, head_dim, head_dim)  # 学習時: ~0.1MB/層
M = torch.zeros(batch, heads, hidden_size, head_dim, head_dim)  # 推論時
```

**理論上は小型モデルでも超長文コンテキストを処理可能**です。

### Layer Configuration

```
Layer 0-9:   SWA only (Lower Decoder)
Layer 10:    SWA + Senri Memory (Group 1)
Layer 11-19: SWA only
Layer 20:    SWA + Senri Memory (Group 2)
Layer 21-29: SWA only
```

### Senri Memory Mechanism

#### 1. 直交基底ベクトル（固定）
```
B = I ∈ R^(d×d)  (単位行列 = 標準直交基底)
d = hidden_size = 576
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

#### 4. Query→メモリ選択（推論時のみ）
```python
# queryとの類似度でtop-kメモリを選択
scores = [|<q, b_i>| for i in 1..d]
selected = top_k(scores, k=top_k_memories)
```

#### 5. 出力計算
```python
# 学習時
output = (M @ q) / (z^T @ q + eps) + local_attention_output

# 推論時
memory_output = Σ_{i ∈ selected} softmax(scores[i]) * (M_i @ q) / (z_i^T @ q + eps)
output = memory_output + local_attention_output
```

## Training Data

### PG19 (Project Gutenberg Books)

HSA論文の知見に基づき、**実効コンテキスト長が長いデータ**を使用。

```yaml
# config/training.yaml
dataset:
  name: "pg19"               # 長編書籍データセット
  niah_ratio: 0.01           # 1%のNIAHタスク混入
  max_train_samples: 1000    # サンプル制限（高速実験用）
```

### NIAH (Needle-in-a-Haystack) Task Injection

HSA論文 Section 3.2: 学習サンプルの1%にNIAHタスクを混入し、長距離検索能力を強化。

```
[長い文章...]
The secret key is: KEY-ABC12345
[さらに長い文章...]

Question: What is the secret key mentioned above?
Answer: KEY-ABC12345
```

## Hyperparameters

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `sliding_window_size` | 1024 | SWAのウィンドウサイズ |
| `chunk_size` | 64 | メモリ更新のチャンクサイズ |
| `top_k_memories` | 64 | 推論時に選択するメモリ数 |
| `num_memory_layers` | 2 | Senri Memoryを持つレイヤー数 |
| `memory_layer_interval` | 10 | メモリレイヤー間のインターバル |
| `niah_ratio` | 0.01 | NIAHタスク混入率 |

## Project Structure

```
senri-llm/
├── src/
│   ├── __init__.py
│   ├── configuration_senri.py      # SenriConfig (extends LlamaConfig)
│   ├── modeling_senri.py           # SenriForCausalLM
│   ├── decoder.py                  # SenriDecoderLayer
│   ├── attention/
│   │   ├── __init__.py
│   │   └── senri_attention.py      # Senri Memory Attention
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── base_memory.py          # TensorMemory (学習用)
│   │   ├── orthogonal_memory.py    # OrthogonalBasisMemory (推論用)
│   │   └── senri_memory.py         # 統合インターフェース
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # SenriTrainer
│   │   └── config.py               # TrainingConfig
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── niah.py                 # Needle-in-a-Haystack評価
│   │   └── multi_query.py          # Multi-Query評価
│   ├── data/
│   │   └── loader.py               # データセットローダー
│   └── config/
│       └── loader.py               # 設定ファイルローダー
├── scripts/
│   ├── convert_to_senri.py         # SmolLM→Senri変換スクリプト
│   └── colab.py                    # Colab実験スクリプト
├── config/
│   ├── model.yaml                  # モデル設定
│   ├── training.yaml               # 学習設定
│   └── experiment.yaml             # 実験設定
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
from scripts.convert_to_senri import convert_to_senri

model = convert_to_senri("HuggingFaceTB/SmolLM-135M")
```

### Training (Google Colab)
```bash
# Clone and install
!git clone https://github.com/YOUR_USERNAME/senri-llm.git
%cd senri-llm
!pip install -e .

# Run training
!python scripts/colab.py train
```

### Evaluation
```bash
!python scripts/colab.py eval
```

### Inference
```python
from src import SenriForCausalLM, SenriConfig

model = SenriForCausalLM.from_pretrained("path/to/senri-model")

# 推論時は自動的に直交基底ルーティングが有効化
model.eval()
output = model.generate(input_ids, max_length=100000)
```

### Training Mode
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
- [SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM-135M)

## License

Apache 2.0
