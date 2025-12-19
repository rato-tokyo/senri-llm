# Senri-LLM Development Guidelines

## Project Overview

Senri-LLMは、**SmolLM-135M**をベースに、シンプルなテンソル積メモリを実装するプロジェクトです。

**現在のステータス**: 3段階学習方式（蒸留→メモリ学習→全体調整）

## 3段階学習アプローチ

### 問題背景

線形Attentionレイヤーは、単純にSoftmax Attentionレイヤーと置換するだけでは機能しません。
出力分布の違いがノイズとして後続レイヤーに伝播し、モデル全体の性能が劣化します。

### 解決策: 3段階学習

**Stage 1: Layer Distillation**
- メモリレイヤーの出力をベースモデルのAttention出力に近づける
- Loss: `MSE(memory_output, base_output.detach())`
- 学習率: 高め (1e-4)

**Stage 2: Memory-only Fine-tuning**
- メモリレイヤー以外をフリーズ
- 言語モデリング損失で学習
- 学習率: 中程度 (5e-5)

**Stage 3: Full Fine-tuning**
- 全パラメータをアンフリーズ
- 低学習率で全体を調整
- 学習率: 低め (1e-5)

### 使用方法

```bash
# 3段階学習を実行
python scripts/colab.py train

# 評価
python scripts/colab.py eval

# 動作確認
python scripts/colab.py test
```

## Architecture Specification

### Core Concept

```
入力 → QKV投影 → メモリ更新 → メモリ検索 → 出力投影
```

**特徴**:
- メモリレイヤーではローカルAttentionなし（メモリのみ）
- 位置エンコーディングなし（NoPE）
- バッチ共有メモリ `[d, d]`
- 完全detach（安定性優先）
- **update → retrieve 順序**

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

## Key Classes

```python
# src/training/three_stage_trainer.py
class ThreeStageTrainer:
    """3段階学習を実行するトレーナー"""

# src/memory/base_memory.py
class TensorMemory:
    """バッチ共有テンソル積メモリ [d, d]"""

# src/attention/senri_attention.py
class SenriAttention:
    """メモリのみのAttention（GQA対応）"""

# src/configuration_senri.py
class SenriConfig(LlamaConfig):
    """Senriモデル設定"""
```

## Configuration Management

設定は `config/*.yaml` で管理。

```yaml
# config/training.yaml
three_stage:
  stage1:
    enabled: true
    num_epochs: 1
    learning_rate: 1.0e-4
  stage2:
    enabled: true
    num_epochs: 2
    learning_rate: 5.0e-5
  stage3:
    enabled: true
    num_epochs: 1
    learning_rate: 1.0e-5
```

## Memory Lifecycle

### Context Manager API（推奨）

```python
with model.new_sequence():
    output = model(input_ids)
    # or
    output = model.generate(input_ids, ...)
```

### 自動リセット

```python
# past_key_values が None の場合、自動でリセット
output = model(input_ids)
```

## Dependencies

```
torch>=2.0.0
transformers>=4.36.0
accelerate>=0.25.0
datasets>=2.14.0
tqdm
```

## Project Structure

```
senri-llm/
├── config/
│   ├── model.yaml          # モデルアーキテクチャ
│   ├── training.yaml       # 3段階学習設定
│   └── experiment.yaml     # 実験設定
├── scripts/
│   ├── colab.py            # メイン実行スクリプト
│   ├── convert_to_senri.py # モデル変換
│   └── poc_memory.py       # PoCテスト
├── src/
│   ├── training/
│   │   └── three_stage_trainer.py  # 3段階トレーナー
│   ├── memory/
│   │   └── base_memory.py  # TensorMemory
│   ├── attention/
│   │   └── senri_attention.py  # SenriAttention
│   ├── modeling_senri.py   # SenriForCausalLM
│   └── configuration_senri.py  # SenriConfig
└── docs/
    └── findings-memory-layer-integration.md  # 知見
```

## Experiment Environment

### Google Colab（推奨）
- GPU: T4 / A100
- スクリプト: `scripts/colab.py`

### ローカル
- 簡単な動作確認のみ
- テスト: `pytest tests/`
