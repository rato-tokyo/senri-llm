# Senri-LLM Development Guidelines

## Project Overview

Senri-LLMは、**SmolLM-135M**をベースに、シンプルなテンソル積メモリを実装するプロジェクトです。

**現在のステータス**: 2段階学習方式（メモリ学習→全体調整）

## 2段階学習アプローチ

### 解決策: 2段階学習

**Stage 1: Memory-only Fine-tuning**
- メモリレイヤー以外をフリーズ
- 言語モデリング損失で学習
- 学習率: 中程度 (5e-5)

**Stage 2: Full Fine-tuning**
- 全パラメータをアンフリーズ
- 低学習率で全体を調整
- 学習率: 低め (1e-5)

### 使用方法

```bash
# 2段階学習を実行
python scripts/colab.py train

# 評価
python scripts/colab.py eval

# 動作確認
python scripts/colab.py test
```

## Architecture Specification

### Core Concept

```
入力 → QKV投影 → スケーリング → メモリ更新 → メモリ検索 → 出力投影
```

**特徴**:
- メモリレイヤーではローカルAttentionなし（メモリのみ）
- 位置エンコーディングなし（NoPE）
- バッチ共有メモリ `[d, d]`
- 完全detach（安定性優先）
- **update → retrieve 順序**
- **スケーリング係数**: `1/sqrt(hidden_size)` で数値安定性を確保

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
Layer 0-14:  Standard Attention (RoPE)
Layer 15:    Memory-only Attention (NoPE)
Layer 16-19: Standard Attention (RoPE)
Layer 20:    Memory-only Attention (NoPE)
Layer 21-24: Standard Attention (RoPE)
Layer 25:    Memory-only Attention (NoPE)
Layer 26-29: Standard Attention (RoPE)
```

### Memory Layer Configuration

- `num_memory_layers`: 3
- `first_memory_layer`: 15
- `memory_layer_interval`: 5
- メモリレイヤーのインデックス: [15, 20, 25]

## Key Classes

```python
# src/training/two_stage_trainer.py
class TwoStageTrainer:
    """2段階学習を実行するトレーナー"""

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
two_stage:
  stage1:
    enabled: true
    num_epochs: 2
    learning_rate: 5.0e-5
  stage2:
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

## 重要な知見

### L2正規化は使用禁止

L2正規化はベクトルの magnitude 情報を破壊し、モデルの言語能力を崩壊させます。
代わりにスケーリング係数を使用してください。

```python
# ❌ 絶対にやってはいけない
keys = F.normalize(keys, p=2, dim=-1)

# ✅ 正しいアプローチ
scale = 1.0 / (hidden_size ** 0.5)
keys = keys * scale
```

詳細は `docs/layer-replacement-guidelines.md` を参照。

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
│   ├── training.yaml       # 2段階学習設定
│   └── experiment.yaml     # 実験設定
├── scripts/
│   ├── colab.py            # メイン実行スクリプト
│   ├── convert_to_senri.py # モデル変換
│   └── debug_generation.py # デバッグ用
├── src/
│   ├── training/
│   │   └── two_stage_trainer.py   # 2段階トレーナー
│   ├── memory/
│   │   └── base_memory.py  # TensorMemory
│   ├── attention/
│   │   └── senri_attention.py  # SenriAttention
│   ├── modeling_senri.py   # SenriForCausalLM
│   └── configuration_senri.py  # SenriConfig
└── docs/
    ├── findings-memory-layer-integration.md  # 知見
    └── layer-replacement-guidelines.md       # レイヤー置換ガイドライン
```

## Experiment Environment

### Google Colab（推奨）
- GPU: T4 / A100
- スクリプト: `scripts/colab.py`

### ローカル
- 簡単な動作確認のみ
- テスト: `pytest tests/`
