# Senri-LLM Development Guidelines

## Project Overview

Senri-LLMは、**SmolLM-135M**をベースに、Infini Attentionを実装するプロジェクトです。

**現在のステータス**: シンプル化フェーズ（単一テンソル積メモリで基本動作を確認中）

## Small Model Philosophy - 重要

### なぜ小型モデルを選択するのか

Senriアーキテクチャは、**コンテキスト長に関係なく固定サイズのメモリ**を使用します。

```python
# メモリサイズは固定（コンテキスト長に依存しない）
M = torch.zeros(batch, heads, head_dim, head_dim)  # 学習・推論共通
```

これにより、**たとえ16M tokensのコンテキストであっても**、最終的には：
- テンソル積メモリ（固定サイズ）
- SWAウィンドウ（固定サイズ）

に収まるため、**理論上は小型モデルでも超長文コンテキストを処理可能**です。

### 複雑なタスク vs コンテキスト記憶

| 観点 | 大型モデル | 小型モデル |
|------|-----------|-----------|
| 複雑な推論 | 得意 | 限定的 |
| コンテキスト記憶 | 得意 | **Senriで対応可能** |
| 学習コスト | 高い | 低い |
| 実験速度 | 遅い | 速い |

**重要**: 「長文を記憶できるか」と「複雑な推論ができるか」は別の能力です。
Senriの目標は**コンテキスト記憶能力の証明**であり、複雑なタスクは対象外です。

### 小型モデルでの長文記憶実証の意義

1. **効率的な実験**: Colab T4で十分な学習・評価が可能
2. **アーキテクチャの検証**: メモリ機構が正しく動作することを証明
3. **スケーラビリティの示唆**: 小型で動けば大型でも動く

### 学習コンテキスト長の重要性

```
メモリを活用する学習のためには:
  学習コンテキスト長 > SWAウィンドウサイズ

現在の設定:
  max_length: 2048 tokens (学習)
  sliding_window_size: 1024 tokens (SWA)
  → メモリが積極的に使用される
```

## Training Data Strategy - 重要

### データセット: PG19（長編書籍）

HSA論文の知見に基づき、**実効コンテキスト長が長いデータ**を使用する。

| 項目 | WikiText-2 | PG19 |
|------|-----------|------|
| 平均長 | 数百トークン | 数万トークン |
| 長距離依存 | ほぼなし | キャラクター追跡、伏線 |
| 用途 | 短文テスト | **長文コンテキスト学習** |

### NIAH（Needle-in-a-Haystack）タスク混入

HSA論文 Section 3.2:
> "Synthetic ruler tasks are randomly inserted into **1% of training samples**"

```yaml
# config/training.yaml
dataset:
  name: "pg19"
  niah_ratio: 0.01  # 1%のNIAHタスク混入
```

NIAHタスクの形式:
```
[長い文章...]
The secret key is: KEY-ABC12345
[さらに長い文章...]

Question: What is the secret key mentioned above?
Answer: KEY-ABC12345
```

### HSA論文からの学び

1. **実効コンテキスト長が汎化に決定的**
   - 単に長いシーケンスではなく、前半を参照しないと後半が理解できない構造が必要
   - 学習データの実効長 > 32K で外挿性能が大幅改善

2. **SWA/メモリのシーソー効果**
   - 大きすぎるSWA窓 → メモリの学習が阻害される
   - 現在の設定（SWA=1024, 学習長=2048）は適切

3. **Warm-upは事前学習済みモデルでは不要**
   - HSAのWarm-upはゼロからの学習用
   - SmolLMは既に学習済みなので、直接長文データで学習可能

## Architecture Specification

### Core Concept（現在: シンプル化版）

```
学習時・推論時: 同じ単一テンソル積メモリ（標準Infini Attention）

将来: 直交基底ベースの動的テンソル積選択（推論時のみ）
```

**シンプル化の理由**: まず基本的なInfini Attentionが正しく動作することを確認する

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
Layer 0-9:   SWA only (Lower Decoder)
Layer 10:    SWA + Senri Memory (Group 1)
Layer 11-19: SWA only
Layer 20:    SWA + Senri Memory (Group 2)
Layer 21-29: SWA only
```

### Memory Layer Configuration

- `num_memory_layers`: 2 (Senri Memoryを持つレイヤー数)
- `first_memory_layer`: 10 (最初のメモリレイヤー)
- `memory_layer_interval`: 10 (メモリレイヤー間隔)
- メモリレイヤーのインデックス: [10, 20]

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

## 論文との相違点と意図的な妥協 - 重要

現在の実装は、Infini-Attention論文と以下の点で異なります。
これらは**意図的な妥協**であり、将来的な修正対象です。

### 相違点一覧

| 項目 | 論文 | 現在の実装 | 影響 | 優先度 |
|------|------|----------|------|--------|
| 活性化関数σ | ELU + 1 | ✅ 実装済み | - | - |
| 更新順序 | retrieve → update | ✅ 実装済み | - | - |
| 正規化分母 | clamp(min=eps) | ✅ 実装済み（new-llm準拠） | - | - |
| Delta更新 | あり（オプション） | 未実装 | 既存バインディング重複時に非効率 | 低 |
| セグメント処理 | チャンク単位で処理 | 全シーケンス一括 | メモリ蓄積が不完全 | 中 |

### 1. 活性化関数σ（実装済み）

**論文**: `σ(K) = ELU(K) + 1` を Keys/Queries に適用
**現在**: ✅ 実装済み（`base_memory.py` の `elu_plus_one` 関数）

```python
# 論文準拠の実装
def elu_plus_one(x):
    return F.elu(x) + 1

sigma_keys = elu_plus_one(keys)
sigma_queries = elu_plus_one(queries)
delta_M = einsum(values, sigma_keys)
```

**効果**: 全ての値が正になり、正規化の分母が常に正（NaN防止）

### 2. retrieve → update 順序（実装済み）

**論文の順序**（因果性維持）:
```
Segment s の処理:
1. M_{s-1} から retrieve（過去のメモリを検索）
2. ローカルAttention計算
3. ゲートで結合
4. M_s に update（現在のKVでメモリ更新）
```

**現在の実装**: ✅ 論文準拠
```
各forward-pass:
1. memory.retrieve(Q)     # 先に検索（メモリ空ならゼロを返す）
2. memory.update(K, V)    # 後で更新
```

**修正履歴（2024-12-19）**:
- 元々は `update → retrieve` 順序だったが、new-llmプロジェクトとの比較により修正
- 空メモリ検出ロジックを追加（`z.abs().sum() < eps` でゼロを返す）

### 3. セグメント単位処理の未実装

**論文**: 長いシーケンスをセグメントに分割し、セグメントごとに処理

```python
# 論文の方式
for segment in chunks(sequence, segment_length=2048):
    mem_output = memory.retrieve(segment_queries)  # 過去から検索
    local_output = local_attention(segment)
    output = gate * mem_output + (1-gate) * local_output
    memory.update(segment_keys, segment_values)    # その後更新
```

**現在**: 全シーケンスを一括処理

**影響**: retrieve→update順序を維持できない
**妥協の理由**: 実装の複雑さを避け、基本動作確認を優先

### 4. Delta更新の未実装

**論文のDelta更新**:
```python
# 既存バインディングを引いてから更新
retrieved = sigma_K @ M / (sigma_K @ z)
M = M + sigma_K.T @ (V - retrieved)
```

**現在**: Linear更新のみ
```python
M = M + K.T @ V
```

**影響**: 同じキーで繰り返し更新すると値が蓄積
**妥協の理由**: 論文でもLinear更新で良い結果を示している（Table 2）

### 将来的な修正ロードマップ

1. **Phase 1（完了）**: 基本動作確認
   - ✅ ELU+1活性化関数
   - ✅ retrieve→update順序
   - ✅ clamp(min=eps)による安定化

2. **Phase 2（将来）**: セグメント処理の実装
   - チャンク単位の処理ループ
   - セグメント間でのメモリ蓄積
   - BPTTによる勾配計算

3. **Phase 3（将来）**: 論文完全準拠
   - Delta更新のオプション実装
   - 32K以上の長文学習

### メモリ勾配についての設計判断

**論文の仕様（BPTT）**:
> "Each Infini-attention layer is trained with back-propagation through time (BPTT) by computing the gradient w.r.t the compressive memory states"

論文はセグメント処理を前提とし、セグメント間でメモリを通じて勾配が流れる。

**現在の実装（簡略化BPTT）**:
```python
# 累積状態のみdetach、現在の更新は勾配を維持
self.M = self.M.detach() + delta_M  # delta_Mには勾配あり
self.z = self.z.detach() + delta_z
```

- 各サンプル内で `update → retrieve` 経路の勾配は流れる
- サンプル間（セグメント間）の勾配は流れない（毎回リセット）
- new-llmは完全detach（勾配なし）だが、senri-llmは現在サンプル内で勾配維持

**比較**:
| 項目 | 論文（BPTT） | new-llm | senri-llm |
|------|-------------|---------|-----------|
| セグメント間勾配 | あり | なし | なし |
| サンプル内勾配 | あり | なし | **あり** |
| メモリ学習 | 完全 | ゲート経由のみ | ゲート＋メモリ経由 |

**妥協の理由**:
- 単一forward-pass学習ではセグメント間BPTTは不可能
- しかし、サンプル内勾配によりメモリの使い方は学習可能

### 現在の期待される精度

- **基本動作**: ✅ 期待できる
- **メモリの効果が観測できる**: ✅ 期待できる
- **NIAHタスクでの改善**: ✅ 期待できる

**根拠**:
- 論文のLinear更新（Delta更新なし）でも良好な結果を示している
- new-llmプロジェクト（完全detach）でも動作している
- senri-llmはサンプル内勾配を維持しており、new-llmより学習効率が良い可能性

## Training vs Inference Mode（現在: シンプル化版）

**現在のシンプル化版では、学習・推論で同じメモリを使用。**

### 比較表（シンプル化版）

| 項目 | 学習時 (`model.train()`) | 推論時 (`model.eval()`) |
|------|-------------------------|------------------------|
| **メモリ構造** | 単一テンソル積メモリ (`TensorMemory`) | 同じ |
| **メモリ形状** | `[batch, heads, head_dim, head_dim]` | 同じ |
| **メモリ更新** | 単純累積: `M = M + v ⊗ k` | 同じ |
| **勾配計算** | あり | なし (`torch.no_grad()`) |
| **メモリリセット** | 毎サンプル（自動） | 各シーケンス開始前（手動） |

### モード切り替えの注意点

```python
# 学習時: 自動的に毎サンプルでメモリリセット
model.train()

# 推論時: 各シーケンス前に手動でリセット
model.eval()
model.reset_memory(batch_size, device, dtype)
outputs = model.generate(**inputs)
```

### メモリリセットのタイミング - 重要

**⚠️ 2024年12月に発見した重大なバグとその修正**

#### 問題

毎回の`forward()`でメモリをリセットすると、長文推論でメモリが機能しない。

```python
# ❌ 間違い: 毎回リセット
def forward(self, ...):
    self.memory.reset(batch_size, device, dtype)  # 過去の情報が消える！
```

#### 正しい実装

```python
# ✅ 正しい: 条件付きリセット
def forward(self, ...):
    M = self.memory.memory.M  # TensorMemoryのM
    needs_reset = (
        M is None
        or self.training  # 学習時は毎回リセット（サンプル独立）
        or M.shape[0] != batch_size  # バッチサイズ変更時
    )
    if needs_reset:
        self.memory.reset(batch_size, device, dtype)
```

#### リセットタイミングの原則

| シナリオ | リセットタイミング | 理由 |
|---------|-------------------|------|
| **学習** | 毎サンプル | 各サンプルは独立、勾配の分離 |
| **推論（短文）** | 毎サンプル | 独立した質問への回答 |
| **推論（長文）** | シーケンス先頭のみ | メモリに過去情報を蓄積 |
| **評価（NIAH等）** | 各テスト前 | テスト間の干渉を防止 |

#### 評価コードでの正しい使用

```python
# NIAH評価など、各テストケース前にリセット
if hasattr(model, "reset_memory"):
    model.reset_memory(batch_size, device, dtype)

# その後generate()を呼ぶ
outputs = model.generate(**inputs)
```

#### 教訓

1. **メモリリセットは呼び出し側の責任**: 推論時は`forward()`内で自動リセットしない
2. **評価コードは明示的にリセット**: 各テストケース前に`reset_memory()`を呼ぶ
3. **学習時は毎回リセット**: 各サンプルの独立性を保証

## Implementation Details

### Tensor Memory（単一メモリ、学習・推論共通）

```python
# メモリ構造
M = torch.zeros(batch, heads, head_dim, head_dim)  # テンソル積
z = torch.zeros(batch, heads, head_dim)            # 正規化係数

# 更新: M = Σ v ⊗ k
M = M + torch.einsum('bhsd,bhse->bhde', v, k)  # 外積の累積
z = z + k.sum(dim=seq)  # 正規化用

# 検索: output = (M @ q) / (z^T @ q + eps)
numerator = torch.einsum('bhde,bhse->bhsd', M, q)
denominator = torch.einsum('bhd,bhsd->bhs', z, q) + eps
output = numerator / denominator.unsqueeze(-1)
```

**テンソル積の意味**:
- `v ⊗ k` は value と key の外積
- key を query として検索すると、対応する value が返る
- 複数の KV ペアを累積することで、連想メモリとして機能

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

```

**注意事項**:
- SVD計算は O(n³) の計算量。頻繁な実行は避ける
- `torch.no_grad()` 内で実行されるため、勾配は計算されない
- メモリ初期化前に呼び出すと `RuntimeError` が発生

### HuggingFace Compatibility

- `SenriConfig`: `LlamaConfig`を継承（SmolLM, Llamaファミリーと互換）
- `SenriForCausalLM`: Llamaアーキテクチャの構造を踏襲
- `from_pretrained`/`save_pretrained`完全対応
- `generate()`メソッドでの推論対応

### Key Classes

```python
# src/configuration_senri.py
class SenriConfig(LlamaConfig):
    model_type = "senri"

    # Senri specific
    sliding_window_size: int = 1024
    chunk_size: int = 64
    num_memory_layers: int = 2
    first_memory_layer: int = 10
    memory_layer_interval: int = 10

# src/memory/base_memory.py
class TensorMemory:
    """単一テンソル積メモリ（学習・推論共通）"""

# src/memory/senri_memory.py
class SenriMemory:
    """TensorMemoryのラッパー（将来の拡張用）"""

# src/attention/senri_attention.py
class SenriAttention(nn.Module):
    """SWA + Senri Memory Attention（単一メモリ版）"""
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
2. `SenriAttention`: メモリリセットの動作
3. `SenriForCausalLM`: SmolLM重みのロード

### Shape Tests
```python
def test_tensor_memory_shapes():
    memory = TensorMemory(hidden_size=576, num_heads=9)
    q = torch.randn(2, 9, 100, 64)
    k = torch.randn(2, 9, 100, 64)
    v = torch.randn(2, 9, 100, 64)

    memory.update(k, v)
    output = memory.retrieve(q)

    assert output.shape == (2, 9, 100, 64)
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
- テンソル積メモリは O(d²) のメモリで任意長のシーケンスを処理可能
- SWAはウィンドウサイズ内のみ計算するため O(n・w) の計算量

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
# モードによるメモリリセットの違いに注意
if self.training:
    # 毎サンプル自動リセット
else:
    # 手動リセットが必要
```

### 3. 位置エンコーディングの混在
```python
# SWA: RoPE適用
# Senri Memory: RoPE適用しない（NoPE）
```

### 4. メモリの毎回リセット（致命的）
```python
# Bad: 推論時にforward()内で毎回リセット
def forward(self, ...):
    self.memory.reset(...)  # ❌ 長文で過去情報が消える

# Good: 条件付きリセット
def forward(self, ...):
    if self.training or memory_not_initialized:
        self.memory.reset(...)  # ✅ 学習時のみ毎回リセット
```

**症状**: NIAH評価で正答率0%、長文生成で前半の情報を参照できない

### 5. retrieve→update順序問題（致命的）
```python
# Bad: retrieve→update順序（学習時にメモリが常にゼロ）
def forward(self, ...):
    self.memory.reset(...)           # 学習時は毎回リセット
    global_out = self.memory.retrieve(q)  # ❌ Mがゼロなので出力もゼロ
    self.memory.update(k, v)         # Mに値を追加（次で消える）

# Good: update→retrieve順序
def forward(self, ...):
    self.memory.reset(...)           # 学習時は毎回リセット
    self.memory.update(k, v)         # ✅ まずMに値を追加
    global_out = self.memory.retrieve(q)  # Mに値があるので正常動作
```

**症状**: 学習中にメモリからの出力が常にゼロ、メモリゲートが学習されない

**注意**: この順序変更は厳密な因果性を緩和するが、単一forward-pass学習では必要。
論文準拠のチャンク単位処理を実装すれば、retrieve→update順序でも動作する。

### 6. メモリ更新時の過度なdetach（致命的）
```python
# Bad: keys/valuesをdetachしてメモリ更新
def update(self, keys, values):
    keys_detached = keys.detach()      # ❌ 勾配経路が完全に切断
    values_detached = values.detach()  # ❌ メモリから学習できない
    delta_M = einsum(values_detached, keys_detached)
    self.M = self.M + delta_M

# Good: 累積状態のみdetach、現在の更新は勾配を維持
def update(self, keys, values):
    delta_M = einsum(values, keys)     # ✅ 現在の入力からの勾配を維持
    self.M = self.M.detach() + delta_M # ✅ 過去の累積のみdetach
```

**症状**: eval_loss が NaN、学習が全く進まない、メモリゲートの学習のみ発生

**原理**:
- 学習時、各サンプルは独立（毎回メモリリセット）
- 現在サンプル内の update → retrieve 勾配経路は必要
- 複数サンプル間での勾配累積は不要（detachで防ぐ）
- `self.M.detach() + delta_M` で「過去をカット、現在は維持」を実現

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
