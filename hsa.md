# Hierarchical Sparse Attention (HSA) - 論文まとめ

**論文**: "Every Token Counts: Generalizing 16M Ultra-Long Context in Large Language Models"
**著者**: Xiang Hu, Zhanchao Zhou, Ruiqi Liang, Zehuan Li, Wei Wu, Jianguo Li (Ant Group, Westlake University)
**arXiv**: 2511.23319v1 [cs.CL] 28 Nov 2025
**GitHub**: https://github.com/ant-research/long-context-modeling

---

## 1. 概要・動機

### 1.1 研究の背景

LLMの知識は静的なパラメータに閉じ込められており、動的な学習や更新が困難。**"Machines that Can Remember"** を構築するため、**超長文コンテキストモデリング**を長期記憶問題として捉える。

### 1.2 超長文コンテキストに必要な3つの性質

| 性質 | 説明 |
|------|------|
| **Sparsity（疎性）** | 人間の長期記憶は選択的活性化で動作。Full attentionでは無限長コンテキストは不可能 |
| **Random-Access Flexibility（ランダムアクセス柔軟性）** | 過去の関連情報を正確に検索するための内在的検索機構が必要 |
| **Length Generalization（長さ汎化）** | 無限長での事前学習は不可能。短いコンテキストから長いコンテキストへの検索能力の汎化が必要 |

### 1.3 既存手法の問題点

| 手法 | 問題点 |
|------|--------|
| **Mamba, Linear Attention** | 可変長情報を固定次元状態ベクトルに圧縮 → 情報ボトルネック、遠方トークンへのランダムアクセス不可 |
| **Sliding Window Attention** | 遠方コンテキストへのアクセス制約 |
| **NSA, MoBA** | チャンク選択が不正確 → in-domain/out-of-domain両方で性能劣化 |

---

## 2. HSA（Hierarchical Sparse Attention）の核心

### 2.1 MoEとの類似性

HSAはMixture of Experts (MoE)と動作が類似：

```
HSA                                    MoE
───────────────────────────────────────────────────────────
Top-K learnable Retrieval      ←→     Top-K learnable Router
Chunk1, Chunk2, ..., ChunkN    ←→     Expert1, Expert2, ..., ExpertN
各チャンクとの個別Attention    ←→     各Expertとの個別FFN
検索スコアで重み付け和         ←→     ルータースコアで重み付け和
```

### 2.2 NSAの限界

NSA（Native Sparse Attention）の実験結果：

| モデル | パラメータ | Single-NIAH (4K→64K) | MQ-NIAH (4K→64K) |
|--------|-----------|---------------------|------------------|
| NSA (w/ RoPE) | 370M | 97.0→60.0 | 72.0→4.0 |
| NSA (w/o RoPE) | 370M | 99.0→73.0 | 83.0→12.0 |

**問題点**: チャンク選択アクションがend-to-endで学習可能でない

### 2.3 HSAの2つの主要貢献

1. **Retrieval-oriented sparse attention**: 各トークンが過去のチャンクと**個別に**attentionを実行し、検索スコアで結果を融合
2. **RoPE for short, NoPE for long**: SWAのKVキャッシュはRoPE使用、HSAはNoPE（No Positional Encoding）使用

---

## 3. HSAの数式定義

### 3.1 基本設定

- 入力シーケンス: `S = {x₀, x₁, ..., xₙ}` （長さn）
- 隠れ状態: `H ∈ ℝⁿˣᵈ` （dは隠れ次元）
- チャンクサイズ: `S = 64`（ハードウェアアライメントのため）
- チャンク数: `n/S`
- チャンクインデックス表記: `H[i] := H_{iS:(i+1)S} ∈ ℝˢˣᵈ`

### 3.2 各チャンクの構成要素

| 要素 | 形状 | 説明 |
|------|------|------|
| `K[i], V[i]` | `ℝˢˣʰˣᵈʰ` | チャンクiのKVキャッシュ |
| `Kᵢˢˡᶜ` | `ℝᵈ` | チャンクiのランドマーク表現（チャンク要約） |
| `Qₜˢˡᶜ` | `ℝᵈ` | トークンtの検索用クエリ |
| `Qₜᵃᵗᵗⁿ` | `ℝʰˣᵈʰ` | トークンtのattention用クエリ |

### 3.3 検索スコアとTop-K選択

```
         ⎧ Qₜˢˡᶜᵀ Kᵢˢˡᶜ / √d,  i ≤ ⌊t/S⌋
sₜ,ᵢ =  ⎨
         ⎩ -∞,                  i > ⌊t/S⌋

Iₜ = {i | rank(sₜ,ᵢ) < K}
```

`rank(·)`: 降順でのランク位置、`Iₜ`: トークンxₜに対する最も関連性の高いK個のチャンクインデックス

### 3.4 Intra-chunk Attention

```
Ōₜ,ᵢ = Attention(Qₜᵃᵗᵗⁿ, K[i], V[i])
     = Softmax( norm(Qₜᵃᵗᵗⁿ) norm(K[i]ᵀ) / √dₕ ) V[i]
```

`norm`: Query-Key Normalization（兆トークン規模学習での安定性に重要）

### 3.5 Inter-chunk Fusion

```
wₜ,ᵢ = exp(sₜ,ᵢ) / Σₖ∈Iₜ exp(sₜ,ₖ)

Oₜ = Σₖ∈Iₜ wₜ,ₖ Ōₜ,ₖ
```

---

## 4. HSA-UltraLongアーキテクチャ

### 4.1 全体構造

```
┌─────────────────────────────────────────────────┐
│                Upper Decoder                     │
│  ┌───────────────────────────────────────────┐  │
│  │  MLP/MoE + SWA                      × R   │  │
│  ├───────────────────────────────────────────┤  │
│  │  MLP/MoE + HSA + SWA                × 1   │  │ × G groups
│  └───────────────────────────────────────────┘  │
├─────────────────────────────────────────────────┤
│                Lower Decoder                     │
│  ┌───────────────────────────────────────────┐  │
│  │  MLP/MoE + SWA                      × L/2 │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

- **Lower Decoder**: L/2層の標準Transformer（SWAのみ）
- **Upper Decoder**: G個のグループに分割
  - 各グループ: 1層のSWA+HSA層 + R層のSWAのみ層

### 4.2 Chunk Encoder

```
H^(L/2)[i] ∈ ℝˢˣᵈ  →  [CLS] token追加  →  Bi-directional Encoder
                                              ↓
                                    E[i] ∈ ℝˢˣᵈ (エンコード済み表現)
                                    Lᵢ ∈ ℝᵈ (ランドマーク表現)
```

- 中間層出力 `H^(L/2)` からチャンク要約表現とKVキャッシュを導出
- **KVキャッシュは全HSAモジュールで共有**（メモリ効率化）

### 4.3 MoE構成

- Ling-2.0の設計に準拠
- 最初の層: Dense MLP
- 以降の層: MoE
- 各MoEブロック: 1つの共有Expert（DeepSeek V3準拠）
- Expert balancing: Training-free balance strategy

---

## 5. 学習手順

### 5.1 5段階学習パイプライン

| 段階 | SWA窓 | HSA top-k | コンテキスト長 | 目的 |
|------|-------|-----------|---------------|------|
| **1. Warm-up** | 512 | 全シーケンスカバー | 16K | HSAの短距離学習 |
| **2. Pre-training** | 4K | sparse | 16K | 通常の事前学習 |
| **3. Long-context mid-training** | 4K | 全シーケンスカバー | 32K | 長文拡張 |
| **4. Annealing** | - | - | 32K | 高品質データでの調整 |
| **5. SFT** | - | - | 8K | 教師あり微調整 |

### 5.2 Warm-upの重要性

**問題**: 4K SWA窓で最初から学習すると、4Kを超えるコンテキストへの汎化に失敗

**仮説**: 大きすぎるSWA窓は短距離情報にランダムアクセスできるため、HSAが短距離パターンを学習する必要がなくなり、意味のある勾配を受け取れない

**解決策**:
- Warm-up中は512トークンの短いSWA窓を使用
- 1%の学習サンプルにRULERタスク（synthetic needle-in-a-haystack）を挿入
- モデルが512トークン窓を超えて高精度なNIAH検索を達成したらWarm-up完了

### 5.3 Warm-up戦略の比較

| 戦略 | PG19 PPL (16K) | MQ-NIAH (1M) |
|------|----------------|--------------|
| なし (BaseLM) | 16.77 | 0.0% |
| Self-copy warm-up | 16.50 | 93.0% |
| Short SWA + Full HSA warm-up | **15.96** | 66.0% |

- **Self-copy**: シーケンスを自分自身と連結して再構成を学習 → 最良の長さ外挿
- **Short SWA + Full HSA**: in-domain性能を維持しつつ合理的な長さ外挿

---

## 6. 学習データ・ハイパーパラメータ

### 6.1 学習データ構成

**第1フェーズ（事前学習）**: 10Tトークン（重複排除済み、マルチドメイン）

| ドメイン | 比率 |
|---------|------|
| Web | 50% |
| Code | 14.4% |
| Math | 12.0% |
| Code-NLP | 5.6% |
| Reason | 5% |
| Multilingual | 4.0% |
| Books | 2.0% |
| Wikipedia | 1.5% |
| Others | 5.5% |

- MoEモデル: 8Tトークン処理
- Denseモデル: 4Tトークン処理

**第2フェーズ**: 32K長文シーケンス 175Bトークン
**第3フェーズ**: 400Bトークン（推論データ高比率）
**SFT**: Grove MoEと同一データセット

### 6.2 ハイパーパラメータ

| パラメータ | MoEモデル | Denseモデル |
|-----------|----------|-------------|
| Optimizer | AdamW | AdamW |
| Weight decay | 0.01 | 0.01 |
| β₁, β₂ | 0.9, 0.95 | 0.9, 0.95 |
| Gradient clipping | 1.0 | 1.0 |
| Learning rate | 3.87e-4 | 4.96e-4 |
| Sequence length | 16,384 | 16,384 |
| Batch size (tokens) | 16.8M | 5.2M |
| 分散学習 | FSDP2 | FSDP2 |

**SFTフェーズ**:
- Dense: LR 5.5e-5, 最大5エポック
- MoE: LR 3.87e-4, 最大3エポック
- スケジュール: Cosine decay

---

## 7. モデル構成

### 7.1 HSA-UltraLongバリアント

| モデル | 総パラメータ | 活性化パラメータ | 学習トークン | アーキテクチャ |
|--------|------------|----------------|-------------|---------------|
| HSA-UL-Base Dense | 0.5B | 0.5B | 4T | Dense |
| HSA-UL-Base MoE | 8B | 1B | 8T | MoE |
| HSA-UL-Inst Dense | 0.5B | 0.5B | 4T | Dense |
| HSA-UL-Inst MoE | 8B | 1B | 8T | MoE |

### 7.2 MoE構成の違い（TRM-MoE vs HSA-UL）

| 項目 | TRM-MoE | HSA-UL-MoE |
|------|---------|------------|
| Expert数 | 32 | 64 |
| 活性化Expert数 | 2 | 4 |
| Expert次元 | 標準 | 半分 |
| 事前学習コンテキスト | 4K | 16K |

---

## 8. 実験結果

### 8.1 ベンチマーク一覧

| カテゴリ | ベンチマーク |
|---------|------------|
| **General** | MMLU, CMMLU, C-Eval, ARC, AGIEval, PIQA, HellaSwag, BBH |
| **Math** | GSM8K, MATH, CMATH, MATH-500, OlympiadBench |
| **Coding** | HumanEval, HumanEval+, MBPP, MBPP+, CRUX-O |
| **Alignment** | IFEval (prompt-level strict accuracy) |

### 8.2 Base Model性能比較

| モデル | アーキ | パラメータ | 学習トークン | AVG |
|--------|-------|----------|------------|-----|
| Qwen2.5 Annealing | Dense | 0.5B | 18T | 41.08 |
| Qwen3 Annealing | Dense | 0.6B | 36T | 48.42 |
| HSA-UL Annealing (Dense) | Dense | 0.5B | 4T | 37.70 |
| TRM-MoE Base | MoE | 8B (1B active) | 8T | 56.58 |
| HSA-UL Base (MoE) | MoE | 8B (1B active) | 8T | 57.27 |
| **HSA-UL Annealing (MoE)** | MoE | 8B (1B active) | 8T | **63.09** |

### 8.3 Instruction Model性能比較

| モデル | アーキ | パラメータ | 学習トークン | AVG |
|--------|-------|----------|------------|-----|
| Qwen3-Inst | Dense | 0.6B | 36T | 40.50 |
| HSA-UL-Inst | Dense | 0.5B | 4T | 36.48 |
| Qwen3-Inst | Dense | 1.7B | 36T | 60.76 |
| **HSA-UL-Inst (MoE)** | MoE | 8B (1B active) | 8T | **62.03** |

**注目点**: HSA-UL-MoE (1B active) がQwen3-1.7Bより1.3ポイント高い

### 8.4 長文コンテキスト評価

#### Needle-in-a-Haystack結果（16Mトークンまで）

**Long-context mid-training後**:
- Single-NIAH: 全深度で90%以上の精度（16Mまで）
- Multi-Query NIAH: MoE-8B-A1Bで16Mまで高精度維持

#### 重要な発見

1. **学習データの実効コンテキスト長が外挿に決定的**
   - 16K窓で学習しても、データの実効長が短いと外挿性能劣化
   - >32Kの実効長データで学習すると大幅改善

2. **HSA/SWAのシーソー効果**
   - 小さいSWA窓（512）→ HSA外挿性能向上
   - 大きいSWA窓（4K）→ HSA外挿性能低下
   - **理由**: 大きいSWA窓は短距離依存を自然に処理するため、HSAが短距離パターンを学習する動機が減少

3. **推論-検索タスクではパラメータ規模が効く**
   - 純粋検索タスク（MQ-NIAH）: MoE-8BとDense-0.5Bで同等
   - 推論+検索タスク（Variable Tracking）: MoE-8Bが明確に優位

---

## 9. 効率性評価

### 9.1 HSA vs FlashAttention-3（H800）

**学習効率**:
| コンテキスト長 | HSA | FlashAttention-3 | 勝者 |
|---------------|-----|------------------|------|
| 4K | ~30ms | ~15ms | FA3 |
| 8K | ~45ms | ~25ms | FA3 |
| 16K | ~70ms | ~50ms | FA3 |
| 32K | ~110ms | ~95ms | FA3 |

**推論効率**:
| コンテキスト長 | HSA | FlashAttention-3 | 勝者 |
|---------------|-----|------------------|------|
| 4K-64K | 遅い | 速い | FA3 |
| 128K | 同等 | 同等 | - |
| 256K | ~400ms | ~800ms | **HSA** |

**HSAが短いシーケンスで不利な理由**:
1. スパース性によりメモリアクセス増加
2. FlashAttention-3はCUDA実装でHopperアーキテクチャを最大活用

---

## 10. 未解決課題と今後の方向性

### 10.1 HSA/SWAシーソー問題

短いSFTデータで学習後、外挿性能が劣化する可能性。長すぎるSWA窓がHSAの短距離依存学習を阻害。

### 10.2 ヘッド比率制約

HSAは現在**16:1のQuery/KVヘッド比率**を要求。深刻な情報ボトルネック。カーネルレベル最適化が必要。

### 10.3 短シーケンスでの効率

短いシーケンスではFlashAttention-3に対する明確な優位性なし。カーネル最適化が必要。

---

## 11. HSAの核心的インサイト

> **"The core insight of HSA is to perform attention chunk by chunk and fuse the results via retrieval scores, rather than selecting chunks and then concatenating them for attention."**

従来のスパースアテンション（NSA等）:
```
チャンク選択 → 選択チャンクを連結 → Attention実行
```

HSA:
```
全チャンクと個別にAttention → 検索スコアで重み付け融合
```

この設計により：
- 検索スコアがforward passに統合される
- バックプロパゲーション中に勾配更新を受ける
- モデルが次トークン予測に有用なチャンクに高いスコアを割り当てることを学習

---

## 12. 長さ汎化に必要な3要素

効果的な長さ汎化には以下の**全て**が必要：

1. **Chunk-wise attention**: チャンクごとの個別attention
2. **Retrieval score-based fusion**: 検索スコアによる結果融合
3. **NoPE (No Positional Encoding)**: HSAレイヤーでの位置エンコーディング除去

---

## 13. Senri-LLMプロジェクトとの関連

### 類似点

| 項目 | Senri | HSA-UltraLong |
|------|-------|---------------|
| グローバルメモリ | Infini Attention | HSA |
| ローカルメモリ | SWA | SWA |
| 位置エンコーディング | SWA: RoPE, Memory: NoPE | SWA: RoPE, HSA: NoPE |
| チャンク処理 | あり | あり |
| 検索メカニズム | 直交基底ルーティング | Landmark-based Top-K |

### 相違点

| 項目 | Senri | HSA-UltraLong |
|------|-------|---------------|
| メモリ表現 | テンソル積 | KVキャッシュ |
| 検索方法 | 直交基底 + Top-K | ランドマーク + Top-K |
| 融合方法 | 直接（単一メモリ）または重み付け | 検索スコアによる重み付け融合 |
| 学習/推論分離 | あり（異なる戦略） | 統一 |

### HSAから学べるポイント

1. **Warm-up戦略**: 短いSWA窓での初期学習が長さ汎化に重要
2. **NoPEの重要性**: グローバルメモリには位置エンコーディングなしが有効
3. **実効コンテキスト長**: 学習データの実効長が外挿能力に影響
4. **シーソー効果**: SWA窓サイズとグローバルメモリ学習のトレードオフ

---

## 14. Self-copy Warm-up戦略の詳細

### 14.1 Self-copy Objective

論文 Section 4.1:
> "Given an input sequence S = {x₁, ..., xₙ}, we construct a target sequence S' = {x₁, ..., xₙ, x₁, ..., xₙ} by concatenating S with itself."

```python
# Self-copy warm-up の擬似コード
def create_self_copy_sample(sequence):
    # 入力: S = [x1, x2, ..., xn]
    # 出力: S' = [x1, x2, ..., xn, x1, x2, ..., xn]
    return sequence + sequence

# モデルは後半部分を再構成することを学習
# → 長距離の prefix 情報を attend して retrieve する能力を獲得
```

**効果**: モデルが長距離のプレフィックス情報に attend し、retrieve する能力を獲得

### 14.2 Warm-up戦略の比較（Table 2詳細）

| 戦略 | パラメータ | PG19 PPL (4K/8K/16K) | MQ-NIAH (4K/8K/64K/1M) |
|------|----------|---------------------|----------------------|
| BaseLM（warm-upなし） | 519.6M | 18.61/17.53/16.77 | 89.0/23.0/5.0/0.0 |
| SWA+HSA (self-copy) | 537.7M | 18.87/17.44/16.50 | **100.0/96.0/93.0/93.0** |
| SWA+HSA (short-swa,full-hsa) | 537.7M | **18.30/17.13/15.96** | 99.0/95.0/90.0/66.0 |

**結論**:
- **Self-copy**: 最良の長さ外挿（1Mで93%）、in-domain性能は若干低下
- **Short SWA + Full HSA**: in-domain性能維持、合理的な外挿能力

---

## 15. Query-Key Normalization（QK Norm）

論文 Section 2.2:
> "norm is the Query-Key Normalization, which we find to be very important for the stability of HSA in practical trillion-token scale training."

### 15.1 QK Normの数式

```
Ōₜ,ᵢ = Softmax( norm(Qₜᵃᵗᵗⁿ) norm(K[i]ᵀ) / √dₕ ) V[i]
```

### 15.2 なぜQK Normが重要か

| 問題 | QK Normなし | QK Normあり |
|------|------------|------------|
| 兆トークン規模学習 | 不安定、発散リスク | 安定 |
| Attention重み分布 | 極端な値が発生 | 均一化 |
| 勾配流 | 消失/爆発 | 安定 |

参考文献:
- Dehghani et al. (2023): "Scaling vision transformers to 22 billion parameters"
- Wortsman et al. (2023): "Small-scale proxies for large-scale transformer training instabilities"

---

## 16. Bi-directional Encoder（チャンクエンコーダ）詳細

### 16.1 構造

```
入力: H^(L/2)[i] ∈ ℝˢˣᵈ  (中間層出力のi番目チャンク)
      ↓
[CLS]トークンを追加: [CLS, H^(L/2)[i]]
      ↓
Bi-directional Encoder
      ↓
出力: E[i] ∈ ℝˢˣᵈ (エンコード済み表現)
      Lᵢ ∈ ℝᵈ (ランドマーク表現、[CLS]の出力)
```

### 16.2 KVキャッシュの導出

```python
# E[i]から線形変換でKeys, Valuesを導出
K[i] = E[i] @ W_K  # Shape: [S, h, d_h]
V[i] = E[i] @ W_V  # Shape: [S, h, d_h]

# ランドマーク表現
K_slc[i] = L[i]  # Shape: [d]、チャンク要約として使用
```

### 16.3 KVキャッシュ共有

> "we share the intermediate layer KV cache among all HSA modules to serve as context memory"

- **中間層（L/2層）** の出力からKVキャッシュを一度だけ計算
- **全てのHSAモジュール** でこのKVキャッシュを共有
- **メリット**: メモリ効率の大幅改善

---

## 17. 詳細なNIAH結果（Figure 4）

### 17.1 Single-NIAH（Long-context mid-training後）

| Depth | 4K | 16K | 64K | 256K | 1M | 4M | 16M |
|-------|-----|-----|-----|------|-----|-----|-----|
| 0% | 98 | 96 | 100 | 98 | 94 | 98 | 95 |
| 11% | 100 | 96 | 100 | 98 | 96 | 98 | 100 |
| 22% | 100 | 96 | 100 | 98 | 100 | 98 | 100 |
| 33% | 100 | 98 | 100 | 98 | 94 | 98 | 100 |
| 44% | 100 | 98 | 100 | 100 | 98 | 98 | 95 |
| 55% | 100 | 96 | 100 | 98 | 98 | 94 | 95 |
| 66% | 100 | 98 | 100 | 98 | 98 | 98 | 100 |
| 77% | 100 | 96 | 100 | 98 | 98 | 96 | 95 |
| 88% | 100 | 98 | 100 | 98 | 98 | 98 | 95 |
| 100% | 100 | 98 | 100 | 98 | 96 | 98 | 100 |

**注目**: 16Mトークンでも95-100%の精度を維持

### 17.2 Multi-Query NIAH（2クエリ、6 KVペア）

| モデル | 4K | 16K | 64K | 256K | 1M | 4M | 16M |
|--------|-----|-----|-----|------|-----|-----|-----|
| MoE-8B-A1B-Annealing | ~100 | ~98 | ~95 | ~90 | ~85 | ~75 | ~65 |
| MoE-8B-A1B | ~100 | ~95 | ~90 | ~80 | ~70 | ~55 | ~45 |
| Dense-0.5B | ~100 | ~95 | ~88 | ~75 | ~60 | ~40 | ~25 |
| Dense-0.5B (SWA 512) | ~100 | ~98 | ~95 | ~90 | ~85 | ~80 | ~70 |

### 17.3 Variable Tracking Task

| モデル | 4K | 16K | 64K | 256K | 1M | 4M | 16M |
|--------|-----|-----|-----|------|-----|-----|-----|
| MoE-8B-A1B | ~95 | ~90 | ~85 | ~75 | ~65 | ~50 | ~35 |
| Dense-0.5B | ~90 | ~80 | ~65 | ~50 | ~35 | ~20 | ~10 |

**発見**: 推論+検索タスクではパラメータ規模が重要

---

## 18. HSA実装詳細

### 18.1 TileLang実装

論文 Section 4.4:
> "HSA implemented using TileLang"

- **TileLang**: AIシステム向けのコンポーザブルなタイルプログラミングモデル
- 参考: Wang et al. (2025) "Tilelang: A composable tiled programming model for AI systems"

### 18.2 効率性ベンチマーク（H800）

**学習効率** (Wall-clock Time in ms):

| コンテキスト長 | HSA | FlashAttention-3 |
|---------------|-----|------------------|
| 4K | ~30 | ~15 |
| 8K | ~45 | ~25 |
| 16K | ~70 | ~50 |
| 32K | ~110 | ~95 |

**推論効率** (Wall-clock Time in ms):

| コンテキスト長 | HSA | FlashAttention-3 |
|---------------|-----|------------------|
| 4K | ~50 | ~20 |
| 8K | ~80 | ~40 |
| 16K | ~120 | ~80 |
| 32K | ~180 | ~150 |
| 64K | ~280 | ~300 |
| 128K | ~400 | ~500 |
| 256K | ~600 | ~800 |

**クロスオーバーポイント**: 推論では約128Kでbreak-even、256K以上でHSA優位

### 18.3 HSAが短いシーケンスで不利な理由

1. **スパース性によるメモリアクセス増加**: 非連続アクセスパターン
2. **FlashAttention-3のCUDA最適化**: Hopperアーキテクチャの機能を最大活用

---

## 19. HSA層のパラメータ増加

論文 Section 4.1:
> "Each HSA layer includes an additional encoder sub-layer, resulting in less than 5% parameter increase compared to BaseLM."

| モデル | パラメータ数 | 増加率 |
|--------|------------|--------|
| BaseLM | 519.6M | - |
| SWA+HSA | 537.7M | +3.5% |

**増加の内訳**:
- Bi-directional Encoder
- ランドマーク射影層
- 検索用クエリ射影層

---

## 20. 長文評価における3つの重要発見

### 20.1 発見1: 学習データの実効コンテキスト長が外挿に決定的

> "Effective context length of training data is critical for HSA extrapolation"

- 16K窓で学習しても、データの実効長が短いと外挿性能劣化
- **>32K** の実効長データで学習すると大幅改善
- Long-context mid-training が必要な理由

### 20.2 発見2: HSA/SWAのシーソー効果

> "A seesaw effect exists between HSA and Sliding Window Attention"

| SWA窓サイズ | HSA外挿性能 | 理由 |
|------------|-----------|------|
| 512 | 高い | HSAが短距離パターンも学習 |
| 4K | 低い | SWAが短距離を処理、HSAの学習動機減少 |

**解決策**: Warm-upフェーズで512窓を使用

### 20.3 発見3: 推論-検索タスクではパラメータ規模が効く

| タスク種類 | Dense-0.5B vs MoE-8B-A1B |
|----------|-------------------------|
| 純粋検索（MQ-NIAH） | 同等 |
| 推論+検索（Variable Tracking） | MoE-8Bが明確に優位 |

---

## 21. 参考文献（主要なもの）

- [11] Dehghani et al. "Scaling vision transformers to 22 billion parameters." 2023. (QK Norm)
- [18] Hu et al. "Hardware-aligned hierarchical sparse attention for efficient long-term memory access." NeurIPS 2025.
- [19] Hu et al. "Efficient length-generalizable attention via causal retrieval for long-context language modeling." ICML 2025.
- [23] Leng et al. "Understanding and improving length generalization in hierarchical sparse attention models." arXiv:2510.17196.
- [29] Mohtashami & Jaggi. "Random-access infinite context length for transformers." NeurIPS 2023.
- [39] Wang et al. "Tilelang: A composable tiled programming model for AI systems." 2025.
- [41] Wortsman et al. "Small-scale proxies for large-scale transformer training instabilities." 2023.
- [47] Yuan et al. "Native sparse attention: Hardware-aligned and natively trainable sparse attention." ACL 2025.
