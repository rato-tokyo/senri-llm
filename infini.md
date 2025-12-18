# Infini-Attention 論文まとめと実装比較

**論文**: "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention"
**著者**: Tsendsuren Munkhdalai, Manaal Faruqui, Siddharth Gopal (Google)
**arXiv**: 2404.07143v2

---

## 1. 論文の核心アイデア

### 1.1 問題設定

標準的なTransformerのAttentionは：
- **メモリ計算量**: O(N²) - シーケンス長の2乗に比例
- **KVキャッシュ**: シーケンス長に比例して増大
- 例: 500Bモデル、バッチ512、コンテキスト2048で **3TB** のメモリフットプリント

### 1.2 解決策: Infini-attention

**圧縮メモリ（Compressive Memory）** を導入し、固定サイズのパラメータでコンテキスト全体を保持：

```
Os, Ms = infini-attention(Xs, Ms-1)
```

- `Xs`: 現在のセグメント入力
- `Ms-1`: 前セグメントからのメモリ状態
- `Os`: 出力
- `Ms`: 更新されたメモリ状態

---

## 2. アーキテクチャ詳細

### 2.1 メモリ構造

**連想行列（Associative Matrix）** を使用：

| 要素 | 形状 | 説明 |
|------|------|------|
| M | `[d_key, d_value]` | KVバインディングを格納 |
| z | `[d_key]` | 正規化項（キーの累積和） |

**メモリフットプリント**: `d_key × d_value + d_key` per head per layer
（シーケンス長に**依存しない**固定サイズ）

### 2.2 メモリ検索（Memory Retrieval）

```
A_mem = σ(Q) @ M_{s-1} / (σ(Q) @ z_{s-1})
```

- `σ`: 活性化関数（**ELU + 1** を使用）
- `Q`: クエリ（位置エンコーディングなし）
- 論文では `σ(Q)` を使用するが、当実装では raw Q を使用

### 2.3 メモリ更新（Memory Update）

**Linear更新**:
```
Ms = M_{s-1} + σ(K)^T @ V
zs = z_{s-1} + Σ σ(K_t)
```

**Delta更新**（改良版）:
```
Ms = M_{s-1} + σ(K)^T @ (V - σ(K) @ M_{s-1} / (σ(K) @ z_{s-1}))
```

Delta更新は、既存のKVバインディングがある場合はメモリを変更しない。

### 2.4 ゲート機構

ローカルAttention（`A_dot`）とメモリ検索（`A_mem`）を学習可能なゲートで結合：

```
A = sigmoid(β) ⊙ A_mem + (1 - sigmoid(β)) ⊙ A_dot
```

- `β`: 学習可能なスカラー（ヘッドごと）
- 論文実験では、学習後にヘッドが **特化型**（β≈0 or β≈1）と **混合型**（β≈0.5）に分化

### 2.5 位置エンコーディング

**重要**: 論文 Section 4.1 (Position Embeddings)

> "we don't use position embeddings for the key and query vectors of the compressive memory to store only global contextual information in the long-term memory. The PEs were applied to the QK vectors only after the compressive memory reading and update."

- **ローカルAttention**: RoPE（位置エンコーディングあり）
- **メモリ**: 位置エンコーディング**なし**（NoPE）

---

## 3. セグメント処理とBPTT

### 3.1 セグメントチャンキング

論文 Section 4.1:
> "we forward-pass the entire input text a Transformer model and then perform segment chunking at each Infini-attention layer"

- 入力全体をモデルに渡す
- 各Infini-attention層でセグメント分割を実行
- セグメントごとに処理し、結合して次の層へ

### 3.2 BPTT（Backpropagation Through Time）

> "Each Infini-attention layer is trained with back-propagation through time (BPTT) by computing the gradient w.r.t the compressive memory states"

- メモリ状態を通じた勾配計算
- メモリ節約のためgradient checkpointing使用

### 3.3 処理順序（重要）

論文の図と式から推測される順序：

```
Segment s:
1. M_{s-1} からメモリ検索（retrieve）
2. ローカルAttention計算
3. ゲートで結合
4. M_{s-1} を M_s に更新（update）
5. M_s を次のセグメント s+1 へ渡す
```

**注意**: retrieve → update の順序（因果性維持）

---

## 4. 実験結果

### 4.1 言語モデリング（PG19, Arxiv-math）

| モデル | メモリサイズ | 圧縮率 | PG19 PPL | Arxiv-math PPL |
|--------|------------|--------|----------|----------------|
| Transformer-XL | 50M | 3.7x | 11.88 | 2.42 |
| Memorizing Transformers | 183M | 1x | 11.37 | 2.26 |
| RMT | 2.5M | 73x | 13.27 | 2.55 |
| **Infini-Transformer (Linear)** | 1.6M | **114x** | **9.65** | 2.24 |
| **Infini-Transformer (Linear + Delta)** | 1.6M | **114x** | 9.67 | **2.23** |

- 設定: 12層、8ヘッド、d=128、セグメント長N=2048、入力長32768
- 100K長学習でArxiv-mathが2.20まで改善

### 4.2 Passkey検索（1Mコンテキスト）

| コンテキスト長 | 32K | 128K | 256K | 512K | 1M |
|--------------|-----|------|------|------|-----|
| Zero-shot (Linear) | 14/13/98 | 11/14/100 | 6/3/100 | 6/7/99 | 8/6/98 |
| Fine-tuned (Linear) | 100/100/100 | 100/100/100 | 100/100/100 | 97/99/100 | 96/94/100 |
| Fine-tuned (Linear + Delta) | 100/100/100 | 100/100/99 | 100/100/99 | 100/100/100 | **100/100/100** |

- 1Bモデル、**5K長**で学習 → **1M長**で評価
- Fine-tuning: 400 steps

### 4.3 書籍要約（BookSum, 500K）

| モデル | Rouge-1 | Rouge-2 | Rouge-L | Overall |
|--------|---------|---------|---------|---------|
| PRIMERA + Unlimiformer | 37.9 | 8.2 | 16.3 | 17.2 |
| **Infini-Transformers (Linear + Delta)** | **40.0** | **8.8** | **17.9** | **18.5** |

- 8Bモデル、8K長で継続事前学習 → 32K長でfine-tune → 500K長で評価

---

## 5. 現在のSenri実装との比較

### 5.1 一致している点

| 項目 | 論文 | Senri実装 | 状態 |
|------|------|----------|------|
| メモリ構造 | 連想行列 M, z | TensorMemory(M, z) | ✅ 一致 |
| メモリサイズ | `[d_key, d_value]` per head | `[heads, head_dim, head_dim]` | ✅ 一致 |
| 正規化項 | z = Σ k | z = keys.sum(dim=seq) | ✅ 一致 |
| 位置エンコーディング | ローカル:RoPE, メモリ:NoPE | ローカル:RoPE, メモリ:NoPE | ✅ 一致 |
| ゲート機構 | sigmoid(β) per head | sigmoid(memory_gate) per head | ✅ 一致 |
| ゲート初期値 | 論文では学習後に分化 | 0.0で初期化 | ✅ 妥当 |

### 5.2 相違点（要修正）

| 項目 | 論文 | Senri実装 | 重要度 | 修正案 |
|------|------|----------|--------|--------|
| **活性化関数σ** | ELU + 1 | なし（raw K, Q） | ⚠️ 中 | σ(K), σ(Q)を追加 |
| **更新順序** | retrieve → update | update → retrieve | 🔴 高 | セグメント処理で解決可能 |
| **Delta更新** | 実装あり | 未実装 | ⚠️ 中 | オプションとして追加 |
| **セグメント処理** | セグメントごとに処理 | 全シーケンス一括 | 🔴 高 | チャンク処理を実装 |

### 5.3 詳細分析

#### 5.3.1 活性化関数σの欠如

論文では `σ(K) = ELU(K) + 1` を使用：
- 負の値を防ぎ、正規化の安定性を向上
- Linear attention の標準的な手法

現在の実装:
```python
# base_memory.py
delta_M = torch.einsum("bhsd,bhse->bhde", values, keys)  # raw keys
```

修正案:
```python
def elu_plus_one(x):
    return F.elu(x) + 1

sigma_k = elu_plus_one(keys)
sigma_q = elu_plus_one(queries)
delta_M = torch.einsum("bhsd,bhse->bhde", values, sigma_k)
```

#### 5.3.2 更新順序（update → retrieve vs retrieve → update）

**論文の順序**: retrieve → update
- 因果性維持：過去のメモリから検索してから、現在のKVで更新
- セグメント`s`の処理で`M_{s-1}`から検索し、`M_s`に更新

**現在の実装**: update → retrieve
- 単一forward-pass学習では必要（メモリが空の状態でretrieveしても意味がない）
- 厳密な因果性は失われるが、現実的な妥協

**根本的な解決策**: セグメント単位の処理を実装
```python
# 長いシーケンスをセグメントに分割
for segment in chunks(sequence, segment_length):
    # 1. 過去のメモリから検索
    mem_output = memory.retrieve(segment_queries)
    # 2. ローカルAttention
    local_output = local_attention(segment)
    # 3. 結合
    output = gate * mem_output + (1-gate) * local_output
    # 4. メモリ更新
    memory.update(segment_keys, segment_values)
```

#### 5.3.3 Delta更新の未実装

Delta更新は既存バインディングの重複を避ける改良版：
```python
# Linear更新
Ms = Ms_prev + sigma_K.T @ V

# Delta更新（改良版）
retrieved = sigma_K @ Ms_prev / (sigma_K @ zs_prev)
Ms = Ms_prev + sigma_K.T @ (V - retrieved)
```

論文実験ではDelta更新が若干良い結果を示している（特にBookSum）。

---

## 6. 実装優先度

### 高優先度（動作に影響）

1. **セグメント処理の実装**
   - 長いシーケンスをチャンクに分割
   - retrieve → update の順序を維持
   - BPTT対応

### 中優先度（性能改善）

2. **活性化関数σの追加**
   - ELU + 1 を K, Q に適用
   - 数値安定性の向上

3. **Delta更新の実装**
   - オプションとして追加
   - 長文での性能改善が期待できる

### 低優先度（最適化）

4. **ゲート初期化の調整**
   - 論文では学習後に自然に分化
   - 現在の0.0初期化で問題なし

---

## 7. 学習設定（論文から）

| 項目 | 値 |
|------|-----|
| オプティマイザ | Adafactor |
| 学習率（LM） | 0.01 |
| 学習率（LLM継続学習） | 0.0001 |
| ウォームアップ | 1000 steps, linear |
| スケジューラ | cosine decay |
| バッチサイズ | 64 |
| セグメント長 | 2048 |
| 入力長（学習） | 32768 |

---

## 8. まとめ

Infini-attentionは：
1. **固定サイズメモリ**で無限長コンテキストを処理
2. **114x圧縮率**でMemorizing Transformersを上回る
3. **5K長学習→1M長評価**の外挿能力を実証

現在のSenri実装は基本的な構造は正しいが、**セグメント処理**と**活性化関数**の追加で論文により忠実な実装が可能。
