# new-llm プロジェクト分析

**場所**: `/Users/sakajiritomoyoshi/Desktop/git/new-llm`

このドキュメントはsenri-llmとの比較のためにnew-llmプロジェクトを分析したものです。
両プロジェクトはInfini-Attention風のメモリ機構を実装していますが、設計思想に違いがあります。

---

## 1. プロジェクト構造

```
new-llm/
├── src/models/
│   ├── model.py              # TransformerLM（メインモデル）
│   ├── memory/
│   │   ├── base.py           # CompressiveMemory（テンソル積メモリ）
│   │   └── mixins.py         # ミックスイン
│   ├── memory_utils.py       # ELU+1, causal_linear_attention
│   └── layers/
│       ├── senri.py          # SenriLayer（メモリ + Linear Attention）
│       └── pythia.py         # PythiaLayer（RoPE + Softmax Attention）
```

---

## 2. メモリ実装の比較

### 2.1 両プロジェクトとも「テンソル積メモリ」

**重要な発見**: new-llmの`CompressiveMemory`もsenri-llmの`TensorMemory`と同じ**テンソル積メモリ**です。

| 項目 | senri-llm (TensorMemory) | new-llm (CompressiveMemory) |
|------|--------------------------|----------------------------|
| メモリ行列 | `M ∈ R^{batch, heads, d, d}` | `M ∈ R^{d, d}` |
| 正規化項 | `z ∈ R^{batch, heads, d}` | `z ∈ R^{d}` |
| バッチ処理 | バッチごとに独立 | **バッチ間で共有** |
| 複数メモリ | 単一メモリ | 複数スロット対応 |

### 2.2 メモリ更新の違い

**senri-llm**:
```python
# 勾配を維持しつつ累積
delta_M = torch.einsum("bhsd,bhse->bhde", values, sigma_keys)
self.M = self.M.detach() + delta_M  # 現在の更新には勾配あり
```

**new-llm**:
```python
# 正規化して完全detach
memory_update = torch.einsum('bsd,bse->de', sigma_k, values) / (batch_size * seq_len)
self.memories[idx] = (memory + memory_update).detach()  # 完全に勾配なし
```

### 2.3 Delta Rule（new-llmのみ実装）

new-llmはDelta Ruleをサポート:
```python
if self.use_delta_rule:
    # 既存のバインディングを引いてから更新
    retrieved_unnorm = torch.matmul(sigma_k, memory)
    norm = torch.matmul(sigma_k, memory_norm).clamp(min=1e-6).unsqueeze(-1)
    delta_v = values - retrieved_unnorm / norm
    memory_update = torch.einsum('bsd,bse->de', sigma_k, delta_v) / (batch_size * seq_len)
```

---

## 3. Attention実装の比較

### 3.1 ローカルAttentionの違い

| 項目 | senri-llm | new-llm |
|------|-----------|---------|
| **方式** | Softmax Dot-product | **Linear Attention** |
| **計算量** | O(N × window) | **O(N)** |
| **位置エンコーディング** | RoPE | **NoPE** |
| **因果性** | マスク適用 | 累積和 |

**new-llmのLinear Attention**:
```python
def causal_linear_attention(q, k, v, eps=1e-6):
    sigma_q = elu_plus_one(q)
    sigma_k = elu_plus_one(k)

    # 外積を累積（因果性を維持）
    kv = torch.einsum('bsd,bse->bsde', sigma_k, v)
    kv_cumsum = torch.cumsum(kv, dim=1)
    k_cumsum = torch.cumsum(sigma_k, dim=1)

    numerator = torch.einsum('bsd,bsde->bse', sigma_q, kv_cumsum)
    denominator = torch.einsum('bsd,bsd->bs', sigma_q, k_cumsum)

    return numerator / denominator.clamp(min=eps).unsqueeze(-1)
```

### 3.2 位置エンコーディングの違い

| レイヤータイプ | senri-llm | new-llm |
|--------------|-----------|---------|
| **SenriLayer** | ローカル: RoPE, メモリ: NoPE | **全体: NoPE** |
| **PythiaLayer** | N/A | RoPE (25%) |

**new-llmの設計思想**: SenriLayerは完全にNoPE（位置情報なし）で、必要に応じてPythiaLayerで位置情報を補完。

---

## 4. モデル構成

### 4.1 new-llmのハイブリッド構成

```python
# new-llm: SenriLayer + PythiaLayerのハイブリッド
model = TransformerLM([
    SenriLayer(...),      # メモリ + Linear Attention (NoPE)
    PythiaLayer(...),     # RoPE + Softmax Attention
    PythiaLayer(...),
    PythiaLayer(...),
    PythiaLayer(...),
    PythiaLayer(...),
])
```

### 4.2 senri-llmの構成

```python
# senri-llm: SmolLM-135Mベースでメモリレイヤーを挿入
# Layer 0-9:  標準Attention
# Layer 10:   SenriAttention (SWA + Memory)
# Layer 11-19: 標準Attention
# Layer 20:   SenriAttention (SWA + Memory)
# Layer 21-29: 標準Attention
```

---

## 5. 追加機能（new-llmのみ）

### 5.1 複数メモリスロット

```python
class CompressiveMemory:
    def __init__(self, memory_dim, num_memories=1, ...):
        # 複数の独立したメモリスロット
        self.memories = [torch.zeros(d, d) for _ in range(num_memories)]
        self.memory_norms = [torch.zeros(d) for _ in range(num_memories)]
```

### 5.2 Landmark選択

複数メモリの場合、クエリとの関連度で重み付け:
```python
def _compute_relevance(self, sigma_q, idx):
    landmark = self.memory_norms[idx]  # Σσ(k)がランドマーク
    rel = torch.einsum('bsd,d->bs', sigma_q, landmark)
    return rel.mean(dim=-1)
```

### 5.3 Freeze/Unfreeze機能

特定のメモリスロットを読み取り専用に:
```python
def freeze(self, indices):
    for idx in indices:
        self.frozen[idx] = True

def unfreeze(self, indices):
    for idx in indices:
        self.frozen[idx] = False
```

### 5.4 メモリExport/Import

学習済みメモリの共有:
```python
# エクスポート
memory_data = model.export_memory()
torch.save(memory_data, "knowledge.pt")

# インポート
memory_data = torch.load("knowledge.pt")
model.import_memory(memory_data, freeze=True)
```

---

## 6. 論文準拠度の比較

| 項目 | Infini-Attention論文 | senri-llm | new-llm |
|------|---------------------|-----------|---------|
| **ローカルAttention** | Softmax + RoPE | ✅ Softmax + RoPE | ❌ Linear + NoPE |
| **メモリAttention** | Linear + NoPE | ✅ Linear + NoPE | ✅ Linear + NoPE |
| **セグメント処理** | チャンク単位 | ✅ 実装済み | ❌ 全シーケンス一括 |
| **ELU+1活性化** | σ(K), σ(Q) | ✅ 実装済み | ✅ 実装済み |
| **retrieve→update順序** | 因果性維持 | ✅ セグメント単位 | ❌ 全体で一括 |
| **Delta Rule** | オプション | ❌ 未実装 | ✅ 実装済み |

**結論**:
- **senri-llm**は論文により忠実
- **new-llm**は独自の効率化設計（Linear Attention + NoPE）

---

## 7. なぜnew-llmが動作するのか

new-llmが論文と異なる設計でも動作する理由:

1. **Linear Attentionの有効性**
   - Katharopoulos et al. (2020)で有効性が示されている
   - O(N)計算量で長いシーケンスに有利

2. **NoPEの有効性**
   - メモリは位置に依存しないグローバル情報を格納
   - PythiaLayerで位置情報を補完

3. **学習可能なパラメータ**
   - QKV投影（`w_q`, `w_k`, `w_v`）が学習可能
   - ゲートパラメータが学習可能
   - これらがメモリの使い方を学習

4. **完全detachでも動作する理由**
   - メモリ自体は連想配列として機能
   - 勾配がなくても「何を格納するか」は学習される

---

## 8. senri-llmへの示唆

### 採用を検討できる機能

1. **Delta Rule**: 既存バインディングの重複を防ぐ（長文で有効）
2. **複数メモリスロット**: 異なる種類の情報を分離格納
3. **Freeze機能**: 学習済み知識の保存

### 採用しない機能

1. **Linear Attention for ローカル**: 論文準拠を優先
2. **NoPE for ローカル**: RoPEで位置情報を維持
3. **バッチ共有メモリ**: バッチ間の独立性を維持

---

## 9. 主要ファイルの概要

| ファイル | 役割 |
|---------|------|
| `model.py` | TransformerLM - レイヤーリストからモデル構築 |
| `memory/base.py` | CompressiveMemory - テンソル積メモリ |
| `memory_utils.py` | ELU+1, causal_linear_attention |
| `layers/senri.py` | SenriLayer - メモリ + Linear Attention |
| `layers/pythia.py` | PythiaLayer - RoPE + Softmax Attention |

---

## 10. 動作確認状況

ユーザーによると、new-llmは動作していたとのこと。
これは以下を示唆:

1. **Linear Attention + メモリの組み合わせは実用的**
2. **完全detachでもメモリは学習に貢献**
3. **独自設計でもInfini-Attentionの恩恵を受けられる**

senri-llmは論文準拠を優先するが、動作しない場合はnew-llmの設計を参考にデバッグ可能。
