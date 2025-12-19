# LLMのレイヤー置換ガイドライン

## 概要

既存の事前学習済みLLMの特定レイヤーを新しいアーキテクチャ（メモリ層、線形Attention等）に置き換える際の注意点をまとめる。

---

## 1. 出力分布の一致

### 問題

事前学習済みモデルの後続レイヤーは、**前のレイヤーの特定の出力分布**を期待している。
新しいレイヤーの出力分布が異なると、モデル全体が崩壊する。

### 事例: Softmax Attention → Linear Attention

| 特性 | Softmax Attention | Linear Attention |
|------|-------------------|------------------|
| 出力範囲 | 暗黙的に正規化 | 制限なし |
| 分布の形状 | 鋭いピーク | 滑らか |
| スケール | 一定 | 可変 |

### 解決策

1. **蒸留学習**: 新レイヤーの出力を元レイヤーの出力に近づける
   ```python
   loss = MSE(new_layer_output, original_layer_output.detach())
   ```

2. **段階的学習**: 蒸留 → 新レイヤーのみ学習 → 全体調整

---

## 2. スケーリングと正規化

### ⚠️ 重大な注意点: L2正規化は使用禁止

**L2正規化はベクトルの magnitude（大きさ）情報を完全に破壊する。**

```python
# ❌ 絶対にやってはいけない
keys = F.normalize(keys, p=2, dim=-1)
values = F.normalize(values, p=2, dim=-1)
```

L2正規化の問題:
- すべてのベクトルが単位球面にマッピングされる
- 元の大きさの情報が完全に失われる
- 後続レイヤーが期待する情報が破壊される
- **結果: モデルの言語能力が完全に崩壊**

### ✅ 正しいアプローチ: スケーリング係数

```python
# ✅ 相対的な大きさを保持しながらスケールダウン
scale = 1.0 / (hidden_size ** 0.5)
keys = keys * scale
values = values * scale
queries = queries * scale
```

スケーリングの利点:
- 相対的な magnitude が保持される
- 数値オーバーフローを防止
- 後続レイヤーとの互換性を維持

### 数値安定性のための追加対策

```python
# クランピングで極端な値を制限
delta = torch.clamp(delta, min=-10.0, max=10.0)
```

---

## 3. fp16/bf16での数値安定性

### 問題

混合精度訓練（fp16）では数値範囲に制限がある。

| 精度 | 最大値 | 最小正の値 |
|------|--------|-----------|
| fp32 | ~3.4e38 | ~1.2e-38 |
| fp16 | ~65,504 | ~6.1e-5 |
| bf16 | ~3.4e38 | ~1.2e-38 |

### 危険な操作

1. **外積の蓄積**
   ```python
   # batch × seq の蓄積で爆発
   delta_M = torch.einsum("bsd,bse->de", keys, values)
   # 例: 17 × 13 × 2 × 2048 ≈ 900,000 > 65,504
   ```

2. **累積和**
   ```python
   # 長いシーケンスで発散
   cumsum = torch.cumsum(values, dim=1)
   ```

### 解決策

1. **事前スケーリング**: 入力を小さくしておく
2. **正規化**: `/ (batch_size * seq_len)` で平均化
3. **クランピング**: 極端な値を制限
4. **NaN/Inf検出**: 異常値をスキップ

```python
# 多重防御の例
delta_M = torch.einsum("bsd,bse->de", sigma_keys, values)
delta_M = delta_M / (batch_size * seq_len)  # 正規化
delta_M = torch.clamp(delta_M, min=-10.0, max=10.0)  # クランピング

if torch.isnan(delta_M).any() or torch.isinf(delta_M).any():
    logger.warning("NaN/Inf detected, skipping update")
    return
```

---

## 4. 蒸留損失関数の選択

### MSE損失（推奨）

```python
loss = F.mse_loss(new_output, original_output.detach())
```

- スケールと方向の両方を一致させる
- 出力分布の完全な再現を目指す

### コサイン類似度損失（非推奨）

```python
# ⚠️ スケール情報が失われている場合のみ使用
cos_sim = F.cosine_similarity(new_output, original_output, dim=-1)
loss = 1 - cos_sim.mean()
```

- 方向のみを一致させる
- スケールが異なる場合に使用（L2正規化後など）
- **L2正規化を使う場合のみの回避策**

### 選択基準

| 状況 | 推奨損失関数 |
|------|-------------|
| スケーリング使用 | MSE |
| L2正規化使用（非推奨） | コサイン類似度 |
| 出力範囲が大きく異なる | Huber損失 |

---

## 5. 学習率の設定

### 3段階学習での推奨値

| ステージ | 目的 | 学習率 |
|----------|------|--------|
| Stage 1 | 蒸留 | 5e-5 (中程度) |
| Stage 2 | 新レイヤーのみ | 5e-5 (中程度) |
| Stage 3 | 全体調整 | 1e-5 (低め) |

### 注意点

- 蒸留の学習率が高すぎると、新レイヤーが不安定になる
- 全体調整の学習率が高すぎると、既存の知識が崩壊する

---

## 6. デバッグとテスト

### 必須のテスト

1. **基本的な言語能力テスト**
   ```python
   prompts = [
       "The capital of France is",
       "1 + 1 =",
       "Hello, my name is",
   ]
   ```

2. **レイヤー出力の比較**
   ```python
   # 新旧レイヤーの出力を比較
   print(f"Original: mean={orig.mean():.4f}, std={orig.std():.4f}")
   print(f"New: mean={new.mean():.4f}, std={new.std():.4f}")
   print(f"Diff max: {(orig - new).abs().max():.4f}")
   ```

3. **NaN/Inf の監視**
   ```python
   if torch.isnan(output).any():
       logger.warning(f"NaN detected in layer {layer_idx}")
   ```

### デバッグのチェックリスト

- [ ] 基本的な言語能力は保持されているか？
- [ ] 出力のスケールは元と同程度か？
- [ ] NaN/Inf は発生していないか？
- [ ] 損失は収束しているか？

---

## 7. 失敗パターンと対策

### パターン1: 言語能力の完全崩壊

**症状**: すべてのプロンプトに対して無関係な出力

**原因**: L2正規化による magnitude 情報の喪失

**対策**: L2正規化をスケーリングに置換

### パターン2: 損失が非常に高い

**症状**: 蒸留損失が 100+ など異常に高い

**原因**: 出力スケールの不一致

**対策**:
- スケーリング係数の調整
- 損失関数の確認（MSE vs コサイン類似度）

### パターン3: NaN/Inf の発生

**症状**: WARNING: NaN/Inf detected

**原因**: fp16でのオーバーフロー

**対策**:
- スケーリング係数を小さくする
- クランピングを追加
- 正規化（/ batch_size * seq_len）を追加

### パターン4: 学習が進まない

**症状**: 損失が変化しない

**原因**: 学習率が低すぎる、または勾配が消失

**対策**:
- 学習率を上げる
- 勾配クリッピングの閾値を確認
- `requires_grad` の設定を確認

---

## 8. チェックリスト

### レイヤー置換前

- [ ] 元レイヤーの出力分布を分析（mean, std, min, max）
- [ ] fp16/bf16での数値範囲を考慮
- [ ] スケーリング戦略を決定（L2正規化は使用しない）

### 実装時

- [ ] スケーリング係数を適用
- [ ] クランピングを追加
- [ ] NaN/Inf検出を追加
- [ ] ログ出力を追加

### 学習時

- [ ] 蒸留損失が収束することを確認
- [ ] 基本的な言語能力テストを実行
- [ ] レイヤー出力を元と比較

### 評価時

- [ ] Perplexity を測定
- [ ] 下流タスク（NIAH等）を評価
- [ ] 長文での動作を確認

---

## 参考文献

- Infini-attention (Google DeepMind, 2024) - ゲート機構によるメモリ統合
- Linear Transformers Are Secretly Fast Weight Programmers - 線形Attentionの理論
- Flash Attention - 効率的なAttention実装と数値安定性
