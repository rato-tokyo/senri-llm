# メモリレイヤー統合に関する知見

## 2024-12-19 診断結果

### 問題の発見

SmolLM-135MにSenriメモリレイヤーを統合した際、NIAHテストが0%という結果になった。
原因を調査した結果、以下の知見を得た。

### 診断プロセス

1. **学習後のモデルでテスト** → 失敗（無関係な会話を生成）
2. **変換直後のモデルでテスト** → 失敗（同様に機能しない）
3. **重みのロード確認** → 正常（Q/K/V projectionは正しくロード）
4. **メモリの動作確認** → 正常（update/retrieveは機能）
5. **レイヤー出力の比較** → **問題発見**

### 根本原因

**メモリレイヤー（Layer 15）の出力がベースモデルと大きく異なる：**

| 指標 | Base (Softmax Attention) | Senri (Linear Attention + Memory) |
|------|--------------------------|-----------------------------------|
| std  | 1.19                     | 0.79                              |
| min  | -21.53                   | -2.67                             |
| max  | 5.51                     | 3.96                              |
| **Diff max** | -                | **21.10**                         |

通常レイヤー（Layer 11など）は完全に一致（Diff = 0.0000）。

### 原因の解釈

1. **Softmax Attention vs Linear Attention の本質的な違い**
   - Softmax: 位置依存の重み付け、鋭いピーク
   - Linear (ELU+1): 位置に依存しない、滑らかな分布

2. **事前学習との不整合**
   - ベースモデルはSoftmax Attentionで事前学習済み
   - 後続レイヤーはSoftmax Attentionの出力分布を期待
   - 線形Attentionの異なる出力分布が「ノイズ」として伝播

3. **結果**
   - メモリレイヤーが情報を破壊
   - モデル全体の出力が劣化
   - NIAHタスクが完全に失敗

### 重要な知見

> **線形Attentionレイヤーは、単純にSoftmax Attentionレイヤーと置換するだけでは機能しない。**
> **出力分布の違いを吸収するための学習が必要。**

### 解決策: 3段階学習アプローチ

#### 第1段階: レイヤー蒸留

メモリレイヤーの出力が、対応するベースレイヤーの出力に近づくように学習。

```python
# 蒸留損失
loss = MSE(memory_layer_output, base_layer_output.detach())
```

**目的**: メモリレイヤーの出力分布をSoftmax Attentionに近づける

#### 第2段階: メモリレイヤーのみファインチューニング

- メモリレイヤー以外を全てフリーズ
- 言語モデリング損失で学習
- メモリレイヤーがコンテキスト内で機能するように調整

```python
# フリーズ
for name, param in model.named_parameters():
    if "layers.15" not in name and "layers.20" not in name and "layers.25" not in name:
        param.requires_grad = False
```

**目的**: メモリ機能の獲得（既存の言語能力を保持しながら）

#### 第3段階: 全体ファインチューニング

- 全パラメータをアンフリーズ
- 低学習率で全体を調整
- 各レイヤー間の協調を最適化

**目的**: メモリレイヤーと通常レイヤーの協調動作

### 実装上の注意点

1. **第1段階の重要性**
   - 蒸留なしでいきなり学習すると、メモリレイヤーの異常出力が他のレイヤーを破壊
   - 蒸留により「安全な初期状態」を確立

2. **学習率の設定**
   - 第1段階: 高め（1e-4）- 迅速に出力分布を近づける
   - 第2段階: 中程度（5e-5）- メモリ機能の獲得
   - 第3段階: 低め（1e-5）- 微調整

3. **評価指標**
   - 第1段階: レイヤー出力のMSE
   - 第2段階: Perplexity + 簡易NIAH
   - 第3段階: 完全なNIAH評価

### 代替アプローチ

1. **ゲート機構の追加（Infini-Attention方式）**
   ```python
   output = gate * memory_output + (1 - gate) * local_attention_output
   ```
   - gateは学習可能なパラメータ
   - 初期値を0に設定することで、最初はlocal attentionのみ使用
   - 徐々にメモリを活用するように学習

2. **残差接続の追加**
   ```python
   output = memory_output + residual_weight * input
   ```
   - 入力を直接出力に加算
   - メモリが失敗しても情報が保持される

### 参考: 関連研究

- **Infini-attention (Google DeepMind, 2024)**: ローカルAttentionとメモリの混合にゲート機構を使用
- **Linear Transformers Are Secretly Fast Weight Programmers**: 線形Attentionの理論的基盤

### 今後の実験計画

1. 3段階学習の実装と検証
2. ゲート機構の追加と比較
3. より大きなモデル（1B+）での検証
4. 長コンテキストベンチマークでの評価

---

## 2024-12-19 追加知見: メモリ更新時のNaN/Inf問題

### 問題の発見

3段階学習の実行中、以下の警告が頻発した：

```
WARNING: NaN/Inf detected in memory update delta_M, skipping update.
keys stats: min=-17.3438, max=17.3750
values stats: min=-10.6484, max=12.8359
```

### 根本原因

**1. QKV投影後の値が大きすぎる**

| テンソル | 最小値 | 最大値 |
|----------|--------|--------|
| keys     | -17.3  | +17.4  |
| values   | -10.6  | +12.8  |

**2. 外積計算での数値爆発**

```python
# メモリ更新の計算
sigma_keys = ELU(keys) + 1  # 正の値に変換、max ≈ 18.4
delta_M = torch.einsum("bsd,bse->de", sigma_keys, values)
```

外積の一要素の計算:
```
≈ sigma_keys_max × values_max × batch_size × seq_len
≈ 18.4 × 12.8 × 2 × 2048
≈ 964,000
```

**3. fp16のオーバーフロー**

- fp16の最大表現可能値: 約65,504
- 計算値（~964,000）がこれを大幅に超過
- Inf発生 → NaN伝播

### 解決策

**1. L2正規化の追加（SenriAttention）**

```python
# 投影後、メモリ操作前にL2正規化を適用
keys = F.normalize(keys, p=2, dim=-1)    # ノルム=1に正規化
values = F.normalize(values, p=2, dim=-1)
queries = F.normalize(queries, p=2, dim=-1)
```

効果: 値の範囲が [-1, +1] に制限され、外積計算の爆発を防止

**2. Clampingの追加（TensorMemory）**

```python
# delta_Mのクリッピング（更新量の制限）
delta_M = torch.clamp(delta_M, min=-10.0, max=10.0)

# メモリMのクリッピング（蓄積の制限）
self.M = torch.clamp(self.M + delta_M, min=-100.0, max=100.0).detach()

# 正規化項zのクリッピング
self.z = torch.clamp(self.z + delta_z, min=self.eps, max=1000.0).detach()
```

効果: 複数レベルでの安全策により、万が一の異常値も抑制

### 修正箇所

| ファイル | 変更内容 |
|----------|----------|
| `src/attention/senri_attention.py` | L2正規化の追加（lines 124-127） |
| `src/memory/base_memory.py` | delta_Mクリッピング（line 146） |
| `src/memory/base_memory.py` | Mクリッピング（line 159） |
| `src/memory/base_memory.py` | zクリッピング（line 173） |

### 教訓

1. **Linear Attentionでは明示的な正規化が必須**
   - Softmax Attentionは暗黙的に正規化（和が1になる）
   - Linear Attentionは正規化がないため、値が発散しやすい

2. **fp16使用時は数値範囲に特に注意**
   - fp16の最大値は約65,000
   - 蓄積操作（累積和など）では特に危険

3. **防御的プログラミングの重要性**
   - 単一の対策ではなく、複数レベルで安全策を講じる
   - 正規化 + クリッピング + NaN/Inf検出・スキップ

4. **ログの重要性**
   - NaN/Inf発生時に入力統計を出力することで、原因特定が容易に
   - 「keys stats: min=-17.3, max=17.3」という情報が問題解決の鍵

### Linear Attention vs Softmax Attention の数値安定性

| 特性 | Softmax Attention | Linear Attention |
|------|-------------------|------------------|
| 正規化 | 暗黙的（softmax） | なし（手動必要） |
| 出力範囲 | 制限あり | 制限なし |
| 勾配 | 安定 | 発散しやすい |
| 推奨対策 | 特になし | L2正規化 + Clamping |
