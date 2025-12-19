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
