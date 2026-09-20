# 金融交易警示與可疑帳戶偵測

以 LightGBM 結合交易行為、時間、幣別與帳戶關係圖特徵，預測需優先調查的可疑帳戶。第二版加入一階／二階可疑鄰居比例、in/out degree 與交易方向統計。

## 特徵

- 交易金額、頻率、時間分布與幣別轉換
- 帳戶流入／流出 degree 與對手方多樣性
- 一階與二階可疑鄰居比例
- 欠採樣、class weight 與閾值選擇
- LightGBM feature importance

## 執行

```powershell
$env:TRANSACTION_DATA_DIR = "D:\path\to\competition-data"
python -m venv .venv
pip install -r requirements.txt
python src/train_graph_features.py
```

資料目錄預期包含 `acct_transaction.csv`、`acct_alert.csv`、`acct_predict.csv` 與選用的 `fx.csv`。

## 版本

- `train_baseline.py`：交易聚合特徵 baseline
- `train_graph_features.py`：加入 graph-neighborhood 特徵的版本

## 評估狀態

原始專案只保留 submission，沒有足以查核的 validation JSON 或 baseline 對照，因此本 repository 不宣稱 F1/AUC。建議下一次實驗保存 split 策略、PR-AUC、F1、recall@alert-budget 與 baseline 差異。約 737 MB 的原始交易資料、帳戶清單與 submission 未提交。
