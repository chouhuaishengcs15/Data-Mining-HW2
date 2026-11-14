# P5 消融實驗與 test 分析程式

本目錄保存可重現的實驗程式和執行說明。書面報告與簡報請在組內共編文件和簡報編輯；本目錄不存放報告正文或簡報稿。協作規則見 `report/README.md`。

## Validation 消融

從 repository 根目錄執行：

```sh
python ablation/run_p5.py
```

程式固定共用資料切分、P3 Autoencoder、normal-only training 與 seeds 0–4。特徵消融比較 ALL21、NO_VARIABILITY17、DROP_WIDTH20、PRUNED18；異常偵測專屬 scoring 消融固定 DROP_WIDTH20，只改 Mahalanobis、MSE、cosine。門檻按 P3/P4 最終 protocol 以 validation Best-F2 選取。AP 和 ROC-AUC 為連續分數指標。輸出在 `results/p5/ablation/`，包括逐 seed 指標、逐筆 validation 分數、比較表、圖和來源版本紀錄。

## Test 錯誤案例分析

P3 已完成一次 Best-F2 test 評估；輸入在 `results/tier2/final_mahalanobis_best_f2_DROP_WIDTH20/`。從 repository 根目錄執行：

```sh
python ablation/analyze_test.py
```

這個程式使用已發布的 test 預測、門檻、原始測試資料及 feature reconstruction errors，不重訓或調整模型。輸出在 `results/p5/analysis/`，包含逐 seed 指標、跨 seed 誤判統計、原始特徵、DP 子群統計與診斷圖。個案選擇規則與輸入檔雜湊記在 `provenance.json`。

逐特徵平方重建誤差只能作為診斷，不能精確分解 Mahalanobis 分數。
