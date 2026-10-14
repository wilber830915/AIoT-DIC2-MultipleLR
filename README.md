# 50 Startups 多元線性迴歸與特徵選擇探索器

以 **CRISP-DM** 流程分析 *50 Startups* 資料集，使用多元線性迴歸預測新創公司利潤（`Profit`），
並系統性比較 **5 種特徵選擇（Feature Selection）方法**在 `k = 1…5` 個特徵下的樣本外表現，
最後以互動式網頁呈現結果。

---

## 目錄

- [專案簡介](#專案簡介)
- [研究問題](#研究問題)
- [資料集說明](#資料集說明)
- [專案結構](#專案結構)
- [環境需求](#環境需求)
- [使用方式](#使用方式)
- [分析流程（CRISP-DM）](#分析流程crisp-dm)
- [5 種特徵選擇方法](#5-種特徵選擇方法)
- [互動式網頁功能](#互動式網頁功能)
- [實驗結果](#實驗結果)
- [關鍵發現與結論](#關鍵發現與結論)
- [統計診斷](#統計診斷)
- [授權與致謝](#授權與致謝)

---

## 專案簡介

本專案同時包含**模型訓練腳本**與**視覺化網頁**兩部分：

1. `main.py`：執行完整的 CRISP-DM 分析流程，輸出 5 種特徵選擇方法在各特徵數量下的 RMSE / R² 對照表。
2. `export_model_data.py`：將各模型（係數、標準誤、t 值、p 值、信賴區間、VIF、殘差）匯出為 `model_data.js`。
3. `index.html`：純 HTML / CSS / JavaScript（搭配 Chart.js）打造的**深色專業儀表板**互動網頁，
   可即時切換特徵選擇方法、特徵數 `k`，並用任一模型試算新創公司利潤。

---

## 研究問題

- 新創公司的利潤能否由 `R&D Spend`、`Administration`、`Marketing Spend` 與 `State` 有效預測？
- 哪些特徵真正有貢獻？是否存在**更精簡**（parsimony）但預測力不減的模型？
- 隨著特徵數增加，樣本外誤差如何變化？是否存在**過擬合轉折點**？

---

## 資料集說明

檔案：`50_Startups.csv`（50 筆、5 個欄位）

| 欄位 | 型別 | 說明 |
| --- | --- | --- |
| `R&D Spend` | 數值 | 研發支出 |
| `Administration` | 數值 | 行政支出 |
| `Marketing Spend` | 數值 | 行銷支出 |
| `State` | 類別 | 所在州別（California / Florida / New York） |
| `Profit` | 數值 | **目標變數**：公司利潤 |

- 無缺失值、無重複列。
- `State` 以 **One-Hot Encoding** 編碼，並設 `drop_first=True` 以避免虛擬變數陷阱
  （以 California 為參考類別，產生 `State_Florida`、`State_New York` 兩個虛擬變數）。

---

## 專案結構

```
L5V2/
├── 50_Startups.csv          # 資料集
├── main.py                  # CRISP-DM 主分析腳本（特徵選擇 + 評估）
├── export_model_data.py     # 匯出模型資料為 model_data.js（供網頁使用）
├── model_data.js            # 網頁資料來源（各方法 × 各 k 的完整模型結果）
├── index.html               # 互動式網頁（深色儀表板）
├── startup_profit_model.joblib  # main.py 訓練後儲存的模型（若有執行）
└── README.md
```

---

## 環境需求

- Python 3.10+
- 套件：

```bash
pip install pandas numpy scikit-learn statsmodels scipy joblib matplotlib
```

主要使用：`pandas`、`numpy`、`scikit-learn`、`statsmodels`、`scipy`、`joblib`。
網頁端使用 CDN 載入 `Chart.js`、Google Fonts 與 FontAwesome，無需額外安裝。

---

## 使用方式

### 1. 執行主分析

```bash
python3 main.py
```

會印出資料探索、5 種特徵選擇方法的完整結果表，以及各方法的最佳特徵組合摘要。

### 2. 重新產生網頁資料

```bash
python3 export_model_data.py
```

會重新計算所有模型並覆寫 `model_data.js`。

### 3. 開啟互動網頁

直接雙擊 `index.html`，或使用本機伺服器（建議，避免瀏覽器跨來源限制）：

```bash
python3 -m http.server 8000
# 瀏覽器開啟 http://localhost:8000/index.html
```

---

## 分析流程（CRISP-DM）

| 階段 | 內容 |
| --- | --- |
| 1. Business Understanding | 以支出與州別預測新創利潤，屬監督式迴歸問題 |
| 2. Data Understanding | 檢視前 5 列、`info()`、描述統計、缺失值、重複列、相關矩陣、利潤分布圖 |
| 3. Data Preparation | One-Hot 編碼 `State`、切分訓練/測試集（80/20, `random_state=42`）、`StandardScaler` 標準化 |
| 4. Modeling | 以 `LinearRegression` 搭配 5 種特徵選擇方法，遍歷 `k = 1…5` |
| 5. Evaluation | 以測試集 RMSE、R² 評估；`statsmodels` OLS 提供係數檢定與 VIF |
| 6. Deployment | 輸出 `model_data.js` 與互動式網頁 `index.html` |

> **註：** 特徵選擇階段在**標準化**特徵上進行（避免尺度影響貪婪搜尋）；
> 最終迴歸係數與統計檢定則在**原始尺度**上以 OLS 重新配適，以維持可解釋性。

---

## 5 種特徵選擇方法

| # | 方法 | 類別 | 說明 |
| --- | --- | --- | --- |
| 1 | **Forward Selection** | Wrapper | 從空集合開始，每次加入能最大幅提升交叉驗證表現的特徵（`SequentialFeatureSelector`, `direction="forward"`） |
| 2 | **Backward Elimination** | Wrapper | 從全部特徵開始，每次移除最不重要的特徵（`direction="backward"`） |
| 3 | **Lasso (L1)** | Embedded | L1 正則化將不重要特徵係數壓縮為 0，依 `|coef|` 排序取前 k 個（`LassoCV`） |
| 4 | **Random Forest Importance** | Embedded | 以隨機森林的 `feature_importances_`（不純度下降）排序取前 k 個 |
| 5 | **Mutual Information** | Filter | 以 `mutual_info_regression` 衡量特徵與目標的非線性相關，排序取前 k 個 |

> `SequentialFeatureSelector` 不允許選取「全部特徵」（`k = n_features`），
> 故 Forward / Backward 的迴圈僅到 `k = 4`；`k = 5` 以「全模型」補上，便於跨方法比較。

---

## 互動式網頁功能

- **方法切換**：5 種特徵選擇方法 + 「比較全部 5 種方法」。
- **特徵數選擇**：`k = 1…5`，甜蜜點自動以綠色標示、過擬合以琥珀色標示。
- **KPI 指標卡**：Test RMSE、R²、Adjusted R²、MAE、F-Statistic（依當前模型即時更新）。
- **雙圖表**：RMSE 與 R² 對特徵數變化圖，**點擊資料點即可切換 k**。
- **迴歸方程式**：以原始尺度呈現當前模型的 OLS 方程式（含正負號著色）。
- **係數統計表**：Estimate、Std. Error、t-Stat、p-Value、95% 信賴區間、VIF。
- **即時利潤預測器**：拖動滑桿輸入 R&D / Marketing / Administration 與州別，
  即時以當前模型輸出預測利潤；未使用的特徵輸入框會自動變灰。
- **殘差與離群值診斷**：測試集 10 筆樣本依絕對殘差排序。

---

## 實驗結果

以下為各方法在測試集上的 **RMSE**（母數越小越好）與 **R²**（越大越好）：

| 方法 \ k | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| Forward Selection | **7,714.33** | 9,152.27 | 8,995.91 | 9,055.96 | 9,055.96 |
| Backward Elimination | **7,714.33** | 9,152.27 | 8,995.91 | 9,055.96 | 9,055.96 |
| Lasso (L1) | **7,714.33** | 8,206.33 | 8,995.91 | 9,015.11 | 9,055.96 |
| RF Importance | **7,714.33** | 8,206.33 | 8,995.91 | 9,055.96 | 9,055.96 |
| Mutual Information | **7,714.33** | 8,206.33 | 8,242.78 | 8,254.69 | 9,055.96 |

對應 R²（以 Lasso 為例）：k=1 → **0.9265**、k=2 → 0.9168、k=3 → 0.9001、k=4 → 0.8996、k=5 → 0.8987。

**各方法最佳特徵組合：全部落在 `k = 1`，即僅使用 `R&D Spend`**。

---

## 關鍵發現與結論

1. **`R&D Spend` 是主導變數**：單一特徵模型（k=1）即達到 R² ≈ 0.927，RMSE ≈ 7,714，
   是 5 種方法、所有 k 中的最佳表現。
2. **精簡性勝出（Occam's Razor）**：無論用哪種選擇方法，加入第二個以上特徵都會使**測試集**誤差上升，
   顯示其餘特徵帶來的是變異而非資訊。
3. **過擬合轉折**：`k = 5`（全模型）在 Lasso / RF / MI 上退化至 RMSE ≈ 9,056，為所有設定的最差表現之一。
4. **`State` 幾乎無貢獻**：州別虛擬變數的係數極小、p 值不顯著，對預測利潤幫助有限。
5. **迴歸方程式（k=1，原始尺度）**：

   ```
   Profit ≈ 49,336.67 + 0.8536 × (R&D Spend)
   ```

---

## 統計診斷

- 單一預測變數模型的 **VIF = 1.0**（無共線性問題）。
- `R&D Spend` 係數的 **t ≈ 26.0、p < 0.0001**，達高度統計顯著。
- 全模型下可觀察 `R&D Spend` 與 `Marketing Spend` 之間存在中度相關（相關係數 ≈ 0.72），
  為多特徵模型不穩定與效能下降的原因之一。

---

## 授權與致謝

- **資料集**：50 Startups Dataset。
- **方法論**：CRISP-DM（Cross-Industry Standard Process for Data Mining）。
- **函式庫**：`scikit-learn`、`statsmodels`、`pandas`、`numpy`、`Chart.js`。
