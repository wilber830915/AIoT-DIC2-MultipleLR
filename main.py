import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 1. DATA PREPARATION (資料預處理)
# ============================================================
df = pd.read_csv("50_Startups.csv")

# 先將 One-Hot Encoding 套用到類別特徵 State (避免虛擬變數陷阱 drop='first')
df_encoded = pd.get_dummies(df, columns=["State"], drop_first=True, dtype=int)

X = df_encoded.drop(columns=["Profit"])
y = df_encoded["Profit"]

feature_names = np.array(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 數值標準化
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

max_features = X_train.shape[1] # 最大特徵數量 (在此資料集中為 5)

# ============================================================
# 2. FEATURE SELECTION SCHEMES (5 種特徵選擇方案)
# ============================================================
results = []

def evaluate_subset(selected_cols, scheme_name, num_f):
    model = LinearRegression()
    model.fit(X_train_scaled[selected_cols], y_train)
    y_pred = model.predict(X_test_scaled[selected_cols])
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "Scheme": scheme_name,
        "Num_Features": num_f,
        "Selected_Features": list(selected_cols),
        "RMSE": rmse,
        "R2": r2
    }

# --- Scheme 1: Forward Selection (SFS) ---
# SFS/SBS require 1 <= n_features_to_select <= n_features - 1
for k in range(1, max_features):
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction="forward")
    sfs.fit(X_train_scaled, y_train)
    cols = feature_names[sfs.get_support()]
    results.append(evaluate_subset(cols, "Forward Selection", k))

# --- Scheme 2: Backward Elimination (SBS) ---
for k in range(1, max_features):
    sbs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction="backward")
    sbs.fit(X_train_scaled, y_train)
    cols = feature_names[sbs.get_support()]
    results.append(evaluate_subset(cols, "Backward Elimination", k))

# --- Scheme 3: Lasso (L1) Importance ---
lasso = LassoCV(cv=5, random_state=42).fit(X_train_scaled, y_train)
lasso_importance = np.abs(lasso.coef_)
lasso_ranked_cols = feature_names[np.argsort(lasso_importance)[::-1]]

for k in range(1, max_features + 1):
    cols = lasso_ranked_cols[:k]
    results.append(evaluate_subset(cols, "Lasso (L1)", k))

# --- Scheme 4: Random Forest Importance ---
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_scaled, y_train)
rf_importance = rf.feature_importances_
rf_ranked_cols = feature_names[np.argsort(rf_importance)[::-1]]

for k in range(1, max_features + 1):
    cols = rf_ranked_cols[:k]
    results.append(evaluate_subset(cols, "RF Importance", k))

# --- Scheme 5: Mutual Information ---
mi_scores = mutual_info_regression(X_train_scaled, y_train, random_state=42)
mi_ranked_cols = feature_names[np.argsort(mi_scores)[::-1]]

for k in range(1, max_features + 1):
    cols = mi_ranked_cols[:k]
    results.append(evaluate_subset(cols, "Mutual Information", k))

results_df = pd.DataFrame(results)

# ============================================================
# 3. VISUALIZATION (圖表繪製)
# ============================================================
SCHEME_SPECS = {
    "Forward Selection":    {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "Backward Elimination": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    "Lasso (L1)":           {"color": "#d62728", "marker": "D", "linestyle": ":"},
    "RF Importance":        {"color": "#9467bd", "marker": "v", "linestyle": "-."},
    "Mutual Information":   {"color": "#2ca02c", "marker": "^", "linestyle": "-"}
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for scheme_name, spec in SCHEME_SPECS.items():
    sub_df = results_df[results_df['Scheme'] == scheme_name].sort_values('Num_Features')
    
    # 左圖：RMSE
    axes[0].plot(
        sub_df['Num_Features'], sub_df['RMSE'], 
        label=scheme_name, color=spec['color'], 
        marker=spec['marker'], linestyle=spec['linestyle'], 
        linewidth=2, markersize=7
    )
    
    # 右圖：R-squared
    axes[1].plot(
        sub_df['Num_Features'], sub_df['R2'], 
        label=scheme_name, color=spec['color'], 
        marker=spec['marker'], linestyle=spec['linestyle'], 
        linewidth=2, markersize=7
    )

# 設定左圖格式
axes[0].set_title("RMSE by Number of Features", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Number of Features", fontsize=11)
axes[0].set_ylabel("RMSE", fontsize=11)
axes[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend(title="Selection Scheme", fontsize=9)

# 設定右圖格式
axes[1].set_title("R-squared by Number of Features", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Number of Features", fontsize=11)
axes[1].set_ylabel("R-squared", fontsize=11)
axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend(title="Selection Scheme", fontsize=9)

plt.tight_layout()
plt.show()

# ============================================================
# 4. OUTPUT TABLES (表格輸出)
# ============================================================
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n=================== 5 個 Scheme 完整測試結果 ===================")
print(results_df.to_string(index=False))

# 找出各 Scheme 的最佳表現特徵組合（最小 RMSE）
best_per_scheme = results_df.loc[results_df.groupby('Scheme')['RMSE'].idxmin()]

print("\n=================== 各 Scheme 最佳特徵組合摘要 ===================")
print(best_per_scheme[['Scheme', 'Num_Features', 'Selected_Features', 'RMSE', 'R2']].to_string(index=False))