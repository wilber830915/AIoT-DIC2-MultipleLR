import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    SequentialFeatureSelector,
    mutual_info_regression,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

RANDOM_STATE = 42


def main():
    df = pd.read_csv("50_Startups.csv")
    df_encoded = pd.get_dummies(df, columns=["State"], drop_first=True, dtype=int)

    X = df_encoded.drop(columns=["Profit"])
    y = df_encoded["Profit"]
    feature_names = np.array(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    max_features = X_train.shape[1]

    selected_by_scheme = {}

    # Scheme 1: Forward Selection
    forward = {}
    for k in range(1, max_features):
        sfs = SequentialFeatureSelector(
            LinearRegression(),
            n_features_to_select=k,
            direction="forward",
            cv=5,
        )
        sfs.fit(X_train_scaled, y_train)
        forward[k] = list(feature_names[sfs.get_support()])
    forward[max_features] = list(feature_names)
    selected_by_scheme["Forward Selection"] = forward

    # Scheme 2: Backward Elimination
    backward = {}
    for k in range(1, max_features):
        sbs = SequentialFeatureSelector(
            LinearRegression(),
            n_features_to_select=k,
            direction="backward",
            cv=5,
        )
        sbs.fit(X_train_scaled, y_train)
        backward[k] = list(feature_names[sbs.get_support()])
    backward[max_features] = list(feature_names)
    selected_by_scheme["Backward Elimination"] = backward

    # Scheme 3: Lasso (L1)
    lasso = LassoCV(cv=5, random_state=RANDOM_STATE).fit(X_train_scaled, y_train)
    lasso_ranked = feature_names[np.argsort(np.abs(lasso.coef_))[::-1]]
    selected_by_scheme["Lasso (L1)"] = {
        k: list(lasso_ranked[:k]) for k in range(1, max_features + 1)
    }

    # Scheme 4: Random Forest Importance
    rf = RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE
    ).fit(X_train_scaled, y_train)
    rf_ranked = feature_names[np.argsort(rf.feature_importances_)[::-1]]
    selected_by_scheme["RF Importance"] = {
        k: list(rf_ranked[:k]) for k in range(1, max_features + 1)
    }

    # Scheme 5: Mutual Information
    mi_scores = mutual_info_regression(
        X_train_scaled, y_train, random_state=RANDOM_STATE
    )
    mi_ranked = feature_names[np.argsort(mi_scores)[::-1]]
    selected_by_scheme["Mutual Information"] = {
        k: list(mi_ranked[:k]) for k in range(1, max_features + 1)
    }

    model_data = {}

    for scheme, by_k in selected_by_scheme.items():
        rows = []
        for k in sorted(by_k.keys()):
            cols = by_k[k]

            # Fit interpretable OLS on RAW feature scale for reporting + predictor.
            Xtr = sm.add_constant(X_train[cols].astype(float), has_constant="add")
            Xte = sm.add_constant(X_test[cols].astype(float), has_constant="add")
            ols = sm.OLS(y_train.astype(float), Xtr).fit()

            y_pred = ols.predict(Xte)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            n = len(y_test)
            p = len(cols)
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else None

            f_stat = ols.fvalue
            f_stat = float(f_stat) if f_stat is not None and np.isfinite(f_stat) else None

            conf = ols.conf_int()
            coefficients = []
            for i, name in enumerate(ols.params.index):
                display = "Intercept" if name == "const" else name

                if name == "const":
                    vif = None
                elif p < 2:
                    vif = 1.0
                else:
                    vif_cols = ["const"] + cols
                    design = sm.add_constant(
                        X_train[cols].astype(float), has_constant="add"
                    )[vif_cols]
                    vif = float(
                        variance_inflation_factor(design.values, vif_cols.index(name))
                    )

                coefficients.append(
                    {
                        "feature": display,
                        "coef": float(ols.params[name]),
                        "stderr": float(ols.bse[name]),
                        "tstat": float(ols.tvalues[name]),
                        "pvalue": float(ols.pvalues[name]),
                        "ci_lower": float(conf.loc[name, 0]),
                        "ci_upper": float(conf.loc[name, 1]),
                        "vif": vif,
                    }
                )

            residuals = y_test.astype(float) - y_pred
            order = np.argsort(-np.abs(residuals.values))
            outliers = []
            for rank, pos in enumerate(order, start=1):
                idx = y_test.index[pos]
                actual = float(y_test.iloc[pos])
                pred = float(y_pred.iloc[pos])
                resid = actual - pred
                outliers.append(
                    {
                        "rank": rank,
                        "index": int(idx),
                        "actual": actual,
                        "predicted": pred,
                        "residual": resid,
                        "abs_error": abs(resid),
                        "pct_error": abs(resid) / actual * 100 if actual else 0.0,
                    }
                )

            rows.append(
                {
                    "k": int(k),
                    "features": list(cols),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "r2": float(r2),
                    "adj_r2": (float(adj_r2) if adj_r2 is not None else None),
                    "f_stat": f_stat,
                    "coefficients": coefficients,
                    "outliers": outliers,
                }
            )

        model_data[scheme] = rows

    with open("model_data.js", "w") as f:
        f.write("window.MODEL_DATA = ")
        f.write(json.dumps(model_data, indent=2))
        f.write(";\n")

    print("Wrote model_data.js")
    for scheme, rows in model_data.items():
        best = min(rows, key=lambda r: r["rmse"])
        print(
            f"  {scheme:<22} best k={best['k']}  RMSE={best['rmse']:,.2f}  "
            f"R2={best['r2']:.4f}  features={best['features']}"
        )


if __name__ == "__main__":
    main()
