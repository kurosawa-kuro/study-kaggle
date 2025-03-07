import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 回帰用モデル
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

RANDOM_STATE = 42

##########################################################
# 1) 関数定義
##########################################################
def load_data(train_path, test_path=None):
    """
    任意のCSV読み込み用。必要があればtrain/test両方を返す。
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path is not None else None
    return train_df, test_df

def fill_missing_values(df):
    """
    汎用的な欠損補完例（回帰タスクでも同じ）。
    - 列ごとに異なる埋め方をするなら、ここをカスタマイズ。
    """
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna('Unknown', inplace=True)
    return df

def feature_engineering(df):
    """
    汎用的な特徴量作成の例。
    - 回帰タスクならではの処理があればここに書く。
    """
    # 例：数値列の組合せ、ログ変換など
    # df['LogFeature'] = np.log1p(df['SomeNumericCol'] + 1)
    return df

def make_features_target(df, target_col, feature_cols):
    """
    引数で「目的変数の列名」と「使いたい特徴量の列名リスト」を受け取り、
    X(features)とy(target)を返す。
    ダミー変数化の処理などもここに含める。
    """
    y = df[target_col]
    X = df[feature_cols].copy()

    # 例: カテゴリ列をダミー化
    cat_cols = [c for c in feature_cols if df[c].dtype == 'object']
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y

def define_models():
    """
    回帰タスク用の主要モデルを定義して返す。
    """
    return {
        "LinearReg": LinearRegression(),
        "SVR": SVR(),
        "KNeighbors": KNeighborsRegressor(),
        "GradientBoost": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(random_state=RANDOM_STATE),
        "LightGBM": LGBMRegressor(random_state=RANDOM_STATE),
        "CatBoost": CatBoostRegressor(verbose=0, random_state=RANDOM_STATE),
    }

def define_param_grids():
    """
    回帰モデルでGridSearchCVで探索するパラメータ。
    必要に応じて各モデルのパラメータを調整。
    """
    return {
        "GradientBoost": {
            "n_estimators": [100, 200],
            "learning_rate": [0.1, 0.05],
            "max_depth": [3, 5],
        },
        "RandomForest": {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
        },
        "XGBoost": {
            "n_estimators": [100, 200],
            "learning_rate": [0.1, 0.05],
            "max_depth": [3, 5],
            "colsample_bytree": [1.0, 0.8],
        },
        "LightGBM": {
            "n_estimators": [100, 200],
            "learning_rate": [0.1, 0.05],
            "max_depth": [5, 10, -1],
            "colsample_bytree": [1.0, 0.8],
        },
        "CatBoost": {
            "iterations": [100, 200],
            "learning_rate": [0.1, 0.05],
            "depth": [3, 5],
        }
    }

def train_and_evaluate(models, param_grids, X_train_scaled, y_train, X_test_scaled, y_test):
    """
    共通の学習・評価フロー。
    - 交差検証(CV)はRMSEを評価指標とし、GridSearchCV あるいは cross_val_score を使用
    - テストデータでは RMSE と R^2 を計算して出力
    """
    results = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def rmse_cv_score(model, X_, y_):
        mse_scores = cross_val_score(model, X_, y_, cv=cv, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-mse_scores)  # neg_mean_squared_error を正にして平方根
        return rmse_scores

    for name, model in models.items():
        if name in param_grids:
            # GridSearchCV
            param_grid = param_grids[name]
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train_scaled, y_train)

            best_estimator = grid_search.best_estimator_
            best_cv_score_neg_mse = grid_search.best_score_
            best_cv_score_rmse = np.sqrt(-best_cv_score_neg_mse)

            # テストデータでのRMSE, R^2
            test_pred = best_estimator.predict(X_test_scaled)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_r2 = r2_score(y_test, test_pred)

            results[name] = {
                "Best Params": grid_search.best_params_,
                "CV Mean RMSE": f"{best_cv_score_rmse:.4f}",
                "Test RMSE": f"{test_rmse:.4f}",
                "Test R^2": f"{test_r2:.4f}"
            }
        else:
            # cross_val_scoreを使ってRMSEを算出
            rmse_scores = rmse_cv_score(model, X_train_scaled, y_train)
            cv_mean_rmse = rmse_scores.mean()

            model.fit(X_train_scaled, y_train)
            test_pred = model.predict(X_test_scaled)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_r2 = r2_score(y_test, test_pred)

            results[name] = {
                "CV Mean RMSE": f"{cv_mean_rmse:.4f}",
                "Test RMSE": f"{test_rmse:.4f}",
                "Test R^2": f"{test_r2:.4f}"
            }
    return results

##########################################################
# ★ データ読み込みを先に実行（必要に応じてパスを修正）
##########################################################
train_df, _ = load_data('train.csv')

##########################################################
# メイン関数
##########################################################
def main():
    # ==== 2) 前処理 & 特徴量作成 ====
    global train_df
    train_df = fill_missing_values(train_df)
    train_df = feature_engineering(train_df)

    # ==== 3) X, yの作成 ====
    target_col = 'SalePrice'
    feature_cols = ['LotArea', 'OverallQual', 'YearBuilt']
    X, y = make_features_target(train_df, target_col, feature_cols)

    # ==== 4) train_test_split ====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # ==== 5) スケーリング ====
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==== 6) モデル & パラメータ定義 ====
    models = define_models()
    param_grids = define_param_grids()

    # ==== 7) 学習 & 評価 ====
    results = train_and_evaluate(models, param_grids, X_train_scaled, y_train, X_test_scaled, y_test)

    # ==== 8) 結果表示 ====
    print("===== Model Results =====")
    for model_name, vals in results.items():
        if "Best Params" in vals:
            print(f"{model_name}:")
            print(f"  Best Params : {vals['Best Params']}")
            print(f"  CV Mean RMSE: {vals['CV Mean RMSE']}")
            print(f"  Test RMSE   : {vals['Test RMSE']}")
            print(f"  Test R^2    : {vals['Test R^2']}")
        else:
            print(f"{model_name}:")
            print(f"  CV Mean RMSE: {vals['CV Mean RMSE']}")
            print(f"  Test RMSE   : {vals['Test RMSE']}")
            print(f"  Test R^2    : {vals['Test R^2']}")

if __name__ == "__main__":
    main()
