import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42

# ==================================================================
# 関数定義
# ==================================================================
def load_data(train_path, test_path=None):
    """
    任意のCSV読み込み用。必要があればtrain/test両方を返す。
    test_pathを使わない場合は、Noneにしておく。
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path is not None else None
    return train_df, test_df

def fill_missing_values(df):
    """
    汎用的な欠損補完例。
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
    汎用的な特徴量作成。
    """
    # df['NewFeature'] = df['SomeNumericCol'] * df['AnotherCol']
    return df

def make_features_target(df, target_col, feature_cols):
    """
    引数で「目的変数の列名」と「使いたい特徴量の列名リスト」を受け取り、
    X(features)とy(target)を返す。
    """
    y = df[target_col]
    X = df[feature_cols].copy()

    # カテゴリ列のダミー変数化
    cat_cols = [c for c in feature_cols if df[c].dtype == 'object']
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y

def define_models():
    """
    主要な分類モデルを一括で定義して返す。
    """
    return {
        "LogisticReg": LogisticRegression(random_state=RANDOM_STATE, max_iter=200),
        "LinearSVC": LinearSVC(random_state=RANDOM_STATE),
        "KNeighbors": KNeighborsClassifier(),
        "GradientBoost": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
    }

def define_param_grids():
    """
    GridSearchCVで探索するパラメータを定義。
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
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score

    results = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        if name in param_grids:
            grid_search = GridSearchCV(
                model,
                param_grids[name],
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train_scaled, y_train)

            best_estimator = grid_search.best_estimator_
            best_cv_score = grid_search.best_score_
            test_pred = best_estimator.predict(X_test_scaled)
            test_acc = accuracy_score(y_test, test_pred)

            results[name] = {
                "Best Params": grid_search.best_params_,
                "CV Mean Acc": f"{best_cv_score:.4f}",
                "Test Acc": f"{test_acc:.4f}"
            }
        else:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
            cv_mean_score = cv_scores.mean()

            model.fit(X_train_scaled, y_train)
            test_pred = model.predict(X_test_scaled)
            test_acc = accuracy_score(y_test, test_pred)

            results[name] = {
                "CV Mean Acc": f"{cv_mean_score:.4f}",
                "Test Acc": f"{test_acc:.4f}"
            }
    return results

# ==================================================================
# ここで先にデータ読み込み (上部に配置しておく)
# ==================================================================
train_df, _ = load_data('train.csv')

# ==================================================================
# main関数
# ==================================================================
def main():
    # 2) 前処理 & 特徴量作成
    global train_df  # train_dfを関数内で更新する場合には必要
    train_df = fill_missing_values(train_df)
    train_df = feature_engineering(train_df)

    # 3) X, yの作成 (列名を適宜書き換える)
    target_col = 'Survived'
    feature_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
    X, y = make_features_target(train_df, target_col, feature_cols)

    # 4) train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=RANDOM_STATE
    )

    # 5) スケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6) モデル & パラメータ定義
    models = define_models()
    param_grids = define_param_grids()

    # 7) 学習 & 評価
    results = train_and_evaluate(models, param_grids, X_train_scaled, y_train, X_test_scaled, y_test)

    # 8) 結果表示
    print("===== Model Results =====")
    for model_name, vals in results.items():
        if "Best Params" in vals:
            print(f"{model_name}:")
            print(f"  Best Params : {vals['Best Params']}")
            print(f"  CV Mean Acc : {vals['CV Mean Acc']}")
            print(f"  Test Acc    : {vals['Test Acc']}")
        else:
            print(f"{model_name}:")
            print(f"  CV Mean Acc : {vals['CV Mean Acc']}")
            print(f"  Test Acc    : {vals['Test Acc']}")

if __name__ == "__main__":
    main()
