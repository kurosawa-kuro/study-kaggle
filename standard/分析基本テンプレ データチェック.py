import pandas as pd
import numpy as np

####################################
# 1) データ読み込み用 関数定義
####################################
def load_data(train_path, test_path=None):
    """
    任意のCSV読み込み用。必要があればtrain/test両方を返す。
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path is not None else None
    return train_df, test_df

####################################
# 2) 基本情報の確認
####################################
def show_basic_info(df, df_name="DataFrame"):
    """
    - データの先頭行、サイズ、info()等を表示
    - カラム一覧・データ型・メモリ使用量の概要も info() で確認
    """
    print(f"\n====== Basic Info: {df_name} ======")
    print(f"Shape: {df.shape}")  # 行数・列数
    print("\n--- Head(5) ---")
    print(df.head(5))
    print("\n--- Info ---")
    print(df.info())  # infoの出力はNoneを返すが、ここで直接print

####################################
# 3) 欠損値の確認
####################################
def check_missing_values(df, df_name="DataFrame"):
    """
    - 欠損値の総数、カラムごとの欠損値数を表示
    """
    print(f"\n====== Missing Values: {df_name} ======")
    total_missing = df.isnull().sum().sum()
    print(f"Total missing values: {total_missing}")
    missing_per_col = df.isnull().sum()
    missing_per_col = missing_per_col[missing_per_col > 0]
    if not missing_per_col.empty:
        print("\n--- Missing per column ---")
        print(missing_per_col)
    else:
        print("No missing values.")

####################################
# 4) 重複データの確認
####################################
def check_duplicates(df, df_name="DataFrame"):
    """
    - 重複行の数を表示。不要な場合は削除検討。
    """
    print(f"\n====== Duplicates Check: {df_name} ======")
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"Found {dup_count} duplicate rows.")
    else:
        print("No duplicate rows found.")

####################################
# 5) カラムごとのユニーク数・カーディナリティ確認
####################################
def check_unique_values(df, df_name="DataFrame", top_n=5):
    """
    - 各カラムのユニーク値数を表示
    - カーディナリティが高いカテゴリ列などを把握するのに役立つ
    - さらに上位N件の値サンプルを表示
    """
    print(f"\n====== Unique Values (Top {top_n} samples): {df_name} ======")
    for col in df.columns:
        unique_count = df[col].nunique(dropna=False)
        print(f"{col}: {unique_count} unique values")
        value_sample = df[col].drop_duplicates().head(top_n).values
        print(f"  Sample: {value_sample}")

####################################
# 6) 基本統計量の確認 (数値/カテゴリ)
####################################
def show_descriptive_stats(df, numeric_cols=None, categorical_cols=None, df_name="DataFrame"):
    """
    数値列・カテゴリ列で分けて describe() や value_counts() などを表示。
    """
    print(f"\n====== Descriptive Stats: {df_name} ======")
    # 数値列
    import numpy as np
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print(f"\n--- Numeric Columns: {numeric_cols} ---")
        print(df[numeric_cols].describe())
    else:
        print("\nNo numeric columns found.")

    # カテゴリ列
    if categorical_cols is None:
        cat_dtypes = ["object", "category"]
        categorical_cols = [col for col in df.columns if df[col].dtype.name in cat_dtypes]
    if categorical_cols:
        print(f"\n--- Categorical Columns: {categorical_cols} ---")
        for col in categorical_cols:
            print(f"\nValue Counts: {col}")
            print(df[col].value_counts(dropna=False))
    else:
        print("\nNo categorical columns found.")

####################################
# 7) 歪度(Skewness)・尖度(Kurtosis) の確認
####################################
def check_skew_kurtosis(df, numeric_cols=None, df_name="DataFrame"):
    """
    数値列に対して、歪度(skew)・尖度(kurtosis)を計算して一覧表示。
    """
    print(f"\n====== Skewness & Kurtosis: {df_name} ======")
    import numpy as np
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found.")
        return
    stats = []
    for col in numeric_cols:
        col_skew = df[col].skew()
        col_kurt = df[col].kurt()
        stats.append((col, col_skew, col_kurt))
    print("Column         Skewness       Kurtosis")
    for (c, s, k) in stats:
        print(f"{c:<14} {s:>10.4f} {k:>10.4f}")

####################################
# 8) 外れ値の簡易チェック (Zスコア or IQR)
####################################
def check_outliers(df, numeric_cols=None, method='zscore', threshold=3.0, df_name="DataFrame"):
    """
    - method='zscore' の場合、Zスコアの絶対値が threshold を超えるデータ数をカウント
    - method='iqr' の場合、1.5倍IQRを超えるデータ数をカウント
    """
    print(f"\n====== Outlier Check ({method}): {df_name} ======")
    import numpy as np
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found.")
        return

    if method == 'zscore':
        df_zscore = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        outlier_count = (df_zscore.abs() > threshold).sum()
        for col in numeric_cols:
            if outlier_count[col] > 0:
                print(f"{col}: {outlier_count[col]} outliers (threshold={threshold})")
    elif method == 'iqr':
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        for col in numeric_cols:
            lb = lower_bound[col]
            ub = upper_bound[col]
            mask_outliers = (df[col] < lb) | (df[col] > ub)
            out_count = mask_outliers.sum()
            if out_count > 0:
                print(f"{col}: {out_count} outliers (IQR method)")
    else:
        print("Unknown method. Use 'zscore' or 'iqr'.")

####################################
# 9) 相関行列の表示
####################################
def check_correlations(df, numeric_cols=None, df_name="DataFrame"):
    """
    数値列同士の相関係数を表示。
    """
    print(f"\n====== Correlation Matrix: {df_name} ======")
    import numpy as np
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        print(corr_matrix)
    else:
        print("Not enough numeric columns for correlation.")

####################################
# 10) ターゲット列との相関・統計など (回帰/分類 タスク前提)
####################################
def check_target_relation(df, target_col, numeric_cols=None, df_name="DataFrame"):
    """
    目的変数が既にわかっている場合に、その変数との相関を出すなどの一元的な確認。
    """
    print(f"\n====== Target Relation Check: {df_name} (Target={target_col}) ======")
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in {df_name}.")
        return

    import numpy as np
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ターゲットが数値かどうか判断
    if np.issubdtype(df[target_col].dtype, np.number):
        # 数値ターゲット: 相関を表示
        if len(numeric_cols) > 1:
            corr_with_target = df[numeric_cols].corrwith(df[target_col]).sort_values(ascending=False)
            print("\n--- Correlation with target (descending) ---")
            print(corr_with_target)
        else:
            print("Not enough numeric columns to check correlation.")
    else:
        # カテゴリターゲット: 分布確認
        print(f"\n--- Category Target: {target_col} Value Counts ---")
        print(df[target_col].value_counts(dropna=False))

#===========================================================
# ★ データ読み込みを先に実行（必要に応じてパスを修正）
#===========================================================
train_df, test_df = load_data('train.csv', 'test.csv')

#===========================================================
# メイン関数
#===========================================================
def main():
    # === 2) 基本情報の確認 ===
    show_basic_info(train_df, df_name="Train")
    if test_df is not None:
        show_basic_info(test_df, df_name="Test")

    # === 3) 欠損値の確認 ===
    check_missing_values(train_df, df_name="Train")
    if test_df is not None:
        check_missing_values(test_df, df_name="Test")

    # === 4) 重複データの確認 ===
    check_duplicates(train_df, df_name="Train")
    if test_df is not None:
        check_duplicates(test_df, df_name="Test")

    # === 5) ユニーク値・カーディナリティ確認 ===
    check_unique_values(train_df, df_name="Train", top_n=5)
    if test_df is not None:
        check_unique_values(test_df, df_name="Test", top_n=5)

    # === 6) 基本統計量の確認 ===
    show_descriptive_stats(train_df, df_name="Train")
    if test_df is not None:
        show_descriptive_stats(test_df, df_name="Test")

    # === 7) 歪度・尖度の確認 ===
    check_skew_kurtosis(train_df, df_name="Train")
    if test_df is not None:
        check_skew_kurtosis(test_df, df_name="Test")

    # === 8) 外れ値の簡易チェック ===
    check_outliers(train_df, method='zscore', threshold=3.0, df_name="Train")
    if test_df is not None:
        check_outliers(test_df, method='zscore', threshold=3.0, df_name="Test")

    # === 9) 相関行列の表示 ===
    check_correlations(train_df, df_name="Train")
    if test_df is not None:
        check_correlations(test_df, df_name="Test")

    # === 10) ターゲット列がある場合の確認 (例: 'SalePrice') ===
    target_col = 'SalePrice'  # 必要に応じて修正
    check_target_relation(train_df, target_col, df_name="Train")

if __name__ == "__main__":
    main()
