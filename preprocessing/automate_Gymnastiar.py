# automate.py

import pandas as pd
import numpy as np
import os
from typing import Tuple
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV atau Excel.
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path, encoding='latin1')
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Format file tidak didukung (hanya .csv atau .xlsx)")


def clean_outliers(df: pd.DataFrame, columns: list, z_thresh: float = 3.0) -> pd.DataFrame:
    df = df.copy()

    # Pastikan kolom-kolom numerik dikonversi ke float
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df[(np.abs(stats.zscore(df[columns])) < z_thresh).all(axis=1)].copy()



def encode_and_scale(df: pd.DataFrame, target_column: str = "Price") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = df.copy()

    # One-hot encoding
    categorical_columns = df.select_dtypes(include='object').columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Split X and y
    X = df_encoded.drop(target_column, axis=1)
    y = df_encoded[target_column]

    # Scaling
    numerical_columns = ['Number of Ratings', 'Number of Reviews']
    numerical_columns = [col for col in numerical_columns if col in X.columns]

    # Convert to float & drop NaN
    for col in numerical_columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Drop rows with NaNs after coercion
    X = X.dropna(subset=numerical_columns)
    y = y.loc[X.index]

    # Skip scaling if data kosong
    if X[numerical_columns].shape[0] == 0:
        raise ValueError("Tidak ada data tersisa setelah preprocessing. Cek apakah outlier cleaning terlalu ketat.")

    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns].astype(np.float64))

    df_clean_encoded = X.copy()
    df_clean_encoded[target_column] = y

    return X, y, df_clean_encoded


def preprocess_pipeline(file_path: str, output_dir: str = "processed") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Pipeline lengkap preprocessing dan simpan hasil ke CSV.
    """
    df = load_dataset(file_path)
    outlier_columns = ['Price', 'Number of Ratings', 'Number of Reviews']
    df_clean = clean_outliers(df, outlier_columns)

    X, y, df_clean_encoded = encode_and_scale(df_clean)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    df_clean_encoded.to_csv(os.path.join(output_dir, "laptopPrice_clean.csv"), index=False)

    print(f"Hasil preprocessing disimpan ke folder '{output_dir}':")
    print(" - X_train.csv")
    print(" - X_test.csv")
    print(" - y_train.csv")
    print(" - y_test.csv")
    print(" - laptopPrice_clean.csv")

    return X_train, X_test, y_train, y_test


def main():
    dataset_path = "laptopPrice_raw.csv"
    output_path = "processed"
    preprocess_pipeline(dataset_path, output_path)


if __name__ == "__main__":
    main()
