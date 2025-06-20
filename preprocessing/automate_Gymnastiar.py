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
        raise ValueError("Format file tidak didukung. Gunakan .csv atau .xlsx")


def clean_outliers(df: pd.DataFrame, columns: list, z_thresh: float = 4.0) -> pd.DataFrame:
    """
    Menghapus outlier berdasarkan z-score dengan threshold longgar (default: 4.0).
    """
    df = df.copy()

    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=columns)

    if df.empty:
        raise ValueError("Data kosong setelah konversi ke numerik.")

    filtered = df[(np.abs(stats.zscore(df[columns])) < z_thresh).all(axis=1)]

    if filtered.empty:
        raise ValueError("Semua data terhapus setelah outlier filtering. Coba naikkan threshold.")

    return filtered


def encode_and_scale(df: pd.DataFrame, target_column: str = "Price") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    One-hot encoding dan scaling kolom numerik.
    """
    df = df.copy()
    categorical_columns = df.select_dtypes(include='object').columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    X = df_encoded.drop(target_column, axis=1)
    y = df_encoded[target_column]

    numerical_columns = ['Number of Ratings', 'Number of Reviews']
    numerical_columns = [col for col in numerical_columns if col in X.columns]

    for col in numerical_columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X.dropna(subset=numerical_columns)
    y = y.loc[X.index]

    if X.empty:
        raise ValueError("Tidak ada data tersisa setelah preprocessing numerik.")

    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    df_clean_encoded = X.copy()
    df_clean_encoded[target_column] = y

    return X, y, df_clean_encoded


def preprocess_pipeline(file_path: str, output_dir: str = "processed") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Pipeline preprocessing lengkap dan menyimpan hasil.
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

    print("✅ Preprocessing selesai. File disimpan ke folder 'processed/'")
    return X_train, X_test, y_train, y_test


def main():
    dataset_path = "laptopPrice_raw.csv"  # Pastikan file ini berada di root repo
    output_path = "processed"
    preprocess_pipeline(dataset_path, output_path)


if __name__ == "__main__":
    main()
