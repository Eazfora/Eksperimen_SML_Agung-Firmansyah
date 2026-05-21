import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH   = os.path.join(BASE_DIR, "student_performance_raw.csv")
OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            "student_performance_preprocessing")
os.makedirs(OUT_DIR, exist_ok=True)

def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data dari: {path}")
    df = pd.read_csv(path)
    logger.info(f"Shape awal: {df.shape}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Memulai data cleaning...")

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"  Duplikat dihapus: {before - len(df)} baris")

    # Handle missing values
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    logger.info(f"  Missing values ditangani. Kolom numerik: {num_cols}")
    logger.info(f"  Kolom kategorik: {cat_cols}")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Feature engineering...")

    # Rata-rata nilai
    df["avg_grade"]     = (df["G1"] + df["G2"] + df["G3"]) / 3
    # Tingkat konsumsi alkohol gabungan
    df["total_alc"]     = df["Dalc"] + df["Walc"]
    # Indeks dukungan keluarga
    df["family_support"] = df["famsup"].map({"yes": 1, "no": 0}) + \
                        df["famrel"]

    logger.info("  Fitur baru: avg_grade, total_alc, family_support")
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Encoding fitur kategorik...")
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        logger.info(f"  Encoded: {col}")
    return df

def scale_features(df: pd.DataFrame, target_col: str = "passed") -> pd.DataFrame:
    logger.info("Scaling fitur numerik...")
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != target_col]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    logger.info(f"  Scaled {len(feature_cols)} fitur")
    return df

def split_and_save(df: pd.DataFrame, target_col: str = "passed"):
    logger.info("Split dan simpan dataset...")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df  = pd.concat([X_test,  y_test],  axis=1)

    train_path = os.path.join(OUT_DIR, "train.csv")
    test_path  = os.path.join(OUT_DIR, "test.csv")
    full_path  = os.path.join(OUT_DIR, "student_performance_preprocessed.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)
    df.to_csv(full_path,        index=False)

    logger.info(f"  Train: {train_df.shape} → {train_path}")
    logger.info(f"  Test : {test_df.shape}  → {test_path}")
    logger.info(f"  Full : {df.shape}       → {full_path}")

def main():
    logger.info("=" * 55)
    logger.info("  AUTOMATED PREPROCESSING - Student Performance")
    logger.info("=" * 55)

    df = load_data(RAW_PATH)
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_features(df)
    df = scale_features(df)
    split_and_save(df)

    logger.info("Preprocessing selesai!")


if __name__ == "__main__":
    main()
