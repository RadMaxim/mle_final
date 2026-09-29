import sys
import pandas as pd
import numpy as np
import scipy.sparse
import os
import gc

def get_memory_usage(top_n=30):

    memory_stats = []

    # Проверяем глобальные переменные текущего notebook
    for name, obj in list(globals().items()):
        if name.startswith("_"):
            continue

        try:
            # Pandas DataFrame
            if isinstance(obj, pd.DataFrame):
                size = obj.memory_usage(deep=True).sum()

            # Pandas Series
            elif isinstance(obj, pd.Series):
                size = obj.memory_usage(deep=True)

            # NumPy array
            elif isinstance(obj, np.ndarray):
                size = obj.nbytes

            # Sparse-матрицы scipy
            elif scipy.sparse.issparse(obj):
                size = (
                    obj.data.nbytes
                    + obj.indices.nbytes
                    + obj.indptr.nbytes
                )

            # Остальные Python-объекты
            else:
                size = sys.getsizeof(obj)

            memory_stats.append(
                (name, type(obj).__name__, size)
            )

        except Exception:
            pass

    memory_df = pd.DataFrame(
        memory_stats,
        columns=["variable", "type", "bytes"]
    )

    memory_df["MB"] = memory_df["bytes"] / 1024**2
    memory_df["GB"] = memory_df["bytes"] / 1024**3

    memory_df = (
        memory_df
        .sort_values("bytes", ascending=False)
        .reset_index(drop=True)
    )

    return memory_df.head(top_n)


def check_numeric_dtypes(df, df_name="DataFrame"):
    print(f"\n=== {df_name} ===")

    numeric_cols = df.select_dtypes(
        include=["number"]
    ).columns

    for col in numeric_cols:
        print(
            f"{col}: "
            f"dtype={df[col].dtype}, "
            f"min={df[col].min()}, "
            f"max={df[col].max()}"
        )



def compare_csv_parquet(files, csv_dir="archive", parquet_dir="parquet"):
    """
    Сравнивает CSV и Parquet версии датасетов.

    Параметры:
        files: список имён файлов без расширения
        csv_dir: директория с CSV-файлами
        parquet_dir: директория с Parquet-файлами

    Возвращает:
        pandas.DataFrame с результатами сравнения
    """

    results = []

    for name in files:
        csv_path = os.path.join(csv_dir, f"{name}.csv")
        parquet_path = os.path.join(parquet_dir, f"{name}.parquet")

        # читаем данные
        df_csv = pd.read_csv(csv_path)
        df_parquet = pd.read_parquet(parquet_path)

        # размер файлов на диске
        csv_disk_mb = os.path.getsize(csv_path) / 1024**2
        parquet_disk_mb = os.path.getsize(parquet_path) / 1024**2

        # размер DataFrame в оперативной памяти
        csv_ram_mb = df_csv.memory_usage(deep=True).sum() / 1024**2
        parquet_ram_mb = df_parquet.memory_usage(deep=True).sum() / 1024**2

        results.append({
            "dataset": name,
            "rows": len(df_csv),
            "columns": df_csv.shape[1],

            "csv_disk_mb": round(csv_disk_mb, 2),
            "parquet_disk_mb": round(parquet_disk_mb, 2),

            "compression": round(
                csv_disk_mb / parquet_disk_mb,
                2
            ),

            "csv_ram_mb": round(csv_ram_mb, 2),
            "parquet_ram_mb": round(parquet_ram_mb, 2),

            "same_shape": df_csv.shape == df_parquet.shape,
            "same_data": df_csv.equals(df_parquet)
        })

    return pd.DataFrame(results)
def check_data_quality(dataframes):
    """
    Проверяет DataFrame на:
    - полные дубликаты строк;
    - пропущенные значения NaN/None;
    - пустые строки.

    dataframes: dict вида
        {
            "events": events,
            "category_tree": category_tree
        }

    Возвращает:
        summary_df — общая статистика
        missing_df — пропуски по колонкам
    """

    summary = []
    missing_details = []

    for name, df in dataframes.items():

        # Полные дубликаты строк
        duplicates = df.duplicated().sum()

        # NaN / None
        null_counts = df.isna().sum()
        total_nulls = null_counts.sum()

        # Пустые строки
        empty_counts = pd.Series(0, index=df.columns, dtype="int64")

        string_columns = df.select_dtypes(
            include=["object", "string"]
        ).columns

        for col in string_columns:
            empty_counts[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .eq("")
                .fillna(False)
                .sum()
            )

        total_empty = empty_counts.sum()

        summary.append({
            "dataset": name,
            "rows": len(df),
            "duplicates": duplicates,
            "null_values": total_nulls,
            "empty_strings": total_empty
        })

        # Детализация по колонкам
        for col in df.columns:
            if null_counts[col] > 0 or empty_counts[col] > 0:
                missing_details.append({
                    "dataset": name,
                    "column": col,
                    "null_values": null_counts[col],
                    "empty_strings": empty_counts[col]
                })

    summary_df = pd.DataFrame(summary)
    missing_df = pd.DataFrame(missing_details)

    return summary_df, missing_df

def check_csv_data_quality(files, csv_dir="archive"):
    """
    Проверяет CSV-файлы на:
    - количество строк;
    - полные дубликаты;
    - NaN / None;
    - пустые строки.
    """

    results = []

    for name in files:
        csv_path = os.path.join(csv_dir, f"{name}.csv")

        df = pd.read_csv(csv_path)

        # Полные дубликаты
        duplicates = df.duplicated().sum()

        # Пропущенные значения
        null_values = df.isna().sum().sum()

        # Пустые строки
        empty_strings = 0

        string_columns = df.select_dtypes(
            include=["object", "string"]
        ).columns

        for col in string_columns:
            empty_strings += (
                df[col]
                .astype("string")
                .str.strip()
                .eq("")
                .fillna(False)
                .sum()
            )

        results.append({
            "dataset": name,
            "rows": len(df),
            "duplicates": duplicates,
            "null_values": null_values,
            "empty_strings": empty_strings
        })

        # освобождаем память перед чтением следующего файла
        del df
        gc.collect()

    return pd.DataFrame(results)
def compare_parquet_memory(
    files,
    original_dir="parquet",
    optimized_dir="parquet_optimized"
):
    results = []

    for name in files:
        original_path = os.path.join(
            original_dir,
            f"{name}.parquet"
        )

        optimized_path = os.path.join(
            optimized_dir,
            f"{name}.parquet"
        )

        # Исходный файл
        df = pd.read_parquet(original_path)

        original_memory_mb = (
            df.memory_usage(deep=True).sum() / 1024**2
        )

        del df
        gc.collect()

        # Оптимизированный файл
        df = pd.read_parquet(optimized_path)

        optimized_memory_mb = (
            df.memory_usage(deep=True).sum() / 1024**2
        )

        del df
        gc.collect()

        saved_mb = (
            original_memory_mb - optimized_memory_mb
        )

        saved_percent = (
            saved_mb / original_memory_mb * 100
        )

        results.append({
            "dataset": name,
            "original_mb": round(original_memory_mb, 2),
            "optimized_mb": round(optimized_memory_mb, 2),
            "saved_mb": round(saved_mb, 2),
            "saved_percent": round(saved_percent, 2)
        })

    result = pd.DataFrame(results)

    return result