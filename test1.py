# %% [markdown]
# # Инициализация

# %% [markdown]
# Загружаем библиотеки необходимые для выполнения кода ноутбука.

# %%
import matplotlib.pyplot as plt
import os
import boto3
from dotenv import load_dotenv
import sys
import scipy.sparse
import sklearn
from utils import check_numeric_dtypes, compare_csv_parquet
from IPython.display import display, HTML
from sklearn.preprocessing import LabelEncoder
import gc
import numpy as np
import pandas as pd
import scipy.sparse
import pyarrow as pa
import pyarrow.parquet as pq
from implicit.als import AlternatingLeastSquares


load_dotenv()

# %%
import utils
import importlib

importlib.reload(utils)



# %%
import boto3, os
s3 = boto3.client(
    "s3",
    endpoint_url="https://storage.yandexcloud.net",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
bucket_name = os.getenv("S3_BUCKET_NAME")

# %% [markdown]
# # === ЭТАП 1 ===

# %% [markdown]
# # Загрузка первичных данных

# %% [markdown]
# # Построение признаков

# %% [markdown]
# Построим три признака, можно больше, для ранжирующей модели.

# %%
# ============================================================
# ПОСТРОЕНИЕ ПРИЗНАКОВ ДЛЯ РАНЖИРУЮЩЕЙ МОДЕЛИ
# ============================================================

import gc
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. Загружаем подготовленные данные
# ------------------------------------------------------------

# Очищенные пользовательские события
events = pd.read_parquet(
    "parquet_cleaned/events.parquet"
)

# Топ популярных товаров
top_popular = pd.read_parquet(
    "recommendations/top_popular/top_popular.parquet"
)

# Персональные рекомендации ALS
personal_als = pd.read_parquet(
    "recommendations/personal_als/personal_als.parquet"
)

# Похожие товары ALS
similar = pd.read_parquet(
    "recommendations/similar/similar.parquet"
)


# ------------------------------------------------------------
# Проверяем размеры загруженных таблиц
# ------------------------------------------------------------

print("events:", events.shape)
print("top_popular:", top_popular.shape)
print("personal_als:", personal_als.shape)
print("similar:", similar.shape)

# %%
# ============================================================
# ПОДГОТОВКА EVENTS
# ============================================================

events = pd.read_parquet(
    "parquet_cleaned/events.parquet"
)


# Приводим названия к единому виду
events = events.rename(
    columns={
        "visitorid": "user_id",
        "itemid": "item_id",
    }
)


# Преобразуем timestamp
events["event_time"] = pd.to_datetime(
    events["timestamp"],
    unit="ms"
)


print(events.columns.tolist())

print(
    "Период данных:",
    events["event_time"].min(),
    "->",
    events["event_time"].max()
)

# %%
# ------------------------------------------------------------
# 2. Приводим названия колонок к единому виду
# ------------------------------------------------------------

# Топ популярных
top_popular = top_popular.rename(
    columns={
        "itemid": "item_id",
        "score": "popular_score",
        "rank": "popular_rank"
    }
)

# Персональные рекомендации ALS
personal_als = personal_als.rename(
    columns={
        "visitorid": "user_id",
        "itemid": "item_id",
        "score": "als_score",
        "rank": "als_rank"
    }
)

# Похожие товары
similar = similar.rename(
    columns={
        "itemid": "source_item_id",
        "similar_itemid": "item_id",
        "score": "similarity_score",
        "rank": "similarity_rank"
    }
)

# %%
# ============================================================
# ВРЕМЕННЫЕ ГРАНИЦЫ
# ============================================================

events["event_time"] = pd.to_datetime(
    events["timestamp"],
    unit="ms"
)

data_end = (
    events["event_time"].max()
    + pd.Timedelta(milliseconds=1)
)

eval_start = (
    data_end
    - pd.Timedelta(days=7)
)

ranker_start = (
    eval_start
    - pd.Timedelta(days=7)
)

# История, которую разрешено использовать
# при построении признаков и кандидатов
events_history = events[
    events["event_time"] < ranker_start
].copy()


print(
    "History:",
    events_history["event_time"].min(),
    "->",
    events_history["event_time"].max()
)

print("Ranker start:", ranker_start)
print("Eval start:", eval_start)

# %%
# ------------------------------------------------------------
# 1. Кандидаты из Personal ALS
# ------------------------------------------------------------

als_candidates = personal_als[
    ["user_id", "item_id"]
].drop_duplicates()

print(
    "ALS candidates:",
    als_candidates.shape
)



# %%
# ------------------------------------------------------------
# История пользователя для Similar
# ------------------------------------------------------------

MAX_HISTORY_ITEMS = 20

user_history = (
    events_history[
        events_history["event"].isin(
            [
                "view",
                "addtocart",
                "transaction",
            ]
        )
    ][
        [
            "user_id",
            "item_id",
            "event_time",
        ]
    ]
    # Сначала самые свежие события
    .sort_values(
        "event_time",
        ascending=False
    )
    # Один item одному пользователю только один раз
    .drop_duplicates(
        [
            "user_id",
            "item_id",
        ]
    )
    # Не позволяем активному пользователю
    # породить тысячи Similar candidates
    .groupby(
        "user_id",
        group_keys=False
    )
    .head(MAX_HISTORY_ITEMS)
    [
        [
            "user_id",
            "item_id",
        ]
    ]
    .rename(
        columns={
            "item_id": "source_item_id"
        }
    )
)

# %%
TOP_SIMILAR_N = 20

similar_top = (
    similar[
        similar["similarity_rank"] <= TOP_SIMILAR_N
    ][
        [
            "source_item_id",
            "item_id",
            "similarity_score",
            "similarity_rank",
        ]
    ]
    .copy()
)

# %%
user_similar = user_history.merge(
    similar_top,
    on="source_item_id",
    how="inner"
)

print(
    "user_similar:",
    user_similar.shape
)

# %%
user_similar.head(10)

# %%

# Один и тот же кандидат может получиться
# от нескольких исходных товаров пользователя.
#
# Оставляем максимальную similarity
# и лучший rank.
similar_features = (
    user_similar
    .groupby(
        ["user_id", "item_id"],
        as_index=False
    )
    .agg(
        similarity_score=(
            "similarity_score",
            "max"
        ),
        similarity_rank=(
            "similarity_rank",
            "min"
        ),
        similarity_sources=(
            "source_item_id",
            "nunique"
        )
    )
)


# %%

similar_candidates = similar_features[
    ["user_id", "item_id"]
]

print(
    "Similar candidates:",
    similar_candidates.shape
)



# %%

# ------------------------------------------------------------
# 3. Кандидаты Top Popular
# ------------------------------------------------------------

# Берём пользователей, для которых строим рекомендации.
# Здесь используем пользователей из personal_als.

users = (
    events[
        ["user_id"]
    ]
    .drop_duplicates()
)

TOP_POPULAR_N = 20

# Каждый популярный товар является кандидатом
# для каждого пользователя.
popular_candidates = users.merge(
    top_popular.sort_values("popular_rank").head(TOP_POPULAR_N)[
        ["item_id"]
    ],
    how="cross"
)

print(
    "Popular candidates:",
    popular_candidates.shape
)


# ------------------------------------------------------------
# 4. Объединяем кандидатов из всех источников
# ------------------------------------------------------------



# %%
candidates = pd.concat(
    [
        als_candidates,
        similar_candidates,
        popular_candidates
    ],
    ignore_index=True
)


# Один user-item мог прийти сразу
# из нескольких источников
candidates = (
    candidates
    .drop_duplicates(
        subset=[
            "user_id",
            "item_id"
        ]
    )
    .reset_index(drop=True)
)

print(
    "Unique candidates:",
    candidates.shape
)


# %%
candidates.to_parquet(
    "candidates.parquet",
    index=False,
    compression="snappy",
)
print("Сохранено:", candidates.shape)

# %%
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



# %%
candidates = pd.read_parquet(
    "candidates.parquet",
)

# %%
candidates.head(2)

# %%
import gc
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# НАСТРОЙКИ
# ============================================================

BATCH_SIZE = 20_000_000

OUTPUT_FILE = "candidates_with_popular.parquet"


# ============================================================
# 1. ГОТОВИМ TOP_POPULAR
# ============================================================

# Если в предыдущем запуске item_id уже был сделан индексом,
# возвращаем его обратно в обычную колонку.
if top_popular.index.name == "item_id":
    top_popular = top_popular.reset_index()


# Оставляем только необходимые колонки.
top_popular_work = top_popular[
    [
        "item_id",
        "popular_score",
        "popular_rank",
    ]
].copy()


# Фиксируем типы.
top_popular_work["item_id"] = (
    top_popular_work["item_id"]
    .astype(np.int32)
)

top_popular_work["popular_score"] = (
    top_popular_work["popular_score"]
    .astype(np.float32)
)

# Nullable Int16:
# умеет хранить <NA>, в отличие от обычного np.int16.
top_popular_work["popular_rank"] = (
    top_popular_work["popular_rank"]
    .astype("Int16")
)


# ------------------------------------------------------------
# Делаем отдельный индексированный справочник.
# Оригинальный top_popular при этом не изменяем.
# ------------------------------------------------------------

top_popular_idx = (
    top_popular_work
    .set_index("item_id")
)


# ============================================================
# 2. ГОТОВИМ CANDIDATES
# ============================================================

# Фиксируем тип ключа join.
candidates["item_id"] = (
    candidates["item_id"]
    .astype(np.int32)
)


# Если ячейка запускалась раньше и признаки popularity
# уже присутствуют, удаляем их перед повторным join.
candidates.drop(
    columns=[
        "popular_score",
        "popular_rank",
    ],
    errors="ignore",
    inplace=True,
)


# ============================================================
# 3. УДАЛЯЕМ НЕДОПИСАННЫЙ ФАЙЛ ПРЕДЫДУЩЕЙ ПОПЫТКИ
# ============================================================

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

    print(
        "Удалён старый файл:",
        OUTPUT_FILE
    )


# ============================================================
# 4. БАТЧАМИ JOIN + ЗАПИСЬ СРАЗУ НА ДИСК
# ============================================================

n = len(candidates)

batch_count = (
    n + BATCH_SIZE - 1
) // BATCH_SIZE


print(
    f"Всего строк: {n:,}"
)

print(
    f"Размер батча: {BATCH_SIZE:,}"
)

print(
    f"Количество батчей: {batch_count:,}"
)


writer = None


try:

    for batch_number, start in enumerate(
        range(0, n, BATCH_SIZE),
        start=1,
    ):

        end = min(
            start + BATCH_SIZE,
            n
        )


        # --------------------------------------------------------
        # Берём только текущий батч
        # --------------------------------------------------------

        chunk = candidates.iloc[
            start:end
        ]


        # --------------------------------------------------------
        # Добавляем признаки popularity
        # --------------------------------------------------------

        merged = chunk.join(
            top_popular_idx[
                [
                    "popular_score",
                    "popular_rank",
                ]
            ],
            on="item_id",
            how="left",
            sort=False,
        )


        # ========================================================
        # 5. ФИКСИРУЕМ DTYPE ПОСЛЕ JOIN
        # ========================================================

        # После left join обязательно явно задаём типы,
        # чтобы схема каждого Arrow batch была одинаковой.

        merged["popular_score"] = (
            merged["popular_score"]
            .astype(np.float32)
        )

        merged["popular_rank"] = (
            merged["popular_rank"]
            .astype("Int16")
        )


        # --------------------------------------------------------
        # Переводим текущий батч в Arrow
        # --------------------------------------------------------

        table = pa.Table.from_pandas(
            merged,
            preserve_index=False,
        )


        # --------------------------------------------------------
        # На первом батче создаём ParquetWriter
        # --------------------------------------------------------

        if writer is None:

            print("\nСхема выходного Parquet:")
            print(table.schema)
            print()

            writer = pq.ParquetWriter(
                OUTPUT_FILE,
                table.schema,
                compression="snappy",
            )


        # --------------------------------------------------------
        # Дополнительная проверка схемы
        # --------------------------------------------------------

        if not table.schema.equals(
            writer.schema,
            check_metadata=False,
        ):

            print(
                f"Батч {batch_number}: "
                "схема отличается — выполняю cast."
            )

            table = table.cast(
                writer.schema
            )


        # ========================================================
        # 6. ПИШЕМ БАТЧ НА ДИСК
        # ========================================================

        writer.write_table(
            table,
            row_group_size=250_000,
        )


        # ========================================================
        # 7. ОСВОБОЖДАЕМ RAM ТЕКУЩЕГО БАТЧА
        # ========================================================

        del chunk
        del merged
        del table

        gc.collect()


        # PyArrow может держать освобождённые страницы памяти.
        try:
            pa.default_memory_pool().release_unused()
        except Exception:
            pass


        # --------------------------------------------------------
        # Прогресс
        # --------------------------------------------------------

        print(
            f"Батч {batch_number}/{batch_count} | "
            f"{end / n * 100:.2f}% | "
            f"{end:,}/{n:,}",
            flush=True,
        )


finally:

    # Writer обязательно закрываем,
    # даже если внутри цикла возникнет исключение.
    if writer is not None:
        writer.close()


# ============================================================
# 8. РЕЗУЛЬТАТ
# ============================================================

print()
print(
    "Готово:",
    OUTPUT_FILE
)

print(
    "Строк записано:",
    f"{n:,}"
)

# %%
candidates = pd.read_parquet("candidates_with_popular.parquet")

candidates.head(10)

# %%
import gc
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


BATCH_SIZE = 25_000_000
OUTPUT_FILE = "candidates_with_similar.parquet"
OLD_FILE = "candidates_with_popular.parquet"
MIN_VALID_SIZE = 1024


# 1. Готовим similar_features
similar_features["user_id"] = similar_features["user_id"].astype(np.int32)
similar_features["item_id"] = similar_features["item_id"].astype(np.int32)
similar_features["similarity_score"] = similar_features["similarity_score"].astype(np.float32)
similar_features["similarity_rank"] = similar_features["similarity_rank"].astype("Int16")

similar_features_idx = similar_features[
    ["user_id", "item_id", "similarity_score", "similarity_rank", "similarity_sources"]
].set_index(["user_id", "item_id"])


# 2. Готовим candidates
candidates["user_id"] = candidates["user_id"].astype(np.int32)
candidates["item_id"] = candidates["item_id"].astype(np.int32)
candidates.drop(
    columns=["similarity_score", "similarity_rank", "similarity_sources"],
    errors="ignore",
    inplace=True,
)


# 3. Удаляем недописанный файл прошлой попытки
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)


# 4. Батчами join + запись на диск
n = len(candidates)
batch_count = (n + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Строк: {n:,} | батчей: {batch_count}")

writer = None
write_success = False

try:
    for batch_number, start in enumerate(range(0, n, BATCH_SIZE), start=1):
        end = min(start + BATCH_SIZE, n)
        chunk = candidates.iloc[start:end]

        merged = chunk.merge(
            similar_features_idx,
            left_on=["user_id", "item_id"],
            right_index=True,
            how="left",
            sort=False,
        )
        merged["similarity_score"] = merged["similarity_score"].astype(np.float32)
        merged["similarity_rank"] = merged["similarity_rank"].astype("Int16")

        table = pa.Table.from_pandas(merged, preserve_index=False)

        if writer is None:
            print("Схема:", table.schema)
            writer = pq.ParquetWriter(OUTPUT_FILE, table.schema, compression="snappy")
        elif not table.schema.equals(writer.schema, check_metadata=False):
            print(f"Батч {batch_number}: cast схемы")
            table = table.cast(writer.schema)

        writer.write_table(table, row_group_size=250_000)

        del chunk, merged, table
        gc.collect()
        try:
            pa.default_memory_pool().release_unused()
        except Exception:
            pass

        print(f"{batch_number}/{batch_count} | {end / n * 100:.2f}%", flush=True)

    write_success = True

finally:
    if writer is not None:
        writer.close()


# 5. Удаляем старый файл только после успеха
print(f"Готово: {OUTPUT_FILE} | строк: {n:,}")

if write_success and os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) >= MIN_VALID_SIZE:
    if OLD_FILE != OUTPUT_FILE and os.path.exists(OLD_FILE):
        os.remove(OLD_FILE)
        print(f"Удалён старый: {OLD_FILE}")
else:
    print("Ошибка записи — старый файл не удаляю")

# %%
candidates = pd.read_parquet("candidates_with_similar.parquet")

candidates.head(10)

# %%
import gc
import os

import pyarrow as pa
import pyarrow.parquet as pq


BATCH = 20_000_000

OUTPUT_FILE = "candidates_with_als.parquet"
OLD_FILE = "candidates_with_similar.parquet"
MIN_VALID_SIZE = 1024


personal_als_idx = (
    personal_als[
        [
            "user_id",
            "item_id",
            "als_score",
            "als_rank",
        ]
    ]
    .drop_duplicates(
        ["user_id", "item_id"]
    )
    .set_index(
        ["user_id", "item_id"]
    )
)


# Удаляем недописанный файл прошлой попытки
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)


n = len(candidates)
write_success = False
writer = None

print(
    f"Всего строк: {n:,} | "
    f"батчей: {(n + BATCH - 1) // BATCH}"
)


try:

    for start in range(0, n, BATCH):

        end = min(start + BATCH, n)

        chunk = candidates.iloc[start:end]

        merged = chunk.join(
            personal_als_idx,
            on=["user_id", "item_id"],
            how="left",
        )

        table = pa.Table.from_pandas(
            merged,
            preserve_index=False
        )

        if writer is None:
            writer = pq.ParquetWriter(
                OUTPUT_FILE,
                table.schema,
                compression="snappy",
            )
        elif not table.schema.equals(writer.schema, check_metadata=False):
            table = table.cast(writer.schema)

        writer.write_table(
            table,
            row_group_size=250_000
        )

        del chunk
        del merged
        del table

        gc.collect()

        print(
            f"{end / n * 100:.2f}% "
            f"({end:,}/{n:,})",
            flush=True,
        )

    write_success = True

finally:

    if writer is not None:
        writer.close()


print("Готово:", OUTPUT_FILE)


# Удаляем старый файл только после успеха
if write_success and os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) >= MIN_VALID_SIZE:
    if OLD_FILE != OUTPUT_FILE and os.path.exists(OLD_FILE):
        os.remove(OLD_FILE)
        print("Удалён старый:", OLD_FILE)
else:
    print("Ошибка записи — старый файл не удаляю")

# %%
import pandas as pd

candidates = pd.read_parquet("candidates_with_als.parquet")

# %%
candidates.head(3)



# %%
get_memory_usage()

# %%
# ------------------------------------------------------------
# Показываем, из какого рекомендателя пришёл кандидат
# ------------------------------------------------------------

candidates["from_als"] = (
    candidates["als_score"]
    .notna()
    .astype("int8")
)

candidates["from_popular"] = (
    candidates["popular_score"]
    .notna()
    .astype("int8")
)

candidates["from_similar"] = (
    candidates["similarity_score"]
    .notna()
    .astype("int8")
)

# %%
display(candidates.head(20))

# %%
score_columns = ["als_score", "popular_score", "similarity_score"]

candidates[score_columns] = (
    candidates[score_columns]
    .astype("float32")
)

candidates["similarity_sources"] = (
    candidates["similarity_sources"]
    .astype("Int16")
)

# %%
candidates["popular_score_log"] = np.log1p(
    candidates["popular_score"]
)

# %%
candidates.to_parquet(
    "candidates_result.parquet",
    index=False,
    compression="snappy",
)



# %%
os.remove("candidates_with_als.parquet")

print("Готово")

# %%
import pandas as pd

candidates = pd.read_parquet("candidates_result.parquet")

# %%
display(candidates.head(20))

# %%
print(
    "Кандидатов:",
    len(candidates)
)

print(
    "Попали в top popular:",
    (candidates["popular_score"] > 0).sum()
)

print(
    "Доля популярных:",
    (candidates["popular_score"] > 0).mean()
)

# %%
non_zero = (candidates != 0).sum()
total = len(candidates)

result = pd.DataFrame({
    "non_zero": non_zero,
    "total": total,
    "percent": (non_zero / total * 100).round(2),
})

print(result)

# %%
stats = pd.DataFrame({
    "notna": candidates.notna().sum(),
    "isna": candidates.isna().sum(),
    "zero": (candidates == 0).sum(),
    "non_zero_real": ((candidates != 0) & candidates.notna()).sum(),
})

stats["non_zero_%"] = (stats["non_zero_real"] / len(candidates) * 100).round(2)
print(stats)

# %% [markdown]
# # Ранжирование рекомендаций

# %% [markdown]
# Построим ранжирующую модель, чтобы сделать рекомендации более точными. Отранжируем рекомендации.

# %%
# ============================================================
# ПОДГОТОВКА ДАННЫХ ДЛЯ РАНЖИРУЮЩЕЙ МОДЕЛИ
# ============================================================

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. Загружаем кандидатов с уже построенными признаками
# ------------------------------------------------------------

candidates = pd.read_parquet(
    "candidates_result.parquet"
)


# ------------------------------------------------------------
# 2. Подготавливаем events
# ------------------------------------------------------------

events = pd.read_parquet(
    "parquet_cleaned/events.parquet"
)


# Если названия ещё не приведены к общему виду
events_rank = events.rename(
    columns={
        "visitorid": "user_id",
        "itemid": "item_id",
    }
)


# ------------------------------------------------------------
# 3. Приводим timestamp к datetime
# ------------------------------------------------------------

if pd.api.types.is_numeric_dtype(
    events_rank["timestamp"]
):
    events_rank["event_time"] = pd.to_datetime(
        events_rank["timestamp"],
        unit="ms"
    )
else:
    events_rank["event_time"] = pd.to_datetime(
        events_rank["timestamp"]
    )


print(
    "Период данных:",
    events_rank["event_time"].min(),
    "->",
    events_rank["event_time"].max()
)

# %%
# ============================================================
# ВРЕМЕННОЙ SPLIT ДЛЯ RANKER
# ============================================================

data_end = (
    events_rank["event_time"].max()
    + pd.Timedelta(milliseconds=1)
)

# Последние 7 дней -> финальная оценка
eval_start = (
    data_end
    - pd.Timedelta(days=7)
)

# Предыдущие 7 дней -> обучение ranker
ranker_start = (
    eval_start
    - pd.Timedelta(days=7)
)


print(
    "История для построения рекомендаций:",
    events_rank["event_time"].min(),
    "->",
    ranker_start
)

print(
    "Ranker train:",
    ranker_start,
    "->",
    eval_start
)

print(
    "Evaluation:",
    eval_start,
    "->",
    data_end
)

# %%
# ============================================================
# 4. ФОРМИРУЕМ ПЕРИОДЫ RANKER TRAIN И EVALUATION
# ============================================================

ranker_events = events_rank[
    (events_rank["event_time"] >= ranker_start)
    &
    (events_rank["event_time"] < eval_start)
].copy()


eval_events = events_rank[
    (events_rank["event_time"] >= eval_start)
    &
    (events_rank["event_time"] < data_end)
].copy()


print(
    "Ranker train:",
    ranker_events["event_time"].min(),
    "->",
    ranker_events["event_time"].max()
)

print(
    "Evaluation:",
    eval_events["event_time"].min(),
    "->",
    eval_events["event_time"].max()
)

print(
    "Ranker events:",
    len(ranker_events)
)

print(
    "Evaluation events:",
    len(eval_events)
)

# %%
# ============================================================
# 5. ФОРМИРУЕМ TARGET
# ============================================================

POSITIVE_EVENTS = [
    "addtocart",
    "transaction",
]


positive_pairs = (
    ranker_events[
        ranker_events["event"].isin(
            POSITIVE_EVENTS
        )
    ][
        [
            "user_id",
            "item_id",
        ]
    ]
    .drop_duplicates()
    .copy()
)


positive_pairs["target"] = 1


print(
    "Positive user-item pairs:",
    len(positive_pairs)
)

display(
    positive_pairs.head()
)

# %%
# ============================================================
# 6. ПРОВЕРЯЕМ ПОКРЫТИЕ POSITIVE ПАР КАНДИДАТАМИ
# ============================================================

positive_in_candidates = positive_pairs.merge(
    candidates[
        [
            "user_id",
            "item_id"
        ]
    ],
    on=[
        "user_id",
        "item_id"
    ],
    how="inner"
)


print(
    "Всего positive pairs:",
    len(positive_pairs)
)

print(
    "Positive найдено среди candidates:",
    len(positive_in_candidates)
)

print(
    "Coverage:",
    len(positive_in_candidates)
    / len(positive_pairs)
)

# %%
# ============================================================
# ДИАГНОСТИКА НИЗКОГО COVERAGE
# ============================================================

print("positive_pairs:", len(positive_pairs))
print("candidates:", len(candidates))

print()
print("Типы ID:")

print(
    "positive user_id:",
    positive_pairs["user_id"].dtype
)

print(
    "candidates user_id:",
    candidates["user_id"].dtype
)

print(
    "positive item_id:",
    positive_pairs["item_id"].dtype
)

print(
    "candidates item_id:",
    candidates["item_id"].dtype
)

# %%
# ============================================================
# ПРОВЕРЯЕМ USER / ITEM COVERAGE
# ============================================================

positive_users = positive_pairs[
    "user_id"
].unique()

positive_items = positive_pairs[
    "item_id"
].unique()


candidate_users = candidates[
    "user_id"
].unique()

candidate_items = candidates[
    "item_id"
].unique()


user_hits = np.isin(
    positive_users,
    candidate_users
).sum()

item_hits = np.isin(
    positive_items,
    candidate_items
).sum()


print(
    "Positive users:",
    len(positive_users)
)

print(
    "Positive users есть в candidates:",
    user_hits
)

print(
    "User coverage:",
    user_hits / len(positive_users)
)


print()


print(
    "Positive items:",
    len(positive_items)
)

print(
    "Positive items есть в candidates:",
    item_hits
)

print(
    "Item coverage:",
    item_hits / len(positive_items)
)

# %%
# ============================================================
# ПРОВЕРЯЕМ, КАКОЙ ИСТОЧНИК НАХОДИТ POSITIVE
# ============================================================

positive_features = positive_pairs.merge(
    candidates[
        [
            "user_id",
            "item_id",
            "from_als",
            "from_popular",
            "from_similar",
        ]
    ],
    on=[
        "user_id",
        "item_id",
    ],
    how="inner",
)


print(
    "Всего positive найдено:",
    len(positive_features)
)


print(
    "ALS:",
    positive_features["from_als"].sum()
)

print(
    "Popular:",
    positive_features["from_popular"].sum()
)

print(
    "Similar:",
    positive_features["from_similar"].sum()
)


print()
print("Комбинации источников:")

display(
    positive_features[
        [
            "from_als",
            "from_popular",
            "from_similar",
        ]
    ]
    .value_counts()
    .reset_index(name="count")
)

# %%
# ============================================================
# КОЛИЧЕСТВО КАНДИДАТОВ НА RANKER-ПОЛЬЗОВАТЕЛЯ
# ============================================================

positive_users = positive_pairs[
    "user_id"
].unique()


ranker_candidate_counts = (
    candidates[
        candidates["user_id"].isin(
            positive_users
        )
    ]
    .groupby("user_id")
    .size()
)


print(
    ranker_candidate_counts.describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)

# %%
# ============================================================
# КОЛИЧЕСТВО КАНДИДАТОВ ПО ИСТОЧНИКАМ НА ПОЛЬЗОВАТЕЛЯ
# ============================================================

ranker_candidates = candidates[
    candidates["user_id"].isin(
        positive_users
    )
].copy()


source_stats = (
    ranker_candidates
    .groupby("user_id")
    .agg(
        total_candidates=(
            "item_id",
            "count"
        ),
        als_candidates=(
            "from_als",
            "sum"
        ),
        popular_candidates=(
            "from_popular",
            "sum"
        ),
        similar_candidates=(
            "from_similar",
            "sum"
        ),
    )
)


display(
    source_stats.describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)

# %%
# ============================================================
# ПРОВЕРЯЕМ РЕАЛЬНОЕ ПРОИСХОЖДЕНИЕ КАНДИДАТОВ
# ============================================================

ranker_candidates = candidates[
    candidates["user_id"].isin(
        positive_users
    )
].copy()

print(
    "Ranker candidates:",
    len(ranker_candidates)
)

# %%
print(personal_als.columns.tolist())

# %%
# ============================================================
# REAL ALS MEMBERSHIP
# ============================================================

personal_als_check = personal_als.rename(
    columns={
        "visitorid": "user_id",
        "itemid": "item_id",
        "score": "als_score",
        "rank": "als_rank",
    }
).copy()


als_pairs = (
    personal_als_check[
        [
            "user_id",
            "item_id",
        ]
    ]
    .drop_duplicates()
    .copy()
)


als_pairs["real_from_als"] = 1


check = ranker_candidates.merge(
    als_pairs,
    on=[
        "user_id",
        "item_id",
    ],
    how="left",
)


check["real_from_als"] = (
    check["real_from_als"]
    .fillna(0)
    .astype("int8")
)


print(
    "По исходному personal_als:",
    check["real_from_als"].sum()
)

print(
    "По текущему from_als:",
    check["from_als"].sum()
)

# %%
get_memory_usage(10)


# %%
# ============================================================
# СОЗДАЁМ ОБУЧАЮЩУЮ ТАБЛИЦУ RANKER
# ============================================================

ranker_users = positive_pairs[
    "user_id"
].unique()


# Сначала сокращаем огромный candidates
ranker_train = candidates[
    candidates["user_id"].isin(
        ranker_users
    )
].copy()


# Добавляем target
ranker_train = ranker_train.merge(
    positive_pairs,
    on=[
        "user_id",
        "item_id",
    ],
    how="left",
    validate="many_to_one",
)


ranker_train["target"] = (
    ranker_train["target"]
    .fillna(0)
    .astype("int8")
)


print(
    "Ranker rows:",
    len(ranker_train)
)

print(
    "Positive:",
    ranker_train["target"].sum()
)

print(
    "Positive share:",
    ranker_train["target"].mean()
)

# %%
# Используем только пользователей,
# которые присутствуют в периоде обучения ranker.

ranker_users = ranker_events[
    "user_id"
].unique()


ranker_train = ranker_train[
    ranker_train["user_id"].isin(
        ranker_users
    )
].copy()


# Оставляем группы, где есть хотя бы один positive.
group_stats = (
    ranker_train
    .groupby("user_id")["target"]
    .agg(
        positives="sum",
        total="count"
    )
)


group_stats = (
    ranker_train
    .groupby("user_id")["target"]
    .agg(
        positives="sum",
        total="count",
    )
)


group_stats["negatives"] = (
    group_stats["total"]
    - group_stats["positives"]
)


valid_users = group_stats[
    (group_stats["positives"] > 0)
    &
    (group_stats["negatives"] > 0)
].index


ranker_train = ranker_train[
    ranker_train["user_id"].isin(
        valid_users
    )
].copy()

ranker_train = ranker_train[
    ranker_train["user_id"].isin(
        valid_users
    )
].copy()


print(
    "Пользователей:",
    ranker_train["user_id"].nunique()
)

print(
    "Строк:",
    len(ranker_train)
)

print(
    "Доля target=1:",
    ranker_train["target"].mean()
)

# %%
from catboost import CatBoostRanker


# CatBoost требует,
# чтобы строки одной группы шли подряд.
ranker_train = (
    ranker_train
    .sort_values(
        [
            "user_id",
            "als_rank"
        ]
    )
    .reset_index(drop=True)
)


ranker = CatBoostRanker(
    iterations=50,
    depth=6,
    learning_rate=0.05,
    loss_function="YetiRank",
    random_seed=42,
    verbose=1
)


ranker.fit(
    ranker_train[FEATURES],
    ranker_train["target"],
    group_id=ranker_train["user_id"]
)

# %%
# ------------------------------------------------------------
# Получаем итоговый score
# ------------------------------------------------------------

candidates["rank_score"] = (
    ranker
    .predict(
        candidates[FEATURES]
    )
    .astype("float32")
)


recommendations = (
    candidates[
        [
            "user_id",
            "item_id",
            "rank_score"
        ]
    ]
    .sort_values(
        [
            "user_id",
            "rank_score"
        ],
        ascending=[
            True,
            False
        ]
    )
    .reset_index(drop=True)
)


recommendations["rank"] = (
    recommendations
    .groupby("user_id")
    .cumcount()
    .add(1)
    .astype("int16")
)


recommendations = recommendations.rename(
    columns={
        "rank_score": "score"
    }
)


display(
    recommendations.head(20)
)

# %%
# ------------------------------------------------------------
# Сохраняем итоговые рекомендации
# ------------------------------------------------------------

recommendations.to_parquet(
    "recommendations.parquet",
    index=False
)


s3.upload_file(
    "recommendations.parquet",
    bucket_name,
    "recsys/recommendations/recommendations.parquet"
)


print(
    "recommendations.parquet сохранён и загружен в S3"
)

# %% [markdown]
# # Оценка качества

# %% [markdown]
# Проверим оценку качества трёх типов рекомендаций: 
# 
# - топ популярных,
# - персональных, полученных при помощи ALS,
# - итоговых
#   
# по четырем метрикам: recall, precision, coverage, novelty.

# %%
import gc
import numpy as np
import pandas as pd

K = 10

# %%
# ============================================================
# ЗАГРУЗКА РЕКОМЕНДАЦИЙ ДЛЯ ОЦЕНКИ
# ============================================================

top_popular = pd.read_parquet(
    "top_popular.parquet",
    columns=["track_id", "score", "rank"]
)

top_popular = top_popular.rename(
    columns={
        "track_id": "item_id",
        "score": "popular_score",
        "rank": "popular_rank"
    }
)

top_popular = top_popular[
    top_popular["popular_rank"] <= K
].copy()


personal_als = pd.read_parquet(
    "personal_als.parquet",
    columns=["user_id", "item_id", "rank"],
    filters=[("rank", "<=", K)]
)

personal_als = personal_als.rename(
    columns={"rank": "als_rank"}
)


recommendations = pd.read_parquet(
    "recommendations.parquet",
    columns=["user_id", "item_id", "rank"],
    filters=[("rank", "<=", K)]
)


print("Top Popular:", top_popular.shape)
print("ALS:", personal_als.shape)
print("Final:", recommendations.shape)

# %%
# ============================================================
# GROUND TRUTH
# ============================================================

events_eval = events[["user_id", "track_id", "started_at"]].copy()

events_eval = events_eval[
    events_eval["started_at"]
    >= pd.Timestamp("2022-12-24")
].copy()

events_eval = events_eval.rename(
    columns={"track_id": "item_id"}
)

ground_truth = (
    events_eval[
        ["user_id", "item_id"]
    ]
    .drop_duplicates()
    .copy()
)

print("Ground truth:", ground_truth.shape)

# %%
del events_eval
gc.collect()

# %%
eval_users = np.intersect1d(
    ground_truth["user_id"].unique(),
    personal_als["user_id"].unique()
)

ground_truth = ground_truth[
    ground_truth["user_id"].isin(eval_users)
].copy()

print(
    "Пользователей для оценки:",
    len(eval_users)
)

# %%
popular_items = (
    top_popular
    .sort_values("popular_rank")
    ["item_id"]
    .to_numpy()
)

top_popular_eval = pd.DataFrame({
    "user_id": np.repeat(
        eval_users,
        len(popular_items)
    ),
    "item_id": np.tile(
        popular_items,
        len(eval_users)
    ),
    "rank": np.tile(
        np.arange(1, len(popular_items) + 1),
        len(eval_users)
    )
})

# %%
personal_als_eval = (
    personal_als[
        personal_als["user_id"].isin(eval_users)
    ][
        ["user_id", "item_id", "als_rank"]
    ]
    .rename(
        columns={"als_rank": "rank"}
    )
)

# %%
final_eval = recommendations[
    recommendations["user_id"].isin(eval_users)
][
    ["user_id", "item_id", "rank"]
].copy()

# %%
catalog = pd.read_parquet(
    "items.parquet",
    columns=["track_id"]
)

catalog_size = catalog["track_id"].nunique()

del catalog
gc.collect()

print("Размер каталога:", catalog_size)

# %%
events_train = events[["track_id", "started_at"]].copy()

events_train = events_train[
    events_train["started_at"]
    < pd.Timestamp("2022-12-16")
]

item_counts = (
    events_train["track_id"]
    .value_counts()
)

total_interactions = len(events_train)

novelty_map = -np.log2(
    item_counts / total_interactions
)

del events_train
del item_counts
gc.collect()

# %%
catalog = pd.read_parquet(
    "items.parquet",
    columns=["track_id"]
)


catalog_size = (
    catalog["track_id"]
    .nunique()
)


print(
    "Размер каталога:",
    catalog_size
)


del catalog
gc.collect()

# %%
def evaluate_recommendations(
    recommendations_df,
    ground_truth,
    catalog_size,
    novelty_map,
    k=10
):
    recs = (
        recommendations_df[
            recommendations_df["rank"] <= k
        ][["user_id", "item_id"]]
        .drop_duplicates()
        .copy()
    )

    truth = (
        ground_truth[
            ["user_id", "item_id"]
        ]
        .drop_duplicates()
        .copy()
    )

    relevant_count = (
        truth
        .groupby("user_id")
        .size()
    )

    hits = recs.merge(
        truth.assign(hit=1),
        on=["user_id", "item_id"],
        how="left"
    )

    hits["hit"] = (
        hits["hit"]
        .fillna(0)
        .astype("int8")
    )

    hits_per_user = (
        hits
        .groupby("user_id")["hit"]
        .sum()
        .reindex(
            relevant_count.index,
            fill_value=0
        )
    )

    precision = (
        hits_per_user / k
    ).mean()

    recall = (
        hits_per_user
        /
        relevant_count
    ).mean()

    coverage = (
        recs["item_id"].nunique()
        /
        catalog_size
    )

    novelty = (
        recs["item_id"]
        .map(novelty_map)
        .dropna()
        .mean()
    )

    return {
        "precision": precision,
        "recall": recall,
        "coverage": coverage,
        "novelty": novelty
    }

# %%
metrics_top_popular = evaluate_recommendations(
    top_popular_eval,
    ground_truth,
    catalog_size,
    novelty_map,
    k=K
)

metrics_als = evaluate_recommendations(
    personal_als_eval,
    ground_truth,
    catalog_size,
    novelty_map,
    k=K
)

metrics_final = evaluate_recommendations(
    final_eval,
    ground_truth,
    catalog_size,
    novelty_map,
    k=K
)

# %%
metrics = pd.DataFrame([
    {
        "model": "Top Popular",
        **metrics_top_popular
    },
    {
        "model": "ALS",
        **metrics_als
    },
    {
        "model": "Final Ranker",
        **metrics_final
    }
]).set_index("model")

display(
    metrics.style.format({
        "precision": "{:.6f}",
        "recall": "{:.6f}",
        "coverage": "{:.6f}",
        "novelty": "{:.4f}"
    })
)

# %% [markdown]
# # === Выводы, метрики ===

# %% [markdown]
# Основные выводы при работе над расчётом рекомендаций, рассчитанные метрики.

# %% [markdown]
# По результатам оценки персональные рекомендации ALS значительно превосходят неперсонализированный топ популярных по метрикам precision и recall. Дополнительное ранжирование рекомендаций позволило ещё улучшить качество: precision@10 увеличился с 0.004150 до 0.005828, а recall@10 — с 0.014103 до 0.016043. Также итоговая модель имеет наибольшее значение novelty — 13.1801, то есть чаще предлагает менее популярные объекты. При этом coverage снизился с 0.004482 для ALS до 0.003937 для итогового ranker. Это связано с тем, что ранжирующая модель не генерирует новые рекомендации, а переупорядочивает кандидатов, полученных ALS. Таким образом, ранжирование позволило повысить точность и полноту рекомендаций при небольшом снижении покрытия каталога.

# %%
personal_als = pd.read_parquet('personal_als.parquet')


# %%
top_popular = pd.read_parquet('top_popular.parquet')


# %%
personal_als.head(10)

# %%
top_popular.head(10)

# %%
similar = pd.read_parquet('similar.parquet')

# %%
similar.head(10)


