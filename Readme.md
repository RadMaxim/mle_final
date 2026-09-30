# Рекомендательная система для интернет-магазина

## Кратко о проекте

Цель проекта — построить воспроизводимый ML-пайплайн рекомендательной системы на основе пользовательских событий `view`, `addtocart` и `transaction`.

Система использует три источника кандидатов:

- **Top Popular** — популярные товары;
- **ALS** — персональные рекомендации по implicit feedback;
- **Similar Items** — похожие товары на основе ALS-представлений.

Кандидаты объединяются, для пары `user-item` рассчитываются признаки, после чего итоговый порядок рекомендаций формирует **CatBoostRanker**.

Основные технологии: **Python, pandas, NumPy, PyArrow, DuckDB, implicit ALS, CatBoost, MLflow, Apache Airflow, PostgreSQL, Redis, Docker Compose, Yandex Object Storage (S3), Prometheus, Grafana**.

---

## Структура проекта

```text
.
├── archive/                         # исходные CSV
├── parquet_cleaned/                 # очищенные Parquet
├── recommendations/                 # Top Popular, ALS, Similar Items
├── models/                          # сохранённые модели
│
├── part1_airflow/
│   ├── dags/
│   │   ├── recsys_data_pipeline.py
│   │   ├── recsys_recommendation_pipeline.py
│   │   ├── recsys_candidates_pipeline.py
│   │   ├── recsys_ranker_pipeline.py
│   │   ├── recsys_full_pipeline.py
│   │   └── test_mlflow_pipeline.py
│   ├── logs/
│   ├── plugins/
│   └── data/
│
├── services/                        # inference-сервис и Prometheus
│
├── 01_data_preprocessing_and_eda.ipynb
├── 02_recommendation_generation.ipynb
├── 03_candidate_generation.ipynb
├── 04_ranker_training_and_inference.ipynb
├── 05_model_evaluation.ipynb
│
├── params.yaml                      # параметры pipeline
├── docker-compose.yaml              # инфраструктура проекта
├── Dockerfile                       # Airflow image
├── DockerfileMLflow                 # MLflow image
├── requirements.txt                 # окружение для локальной работы
├── airflow_requirements.txt         # зависимости Airflow
├── mlflow_req.txt                   # зависимости MLflow
├── server_model.sh                  # запуск MLflow server
├── start_project.sh                 # запуск инфраструктуры
└── Readme.md
```

### Форматы файлов

- `.csv` — исходные данные;
- `.parquet` — подготовленные данные, кандидаты и рекомендации;
- `.cbm` — модель CatBoostRanker;
- `.yaml` — конфигурация;
- `.py` — Airflow DAG-и и Python-код;
- `.sh` — shell-скрипты запуска;
- `.env` — переменные окружения и секреты.

---

## Установка и запуск

### 1. Клонирование проекта

```bash
git clone https://github.com/RadMaxim/mle_final.git
cd mle_final
```

### 2. Локальное Python-окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Переменные окружения

Создайте `.env` в корне проекта. Основные переменные:

```env
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
S3_BUCKET_NAME=

AIRFLOW_UID=50000
AIRFLOW_PROJ_DIR=.

MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net

APP_PORT=8000
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_USER=admin
GRAFANA_PASS=admin

DB_DESTINATION_USER=
DB_DESTINATION_PASSWORD=
DB_DESTINATION_HOST=
DB_DESTINATION_PORT=
DB_DESTINATION_NAME=
```

### 4. Запуск инфраструктуры

Основной способ:

```bash
chmod +x start_project.sh
./start_project.sh
```

Либо вручную:

```bash
docker compose build
docker compose up airflow-init
docker compose up -d
```

После запуска доступны:

- Airflow: `http://localhost:8080`
- MLflow: `http://localhost:5000`
- inference API: порт из `APP_PORT`
- Prometheus: порт из `PROMETHEUS_PORT`
- Grafana: порт из `GRAFANA_PORT`

---

# Руководство по проекту

## 1. Трансляция бизнес-задачи в техническую задачу

Бизнес-задача — повысить релевантность товарных рекомендаций для пользователя.

Технически задача решается в два этапа:

1. **Candidate generation** — формирование набора возможных рекомендаций из Top Popular, ALS и Similar Items.
2. **Ranking** — переупорядочивание кандидатов с помощью CatBoostRanker.

Для оценки используются:

- `precision@10`;
- `recall@10`;
- `coverage`;
- `novelty`.

Финальная модель показала:

| Модель | Precision@10 | Recall@10 | Coverage | Novelty |
|---|---:|---:|---:|---:|
| ALS | 0.009504 | 0.067424 | 0.001537 | 12.2737 |
| Final Ranker | 0.010744 | 0.077755 | 0.001942 | 12.5657 |

Итоговый ranker улучшил основные offline-метрики относительно ALS.

---

## 2. Разворачивание инфраструктуры обучения

Инфраструктура запускается через Docker Compose и включает:

- **Airflow** — оркестрация pipeline;
- **PostgreSQL** — metadata DB для Airflow и MLflow;
- **Redis** — broker для CeleryExecutor;
- **MLflow** — хранение параметров, метрик и моделей;
- **Yandex Object Storage** — хранение Parquet-файлов и артефактов.

Основной запуск:

```bash
./start_project.sh
```

Параметры моделей, пути и размеры batch-обработки вынесены в `params.yaml`.

---

## 3. EDA

EDA находится в:

```text
01_data_preprocessing_and_eda.ipynb
```

Основные шаги:

- конвертация CSV → Parquet;
- проверка пропусков и дубликатов;
- оптимизация типов данных;
- анализ временного диапазона;
- анализ событий `view`, `addtocart`, `transaction`;
- анализ товарных свойств и категорий;
- проверка покрытия товаров метаданными.

Ключевые выводы:

- пользовательские события сильно несбалансированы: просмотры значительно преобладают;
- `transactionid` отсутствует у `view` и `addtocart` по структуре данных, а не из-за ошибки;
- часть товаров отсутствует в `item_properties`, но остаётся полезной для коллаборативной модели;
- Parquet существенно уменьшает объём данных на диске;
- для implicit-feedback модели события имеют разную силу и используются с разными весами.

---

## 4. Генерация признаков и обучение модели

### Ноутбуки

```text
01_data_preprocessing_and_eda.ipynb
        ↓
02_recommendation_generation.ipynb
        ↓
03_candidate_generation.ipynb
        ↓
04_ranker_training_and_inference.ipynb
        ↓
05_model_evaluation.ipynb
```

Назначение:

- `01_...` — подготовка данных и EDA;
- `02_...` — Top Popular, ALS и Similar Items;
- `03_...` — объединение кандидатов и построение признаков;
- `04_...` — обучение CatBoostRanker и inference;
- `05_...` — сравнение моделей по offline-метрикам.

### Признаки ranker

Используются признаки:

```text
als_score
als_rank
similarity_score
similarity_rank
popular_score
popular_rank
```

Target формируется по событиям:

```text
addtocart
transaction
```

Модель:

```text
CatBoostRanker
loss_function = YetiRank
```

### MLflow

В MLflow логируются:

- параметры ALS и CatBoostRanker;
- параметры генерации кандидатов;
- статистики candidate set;
- coverage положительных user-item пар;
- модель CatBoostRanker;
- метрики обучения и экспериментов.

Эксперименты разделены на:

```text
recsys_als
recsys_candidates
recsys_ranker
```

---

## 5. Airflow DAG-и

Логика ноутбуков перенесена в отдельные DAG-и:

### `recsys_data_pipeline.py`

Подготовка исходных данных:

- CSV → Parquet;
- очистка и проверка данных;
- сохранение подготовленных файлов.

### `recsys_recommendation_pipeline.py`

Генерация базовых рекомендаций:

- Top Popular;
- Personal ALS;
- Similar Items;
- логирование ALS в MLflow.

### `recsys_candidates_pipeline.py`

Формирование итогового candidate set и признаков ranker.

### `recsys_ranker_pipeline.py`

- подготовка train-выборки;
- обучение CatBoostRanker;
- scoring кандидатов;
- построение `recommendations.parquet`;
- логирование модели в MLflow.

### `recsys_full_pipeline.py`

Master DAG, который запускает основные DAG-и последовательно:

```text
recsys_data_pipeline
        ↓
recsys_recommendation_pipeline
        ↓
recsys_candidates_pipeline
        ↓
recsys_ranker_pipeline
```

Для полного обновления модели достаточно запустить `recsys_full_pipeline` в Airflow.

---

## 6. Разворачивание инфраструктуры применения модели

Inference-сервис запускается вместе с инфраструктурой через Docker Compose.

Основные компоненты:

- **main-app** — API для применения модели;
- **CatBoostRanker** — модель ранжирования;
- **recommendations.parquet** — итоговые рекомендации;
- **Prometheus** — сбор технических метрик;
- **Grafana** — визуализация мониторинга;
- **Airflow** — обновление данных, кандидатов и модели;
- **MLflow** — история экспериментов и модельные артефакты.

Запуск:

```bash
./start_project.sh
```

Проверка контейнеров:

```bash
docker compose ps
```

---

## Основные артефакты

```text
parquet_cleaned/events.parquet
parquet_cleaned/item_properties.parquet
recommendations/top_popular/top_popular.parquet
recommendations/personal_als/personal_als.parquet
recommendations/similar/similar.parquet
candidates_result.parquet
models/catboost_ranker.cbm
recommendations.parquet
```

## Жизненный цикл модели

Проект реализует полный цикл работы рекомендательной системы:

1. **Обработка данных**
   - исходные CSV преобразуются в Parquet;
   - выполняются очистка, проверка качества и оптимизация типов данных;
   - подготовленные данные сохраняются локально и в S3.

2. **Создание модели**
   - строятся Top Popular, Personal ALS и Similar Items;
   - формируется объединённый набор кандидатов;
   - для кандидатов рассчитываются признаки;
   - CatBoostRanker обучается на событиях `addtocart` и `transaction`;
   - эксперименты, параметры и модель логируются в MLflow.

3. **Выкатка модели**
   - итоговая модель сохраняется в формате `.cbm`;
   - формируется `recommendations.parquet`;
   - inference-сервис запускается через Docker Compose;
   - сервис использует подготовленные рекомендации и модель для выдачи результата.

4. **Сопровождение**
   - Airflow автоматизирует последовательное обновление данных, рекомендаций, кандидатов и ranker;
   - `recsys_full_pipeline` запускает весь ML pipeline;
   - MLflow используется для отслеживания экспериментов и хранения модельных артефактов;
   - Prometheus собирает технические метрики;
   - Grafana используется для мониторинга состояния сервиса.