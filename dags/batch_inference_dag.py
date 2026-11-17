"""
DAG batch-инференса недельных продаж сети дарксторов "Прилавок"

Этот DAG выполняет следующие задачи:
1. Загрузка данных для инференса из Postgres (таблицы plan, stores, features)
2. Предобработка признаков (идентичная логике обучения)
3. Загрузка обученной модели CatBoost из S3
4. Batch-инференс на подготовленных данных. Пост-обработка предсказаний
6. Запись результатов в таблицу predictions в Postgres

ВАЖНО: Креденшелы подставляются через Airflow Variables/Connections!

============= Полезные инструкции =============

1. НАСТРОЙКА AIRFLOW VARIABLES (Admin → Variables в UI):
   - s3_bucket_name = ваше_имя_бакета
   - s3_model_key = путь_к_модели.pkl
   - s3_access_key = ваш_access_key
   - s3_secret_key = ваш_secret_key

2. НАСТРОЙКА AIRFLOW CONNECTION (Admin → Connections):
   - Conn Id: postgres_sales_db
   - Conn Type: Postgres
   - Host, Port, Schema, Login, Password из параметров вашей БД

3. ПОЛЕЗНЫЕ ССЫЛКИ:
   - Документация Airflow XCom: https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/xcoms.html
   - PostgresHook: https://airflow.apache.org/docs/apache-airflow-providers-postgres/stable/_api/airflow/providers/postgres/hooks/postgres/index.html

4. ПОДСКАЗКИ:
   - Используйте context['ti'].xcom_push(key='название', value=данные) для передачи данных
   - Используйте context['ti'].xcom_pull(key='название', task_ids='имя_задачи') для получения
"""

from datetime import datetime, timedelta
import io
import logging
import os
import pickle
import tempfile
from airflow import DAG
from airflow.hooks.postgres_hook import PostgresHook
from airflow.models import Variable
from airflow.operators.python import PythonOperator
import boto3
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

# Импорт функции предобработки
from preprocessing import preprocess_data

# ========== Конфигурация ==========

S3_BUCKET = Variable.get('s3_bucket_name')
S3_ACCESS_KEY = Variable.get('s3_access_key', default_var=None)
S3_SECRET_KEY = Variable.get('s3_secret_key', default_var=None)
S3_MODEL_KEY = Variable.get('s3_model_key', default_var='catboost_model.pkl')

POSTGRES_CONN_ID = 'postgres_sales_db'

logger = logging.getLogger(__name__)


# ========== Задачи DAG ==========


def load_data_from_postgres(**context):
  """Этап 1.

  Загрузка данных для инференса из Postgres в таблицу inference_data_temp.
  """
  hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

  drop_table_query = 'DROP TABLE IF EXISTS inference_data_temp;'

  create_table_query = """
    CREATE TABLE inference_data_temp AS
    WITH combined_sales_plan AS (
        SELECT store, dept, date, weekly_sales, is_holiday FROM sales
        UNION ALL
        SELECT store, dept, date, NULL AS weekly_sales, is_holiday FROM plan
    )
    SELECT 
        c.store, c.dept, c.date, c.weekly_sales, c.is_holiday,
        s.type, s.size,
        f.temperature, f.fuel_price, f.cpi, f.unemployment,
        f.factor1, f.factor2, f.factor3, f.factor4, f.factor5
    FROM combined_sales_plan c
    LEFT JOIN stores s ON c.store = s.store
    LEFT JOIN features f ON c.store = f.store AND c.dept = f.dept AND c.date = f.date;
    """

  logging.info('Пересоздание таблицы inference_data_temp в PostgreSQL...')
  hook.run(drop_table_query)
  hook.run(create_table_query)

  # Вывод статистики
  stats_query = 'SELECT COUNT(*), MIN(date), MAX(date) FROM inference_data_temp;'
  stats = hook.get_first(stats_query)

  logging.info('Информация о таблице inference_data_temp:')
  logging.info(f'- Количество строк: {stats[0]}')
  logging.info(f'- Минимальная дата: {stats[1]}')
  logging.info(f'- Максимальная дата: {stats[2]}')

  # Передача первой даты плана в XCom
  first_plan_date_query = 'SELECT MIN(date) FROM plan;'
  first_plan_date = str(hook.get_first(first_plan_date_query)[0])

  ti = context['ti']
  ti.xcom_push(key='first_plan_date', value=first_plan_date)
  logging.info(f'Первая дата плана {first_plan_date} отправлена в XCom.')


def preprocess_features(**context):
  """Этап 2.

  Предобработка признаков для инференса.
  """
  ti = context['ti']
  first_plan_date = ti.xcom_pull(key='first_plan_date', task_ids='load_data')
  logging.info(f'Получена первая дата плана: {first_plan_date}')

  hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
  df = hook.get_pandas_df('SELECT * FROM inference_data_temp;')

  # Предобработка всех данных (история + план)
  df_preprocessed = preprocess_data(df)

  # Фильтрация строго по дате плана
  df_preprocessed['date'] = pd.to_datetime(df_preprocessed['date'])
  plan_date_dt = pd.to_datetime(first_plan_date)

  df_plan = df_preprocessed[df_preprocessed['date'] >= plan_date_dt].copy()

  # Удаляем weekly_sales
  if 'weekly_sales' in df_plan.columns:
    df_plan = df_plan.drop(columns=['weekly_sales'])

  # Заполняем лаговые и скользящие колонки нулями
  lag_cols = [
      c
      for c in df_plan.columns
      if 'sales_' in c or 'mean_sales_' in c or 'avg_sales_' in c
  ]
  df_plan[lag_cols] = df_plan[lag_cols].fillna(0)

  # Удаляем оставшиеся NaN
  df_plan = df_plan.dropna().reset_index(drop=True)

  # Сохраняем обратно строковый формат даты перед сериализацией
  df_plan['date'] = df_plan['date'].dt.strftime('%Y-%m-%d')

  ti.xcom_push(key='preprocessed_data', value=df_plan.to_json(orient='split'))

  # Удаляем временную таблицу
  hook.run('DROP TABLE IF EXISTS inference_data_temp;')
  logging.info('Предобработка завершена, временная таблица удалена.')


def load_model_from_s3(**context):
  """Этап 3.

  Загрузка модели из S3 Yandex Cloud.
  """
  s3_client = boto3.client(
      's3',
      endpoint_url='https://storage.yandexcloud.net',
      aws_access_key_id=S3_ACCESS_KEY,
      aws_secret_access_key=S3_SECRET_KEY,
      region_name='ru-central1',
  )

  model_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_MODEL_KEY)
  model_bytes = model_obj['Body'].read()
  bytes_buffer = io.BytesIO(model_bytes)

  temp_dir = tempfile.gettempdir()
  temp_model_path = os.path.join(temp_dir, 'catboost_model.pkl')

  with open(temp_model_path, 'wb') as f:
    f.write(bytes_buffer.getbuffer())

  ti = context['ti']
  ti.xcom_push(key='model_path', value=temp_model_path)
  logging.info(f'Модель успешно загружена в {temp_model_path}')


def run_batch_inference(**context):
  """Этап 4.

  Batch-инференс предсказаний.
  """
  ti = context['ti']
  json_data = ti.xcom_pull(
      key='preprocessed_data', task_ids='preprocess_features'
  )
  model_path = ti.xcom_pull(key='model_path', task_ids='load_model')

  if not json_data or not model_path:
    raise ValueError('Данные или путь к модели не получены из XCom!')

  df = pd.read_json(json_data, orient='split')

  with open(model_path, 'rb') as f:
    model = pickle.load(f)

  # Выделяем фичи для предсказания
  if hasattr(model, 'feature_names_'):
    feature_cols = model.feature_names_
    X = df[feature_cols]
  else:
    ignore_cols = ['store', 'dept', 'date']
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    X = df[feature_cols]

  # Прогноз и клиппинг аномалий
  predictions = model.predict(X)
  df['predicted_weekly_sales'] = predictions
  df['predicted_weekly_sales'] = df['predicted_weekly_sales'].clip(lower=0)

  ti.xcom_push(key='predictions', value=df.to_json(orient='split'))
  logging.info(f'Батч-инференс выполнен. Размер: {df.shape}')


def save_predictions_to_postgres(**context):
  """Этап 5.

  Сохранение предсказаний в PostgreSQL.
  """
  ti = context['ti']
  predictions_json = ti.xcom_pull(
      key='predictions', task_ids='run_inference'
  )
  predictions_df = pd.read_json(predictions_json, orient='split')

  predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.date

  hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

  hook.run('DROP TABLE IF EXISTS predictions;')
  create_table_query = """
    CREATE TABLE IF NOT EXISTS predictions (
        store INT,
        dept INT,
        date DATE,
        predicted_weekly_sales FLOAT,
        prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (store, dept, date)
    );
    """
  hook.run(create_table_query)

  values = (
      predictions_df[['store', 'dept', 'date', 'predicted_weekly_sales']]
      .to_records(index=False)
      .tolist()
  )

  insert_query = """
    INSERT INTO predictions (store, dept, date, predicted_weekly_sales)
    VALUES %s
    ON CONFLICT (store, dept, date) 
    DO UPDATE SET 
        predicted_weekly_sales = EXCLUDED.predicted_weekly_sales,
        prediction_timestamp = CURRENT_TIMESTAMP;
    """

  conn = hook.get_conn()
  with conn.cursor() as cur:
    execute_values(cur, insert_query, values)
    conn.commit()

  # Статистика
  row_count = hook.get_first('SELECT COUNT(*) FROM predictions;')[0]
  top5_rows = hook.get_records('SELECT * FROM predictions LIMIT 5;')

  logging.info(f'Записано строк в витрину predictions: {row_count}')
  logging.info('Первые 5 строк таблицы predictions:')
  for row in top5_rows:
    logging.info(row)


# ========== Определение DAG ==========

default_args = {
    'owner': 'DataScience_Team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'sales_prediction_batch_inference',
    default_args=default_args,
    description='Batch-инференс прогнозирования продаж для Прилавка',
    schedule_interval='0 20 * * 0',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['sales', 'ml', 'batch-inference', 'production'],
)

task_load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data_from_postgres,
    provide_context=True,
    dag=dag,
)

task_preprocess = PythonOperator(
    task_id='preprocess_features',
    python_callable=preprocess_features,
    provide_context=True,
    dag=dag,
)

task_load_model = PythonOperator(
    task_id='load_model',
    python_callable=load_model_from_s3,
    provide_context=True,
    dag=dag,
)

task_inference = PythonOperator(
    task_id='run_inference',
    python_callable=run_batch_inference,
    provide_context=True,
    dag=dag,
)

task_save_predictions = PythonOperator(
    task_id='save_predictions',
    python_callable=save_predictions_to_postgres,
    provide_context=True,
    dag=dag,
)

task_load_data >> task_preprocess
task_load_model >> task_inference
task_preprocess >> task_inference >> task_save_predictions