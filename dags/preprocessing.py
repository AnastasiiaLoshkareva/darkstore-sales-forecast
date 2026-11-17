"""Модуль предобработки данных и инжиниринга признаков.

Выполняет базовую очистку данных (обработку аномальных продаж и пропусков)
и генерирует временные, накопительные, лаговые и скользящие признаки 
для обучения модели и применения batch-инференса в прогнозировании продаж.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_temporal_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    return df


def create_avg_sales_feature(df):
    df = df.copy()
    df = df.sort_values(by=['store', 'dept', 'date']).reset_index(drop=True)

    df['avg_sales_before'] = df.groupby(['store', 'dept'])[
        'weekly_sales'
    ].transform(lambda x: x.expanding().mean().shift(1))

    return df


def create_lag_features(df):
    df_temp = df.copy()
    df_temp = df_temp.sort_values(by=['store', 'dept', 'date']).reset_index(
        drop=True
    )

    df_temp['sales_1week_ago'] = df_temp.groupby(['store', 'dept'])[
        'weekly_sales'
    ].transform(lambda x: x.shift(1))
    df_temp['sales_2week_ago'] = df_temp.groupby(['store', 'dept'])[
        'weekly_sales'
    ].transform(lambda x: x.shift(2))
    df_temp['sales_4week_ago'] = df_temp.groupby(['store', 'dept'])[
        'weekly_sales'
    ].transform(lambda x: x.shift(4))

    return df_temp


def create_rolling_features(df):
    df_temp = df.copy()
    df_temp = df_temp.sort_values(by=['store', 'dept', 'date']).reset_index(
        drop=True
    )

    df_temp['mean_sales_2week'] = df_temp.groupby(['store', 'dept'])[
        'weekly_sales'
    ].transform(lambda x: x.rolling(window=2).mean().shift(1))
    df_temp['mean_sales_4week'] = df_temp.groupby(['store', 'dept'])[
        'weekly_sales'
    ].transform(lambda x: x.rolling(window=4).mean().shift(1))
    return df_temp


def preprocess_data(df):
    """Полная предобработка данных (идентичная логике обучения)."""
    df = df.copy()

    # 1. Обработка аномальных (отрицательных) продаж в исторической части
    # Заменяем отрицательные продажи на NaN, чтобы они не искажали средние и лаги
    if 'weekly_sales' in df.columns:
        df.loc[df['weekly_sales'] < 0, 'weekly_sales'] = np.nan

    # 2. Обработка пропусков во ВСЕХ числовых признаках, КРОМЕ weekly_sales
    feature_cols = [col for col in df.columns if col != 'weekly_sales']
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(
        df[numeric_cols].mean(numeric_only=True)
    )

    # 3. Создаем временные признаки
    df = create_temporal_features(df)

    # 4. Создаем признаки среднего значения продаж
    df = create_avg_sales_feature(df)

    # 5. Создаем лаговые признаки
    df = create_lag_features(df)

    # 6. Создаем скользящие признаки
    df = create_rolling_features(df)

    return df
