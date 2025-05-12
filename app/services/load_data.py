# app/services/load_data.py

import io
import os
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from app.services.calculate_indicators import calculate_indicators
from app.utils.logger import logger
from app.extensions import db



def remove_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Обычная фильтрация выбросов по одному столбцу.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

def remove_outliers_multi(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Фильтрация выбросов сразу по нескольким столбцам без цикла вызовов.
    """
    mask = pd.Series(True, index=df.index)
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask &= df[col].between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return df[mask]

def clean_nbsp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заменяем все неразрывные пробелы на обычные в любых текстовых колонках.
    """
    return df.replace({u'\u00A0': ' '}, regex=True)

# категории возраста и дохода
AGE_BINS = [0, 18, 25, 35, 45, 55, 64, np.inf]
AGE_LABELS = [
    "до 18 лет",
    "от 18 до 24 лет",
    "от 25 до 34 лет",
    "от 35 до 44 лет",
    "от 45 до 54 лет",
    "от 55 до 63 лет",
    "свыше 64 лет",
]
INC_BINS = [0, 30_000, 50_000, 70_000, 100_000, np.inf]
INC_LABELS = [
    "до 30 тыс. руб.",
    "от 30 до 50 тыс. руб.",
    "от 50 до 70 тыс. руб.",
    "от 70 до 100 тыс. руб.",
    "свыше 100 тыс. руб.",
]

def _parse_age(age_str: str) -> float:
    if pd.isna(age_str) or not age_str.strip():
        return np.nan
    nums = re.findall(r'\d+', age_str)
    if len(nums) == 2:
        return (int(nums[0]) + int(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return np.nan

def _parse_income(income_str: str) -> float:
    if pd.isna(income_str) or not income_str.strip():
        return np.nan
    s = income_str.replace(' ', '').lower()
    nums = re.findall(r'\d+', s)
    if not nums:
        return np.nan
    val = float(nums[0])
    if 'млн' in s:
        return val * 1_000_000
    if 'тыс' in s:
        return val * 1_000
    return val

def _num_to_age_cat(x: float) -> str:
    return pd.cut([x], bins=AGE_BINS, labels=AGE_LABELS)[0]

def _num_to_inc_cat(x: float) -> str:
    return pd.cut([x], bins=INC_BINS, labels=INC_LABELS)[0]

def _read_and_preprocess(csv_path: str) -> pd.DataFrame:
    logger.info(f'Загрузка данных из файла: {csv_path}')

    # 1) Загрузка данных и базовая фильтрация
    usecols = [
        'DATE_OF_ARRIVAL','DAYS_CNT','VISITORS_CNT','SPENT',
        'HOME_REGION','HOME_CITY','HOME_COUNTRY',
        'AGE','GENDER','INCOME','GOAL'
    ]
    dtype = {c: str for c in ['HOME_REGION','HOME_CITY','HOME_COUNTRY',
                               'AGE','GENDER','INCOME','GOAL']}
    df = pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=dtype,
        parse_dates=['DATE_OF_ARRIVAL'],
        low_memory=False
    )
    logger.info(f'Записей до обработки: {len(df)}')

    # 2) Оставляем только записи с целью "Туризм"
    if 'GOAL' not in df.columns:
        logger.error("Отсутствует столбец GOAL")
        raise ValueError("Не найден столбец GOAL")
    before = len(df)
    df = df[df['GOAL'] == 'Туризм'].drop(columns=['GOAL'])
    logger.info(f'После фильтрации по GOAL="Туризм": {len(df)} из {before}')

    # 3) Удаление строк-заголовков и очистка текста
    df = df[df.ne(df.columns).all(axis=1)]
    df = clean_nbsp(df)
    logger.info(f'После удаления артефактов: {len(df)}')

    # 4) Отметка иностранцев и фильтрация россиян по возрасту и полу
    df['is_foreign'] = df['HOME_COUNTRY'].str.lower() != 'россия'
    rus_mask = (
        (~df['is_foreign']) &
        df['AGE'].notna() & df['GENDER'].notna() &
        df['AGE'].str.strip().ne('') & df['GENDER'].str.strip().ne('')
    )
    df = df[rus_mask | df['is_foreign']].copy()
    logger.info(f'После отбора россиян с AGE/GENDER: {len(df)}')

    # 5) Преобразование AGE и INCOME в числовой формат
    df['AGE_NUM'] = df['AGE'].apply(_parse_age)
    df['INC_NUM'] = df['INCOME'].apply(_parse_income)

    # 6) Проверка и очистка дат
    df['DATE_OF_ARRIVAL'] = pd.to_datetime(df['DATE_OF_ARRIVAL'], errors='coerce')
    df = df[df['DATE_OF_ARRIVAL'].notna()]
    logger.info(f'После обработки дат: {len(df)}')

    # 7) Заполнение пропусков в числовых полях
    for col in ['DAYS_CNT','VISITORS_CNT','SPENT']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['SPENT_MISSING'] = df['SPENT'].isna()
    df['DAYS_CNT'] = df['DAYS_CNT'].fillna(df['DAYS_CNT'].median())
    df['VISITORS_CNT'] = df['VISITORS_CNT'].fillna(df['VISITORS_CNT'].median())
    df['SPENT'] = df['SPENT'].fillna(0)
    logger.info(f'После заполнения пропусков: {len(df)}')

    # 8) Удаление выбросов по ключевым показателям
    df = remove_outliers_multi(df, ['DAYS_CNT','VISITORS_CNT','SPENT'])
    logger.info(f'После удаления выбросов: {len(df)}')

    # 9) Обучение моделей на российской выборке и заполняем пропуски для иностранцев
    rus_full = df[(~df['is_foreign']) & df['AGE_NUM'].notna() & df['INC_NUM'].notna()].copy()
    FEATS = ['DAYS_CNT','VISITORS_CNT']

    age_model = RandomForestRegressor(n_estimators=100, random_state=0)
    age_model.fit(rus_full[FEATS], rus_full['AGE_NUM'])
    inc_model = RandomForestRegressor(n_estimators=100, random_state=0)
    inc_model.fit(rus_full[FEATS], rus_full['INC_NUM'])
    sp_model = RandomForestRegressor(n_estimators=100, random_state=0)
    sp_model.fit(rus_full[FEATS], rus_full['SPENT'])
    gen_model = RandomForestClassifier(n_estimators=100, random_state=0)
    gen_model.fit(rus_full[FEATS], rus_full['GENDER'])

    foreign = df['is_foreign']
    mask_a = foreign & df['AGE'].isna()
    df.loc[mask_a, 'AGE'] = [_num_to_age_cat(x) for x in age_model.predict(df.loc[mask_a, FEATS])]
    mask_i = foreign & df['INCOME'].isna()
    df.loc[mask_i, 'INCOME'] = [_num_to_inc_cat(x) for x in inc_model.predict(df.loc[mask_i, FEATS])]
    mask_s = foreign & df['SPENT_MISSING']
    df.loc[mask_s, 'SPENT'] = sp_model.predict(df.loc[mask_s, FEATS])
    mask_g = foreign & df['GENDER'].isna()
    df.loc[mask_g, 'GENDER'] = gen_model.predict(df.loc[mask_g, FEATS])

    # 10) Очистка вспомогательных столбцов и вычисление финальных признаков
    df.drop(columns=['AGE_NUM','INC_NUM','SPENT_MISSING','is_foreign'], inplace=True)
    df['avg_spent_per_visitor'] = np.where(
        df['VISITORS_CNT'] > 0,
        df['SPENT'] / df['VISITORS_CNT'],
        0.0
    )

    return df


def load_data_to_db(database_url: str, csv_path: str = None, default_dir: str = None) -> pd.DataFrame:
    """
    Загружает CSV в БД через staging + транзакционный swap,
    с нормализацией часто повторяющихся полей в измерительные таблицы,
    и всегда пересчитывает индикаторы (перезаписывает JSON).
    """
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'output')
    )
    output_path = os.path.join(output_dir, 'indicators.json')

    # Проверяем путь к CSV
    if not csv_path:
        logger.error("Путь к CSV не задан (параметр csv_path обязательный).")
        raise FileNotFoundError("Путь к CSV не задан: csv_path is required")
    if not os.path.exists(csv_path):
        logger.error(f"CSV-файл не найден: {csv_path}")
        raise FileNotFoundError(f"CSV-файл не найден: {csv_path}")

    logger.info(f"Используется CSV: {csv_path}")
    # Предобработка исходных данных
    df = _read_and_preprocess(csv_path)

    # Приводим все имена колонок к нижнему регистру для унификации
    df_norm = df.rename(columns=lambda c: c.lower())

    # --- Построение размерностей (справочников) ---
    df_regions   = pd.DataFrame(df_norm['home_region'].dropna().unique(),  columns=['home_region'])
    df_cities    = pd.DataFrame(df_norm['home_city'].dropna().unique(),    columns=['home_city'])
    df_countries = pd.DataFrame(df_norm['home_country'].dropna().unique(), columns=['home_country'])
    df_ages      = pd.DataFrame(df_norm['age'].dropna().unique(),          columns=['age'])
    df_genders   = pd.DataFrame(df_norm['gender'].dropna().unique(),       columns=['gender'])
    df_incomes   = pd.DataFrame(df_norm['income'].dropna().unique(),       columns=['income'])

    # Добавляем суррогатные ключи
    for df_dim, col in [
        (df_regions,   'home_region'),
        (df_cities,    'home_city'),
        (df_countries, 'home_country'),
        (df_ages,      'age'),
        (df_genders,   'gender'),
        (df_incomes,   'income'),
    ]:
        df_dim[f"{col}_id"] = range(1, len(df_dim) + 1)

    # --- Фактовая таблица с внешними ключами ---
    df_fact = (
        df_norm
          .merge(df_regions,   on='home_region',    how='left')
          .merge(df_cities,    on='home_city',      how='left')
          .merge(df_countries, on='home_country',   how='left')
          .merge(df_ages,      on='age',            how='left')
          .merge(df_genders,   on='gender',         how='left')
          .merge(df_incomes,   on='income',         how='left')
    )
    # Оставляем только поля для выгрузки
    df_fact = df_fact[[
        'date_of_arrival', 'days_cnt', 'visitors_cnt', 'spent',
        'home_region_id', 'home_city_id', 'home_country_id',
        'age_id', 'gender_id', 'income_id',
        'avg_spent_per_visitor'
    ]]

    # Заполняем отсутствующие id нулями и приводим к int, чтобы в CSV не было "1.0"
    id_cols = ['home_region_id','home_city_id','home_country_id','age_id','gender_id','income_id']
    df_fact[id_cols] = df_fact[id_cols].fillna(0).astype(int)

    engine = db.get_engine()
    conn = engine.raw_connection()
    cur = conn.cursor()

    try:
        conn.autocommit = False

        # staging-таблицы для размерностей
        staging_dims = [
            ('dim_home_region_staging',   ['home_region_id','home_region']),
            ('dim_home_city_staging',     ['home_city_id','home_city']),
            ('dim_home_country_staging',  ['home_country_id','home_country']),
            ('dim_age_staging',           ['age_id','age']),
            ('dim_gender_staging',        ['gender_id','gender']),
            ('dim_income_staging',        ['income_id','income']),
        ]
        for table, cols in staging_dims:
            cur.execute(f"DROP TABLE IF EXISTS {table};")
            cur.execute(f"CREATE TABLE {table} ({cols[0]} INT PRIMARY KEY, {cols[1]} TEXT UNIQUE);")

        # staging-таблица для фактов
        cur.execute("DROP TABLE IF EXISTS tourism_data_staging;")
        cur.execute("""
            CREATE TABLE tourism_data_staging (
                date_of_arrival         TIMESTAMP,
                days_cnt                NUMERIC,
                visitors_cnt            NUMERIC,
                spent                   NUMERIC,
                home_region_id          INT,
                home_city_id            INT,
                home_country_id         INT,
                age_id                  INT,
                gender_id               INT,
                income_id               INT,
                avg_spent_per_visitor   NUMERIC
            );
        """)

        # COPY для размерностей
        for df_dim, (table, cols) in zip(
            [df_regions, df_cities, df_countries, df_ages, df_genders, df_incomes], staging_dims
        ):
            buf = io.StringIO()
            df_dim.to_csv(buf, sep='\t', header=False, index=False, columns=cols)
            buf.seek(0)
            cur.copy_expert(
                f"COPY {table} ({', '.join(cols)}) FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t')",
                buf
            )

        # COPY для фактов
        buf = io.StringIO()
        df_fact.to_csv(buf, sep='\t', header=False, index=False)
        buf.seek(0)
        cur.copy_expert(
            "COPY tourism_data_staging FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t')",
            buf
        )

        conn.commit()
        logger.info("Нормализованные данные загружены в staging через COPY.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Ошибка при bulk-load нормализованных данных: {e}")
        raise
    finally:
        cur.close()
        conn.close()

    # транзакционный swap для размерностей и фактов
    try:
        with engine.begin() as trx:
            # размерности
            for table in [
                'dim_home_region','dim_home_city','dim_home_country',
                'dim_age','dim_gender','dim_income'
            ]:
                trx.execute(db.text(f"ALTER TABLE IF EXISTS {table} RENAME TO {table}_old;"))
                trx.execute(db.text(f"ALTER TABLE {table}_staging RENAME TO {table};"))
                trx.execute(db.text(f"DROP TABLE IF EXISTS {table}_old;"))
            # факт
            trx.execute(db.text("ALTER TABLE IF EXISTS tourism_data RENAME TO tourism_data_old;"))
            trx.execute(db.text("ALTER TABLE tourism_data_staging RENAME TO tourism_data;"))
            trx.execute(db.text("DROP TABLE IF EXISTS tourism_data_old;"))
        logger.info("Стейджинг успешно переименован в основные таблицы.")
    except Exception as e:
        logger.error(f"Ошибка транзакционного swap нормализованных таблиц: {e}")
        raise

    # пересчёт и сохранение JSON индикаторов (используем оригинальный df)
    os.makedirs(output_dir, exist_ok=True)
    required = [
        'VISITORS_CNT','DATE_OF_ARRIVAL','HOME_REGION',
        'HOME_CITY','AGE','GENDER','SPENT','INCOME'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"Невозможно пересчитать индикаторы — нет столбцов: {missing}")
    else:
        try:
            logger.info("Запуск calculate_indicators...")
            calculate_indicators(df, output_path)
            logger.info(f"Индикаторы сохранены: {output_path}")
        except Exception as e:
            logger.error(f"Ошибка при расчёте индикаторов: {e}")

    return df