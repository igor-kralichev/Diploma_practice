# app/services/calculate_indicators.py

import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def format_age_confidence_interval(lower: str, upper: str) -> str:
    """
    Форматирует возрастной интервал:
    - Если верхняя и нижняя граница одинаковые, возвращает одно значение.
    - Если разные — возвращает диапазон.
    """
    if lower == upper:
        return f"{lower}"
    else:
        return f"{lower} - {upper}"

def map_age_to_category(age_value: float) -> str:
    for (low, high), label in {
        (0, 18): 'до 18 лет',
        (18, 25): 'от 18 до 24 лет',
        (25, 35): 'от 25 до 34 лет',
        (35, 45): 'от 35 до 44 лет',
        (45, 55): 'от 45 до 54 лет',
        (55, 65): 'от 55 до 64 лет',
        (65, 200): 'свыше 64 лет'
    }.items():
        if low < age_value <= high:
            return label
    return 'неизвестно'

# вспомогательная функция для profit_index
def _best_by_profit_index(grp: pd.DataFrame):
    metrics = grp.copy()
    metrics['avg_spent'] = metrics['total_spent'] / metrics['total_visitors']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(metrics[['total_spent', 'total_visitors', 'avg_spent']])
    metrics[['s_spent', 's_vis', 's_avg']] = scaled
    metrics['profit_index'] = metrics[['s_spent', 's_vis', 's_avg']].mean(axis=1)
    best = metrics['profit_index'].idxmax()
    row = metrics.loc[best, ['total_spent', 'total_visitors', 'avg_spent', 'profit_index']]
    return {
        'category': best,
        'total_spent': float(row['total_spent']),
        'total_visitors': int(row['total_visitors']),
        'avg_spent': float(row['avg_spent']),
        'profit_index': float(row['profit_index'])
    }

def promethee_best_choice(dataframe, criteria_columns, maximize_columns, weights):
    """
    PROMETHEE для выбора наилучшей альтернативы с объяснением.
    """
    df_promethee = dataframe.copy()
    preference_matrix = pd.DataFrame(0, index=df_promethee.index, columns=df_promethee.index, dtype=float)

    # Преобразуем критерии (инвертируем, если это 'min')
    for i, column in enumerate(criteria_columns):
        if column not in maximize_columns:
            df_promethee[column] = -df_promethee[column]

    # Заполняем матрицу предпочтений
    for i in df_promethee.index:
        for j in df_promethee.index:
            if i != j:
                for k, column in enumerate(criteria_columns):
                    if df_promethee.loc[i, column] > df_promethee.loc[j, column]:
                        preference_matrix.loc[i, j] += weights[k]

    # Расчёт Net Flow
    df_promethee['PositiveFlow'] = preference_matrix.sum(axis=1) / (len(df_promethee) - 1)
    df_promethee['NegativeFlow'] = preference_matrix.sum(axis=0) / (len(df_promethee) - 1)
    df_promethee['NetFlow'] = df_promethee['PositiveFlow'] - df_promethee['NegativeFlow']

    # Сортировка по Net Flow
    df_promethee = df_promethee.sort_values(by='NetFlow', ascending=False)
    """
    # Формирование объяснения
    explanation = {}
    for index, row in df_promethee.iterrows():
        explanation[index] = {
            'NetFlow': row['NetFlow'],
            'Criteria': {}
        }
        for col in criteria_columns:
            explanation[index]['Criteria'][col] = float(df_promethee.loc[index, col])
    """
    # Возвращаем лучший выбор и полное объяснение
    best_choice = df_promethee.index[0] if not df_promethee.empty else "Не определено"
    """
    # Полное ранжирование для анализа
    ranking = df_promethee[['NetFlow']].to_dict(orient='index')
    """
    return best_choice



def calculate_indicators(df: pd.DataFrame, output_path: str):
    # 1. Сколько туристов за весь период
    total_tourists = int(df['VISITORS_CNT'].sum())

    # 2. Сколько туристов по месяцам
    df['DATE_OF_ARRIVAL'] = pd.to_datetime(df['DATE_OF_ARRIVAL'], errors='coerce')
    monthly = (
         df
         .set_index('DATE_OF_ARRIVAL')['VISITORS_CNT']
         .resample('ME')  # ежемесячно
         .sum()
    )
    tourists_per_month = {
        dt.strftime('%Y-%m'): int(cnt)
        for dt, cnt in monthly.items()
        if pd.notna(dt)
    }

    # 3. Территориальное распределение
    territorial_by_region = {
        region: int(cnt)
        for region, cnt in df.groupby('HOME_REGION')['VISITORS_CNT'].sum().items() if region != ''
    }
    territorial_by_district = {
        city: int(cnt)
        for city, cnt in df.groupby('HOME_CITY')['VISITORS_CNT'].sum().items() if city != ''
    }
    territorial_by_country = {
        country: int(cnt)
        for country, cnt in df.groupby('HOME_COUNTRY')['VISITORS_CNT'].sum().items() if country != 'Россия'
    }

    # 4. Демографическое распределение (абсолютное и относительное)
    age_counts = df.groupby('AGE')['VISITORS_CNT'].sum()
    gender_counts = df.groupby('GENDER')['VISITORS_CNT'].sum()

    demographic_distribution = {
        'by_age': {
            'absolute': {age: int(cnt) for age, cnt in age_counts.items() if age != ''},
            'relative': {age: round(cnt / total_tourists, 3) for age, cnt in age_counts.items() if age != ''}
        },
        'by_gender': {
            'absolute': {gender if gender != '' else 'Не определено': int(cnt) for gender, cnt in gender_counts.items()},
            'relative': {gender if gender != '' else 'Не определено': round(cnt / total_tourists, 3) for gender, cnt in gender_counts.items()}
        }
    }

    # 5. Выгодные категории по всем параметрам
    profitable = {}
    for field, key in [
        ('HOME_COUNTRY', 'country'),
        ('HOME_REGION', 'region'),
        ('HOME_CITY', 'city'),
        ('AGE', 'age'),
        ('INCOME', 'income')]:
        df_f = df[df[field] != '']
        agg = df_f.groupby(field).agg(
            total_spent=('SPENT', 'sum'),
            total_visitors=('VISITORS_CNT', 'sum')
        )
        profitable[key] = _best_by_profit_index(agg)

    # 6. Профиль среднестатистического туриста
    # отображаем категориальные признаки как мода, числовые через Promethee
    df2 = df.copy()

    # для удобства расчётов – отображение категорий возраста в числа
    age_map = {
        'до 18 лет': 15,
        'от 18 до 24 лет': 21,
        'от 25 до 34 лет': 30,
        'от 35 до 44 лет': 40,
        'от 45 до 54 лет': 50,
        'от 55 до 64 лет': 60,
        'свыше 64 лет': 70
    }
    df2['AGE_NUM'] = df2['AGE'].map(age_map)

    average_tourist_profile = {}

    # 6.1 Promethee по возрасту (в кодах)
    age_df = df2.groupby('AGE_NUM')['VISITORS_CNT'] \
        .sum() \
        .rename('visitors') \
        .to_frame()
    best_age_num = promethee_best_choice(
        age_df,
        criteria_columns=['visitors'],
        maximize_columns=['visitors'],
        weights=[1]
    )
    # обратное отображение числового кода в текстовую категорию
    average_tourist_profile['age'] = map_age_to_category(best_age_num)

    # 6.2 Promethee по полу
    gender_df = df2.groupby('GENDER')['VISITORS_CNT'] \
        .sum() \
        .rename('visitors') \
        .to_frame()
    average_tourist_profile['gender'] = promethee_best_choice(
        gender_df,
        ['visitors'],
        ['visitors'],
        [1]
    )

    # 6.3 Promethee по стране
    country_df = df2.groupby('HOME_COUNTRY')['VISITORS_CNT'] \
        .sum() \
        .rename('visitors') \
        .to_frame()
    average_tourist_profile['country'] = promethee_best_choice(
        country_df,
        ['visitors'],
        ['visitors'],
        [1]
    )

    # 6.4 Promethee по региону
    region_df = df2.groupby('HOME_REGION')['VISITORS_CNT'] \
        .sum() \
        .rename('visitors') \
        .to_frame()
    average_tourist_profile['region'] = promethee_best_choice(
        region_df,
        ['visitors'],
        ['visitors'],
        [1]
    )

    # 6.5 Если регион — Нижегородская область, Promethee по району
    if average_tourist_profile['region'] == 'Нижегородская область':
        city_df = (
            df2[df2['HOME_REGION'] == 'Нижегородская область']
            .groupby('HOME_CITY')['VISITORS_CNT']
            .sum()
            .rename('visitors')
            .to_frame()
        )
        average_tourist_profile['district'] = promethee_best_choice(
            city_df,
            ['visitors'],
            ['visitors'],
            [1]
        )
    else:
        average_tourist_profile['district'] = None

    # 6.6 Promethee по доходу
    income_df = df2.groupby('INCOME')['VISITORS_CNT'] \
        .sum() \
        .rename('visitors') \
        .to_frame()
    average_tourist_profile['income'] = promethee_best_choice(
        income_df,
        ['visitors'],
        ['visitors'],
        [1]
    )
    # Финальный сборник индикаторов
    indicators = {
        '1_total_tourists': total_tourists,
        '2_tourists_per_month': tourists_per_month,
        '3_territorial_distribution': {
            'by_region': territorial_by_region,
            'by_district': territorial_by_district,
            'by_country': territorial_by_country
        },
        '4_demographic_distribution': demographic_distribution,
        '5_profitable_categories': profitable,
        '6_average_tourist_profile': average_tourist_profile
    }

    # Сохранение в JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(indicators, f, ensure_ascii=False, indent=4)

    print(f"Ключевые показатели сохранены в {output_path}")