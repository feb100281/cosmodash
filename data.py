# В данный модуле находяться классы обработки данных

import pickle
import redis
import pandas as pd
import os
import uuid
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Не забыть поменять на джанго потом
r = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT"),
    db=os.getenv("REDIS_DB"),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=False,
)


COLS_DICT = {
    "date": "Дата",  # Дата операции
    "client_order_date": "Дата заказа",
    "client_order_number": "Номер заказа",
    "client_order": "Заказ клиента",
    "operation": "Операция",
    "dt": "Продажи",
    "cr": "Возвраты",
    "amount": "Чистая выручка",
    "quant_dt": "quant_dt",  # Количество ед продажи
    "quant_cr": "quant_cr",  # Количество ед возвраты
    "quant": "Количество",
    "warehouse": "Склад",
    "spec": "Спецификация",
    "fullname": "Номенклатура",
    "imname": "Название в ИМ",
    "article": "Артикл",
    "onec_cat": "onec_cat",  # Это категория 1С
    "onec_subcat": "onec_subcat",  # Это подкатегоря 1С
    "init_date": "Дата первого заказа",
    "im_id": "Код товара в ИМ",
    "cat": "Категория",
    "cat_icon": "cat_icon",
    "parent_cat": "Группа",
    "parent_icon": "parent_icon",
    "manu": "Производитель",
    "manu_origin": "Стана происхождения",
    "brend": "Бренд",
    "brend_origin": "Страна бренда",
    "subcat": "Подкатегоря",
    "store": "Торговая точка",
    "chanel": "Канал продаж",
    "store_gr_name": "Магазин",
    "store_region": "Регион",
    "agent": "_Агент",
    "agent_name": "Агент",  # report_name
    "manager": "_Менеджер",
    "manager_name": "Менеджер",  # report_name
    # "eom": "Отчетный период",
    "month_fmt": "Месяц",
    "quarter_fmt": "Квартал",
    "week_fmt": "_Неделя",  # сокращенная
    "week_fullname": "Неделя",
    "month_id": "month_id",
    "year":"год",
    "store_gr_name_dt_ytd":"Продажи YTD",
    "store_gr_name_cr_ytd": "Возвраты YTD",
    "store_gr_name_quant_dt_ytd":"Количество проданных товароы YTD",
    "store_gr_name_quant_cr_ytd":"Количество возвращенных товароы YTD", 
    "store_gr_name_amount_ytd":'Чистая выручка YTD',
    "store_gr_name_quant_ytd":"Количество YTD",
    #Далее в том же духе
    # "manager_name_dt_ytd",
    # "manager_name_cr_ytd",
    # "manager_name_quant_dt_ytd",
    # "manager_name_quant_cr_ytd",
    # "manager_name_amount_ytd",
    # "manager_name_quant_ytd",
    # "agent_name_dt_ytd",
    # "agent_name_cr_ytd",
    # "agent_name_quant_dt_ytd",
    # "agent_name_quant_cr_ytd",
    # "agent_name_amount_ytd",
    # "agent_name_quant_ytd",
    # "fullname_dt_ytd",
    # "fullname_cr_ytd",
    # "fullname_quant_dt_ytd",
    # "fullname_quant_cr_ytd",
    # "fullname_amount_ytd",
    # "fullname_quant_ytd",
    # "brend_dt_ytd",
    # "brend_cr_ytd",
    # "brend_quant_dt_ytd",
    # "brend_quant_cr_ytd",
    # "brend_amount_ytd",
    # "brend_quant_ytd",
    # "manu_dt_ytd",
    # "manu_cr_ytd",
    # "manu_quant_dt_ytd",
    # "manu_quant_cr_ytd",
    # "manu_amount_ytd",
    # "manu_quant_ytd",
}

COLORS = [
    "indigo.6",
    "teal.6",
    "gray.6",
    "blue.6",
    "cyan.6",
    "pink.6",
    "lime.6",
    "orange.6",
    "violet.6",
    "grape.6",
    "red.6",
    "green.6",
    "yellow.6",
    "sky.6",
    "purple.6",
    "brand.6",
    "dark.6",
    "brown.6",
    "azure.6",
    "magenta.6",
]

cols = [
    "date",
    "client_order_date",
    "client_order_number",
    "client_order",
    "operation",
    "dt",
    "cr",
    "amount",
    "quant_dt",
    "quant_cr",
    "quant",
    "warehouse",
    "spec",
    "item_id",
    "fullname",
    "imname",
    "article",
    "onec_cat",
    "onec_subcat",
    "init_date",
    "im_id",
    "cat_id",
    "cat",
    "parent_cat_id",
    "parent_cat",
    "manu",
    "manu_origin",
    "brend",
    "brend_origin",
    "subcat_id",
    "subcat",
    "store",
    "chanel",
    "store_gr_name",
    "store_region",
    "agent",
    "agent_name",
    "manager",
    "manager_name",
    "eom",
    "month_fmt",
    "quarter_fmt",
    "week_fmt",
    "week_fullname",
    "month_id",
    "year",
    "store_gr_name_dt_ytd",
    "store_gr_name_cr_ytd",
    "store_gr_name_quant_dt_ytd",
    "store_gr_name_quant_cr_ytd",
    "store_gr_name_amount_ytd",
    "store_gr_name_quant_ytd",
    "manager_name_dt_ytd",
    "manager_name_cr_ytd",
    "manager_name_quant_dt_ytd",
    "manager_name_quant_cr_ytd",
    "manager_name_amount_ytd",
    "manager_name_quant_ytd",
    "agent_name_dt_ytd",
    "agent_name_cr_ytd",
    "agent_name_quant_dt_ytd",
    "agent_name_quant_cr_ytd",
    "agent_name_amount_ytd",
    "agent_name_quant_ytd",
    "fullname_dt_ytd",
    "fullname_cr_ytd",
    "fullname_quant_dt_ytd",
    "fullname_quant_cr_ytd",
    "fullname_amount_ytd",
    "fullname_quant_ytd",
    "brend_dt_ytd",
    "brend_cr_ytd",
    "brend_quant_dt_ytd",
    "brend_quant_cr_ytd",
    "brend_amount_ytd",
    "brend_quant_ytd",
    "manu_dt_ytd",
    "manu_cr_ytd",
    "manu_quant_dt_ytd",
    "manu_quant_cr_ytd",
    "manu_amount_ytd",
    "manu_quant_ytd",
]


def to_str_date(d):
    if isinstance(d, str):
        return d
    # datetime.date или pd.Timestamp
    return d.strftime("%Y-%m-%d")


def load_column_range(column_name, start_eom, end_eom):
    chunks = pickle.loads(r.get(f"mydf:{column_name}:__chunks__"))

    # приводим start/end к строкам
    start_str = to_str_date(start_eom)
    end_str = to_str_date(end_eom)

    # приводим все chunks к строкам
    chunks_str = [to_str_date(c) for c in chunks]

    # фильтруем
    needed_chunks = [c for c in chunks_str if start_str <= c <= end_str]

    series_list = []
    for chunk in needed_chunks:
        data = r.get(f"mydf:{column_name}:{chunk}")
        if data:
            series_list.append(pickle.loads(data))

    if series_list:
        return pd.concat(series_list, ignore_index=True)
    else:
        return pd.Series(dtype=object)


def load_columns_df(columns, start_eom, end_eom):
    data = {}

    for col in columns:
        data[col] = load_column_range(col, start_eom, end_eom)
    return pd.DataFrame(data)


def save_df_to_redis(df, expire_seconds=600):
    df_id = str(uuid.uuid4())
    r.set(df_id, pickle.dumps(df), ex=expire_seconds)
    return df_id


def load_df_from_redis(df_id) -> pd.DataFrame:
    data = r.get(df_id)
    return pickle.loads(data) if data else None


def delete_df_from_redis(df_id):
    r.delete(df_id)


def load_column_dates(column_name, dates):
    dates_str = pd.to_datetime(dates, errors="coerce").strftime("%Y-%m-%d").tolist()

    series_list = []
    for d in dates_str:
        key = f"mydf:{column_name}:{d}"

        data = r.get(key)

        if data:
            series_list.append(pickle.loads(data))
    if series_list:
        return pd.concat(series_list, ignore_index=True)
    else:
        return pd.Series(dtype=object)


def load_columns_dates(columns, dates):
    """
    Загружает DataFrame по списку дат и списку колонок.
    """
    data = {}
    for col in columns:
        data[col] = load_column_dates(col, dates)
    return pd.DataFrame(data)


# cols = ['manager','date','amount','store','eom','chanel','cat']


# import time

# df_jan_feb = load_columns_df(r, cols, "2024-01-31", "2025-08-31")
# # print(df_jan_feb)
# print(df_jan_feb.memory_usage(deep=True).sum() / 1024**2, "MB")

# start = time.time()


# #a = save_df_to_redis(df_jan_feb)
# #print(a)
# b = load_df_from_redis('e3d09ccf-6538-4d2a-b42f-f93f01b3e6a0')


# end = time.time()

# print(f"Время выполнения: {end - start:.3f} секунд")


# g = r.get('all')
# df = pickle.loads(g)
# print(df)


# df = df_jan_feb.pivot_table(
#     index='eom',
#     columns=['chanel','store'],
#     values='dt',
#     aggfunc='sum'

# ).reset_index().sort_values(by='eom')
# # print(df)


# class RedisLoader:
#     @staticmethod
#     def load_df(key, default=None):
#         pickled = r.get(key)
#         if pickled is None:
#             return default if default is not None else pd.DataFrame()
#         return pickle.loads(pickled)


# class DataMart:
#     pass


# class SalesDynamix:
#     def __init__(
#         self,
#         store_filter=None,
#         chanel_filter=None,
#         store_gr_name_filter=None,
#         month_id_filter=None,
#         period_filter="monthly",
#         store_breadown=False,
#         chanel_breakdon=False,
#     ):

#         self.KEY = (
#             "sd_monthly_data"
#             if period_filter != "weekly"
#             else "sd_weekly_data"
#         )
#         self.store_filter = store_filter
#         self.chanel_filter = chanel_filter
#         self.store_gr_name_filter = store_gr_name_filter
#         self.month_id_filter = month_id_filter
#         self.period_filter = period_filter
#         self.store_breakdown = store_breadown
#         self.chanel_breakdon = chanel_breakdon

#     def data(self):
#         df = RedisLoader.load_df(self.KEY)
#         if df.empty:
#             return None
#         total_chart = df.pivot_table(
#             index=['eom','month_fmt'],
#             values=['dt','cr','amount','quant_dt','quant_cr','quant','client_order'],
#             aggfunc='sum'
#         ).fillna(0).reset_index().sort_values(by='eom')
#         total_chart['average_check'] = total_chart['amount'] / total_chart['client_order']


#         if self.period_filter == "monthly":
#             if self.month_id_filter:
#                 if isinstance(self.month_id_filter, list):
#                     start, finish = self.month_id_filter
#                     df = df[(df["month_id"] >= start) & (df["month_id"] <= finish)]
#                 elif isinstance(self.month_id_filter, int):
#                     df = df[df["month_id"] == self.month_id_filter]
#             else:
#                 finish = df["month_id"].max()
#                 start = finish - 12
#                 df = df[(df["month_id"] >= start) & (df["month_id"] <= finish)]

#         if self.store_gr_name_filter:
#             if isinstance(self.store_gr_name_filter, list):
#                 df = df[df["store_group"].isin(self.store_gr_name_filter)]
#             elif isinstance(self.store_gr_name_filter, str):
#                 df = df[df["store_group"] == self.store_gr_name_filter]
#         else:
#             if not self.store_breakdown:
#                 df["store_group"] = "Все данные"

#         if self.chanel_filter:
#             if isinstance(self.chanel_filter, list):
#                 df = df[df["store_group"].isin(self.store_gr_name_filter)]
#             elif isinstance(self.chanel_filter, str):
#                 df = df[df["store_group"] == self.chanel_filter]
#         else:
#             if not self.chanel_breakdon:
#                 df["store_group"] = "Все данные"

#         return df, total_chart

# class SegmentAnalisys:
#     def __init__(self):
#         self.KEY = 'ia_data'

#     def data(self):
#         def df_to_nested_dict(df: pd.DataFrame, cols: list[str]) -> dict:
#             """
#             Преобразует DataFrame в вложенный словарь по указанным колонкам.
#             """
#             result = {}
#             for row in df[cols].drop_duplicates().itertuples(index=False):
#                 d = result
#                 for col in row[:-1]:  # все уровни кроме последнего
#                     d = d.setdefault(col, {})
#                 d[row[-1]] = {}  # fullname — лист
#             return result


#         def dict_to_dmc_tree(d: dict) -> list:
#             """
#             Преобразует вложенный словарь в список узлов для dmc.Tree.
#             """
#             nodes = []
#             for key, val in d.items():
#                 node = {"value": key, "label": key}
#                 if val:  # есть дети
#                     node["children"] = dict_to_dmc_tree(val)
#                 nodes.append(node)
#             return nodes


#         def df_to_dmc_tree(df: pd.DataFrame, cols: list[str]) -> list:
#             nested = df_to_nested_dict(df, cols)
#             return dict_to_dmc_tree(nested)

#         df = RedisLoader.load_df(self.KEY)
#         cats_list = ['parent_cat','cat','subcat','fullname']
#         dff = df[cats_list].drop_duplicates()
#         dff = dff.sort_values(by='parent_cat',ascending=False)
#         valid_data = df_to_dmc_tree(dff, cats_list)

#         return df,valid_data


# df = RedisLoader.load_df('segment_analisys_monthly')
# print(df['parent_cat'].unique())


# a = SegmentAnalisys()
# print(a.data()[0])
