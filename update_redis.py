import pickle
import locale
import numpy as np
import pandas as pd
import locale

import os
import redis
locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")
from data import r

from sqlalchemy import create_engine
import pymysql


from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


config = {
    "user": os.getenv("MYSQL_USER"),  
    "password": os.getenv("MYSQL_PASSWORD"),  
    "host": os.getenv("MYSQL_HOST"),  
    "database": os.getenv("MYSQL_DATABASE"), 
}
print(config) 
conn = pymysql.connect(**config)
cur = conn.cursor()


# Создание строки подключения для SQLAlchemy
connection_string = f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}/{config['database']}"

# Создание движка SQLAlchemy
engine = create_engine(connection_string)

SALES_DOMAIN = pd.read_sql("SELECT * FROM sales_domain order by eom", engine)

r = redis.Redis(
    host=os.getenv("REDIS_HOST"),  
    port=os.getenv("REDIS_PORT"),
    db=os.getenv("REDIS_DB"),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=False
)

r.flushall()

print(SALES_DOMAIN.memory_usage(deep=True).sum() / 1024**2, "MB")

# r.set('all', pickle.dumps(SALES_DOMAIN))

# Фиксим даты
SALES_DOMAIN['client_order_date'] = SALES_DOMAIN['client_order_date'].fillna(SALES_DOMAIN['date'])

SALES_DOMAIN['date'] = pd.to_datetime(SALES_DOMAIN['date'],errors='coerce').dt.normalize()

SALES_DOMAIN['client_order_date'] = pd.to_datetime(SALES_DOMAIN['client_order_date'],errors='coerce').dt.normalize()

SALES_DOMAIN['year'] = SALES_DOMAIN['date'].dt.year

# Убираем NaN в строковых колонках

cols_for_non_data = [
    'client_order_number','client_order','warehouse','spec','imname','article',"onec_cat", "onec_subcat",
    'manu','manu_origin','brend','brend_origin','agent','agent_name','manager','manager_name','store_region'    
]

# колонки для аналитики
a_cols = [
    'store_gr_name',
    'chanel',
    'store_region',
    'manager_name',
    'agent_name',
    'fullname',
    'brend',
    'brend_origin',
    'manu',
    'manu_origin',
    'parent_cat',
    'cat',
    'subcat'      
]

d_cols = [
    'dt',
    'cr',
    'quant_dt',
    'quant_cr',
    'amount',
    'quant',
]

# for col in cols_for_non_data:
#     SALES_DOMAIN[col] = SALES_DOMAIN[col].fillna('Нет данных')



SALES_DOMAIN['parent_cat'] = SALES_DOMAIN['parent_cat'].fillna('Группа не указана')
SALES_DOMAIN['cat'] = SALES_DOMAIN['cat'].fillna('Категория не указана')
SALES_DOMAIN['subcat'] = SALES_DOMAIN['subcat'].fillna('Подкатегория не указана')

SALES_DOMAIN['store'] = SALES_DOMAIN['store'].fillna('Магазин не указан')
SALES_DOMAIN['store_gr_name'] = SALES_DOMAIN['store_gr_name'].fillna('Магазин не указан')
SALES_DOMAIN['chanel'] = SALES_DOMAIN['chanel'].fillna('Канал не указан')
# упорядочиваем на всяк
SALES_DOMAIN = SALES_DOMAIN.sort_values(by='date')

# # Делаем суммы заказов 
# SALES_DOMAIN['orders_per_day'] = SALES_DOMAIN.groupby('date')['client_order'].transform('nunique')
# SALES_DOMAIN['orders_per_day_store'] = SALES_DOMAIN.groupby(['store_gr_name','date'])['client_order'].transform('nunique')
# SALES_DOMAIN['orders_per_day_manager'] = SALES_DOMAIN.groupby(['manager_name','date'])['client_order'].transform('nunique')

# d_cols = [
#     'dt',
#     'cr',
#     'quant_dt',
#     'quant_cr',
#     'amount',
#     'quant',
#     'orders_per_day'
# ]



# # Делаем дневные значения для колонок с датами
# for col in d_cols:
#     col_name = f"{col}"+'_per_day'
#     SALES_DOMAIN[col_name] = SALES_DOMAIN.groupby('date')[col].transform('sum')
    
#     daily = SALES_DOMAIN[["date", col_name]].drop_duplicates()

#     # MTD
#     daily["mtd"] = daily.groupby(daily["date"].dt.to_period("M"))[col_name].cumsum()
#     # YTD
#     daily["ytd"] = daily.groupby(daily["date"].dt.to_period("Y"))[col_name].cumsum()

#     # обратно мержим по дате
#     SALES_DOMAIN = SALES_DOMAIN.merge(
#         daily[["date", "mtd", "ytd"]],
#         on="date",
#         how="left",
#         suffixes=("", f"_{col}")
#     )

#     SALES_DOMAIN.rename(
#         columns={
#             f"mtd": f"{col}_mtd",
#             f"ytd": f"{col}_ytd"
#         },
#         inplace=True
#     )


#Делаем матрицу дата магазин



    

    # for item in a_cols:
    #     sub_col_name = f"{col}_per_day_{item}"
    #     SALES_DOMAIN[sub_col_name] = SALES_DOMAIN.groupby([item, "date"])[col].transform("sum")

    #     daily = SALES_DOMAIN[[item, "date", sub_col_name]].drop_duplicates()

    #     # MTD
    #     daily["mtd"] = daily.groupby([item, daily["date"].dt.to_period("M")])[sub_col_name].cumsum()
    #     # YTD
    #     daily["ytd"] = daily.groupby([item, daily["date"].dt.to_period("Y")])[sub_col_name].cumsum()

    #     SALES_DOMAIN = SALES_DOMAIN.merge(
    #         daily[[item, "date", "mtd", "ytd"]],
    #         on=[item, "date"],
    #         how="left",
    #         suffixes=("", f"_{col}_{item}")
    #     )

    #     SALES_DOMAIN.rename(
    #         columns={
    #             "mtd": f"{col}_{item}_mtd",
    #             "ytd": f"{col}_{item}_ytd"
    #         },
    #         inplace=True
    #     )
        



# print(SALES_DOMAIN.columns.to_list())
# print(SALES_DOMAIN.memory_usage(deep=True).sum() / 1024**2, "MB")


# df = SALES_DOMAIN[SALES_DOMAIN['date']==pd.to_datetime('2025-07-31').normalize()].copy()
# df = df[['date','store_gr_name','amount_mtd','amount_ytd']]
# pd.options.display.float_format = '{:,.2f}'.format
# print(df)

# #Теперь коммулитивные суммы






for eom in SALES_DOMAIN["eom"].unique():
    chunk_df = SALES_DOMAIN[SALES_DOMAIN["eom"] == eom]
    for col in chunk_df.columns:
        key = f"mydf:{col}:{eom}"
        # сохраняем как Series
        r.set(key, pickle.dumps(chunk_df[col]))
        print(key + ' saved')
        
        # обновляем мета с доступными чанками
        meta_key = f"mydf:{col}:__chunks__"
        chunks_list = pickle.loads(r.get(meta_key)) if r.exists(meta_key) else []
        if eom not in chunks_list:
            chunks_list.append(eom)
            r.set(meta_key, pickle.dumps(chunks_list))





KEYS_LIST = [b"sales_dynamix_monthly", b"first_date", b"last_date", b"sales_data"]

COLS_DICT = {
    "date": "Дата",
    "client_order_date": "Дата заказа",
    "client_order_number": "Номер заказа",
    "client_order": "Заказ клиента",
    "operation": "Операция",
    "dt": "dt",
    "cr": "cr",
    "amount": "Сумма",
    "quant_dt": "quant_dt",
    "quant_cr": "quant_cr",
    "quant": "Количество",
    "warehouse": "Склад",
    "spec": "Спецификация",
    "fullname": "Номенклатура",
    "imname": "Название в ИМ",
    "article": "Артикл",
    "onec_cat": "onec_cat",
    "onec_subcat": "onec_subcat",
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
    "eom": "Отчетный период",
    "month_fmt": "Месяц",
    "quarter_fmt": "Квартал",
    "week_fmt": "_Неделя",  # сокращенная
    "week_fullname": "Неделя",
    "month_id": "month_id",
    "client_order_num": "Количество заказов",
    "last_trade":'Дата последней продажи'
}

COLS_LIST = [
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
    'item_id',
    "fullname",
    "imname",
    "article",
    "onec_cat",
    "onec_subcat",
    "init_date",
    "im_id",
    "cat_id"
    "cat",
    "cat_icon",
    "parent_cat_id"
    "parent_cat",
    "parent_icon",
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
]


# #Витрина данных для отчета по динамики магазинов
# def stores_dynamix_monthly(key):
#     data_key = key[0]
#     filter_key = key[1]
#     print("sales_dynamix_monthly start")
#     cols_to_keep = [
#         "client_order",
#         "dt",
#         "cr",
#         "amount",
#         "quant_dt",
#         "quant_cr",
#         "quant",
#         "chanel",
#         "store_gr_name",
#         "store",
#         "store_region",
#         "month_id",
#         "eom",
#         "quarter_fmt",
#     ]   
#     df = SALES_DOMAIN[cols_to_keep].copy()
#     df['eom'] = pd.to_datetime(df['eom'])
#     df["month_fmt"] = df["eom"].dt.strftime("%b %y").str.capitalize()
#     index_fields = [        
#         "quarter_fmt",
#         "eom",
#         "month_id",
#         "month_fmt",
#         'store_region',
#         'chanel',
#         'store_gr_name',
#         'store',             
#     ]
#     values_dict = {
#         'dt':'sum',
#         'cr':'sum',
#         'amount':'sum',
#         'quant_dt':'sum',
#         'quant_cr':'sum',
#         'client_order':'nunique'
#     }
#     vals_cols = list(values_dict.keys())
    
#     df.store_region = np.where(df.store_region == None,'Нет данных', df.store_region)
#     df.chanel = np.where(df.chanel == None,'Нет данных', df.chanel)
#     df.store_gr_name = np.where(df.store_gr_name == None,'Нет данных', df.store_gr_name)
#     df.store = np.where(df.store == None,'Нет данных', df.store)   
    
#     grouped = df.pivot_table(
#         index = index_fields,
#         values = vals_cols,
#         aggfunc=values_dict
#     ).reset_index().fillna(0).sort_values(by='eom')
#     print("sales_dynamix_monthly saving")
#     pickled = pickle.dumps(grouped)
#     r.set(data_key, pickled)
#     print("sales_dynamix_monthly done")
    
#     filter_df = df[['store_gr_name','store','chanel','store_region']].drop_duplicates().dropna()
#     print(filter_df.head(50))
#     pickled = pickle.dumps(filter_df)
#     r.set(filter_key, pickled)
    

# # Данные для range_slider
# def range_slider_data(key="range_slider_data"):
#     pass


# def items_analisys(key="segment_analisys_monthly"):
#     cols_to_keep = [        
#         "parent_cat",
#         "cat",
#         "subcat",
#         "fullname",
#         "init_date",
#         "amount",
#         "quant",
#         "date",
#     ]
#     df = SALES_DOMAIN[cols_to_keep].copy()
    
#     df['price'] = df['amount'] / df['quant']
#     df['parent_cat'] = df['parent_cat'].fillna('Нет группы')    
#     df['cat'] = df['cat'].fillna(df['parent_cat'] + 'Категория не задана')
#     df['subcat'] = df['subcat'].fillna(df['cat'] + 'Нет подкатегории')
#     df['fullname'] = df['fullname'].fillna(df['subcat'] + 'Нет номенклатуры')
#     grouped = df.pivot_table(
#         index=['parent_cat','cat','subcat','fullname'],
#         values=['init_date','date','amount','quant','price'],
#         aggfunc={
#             'init_date': 'first',
#             'date': list,
#             'amount': list,
#             'quant': list,
#             'price': list
#         }
#     ).reset_index()
#     print(grouped['parent_cat'].unique())
#     #print(grouped)
#     print("segment_analisys_monthly saving")
#     pickled = pickle.dumps(grouped)
#     r.set(key, pickled)
#     print("segment_analisys_monthly done")


# НЕ ТРОГАТЬ
def old_data(key="sales_data"):
    df = pd.read_sql("SELECT * FROM sales_summary", engine)
    df["date"] = pd.to_datetime(df["date"])
    pickled = pickle.dumps(df)
    r.set(key, pickled)
    last_date = df["date"].max()
    first_date = df["date"].min()

    r.set("last_date", last_date.strftime("%Y-%m-%d"))
    r.set("first_date", first_date.strftime("%Y-%m-%d"))
    print("old data saved")

# RegisterUpdetesFunction = {
#     'sales_data':old_data,
#     ('sd_monthly_data','store_filters'):stores_dynamix_monthly,
#     'ia_data':items_analisys,
#     }


#

# for k,v in RegisterUpdetesFunction.items():
#     v(k)



old_data()
# sales_dynamix_monthly()
# segment_analisys_monthly()

