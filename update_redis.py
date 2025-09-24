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
    'manager_name',
    'agent_name',
    'fullname',
    'brend',
    'manu',
      
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

SALES_DOMAIN['fullname'] = SALES_DOMAIN['fullname'].fillna('Номенклатура не указана')  

# SALES_DOMAIN['subcat_agg'] = SALES_DOMAIN['parent_cat'] + '-' + SALES_DOMAIN['cat'] + SALES_DOMAIN['subcat'] #Для аггрегации по подгатегориям

SALES_DOMAIN['store'] = SALES_DOMAIN['store'].fillna('Магазин не указан')
SALES_DOMAIN['store_gr_name'] = SALES_DOMAIN['store_gr_name'].fillna('Магазин не указан')
SALES_DOMAIN['chanel'] = SALES_DOMAIN['chanel'].fillna('Канал не указан')
# упорядочиваем на всяк
SALES_DOMAIN = SALES_DOMAIN.sort_values(by='date')

# Делаем значения YTD для отмеченных колонок

for analitics in a_cols:
    for item in d_cols:
        col_name = f"{analitics}_{item}_ytd"
        SALES_DOMAIN[col_name] = (
            SALES_DOMAIN
            .sort_values("eom")  # чтобы cum правильно шел
            .groupby(["year", analitics])[item]
            .cumsum()
        )


print(SALES_DOMAIN.memory_usage(deep=True).sum() / 1024**2, "MB использует REDIS для всей херни")
# SALES_DOMAIN.to_csv('data.csv',sep='|',index=False)

print(SALES_DOMAIN.columns.to_list())




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

old_data()

