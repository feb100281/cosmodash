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
SALES_DOMAIN['client_order'] = SALES_DOMAIN['client_order'].fillna('без заказа')
SALES_DOMAIN['manager_name'] = SALES_DOMAIN['manager_name'].fillna('нет данных')

cols_list = [
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
]

cummulitive_fields = {
    'orders_per_day':''
}




df = SALES_DOMAIN[['date','client_order','store_gr_name','dt','manager_name']].copy()

# Убедимся, что 'date' в формате datetime
df['date'] = pd.to_datetime(df['date'])

# Считаем уникальные заказы на дату
df['orders_per_day'] = df.groupby('date')['client_order'].transform('nunique')
df['orders_per_day_store'] = df.groupby(['store_gr_name','date'])['client_order'].transform('nunique')
df['orders_per_day_manager'] = df.groupby(['manager_name','date'])['client_order'].transform('nunique')


df['dt_per_day_stores'] = df.groupby(['date','store_gr_name'])['dt'].transform('sum')

df = df.sort_values(by='date')

dff = df[df['date'] == '2025-07-31']
print(dff[['store_gr_name','manager_name','orders_per_day','orders_per_day_manager','dt']])