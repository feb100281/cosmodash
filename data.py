# В данный модуле находяться классы обработки данных

import pickle
import redis
import pandas as pd
import os
import uuid
from dotenv import load_dotenv
from pathlib import Path
import pymysql
from sqlalchemy import create_engine

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Не забыть поменять на джанго потом
r = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT"),
    db=os.getenv("REDIS_DB"),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=False,
    socket_connect_timeout=30,      # Таймаут подключения
    socket_timeout=30,              # Таймаут операций
    retry_on_timeout=True,          # Повтор при таймауте
    max_connections=50,             # Больше соединений
    health_check_interval=30,       # Проверка здоровья
)

# подключаем к базе данных
config = {
    "user": os.getenv("MYSQL_USER"),  
    "password": os.getenv("MYSQL_PASSWORD"),  
    "host": os.getenv("MYSQL_HOST"),  
    "database": os.getenv("MYSQL_DATABASE"), 
}


# Создание строки подключения для SQLAlchemy
connection_string = f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}/{config['database']}"

ENGINE = create_engine(
    connection_string,
    pool_pre_ping=True,
    pool_recycle=3600,
)

def get_connection():
    return pymysql.connect(**config)


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


def load_sql_df(start_eom,end_eom):
    qs = f"""
    SELECT 
    s.date,
    s.dt,
    s.cr,
    (s.dt - s.cr) AS amount,
    coalesce(st.name,'Магазин не указан') AS store,
    coalesce(sg.name,'Магазин не указан') AS store_gr_name,
    coalesce(st.chanel,'Канал не указан') AS chanel,
    mn.report_name AS manager,
    an.report_name AS agent,
    coalesce(parent.name,'Группа не указана') as parent,
    coalesce(cat.name,'Категория не указана') AS cat,
    coalesce(sc.name,'Подкатегория не указана') AS subcat,
    s.client_order AS client_order,
    (s.quant_dt - s.quant_cr) AS quant,
    s.client_order_number AS client_order_number,
    sg.region AS store_region,
    s.quant_dt AS quant_dt,
    s.quant_cr AS quant_cr,
    i.fullname AS fullname,
    m.name AS manu,
    b.name AS brend
FROM
    sales_salesdata AS s
        LEFT JOIN
    corporate_stores AS st ON st.id = s.store_id
    left join corporate_storegroups as sg on sg.id = st.gr_id
    left join corporate_managers as mn on mn.id = s.manager_id
    left join corporate_agents as an on an.id = s.agent_id
    left join corporate_items as i on i.id = s.item_id
    left join corporate_itembrend as b on b.id = i.brend_id
    left join corporate_itemmanufacturer as m on m.id = i.manufacturer_id
    left join corporate_cattree as cat on cat.id = i.cat_id
    left join corporate_cattree as parent on parent.id = cat.parent_id
    left join corporate_subcategory as sc on sc.id = i.subcat_id
    where date between '{start_eom}' and '{end_eom}'   
    
    """
    df = pd.read_sql(qs,ENGINE)
    
    df['eom'] = pd.to_datetime(df.date) + pd.offsets.MonthEnd(0)
    return df


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

REPORTS = {}

def save_report(report):
    rid = str(uuid.uuid4())
    REPORTS[rid] = report
    return rid

def load_report(rid):
    return REPORTS.get(rid)

def delete_report(rid):
    """Удаляет отчёт из памяти, если он есть"""
    if rid in REPORTS:
        del REPORTS[rid]
    
    