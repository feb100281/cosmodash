import pymysql
from dotenv import load_dotenv
from pathlib import Path
import os
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

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

def get_items(ids_int):
    if not ids_int:
        return []

    placeholders = ','.join(ids_int)
    query = f"""
        select 
        max(s.date) as last_sales_date,
        sum(dt) as dt,
        sum(cr) as cr,
        sum(quant_dt) as quant_dt,
        sum(quant_cr) as quant_cr,
        i.fullname,
        i.init_date,
        GROUP_CONCAT(DISTINCT m.name ORDER BY m.name SEPARATOR ',') AS manu,
        GROUP_CONCAT(DISTINCT b.name ORDER BY b.name SEPARATOR ',') AS brend
        from sales_salesdata as s 
        left join corporate_items as i on i.id = s.item_id
        left join corporate_itemmanufacturer as m on m.id = i.manufacturer_id
        left join corporate_itembrend as b on b.id = i.brend_id

        where item_id in  ({placeholders})
        group by fullname, init_date
    """

    return pd.read_sql(query, engine)

    

