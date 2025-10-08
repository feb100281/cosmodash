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

def fletch_dataset(start, end):
    
    q = f"""
    SELECT 
            s.date,
            s.dt,
            s.cr,
            s.quant_dt,
            s.quant_cr,
            s.item_id,
            i.fullname AS fullname,
            i.article,
            i.init_date,
            i.manufacturer_id as manu_id,
            m.name AS manu,
            i.brend_id,
            b.name as brend,
            cat.id AS cat_id,
            cat.name AS cat,
            parent.id AS parent_cat_id,
            parent.name AS parent_cat,
            subcat.id AS subcat_id,
            subcat.name AS subcat
        FROM
            sales_salesdata AS s
                LEFT JOIN
            corporate_items AS i ON i.id = s.item_id
                LEFT JOIN
            corporate_itemmanufacturer AS m ON m.id = i.manufacturer_id
                LEFT JOIN
            corporate_itembrend AS b ON b.id = i.brend_id
                LEFT JOIN
            corporate_cattree AS cat ON cat.id = i.cat_id
                LEFT JOIN
            corporate_cattree AS parent ON parent.id = cat.parent_id
                LEFT JOIN
            corporate_subcategory AS subcat ON subcat.id = i.subcat_id
        WHERE
            date BETWEEN '{start}' AND '{end}'
    """
    
    return pd.read_sql(q, engine)

def fleching_cats():
    q = f"""
        SELECT 
            parent.id AS parent_cat_id,
            parent.name AS parent_cat,
            cat.id AS cat_id,
            cat.name AS cat,
            subcat.id AS subcat_id,
            subcat.name AS subcat
        FROM
            corporate_cattree AS cat
                JOIN
            corporate_cattree AS parent ON parent.id = cat.parent_id
                LEFT JOIN
            corporate_subcategory AS subcat ON subcat.category_id = cat.id
    """
    return pd.read_sql(q, engine)

def assign_cat(ids,cat_id,subcat_id):
    placeholders = ','.join(ids)
    cat_id = int(float(cat_id))
    subcat_id = int(float(subcat_id)) if subcat_id else 'NULL'
    q = f"""
    update corporate_items
    set cat_id = {cat_id}, subcat_id = {subcat_id}
    where id in ({placeholders})
    
    """
    cur.execute(q)
    conn.commit()

