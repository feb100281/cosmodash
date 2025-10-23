import numpy as np
import pandas as pd
from data import ENGINE

def fletch_cats():
    q = """
    with parent as (
    select
    id as parent_id,
    name as parent
    from corporate_cattree
    where parent_id is Null
    )

    select 
    p.parent_id,
    p.parent,
    cat.id as cat_id,
    cat.name as cat,
    COALESCE(sc.id,0) as subcat_id,
    COALESCE(sc.name,'Нет подкатегории') as subcat
    from corporate_cattree as cat

    left join parent as p on p.parent_id = cat.parent_id
    left join corporate_subcategory as sc on sc.category_id = cat.id
    where cat.parent_id is not Null       
    """
    return pd.read_sql(q,ENGINE)


def get_df(start, end):
    q = f"""
    select
    date,
    sum(s.dt) as dt,
    sum(s.cr) as cr,
    sum(s.dt - s.cr) as amount,
    parent.id as parent_id,
    parent.name as parent,    
    cat.id as cat_id,
    cat.name as cat,
    coalesce(subcat.id,0) as subcat_id,
    coalesce(subcat.name,'Нет подкатегории') as subcat

    from sales_salesdata as s
    left join corporate_items as i on i.id = s.item_id
    left join corporate_cattree as cat on cat.id = i.cat_id
    left join corporate_cattree as parent on parent.id = cat.parent_id
    left join corporate_subcategory as subcat on subcat.id = i.subcat_id
    where date between '{start}' and '{end}'

    GROUP BY date, parent_id, parent, cat_id, cat, subcat_id, subcat;
    """
    return pd.read_sql(q,ENGINE)