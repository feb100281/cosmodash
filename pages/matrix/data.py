import pandas as pd
from data import ENGINE, get_connection

def fletch_cats() ->pd.DataFrame:
    q = """
    SELECT
    g.id as gr_id, g.name as gr_name, c.id as cat_id,c.name as cat_name
    from corporate_cattree as c
    join corporate_cattree as g on c.parent_id = g.id
    order by 1    
    """
    return pd.read_sql(q, ENGINE)
    