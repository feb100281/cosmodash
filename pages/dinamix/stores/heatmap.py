# queries/heatmap.py
import pandas as pd
from data import ENGINE

def fetch_daily_amount(start_sql: str, end_sql: str, stores: list[str] | None) -> pd.DataFrame:
    """Возвращает дневные суммы amount по каждому дню.
       Колонки: date (datetime64[ns]), amount (float)."""
    where_stores = ""
    if stores:
        in_list = ",".join(f"'{s}'" for s in stores)
        where_stores = f" AND sg.name IN ({in_list})"

    q = f"""
        SELECT
            DATE(s.date) AS date,
            SUM(s.dt - s.cr) AS amount
        FROM sales_salesdata s
        LEFT JOIN corporate_stores st ON st.id = s.store_id
        LEFT JOIN corporate_storegroups sg ON sg.id = st.gr_id
        WHERE s.date BETWEEN '{start_sql}' AND '{end_sql}'
        {where_stores}
        GROUP BY DATE(s.date)
        ORDER BY DATE(s.date)
    """
    df = pd.read_sql(q, ENGINE)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    return df

