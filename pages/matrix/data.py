import pandas as pd
import numpy as np
from data import ENGINE, get_connection
import json

# для выгрузки категорий и групп
def fletch_cats() ->pd.DataFrame:
    q = """
    SELECT
    g.id as gr_id, g.name as gr_name, c.id as cat_id,c.name as cat_name
    from corporate_cattree as c
    join corporate_cattree as g on c.parent_id = g.id
    order by 1    
    """
    return pd.read_sql(q, ENGINE)
    
    
#Считаем ABC
def fletch_data(start, end,cats) ->pd.DataFrame:
    cat = "" if not cats else f"and i.cat_id in ({cats})"
    q = f"""
    SELECT 
    s.item_id,
    sum(s.dt-s.cr) as amount,
    JSON_ARRAYAGG(s.date) AS date_json,
    JSON_ARRAYAGG(s.quant_dt-s.quant_cr) AS quant_json,
    coalesce(case when i.article = "" then 'Нет арт.' else i.article end , 'Нет арт.') as article,
    i.fullname,
    i.cat_id,
    cat.name as cat_name,
    i.subcat_id,
    coalesce(sc.name,'Нет подкатегории') as sc_name
    from sales_salesdata as s
    join corporate_items as i on i.id = s.item_id
    left join corporate_cattree as cat on cat.id = i.cat_id
    left join corporate_subcategory as sc on sc.id = i.subcat_id
    where date between '{start}' and '{end}' {cat}  
    group by s.item_id, article, i.fullname,i.cat_id,i.subcat_id
    """
    return pd.read_sql(q, ENGINE)


def assign_abc(df: pd.DataFrame, thresholds) -> pd.DataFrame:
    df = df.copy()

    # сортируем по вкладу
    df = df.sort_values("amount", ascending=False)

    # доля каждого товара
    total = df["amount"].sum()
    df["share"] = df["amount"] / total

    # накопительная доля
    df["cum_share"] = df["share"].cumsum()
    
    # ABC
    a_thr = thresholds["a"] / 100
    b_thr = thresholds["b"] / 100

    df["abc"] = np.select(
        [
            df["cum_share"] <= a_thr,
            df["cum_share"] <= (b_thr+a_thr),
        ],
        ["A", "B"],
        default="C",
    )

    return df

def xyz_stats_from_json(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    date_json,
    quant_json,
    *,
    end_inclusive: bool = False,
    fill_missing_days: bool = True,
) -> dict:
    """
    start, end: границы периода.
    date_json, quant_json: либо list (уже распарсенные), либо JSON-строка.
    end_inclusive=False: timeline = [start, end) как в SQL: date >= start and date < end
    fill_missing_days=True: добавляем нули в дни без продаж

    Возвращает: mean_month, std_month, cv, months_count
    """

    # --- normalize inputs (json string -> python list) ---
    if isinstance(date_json, str):
        date_list = json.loads(date_json)
    else:
        date_list = date_json

    if isinstance(quant_json, str):
        quant_list = json.loads(quant_json)
    else:
        quant_list = quant_json

    if not date_list or not quant_list:
        return {"mean_month": 0.0, "std_month": 0.0, "cv": np.nan, "months_count": 0}

    if len(date_list) != len(quant_list):
        raise ValueError(f"date_json and quant_json lengths differ: {len(date_list)} vs {len(quant_list)}")

    # --- parse dates ---
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if end_inclusive:
        timeline = pd.date_range(s, e, freq="D")
    else:
        # [start, end)
        timeline = pd.date_range(s, e - pd.Timedelta(days=1), freq="D") if e > s else pd.DatetimeIndex([])

    # --- sales df (daily facts) ---
    sales = pd.DataFrame({"date": pd.to_datetime(date_list), "qty": pd.to_numeric(quant_list, errors="coerce")})
    sales = sales.dropna(subset=["date", "qty"])

    # если в json есть несколько записей в один день — суммируем
    sales = sales.groupby("date", as_index=False)["qty"].sum()

    # ограничим периодом (на всякий)
    if end_inclusive:
        sales = sales[(sales["date"] >= s) & (sales["date"] <= e)]
    else:
        sales = sales[(sales["date"] >= s) & (sales["date"] < e)]

    if sales.empty and fill_missing_days:
        # всё нули
        months = pd.Series(0.0, index=timeline).resample("MS").sum()
    else:
        if fill_missing_days:
            tl = pd.DataFrame({"date": timeline})
            daily = tl.merge(sales, on="date", how="left").fillna({"qty": 0.0})
        else:
            daily = sales.copy()

        daily = daily.sort_values("date").set_index("date")

        # --- monthly aggregation ---
        # "MS" = month start; sum -> месячные продажи
        months = daily["qty"].resample("MS").sum()

    # --- stats ---
    mean_month = float(months.mean()) if len(months) else 0.0
    std_month = float(months.std(ddof=0)) if len(months) else 0.0  # ddof=0 стабильнее для коротких рядов
    cv = (std_month / mean_month) if mean_month != 0 else np.nan

    return {
        "mean_month": mean_month,
        "std_month": std_month,
        "cv": float(cv) if cv == cv else np.nan,  # NaN-safe
        "months_count": int(len(months)),
    }





def matrix_calculation(start,end,cats,threholds) -> pd.DataFrame:
    
    df = fletch_data(start,end,cats)
    df = assign_abc(df,threholds)
    
    out = df.copy()

    stats = out.apply(
        lambda r: xyz_stats_from_json(
            start=start,
            end=end,
            date_json=r["date_json"],
            quant_json=r["quant_json"],
            end_inclusive=False,
            fill_missing_days=True,
        ),
        axis=1,
    )

    stats_df = pd.DataFrame(list(stats))
    out = pd.concat([out.reset_index(drop=True), stats_df], axis=1)
    
    x_thr = threholds["x"] 
    y_thr = threholds["y"] 

    out["xyz"] = np.select(
        [
            out["cv"] <= x_thr,
            out["cv"] <= y_thr,
        ],
        ["X", "Y"],
        default="Z",
    )
    
    out = out.sort_values(by=["abc","cv"])
    
    pt = out.pivot_table(
    index="abc",
    columns="xyz",
    values="fullname",
    aggfunc=lambda x: "\n".join(sorted(x))
).reset_index()
    
    # print(out.columns.to_list())
    return out
   
    
    