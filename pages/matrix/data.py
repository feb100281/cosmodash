import pandas as pd
import numpy as np
from data import ENGINE, get_connection
import locale
from scipy.stats import norm

locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")


# для выгрузки категорий и групп
def fletch_cats() -> pd.DataFrame:
    q = """
    SELECT
    g.id as gr_id, g.name as gr_name, c.id as cat_id,c.name as cat_name
    from corporate_cattree as c
    join corporate_cattree as g on c.parent_id = g.id
    order by 1    
    """
    return pd.read_sql(q, ENGINE)


# Считаем ABC
def fletch_data(start, end, cats) -> pd.DataFrame:
    cat = "" if not cats else f"and i.cat_id in ({cats})"
    q = f"""
    WITH sales as (
    select
    item_id,
    LAST_DAY(date) AS month_end,
    sum(dt-cr) as amount,
    sum(quant_dt-quant_cr) as quant
    from sales_salesdata
    group by item_id, month_end
    ),
    barcode as (
    select 
	i.id,
	i.article,
	i.fullname,
	count(distinct b.barcode) as barcode_count,
	GROUP_CONCAT(b.barcode) as barcode

	from corporate_items_barcode as t
	join corporate_barcode as b on b.id = t.barcode_id
	join corporate_items as i on i.id = t.items_id
	group by i.id, i.fullname
    )

    SELECT 
    s.item_id,
    sum(s.amount) as amount,
    sum(s.quant) as quant,
    JSON_ARRAYAGG(s.month_end) AS date_json,
    JSON_ARRAYAGG(quant) AS quant_json,
    coalesce(case when i.article = "" then 'Нет арт.' else i.article end , 'Нет арт.') as article,
    i.fullname,
    i.cat_id,
    cat.name as cat_name,
    i.subcat_id,
    coalesce(sc.name,'Нет подкатегории') as sc_name,
    bc.barcode 

    from sales as s
    join corporate_items as i on i.id = s.item_id
    left join corporate_cattree as cat on cat.id = i.cat_id
    left join corporate_subcategory as sc on sc.id = i.subcat_id
    left join barcode as bc on bc.id = s.item_id
    where month_end between '{start}' and '{end}' {cat}  
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
            df["cum_share"] <= (b_thr + a_thr),
        ],
        ["A", "B"],
        default="C",
    )

    return df


def count_month_gaps(dates):
    # dates: list[datetime64] (могут быть NaT)
    ds = [d for d in dates if pd.notna(d)]
    if len(ds) <= 1:
        return 0

    s = pd.Series(ds).sort_values()

    # приводим к "месячным" индексам (целое число месяцев)
    m = s.dt.year * 12 + s.dt.month

    diffs = m.diff().dropna()  # разницы между соседними месяцами
    gaps = (diffs - 1).clip(lower=0)  # сколько пустых месяцев между ними
    return int(gaps.sum())


def matrix_calculation(start, end, cats, threholds,lt,sr) -> pd.DataFrame:

    df = fletch_data(start, end, cats)
    df = assign_abc(df, threholds)

    df["ls_quant"] = (
        df["quant_json"]
        .str.strip("[]")
        .str.split(",")
        .apply(lambda xs: [float(x) for x in xs])
    )
    df["ls_date"] = (
        df["date_json"]
        .str.strip("[]")
        .str.split(",")
        .apply(
            lambda xs: [
                pd.to_datetime(x.strip().strip("'").strip('"'), errors="coerce")
                for x in xs
            ]
        )
    )

    out = df.copy()

    out["mean_month"] = out["ls_quant"].map(np.mean)
    out["std_month"] = out["ls_quant"].map(np.std)
    out["cv"] = out["std_month"] / out["mean_month"]
    out["month_count"] = out["ls_quant"].map(np.count_nonzero)
    out["max_month"] = out["ls_quant"].map(np.max)
    out["min_month"] = out["ls_quant"].map(np.min)
    out["missing_months"] = out["ls_date"].apply(count_month_gaps)

    out["min_date"] = out["ls_date"].map(lambda x: min(d for d in x if pd.notna(d)))

    out["max_date"] = out["ls_date"].map(lambda x: max(d for d in x if pd.notna(d)))

    out["sales_period_months"] = (
        (out["max_date"].dt.year * 12 + out["max_date"].dt.month)
        - (out["min_date"].dt.year * 12 + out["min_date"].dt.month)
        + 1
    )
    out["mean_amount"] = out["amount"] / out["sales_period_months"]
    out = out.sort_values("mean_amount", ascending=False)
    total = out["mean_amount"].sum()
    out["share_mean"] = out["mean_amount"] / total

    
    

    check_date = pd.to_datetime(end)

    # Вне рейтинга
    out["abc"] = np.where(
        out["sales_period_months"] == 1,
        np.where(out["max_date"] == check_date, "Новый", "Редкий"),
        None,
    )
    out['_amount'] = np.where(out["abc"].isna(),out["mean_amount"],0)
    _total = out['_amount'].sum()
    out['_share'] = out['_amount'] / _total
    # накопительная доля
    out['cum_share'] = out['_share'].cumsum()

    out["xyz"] = np.where(
        out["sales_period_months"] == 1,
        np.where(out["max_date"] == check_date, "Новый", "Редкий"),
        None,
    )

    # ABC
    a_thr = threholds["a"] / 100
    b_thr = threholds["b"] / 100

    out["abc"] = np.where(
        out["abc"].isna(),
        np.select(
            [
                out["cum_share"] <= a_thr,
                out["cum_share"] <= (b_thr + a_thr),
            ],
            ["A", "B"],
            default="C",
        ),
        out["abc"],
    )

    # XYZ
    x_thr = threholds["x"]
    y_thr = threholds["y"]
    out["xyz"] = np.where(
        out["xyz"].isna(),
        np.select(
            [
                out["cv"] <= x_thr,
                out["cv"] <= y_thr,
            ],
            ["X", "Y"],
            default="Z",
        ),
        out["xyz"],
    )
    
    # Считаем стоки
    z = norm.ppf(sr/100)
    out['ss'] = out["std_month"]*z
    out['rop'] = out["mean_month"]*z + out['ss']
    out['ss'] = out['ss'].map(np.ceil)
    out['rop'] = out['rop'].map(np.ceil)
    
    out = out.sort_values(by=["abc", "xyz", "share"], ascending=[True, True, False])

    out["min_date"] = out["min_date"].dt.strftime("%b %Y").str.capitalize()
    out["max_date"] = out["max_date"].dt.strftime("%b %Y").str.capitalize()

    
    return out
