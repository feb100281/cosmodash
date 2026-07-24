# pages/matrix/data.py
from __future__ import annotations

import json
import locale
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from data import ENGINE


locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")


# ============================================================================
# Категории / группы
# ============================================================================

def fletch_cats() -> pd.DataFrame:
    q = """
    SELECT
        g.id AS gr_id,
        g.name AS gr_name,
        c.id AS cat_id,
        c.name AS cat_name
    FROM corporate_cattree AS c
    JOIN corporate_cattree AS g
        ON c.parent_id = g.id
    ORDER BY g.id
    """
    return pd.read_sql(q, ENGINE)


# ============================================================================
# Продажи
# ============================================================================

def fletch_data(start, end, cats) -> pd.DataFrame:
    cat = "" if not cats else f"AND i.cat_id IN ({cats})"

    q = f"""
    WITH sales AS (
        SELECT
            item_id,
            LAST_DAY(date) AS month_end,
            SUM(dt - cr) AS amount,
            SUM(quant_dt - quant_cr) AS quant
        FROM sales_salesdata
        GROUP BY
            item_id,
            LAST_DAY(date)
    ),
    barcode AS (
        SELECT
            i.id,
            i.article,
            i.fullname,
            COUNT(DISTINCT b.barcode) AS barcode_count,
            GROUP_CONCAT(DISTINCT b.barcode ORDER BY b.barcode SEPARATOR ', ') AS barcode
        FROM corporate_items_barcode AS t
        JOIN corporate_barcode AS b
            ON b.id = t.barcode_id
        JOIN corporate_items AS i
            ON i.id = t.items_id
        GROUP BY
            i.id,
            i.article,
            i.fullname
    )
    SELECT
        s.item_id,
        SUM(s.amount) AS amount,
        SUM(s.quant) AS quant,
        JSON_ARRAYAGG(s.month_end) AS date_json,
        JSON_ARRAYAGG(s.quant) AS quant_json,

        COALESCE(
            CASE
                WHEN i.article = '' THEN 'Нет арт.'
                ELSE i.article
            END,
            'Нет арт.'
        ) AS article,

        i.fullname,
        COALESCE(manu.name, 'Нет производителя') AS manu,
        i.cat_id,
        cat.name AS cat_name,
        i.subcat_id,
        COALESCE(sc.name, 'Нет подкатегории') AS sc_name,
        bc.barcode

    FROM sales AS s
    JOIN corporate_items AS i
        ON i.id = s.item_id
    LEFT JOIN corporate_itemmanufacturer AS manu
        ON manu.id = i.manufacturer_id
    LEFT JOIN corporate_cattree AS cat
        ON cat.id = i.cat_id
    LEFT JOIN corporate_subcategory AS sc
        ON sc.id = i.subcat_id
    LEFT JOIN barcode AS bc
        ON bc.id = s.item_id

    WHERE s.month_end BETWEEN '{start}' AND '{end}'
    {cat}

    GROUP BY
        s.item_id,
        i.article,
        i.fullname,
        manu.name,
        i.cat_id,
        cat.name,
        i.subcat_id,
        sc.name,
        bc.barcode
    """

    return pd.read_sql(q, ENGINE)


# ============================================================================
# Текущие остатки
# ============================================================================

def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _parse_stock_qty(value: Any) -> dict[str, float]:
    """
    Разбирает значения остатков вида:

        ["Европарк - 2 шт.", "ОСНОВНОЙ склад - 1364 шт."]

    или:

        ["2000000001944 - 23 шт.", "2000000001944 - 984 шт."]

    Если один и тот же склад / штрихкод встречается несколько раз,
    количества суммируются.
    """
    if _is_empty(value):
        return {}

    raw_values = None

    if isinstance(value, (list, tuple, np.ndarray)):
        raw_values = list(value)
    else:
        text = str(value).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                raw_values = parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            raw_values = None

    result: dict[str, float] = {}

    if raw_values is not None:
        for item in raw_values:
            match = re.match(
                r"^\s*(.+?)\s*-\s*(-?\d+(?:[.,]\d+)?)\s*шт\.\s*$",
                str(item),
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            name = match.group(1).strip().strip("'\"")
            if not name:
                continue

            try:
                qty_value = float(match.group(2).replace(",", "."))
            except (TypeError, ValueError):
                continue

            result[name] = result.get(name, 0.0) + qty_value

        return result

    # Fallback для старого строкового представления массива.
    text = str(value)
    matches = re.findall(
        r"[\"']?([^\"'\[\],]+?)[\"']?\s*-\s*(-?\d+(?:[.,]\d+)?)\s*шт\.",
        text,
        flags=re.IGNORECASE,
    )

    for name, qty in matches:
        name = str(name).strip().strip("'\"")
        if not name:
            continue

        try:
            qty_value = float(str(qty).replace(",", "."))
        except (TypeError, ValueError):
            continue

        result[name] = result.get(name, 0.0) + qty_value

    return result


def _format_stock_qty(value: Any) -> str:
    """Компактное многострочное представление остатков для AG Grid."""
    values = _parse_stock_qty(value)

    if not values:
        return ""

    parts: list[str] = []

    for name, qty in sorted(values.items(), key=lambda item: str(item[0]).lower()):
        qty_text = f"{qty:,.0f}".replace(",", " ")
        parts.append(f"{name} — {qty_text} шт.")

    return "\n".join(parts)


def fetch_current_stocks() -> pd.DataFrame:
    """
    Возвращает последние остатки для каждого item_id.

    Основные колонки:
        stock_date
        stock_available
        stock_ordered
        stock_total
        barcode_stocks
        barcode_ordered

    Динамические колонки:
        stock_wh::<название склада>
        ordered_wh::<название склада>

    Важно:
    warehouse_stocks / warehouse_ordered могут содержать один склад несколько
    раз. При разборе значения по одинаковому складу суммируются.
    """
    q = """
    WITH ranked AS (
        SELECT
            item_id,
            init_date,
            tot_available,
            tot_ordered,
            total,
            barcode_stocks,
            barcode_ordered,
            warehouse_stocks,
            warehouse_ordered,
            ROW_NUMBER() OVER (
                PARTITION BY item_id
                ORDER BY init_date DESC
            ) AS rn
        FROM djangodb.stocks_data
    )
    SELECT
        item_id,
        init_date,
        tot_available,
        tot_ordered,
        total,
        barcode_stocks,
        barcode_ordered,
        warehouse_stocks,
        warehouse_ordered
    FROM ranked
    WHERE rn = 1
    """

    stocks = pd.read_sql(q, ENGINE)

    if stocks.empty:
        return pd.DataFrame(
            columns=[
                "item_id",
                "stock_date",
                "stock_available",
                "stock_ordered",
                "stock_total",
                "barcode_stocks",
                "barcode_ordered",
            ]
        )

    stocks["item_id"] = pd.to_numeric(stocks["item_id"], errors="coerce")
    stocks = stocks.dropna(subset=["item_id"]).copy()
    stocks["item_id"] = stocks["item_id"].astype(int)

    stocks = stocks.rename(
        columns={
            "init_date": "stock_date",
            "tot_available": "stock_available",
            "tot_ordered": "stock_ordered",
            "total": "stock_total",
        }
    )

    stocks["stock_date"] = pd.to_datetime(
        stocks["stock_date"],
        errors="coerce",
    )

    for column in ("stock_available", "stock_ordered", "stock_total"):
        stocks[column] = pd.to_numeric(
            stocks[column],
            errors="coerce",
        ).fillna(0.0)

    # ----------------------------------------------------------------------
    # Остатки по складам
    # ----------------------------------------------------------------------
    stock_dicts = stocks["warehouse_stocks"].apply(_parse_stock_qty)

    stock_warehouses = sorted(
        {
            warehouse
            for warehouses in stock_dicts
            for warehouse in warehouses.keys()
        },
        key=str.lower,
    )

    for warehouse in stock_warehouses:
        stocks[f"stock_wh::{warehouse}"] = stock_dicts.apply(
            lambda values, wh=warehouse: float(values.get(wh, 0.0))
        )

    # ----------------------------------------------------------------------
    # Заказы по складам
    # ----------------------------------------------------------------------
    ordered_dicts = stocks["warehouse_ordered"].apply(_parse_stock_qty)

    ordered_warehouses = sorted(
        {
            warehouse
            for warehouses in ordered_dicts
            for warehouse in warehouses.keys()
        },
        key=str.lower,
    )

    for warehouse in ordered_warehouses:
        stocks[f"ordered_wh::{warehouse}"] = ordered_dicts.apply(
            lambda values, wh=warehouse: float(values.get(wh, 0.0))
        )

    # ----------------------------------------------------------------------
    # Остатки по штрихкодам для компактного отображения в Grid
    # ----------------------------------------------------------------------
    stocks["barcode_stocks_display"] = stocks["barcode_stocks"].apply(
        _format_stock_qty
    )

    stocks = stocks.drop(
        columns=["warehouse_stocks", "warehouse_ordered"],
        errors="ignore",
    )

    return stocks



def fetch_items_metadata(item_ids: list[int]) -> pd.DataFrame:
    """
    Метаданные товаров для SKU, которые есть в остатках, но не имели продаж
    в выбранном периоде.
    """
    if not item_ids:
        return pd.DataFrame()

    ids_sql = ",".join(str(int(v)) for v in item_ids)

    q = f"""
    WITH barcode AS (
        SELECT
            i.id,
            GROUP_CONCAT(
                DISTINCT b.barcode
                ORDER BY b.barcode
                SEPARATOR ', '
            ) AS barcode
        FROM corporate_items_barcode AS t
        JOIN corporate_barcode AS b
            ON b.id = t.barcode_id
        JOIN corporate_items AS i
            ON i.id = t.items_id
        WHERE i.id IN ({ids_sql})
        GROUP BY i.id
    )
    SELECT
        i.id AS item_id,
        COALESCE(
            CASE
                WHEN i.article = '' THEN 'Нет арт.'
                ELSE i.article
            END,
            'Нет арт.'
        ) AS article,
        i.fullname,
        COALESCE(manu.name, 'Нет производителя') AS manu,
        i.cat_id,
        cat.name AS cat_name,
        i.subcat_id,
        COALESCE(sc.name, 'Нет подкатегории') AS sc_name,
        bc.barcode
    FROM corporate_items AS i
    LEFT JOIN corporate_itemmanufacturer AS manu
        ON manu.id = i.manufacturer_id
    LEFT JOIN corporate_cattree AS cat
        ON cat.id = i.cat_id
    LEFT JOIN corporate_subcategory AS sc
        ON sc.id = i.subcat_id
    LEFT JOIN barcode AS bc
        ON bc.id = i.id
    WHERE i.id IN ({ids_sql})
    """

    return pd.read_sql(q, ENGINE)


# ============================================================================
# ABC
# ============================================================================

def assign_abc(df: pd.DataFrame, thresholds) -> pd.DataFrame:
    df = df.copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    df = df.sort_values("amount", ascending=False)

    total = float(df["amount"].sum())

    if total > 0:
        df["share"] = df["amount"] / total
    else:
        df["share"] = 0.0

    df["cum_share"] = df["share"].cumsum()

    a_thr = thresholds["a"] / 100
    b_thr = thresholds["b"] / 100

    df["abc"] = np.select(
        [
            df["cum_share"] <= a_thr,
            df["cum_share"] <= (a_thr + b_thr),
        ],
        ["A", "B"],
        default="C",
    )

    return df


# ============================================================================
# Месячные ряды / XYZ
# ============================================================================

def _parse_quant_list(value: Any) -> list[float]:
    if _is_empty(value):
        return []

    if isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    else:
        raw = str(value).strip("[]").split(",")

    result: list[float] = []

    for item in raw:
        try:
            result.append(float(str(item).strip().strip("'\"")))
        except (TypeError, ValueError):
            result.append(0.0)

    return result


def _parse_date_list(value: Any) -> list[pd.Timestamp]:
    if _is_empty(value):
        return []

    if isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    else:
        raw = str(value).strip("[]").split(",")

    result: list[pd.Timestamp] = []

    for item in raw:
        dt = pd.to_datetime(
            str(item).strip().strip("'\""),
            errors="coerce",
        )
        result.append(dt)

    return result


def _monthly_series_with_zeros(
    dates: list[pd.Timestamp],
    quantities: list[float],
) -> pd.Series:
    """
    Создаёт непрерывный месячный ряд между первой и последней продажей.

    Пример:
        Янв = 10
        Фев = нет строки
        Мар = 20

    превращается в:
        [10, 0, 20]

    Это важно для корректного XYZ/CV: месяц без продаж должен участвовать
    в среднем и стандартном отклонении как 0.
    """
    pairs = []

    for d, q in zip(dates, quantities):
        if pd.isna(d):
            continue

        try:
            qty = float(q)
        except (TypeError, ValueError):
            qty = 0.0

        month = pd.Timestamp(d).to_period("M")
        pairs.append((month, qty))

    if not pairs:
        return pd.Series(dtype="float64")

    grouped: dict[pd.Period, float] = {}
    for month, qty in pairs:
        grouped[month] = grouped.get(month, 0.0) + qty

    start = min(grouped)
    end = max(grouped)

    periods = pd.period_range(start=start, end=end, freq="M")

    return pd.Series(
        [grouped.get(period, 0.0) for period in periods],
        index=periods,
        dtype="float64",
    )


def count_month_gaps(dates):
    ds = [d for d in dates if pd.notna(d)]

    if len(ds) <= 1:
        return 0

    months = sorted({pd.Timestamp(d).to_period("M") for d in ds})

    if len(months) <= 1:
        return 0

    full = pd.period_range(min(months), max(months), freq="M")
    return int(len(full) - len(months))


# ============================================================================
# Основной расчёт
# ============================================================================

def matrix_calculation(start, end, cats, threholds, lt, sr) -> pd.DataFrame:
    # ----------------------------------------------------------------------
    # 1. Продажи выбранного периода
    # ----------------------------------------------------------------------
    sales_df = fletch_data(start, end, cats)

    # ----------------------------------------------------------------------
    # 2. Актуальные остатки
    # ----------------------------------------------------------------------
    stocks = fetch_current_stocks()

    if not stocks.empty:
        stocks["item_id"] = pd.to_numeric(
            stocks["item_id"],
            errors="coerce",
        ).astype("Int64")

    # ----------------------------------------------------------------------
    # 3. Добавляем SKU без продаж, но с текущим остатком/заказом
    # ----------------------------------------------------------------------
    if sales_df.empty:
        sold_item_ids: set[int] = set()
    else:
        sales_df["item_id"] = pd.to_numeric(
            sales_df["item_id"],
            errors="coerce",
        ).astype("Int64")

        sold_item_ids = set(
            sales_df["item_id"]
            .dropna()
            .astype(int)
            .tolist()
        )

    if stocks.empty:
        stock_item_ids: set[int] = set()
    else:
        stock_mask = (
            pd.to_numeric(
                stocks["stock_available"],
                errors="coerce",
            ).fillna(0) != 0
        ) | (
            pd.to_numeric(
                stocks["stock_ordered"],
                errors="coerce",
            ).fillna(0) != 0
        ) | (
            pd.to_numeric(
                stocks["stock_total"],
                errors="coerce",
            ).fillna(0) != 0
        )

        stock_item_ids = set(
            stocks.loc[stock_mask, "item_id"]
            .dropna()
            .astype(int)
            .tolist()
        )

    missing_item_ids = sorted(stock_item_ids - sold_item_ids)

    if missing_item_ids:
        metadata = fetch_items_metadata(missing_item_ids)

        if not metadata.empty:
            # Применяем тот же фильтр категории, что и для продаж.
            if cats:
                selected_cat_ids = {
                    int(v.strip())
                    for v in str(cats).split(",")
                    if str(v).strip()
                }

                metadata = metadata[
                    pd.to_numeric(
                        metadata["cat_id"],
                        errors="coerce",
                    ).isin(selected_cat_ids)
                ].copy()

            if not metadata.empty:
                # Для товаров без продаж создаём корректные нулевые показатели.
                metadata["amount"] = 0.0
                metadata["quant"] = 0.0
                metadata["date_json"] = "[]"
                metadata["quant_json"] = "[]"
                metadata["share"] = 0.0
                metadata["cum_share"] = 0.0

                if sales_df.empty:
                    sales_df = metadata
                else:
                    sales_df = pd.concat(
                        [sales_df, metadata],
                        ignore_index=True,
                        sort=False,
                    )

    # Если нет ни продаж, ни остатков — возвращаем пустой DataFrame.
    if sales_df.empty:
        return sales_df

    df = sales_df.copy()

    # ABC предварительно считаем только по продажам.
    # Строки без продаж позже получат отдельный класс "Без продаж".
    df = assign_abc(df, threholds)

    # ----------------------------------------------------------------------
    # 4. Присоединяем текущие остатки
    # ----------------------------------------------------------------------
    df["item_id"] = pd.to_numeric(
        df["item_id"],
        errors="coerce",
    ).astype("Int64")

    if not stocks.empty:
        df = df.merge(
            stocks,
            on="item_id",
            how="left",
            validate="many_to_one",
        )

    for column in ("stock_available", "stock_ordered", "stock_total"):
        if column not in df.columns:
            df[column] = 0.0

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0.0)

    for column in df.columns:
        if column.startswith("stock_wh::") or column.startswith("ordered_wh::"):
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            ).fillna(0.0)

    # ----------------------------------------------------------------------
    # Месячные продажи
    # ----------------------------------------------------------------------
    df["ls_quant"] = df["quant_json"].apply(_parse_quant_list)
    df["ls_date"] = df["date_json"].apply(_parse_date_list)

    out = df.copy()

    out["_monthly_series"] = [
        _monthly_series_with_zeros(dates, quantities)
        for dates, quantities in zip(
            out["ls_date"],
            out["ls_quant"],
        )
    ]

    out["mean_month"] = out["_monthly_series"].apply(
        lambda s: float(s.mean()) if len(s) else 0.0
    )

    # ddof=0 — считаем σ по всему наблюдаемому периоду как генеральную совокупность
    out["std_month"] = out["_monthly_series"].apply(
        lambda s: float(s.std(ddof=0)) if len(s) else 0.0
    )

    out["cv"] = np.where(
        out["mean_month"] > 0,
        out["std_month"] / out["mean_month"],
        np.nan,
    )

    out["month_count"] = out["_monthly_series"].apply(
        lambda s: int((s != 0).sum()) if len(s) else 0
    )

    out["missing_months"] = out["_monthly_series"].apply(
        lambda s: int((s == 0).sum()) if len(s) else 0
    )

    out["max_month"] = out["_monthly_series"].apply(
        lambda s: float(s.max()) if len(s) else 0.0
    )

    out["min_month"] = out["_monthly_series"].apply(
        lambda s: float(s.min()) if len(s) else 0.0
    )

    # Есть ли вообще продажи у SKU в выбранном периоде.
    out["has_sales_period"] = (
        pd.to_numeric(out["quant"], errors="coerce").fillna(0) != 0
    ) | (
        pd.to_numeric(out["amount"], errors="coerce").fillna(0) != 0
    )

    def _safe_min_date(values):
        valid = [d for d in values if pd.notna(d)]
        return min(valid) if valid else pd.NaT

    def _safe_max_date(values):
        valid = [d for d in values if pd.notna(d)]
        return max(valid) if valid else pd.NaT

    out["min_date"] = out["ls_date"].apply(_safe_min_date)
    out["max_date"] = out["ls_date"].apply(_safe_max_date)

    out["sales_period_months"] = np.where(
        out["min_date"].notna() & out["max_date"].notna(),
        (
            out["max_date"].dt.year * 12
            + out["max_date"].dt.month
            - out["min_date"].dt.year * 12
            - out["min_date"].dt.month
            + 1
        ),
        0,
    )

    out["sales_period_months"] = pd.to_numeric(
        out["sales_period_months"],
        errors="coerce",
    ).fillna(0).astype(int)

    out["mean_amount"] = np.where(
        out["sales_period_months"] > 0,
        out["amount"] / out["sales_period_months"],
        0.0,
    )

    # ----------------------------------------------------------------------
    # Финальный ABC по средней месячной выручке
    # ----------------------------------------------------------------------
    out = out.sort_values("mean_amount", ascending=False)

    total_mean_amount = float(out["mean_amount"].sum())

    if total_mean_amount > 0:
        out["share_mean"] = out["mean_amount"] / total_mean_amount
    else:
        out["share_mean"] = 0.0

    check_date = pd.to_datetime(end)

    out["abc"] = np.where(
        ~out["has_sales_period"],
        "Без продаж",
        np.where(
            out["sales_period_months"] == 1,
            np.where(
                out["max_date"] == check_date,
                "Новый",
                "Редкий",
            ),
            None,
        ),
    )

    out["_amount"] = np.where(
        out["abc"].isna(),
        out["mean_amount"],
        0.0,
    )

    rating_total = float(out["_amount"].sum())

    if rating_total > 0:
        out["_share"] = out["_amount"] / rating_total
    else:
        out["_share"] = 0.0

    out["cum_share"] = out["_share"].cumsum()

    out["xyz"] = np.where(
        ~out["has_sales_period"],
        "Без продаж",
        np.where(
            out["sales_period_months"] == 1,
            np.where(
                out["max_date"] == check_date,
                "Новый",
                "Редкий",
            ),
            None,
        ),
    )

    a_thr = threholds["a"] / 100
    b_thr = threholds["b"] / 100

    out["abc"] = np.where(
        out["abc"].isna(),
        np.select(
            [
                out["cum_share"] <= a_thr,
                out["cum_share"] <= (a_thr + b_thr),
            ],
            ["A", "B"],
            default="C",
        ),
        out["abc"],
    )

    x_thr = float(threholds["x"])
    y_thr = float(threholds["y"])

    out["xyz"] = np.where(
        out["xyz"].isna(),
        np.select(
            [
                out["cv"].fillna(np.inf) <= x_thr,
                out["cv"].fillna(np.inf) <= y_thr,
            ],
            ["X", "Y"],
            default="Z",
        ),
        out["xyz"],
    )

    # ----------------------------------------------------------------------
    # Safety Stock / ROP
    # ----------------------------------------------------------------------
    lead_time = max(float(lt or 0), 0.0)
    service_ratio = min(max(float(sr or 0), 0.01), 99.99)

    z = norm.ppf(service_ratio / 100.0)

    # σ задано в "ед./месяц", поэтому для LT месяцев:
    # SS = z * σ * sqrt(LT)
    out["ss"] = (
        z
        * out["std_month"].fillna(0.0)
        * np.sqrt(lead_time)
    )

    # ROP = ожидаемый спрос за Lead Time + Safety Stock
    out["rop"] = (
        out["mean_month"].fillna(0.0) * lead_time
        + out["ss"]
    )

    out["ss"] = np.ceil(out["ss"].clip(lower=0))
    out["rop"] = np.ceil(out["rop"].clip(lower=0))

    # ----------------------------------------------------------------------
    # Аналитика текущих запасов
    # ----------------------------------------------------------------------
    out["stock_cover_months"] = np.where(
        out["mean_month"] > 0,
        out["stock_available"] / out["mean_month"],
        np.nan,
    )

    out["stock_cover_months_total"] = np.where(
        out["mean_month"] > 0,
        out["stock_total"] / out["mean_month"],
        np.nan,
    )

    out["stock_vs_rop"] = out["stock_available"] - out["rop"]
    out["stock_total_vs_rop"] = out["stock_total"] - out["rop"]

    # Сколько ещё нужно заказать ПОСЛЕ учёта уже заказанного товара.
    out["order_need"] = np.ceil(
        np.maximum(
            out["rop"] - out["stock_total"],
            0.0,
        )
    )

    no_available = out["stock_available"] <= 0
    has_order = out["stock_ordered"] > 0
    below_rop = out["stock_available"] < out["rop"]
    total_covers_rop = out["stock_total"] >= out["rop"]
    excess = out["stock_cover_months"] > 6

    # Отдельно выделяем товар, который лежит/едет, но в выбранном периоде
    # вообще не продавался. Это важный сигнал для ассортиментной матрицы.
    no_sales_with_stock = (
        ~out["has_sales_period"]
        & (out["stock_total"] > 0)
    )

    out["stock_status"] = np.select(
        [
            no_sales_with_stock,
            no_available & ~has_order,
            no_available & has_order,
            below_rop & ~total_covers_rop,
            below_rop & total_covers_rop,
            excess,
        ],
        [
            "Без продаж, есть запас",
            "Нет остатка",
            "Нет остатка, товар заказан",
            "Дефицит",
            "Ниже ROP, заказ закрывает",
            "Избыток > 6 мес.",
        ],
        default="Достаточно",
    )

    # ----------------------------------------------------------------------
    # Финальная сортировка
    # ----------------------------------------------------------------------
    abc_order = {
        "A": 0,
        "B": 1,
        "C": 2,
        "Новый": 3,
        "Редкий": 4,
        "Без продаж": 5,
    }
    xyz_order = {
        "X": 0,
        "Y": 1,
        "Z": 2,
        "Новый": 3,
        "Редкий": 4,
        "Без продаж": 5,
    }

    out["_abc_order"] = out["abc"].map(abc_order).fillna(99)
    out["_xyz_order"] = out["xyz"].map(xyz_order).fillna(99)

    out = out.sort_values(
        by=["_abc_order", "_xyz_order", "share"],
        ascending=[True, True, False],
    )

    out["min_date"] = out["min_date"].dt.strftime("%b %Y").str.capitalize()
    out["max_date"] = out["max_date"].dt.strftime("%b %Y").str.capitalize()

    if "stock_date" in out.columns:
        out["stock_date"] = pd.to_datetime(
            out["stock_date"],
            errors="coerce",
        ).dt.strftime("%d.%m.%Y")

    out = out.drop(
        columns=[
            "_monthly_series",
            "_abc_order",
            "_xyz_order",
        ],
        errors="ignore",
    )

    return out
