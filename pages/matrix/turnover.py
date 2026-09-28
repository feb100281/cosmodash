# pages/matrix/turnover.py
"""
Оборачиваемость SKU: продажи периода, текущие остатки, приходы (stock_receipts).

turnover_days  = доступный остаток / среднедневные продажи
sell_through   = продано с последнего прихода / (продано с прихода + остаток)
stock_value_purchase — по цене последнего прихода, если цена есть.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .data import ENGINE


FAST_DAYS = 60
NORMAL_DAYS = 120
SLOW_DAYS = 365
NEW_RECEIPT_DAYS = 60
STALE_RECEIPT_DAYS = 90
LOW_SELL_THROUGH = 0.30

STATUS_NO_STOCK = "Нет остатка"
STATUS_FAST = f"Быстрая (≤{FAST_DAYS} дн.)"
STATUS_NORMAL = f"Нормальная ({FAST_DAYS + 1}–{NORMAL_DAYS} дн.)"
STATUS_SLOW = f"Медленная ({NORMAL_DAYS + 1}–{SLOW_DAYS} дн.)"
STATUS_VERY_SLOW = f"Очень медленная (>{SLOW_DAYS} дн.)"
STATUS_NEW = "Новый приход, продаж ещё нет"
STATUS_DEAD = "Неликвид: нет продаж"

STATUS_ORDER = [
    STATUS_DEAD,
    STATUS_VERY_SLOW,
    STATUS_SLOW,
    STATUS_NORMAL,
    STATUS_FAST,
    STATUS_NEW,
    STATUS_NO_STOCK,
]

STATUS_COLORS = {
    STATUS_DEAD: "#7B4437",
    STATUS_VERY_SLOW: "#9A6100",
    STATUS_SLOW: "#C9A227",
    STATUS_NORMAL: "#3D7A67",
    STATUS_FAST: "#1F5E4E",
    STATUS_NEW: "#2F75B5",
    STATUS_NO_STOCK: "#8A8A8A",
}

TURNOVER_COLUMNS = [
    "avg_daily_sales",
    "turnover_days",
    "turns_per_year",
    "first_receipt_date",
    "last_receipt_date",
    "days_since_receipt",
    "receipts_docs",
    "receipt_qty_total",
    "receipt_qty_period",
    "last_receipt_qty",
    "sold_since_receipt",
    "sell_through",
    "purchase_price",
    "stock_value_purchase",
    "turnover_status",
]


# ---------------------------------------------------------------------------
# Данные о приходах
# ---------------------------------------------------------------------------

def fetch_receipts_summary(period_start: str, period_end: str) -> pd.DataFrame:
    """Агрегаты приходов по item_id; пустой DataFrame, если таблицы нет."""
    q = f"""
    WITH r AS (
        SELECT item_id, receipt_guid, receipt_date, qty, price_purchase
        FROM djangodb.stock_receipts
        WHERE item_id IS NOT NULL
    ),
    agg AS (
        SELECT
            item_id,
            MIN(receipt_date) AS first_receipt_date,
            MAX(receipt_date) AS last_receipt_date,
            COUNT(DISTINCT receipt_guid) AS receipts_docs,
            SUM(qty) AS receipt_qty_total,
            SUM(CASE WHEN receipt_date BETWEEN '{period_start}' AND '{period_end}'
                     THEN qty ELSE 0 END) AS receipt_qty_period
        FROM r
        GROUP BY item_id
    ),
    last_qty AS (
        SELECT r.item_id, SUM(r.qty) AS last_receipt_qty
        FROM r
        JOIN agg ON agg.item_id = r.item_id AND agg.last_receipt_date = r.receipt_date
        GROUP BY r.item_id
    ),
    priced AS (
        SELECT
            item_id,
            price_purchase,
            ROW_NUMBER() OVER (PARTITION BY item_id ORDER BY receipt_date DESC) AS rn
        FROM r
        WHERE price_purchase IS NOT NULL AND price_purchase > 0
    ),
    sold_since AS (
        SELECT s.item_id, SUM(s.quant_dt - s.quant_cr) AS sold_since_receipt
        FROM sales_salesdata AS s
        JOIN agg ON agg.item_id = s.item_id AND s.date >= agg.last_receipt_date
        GROUP BY s.item_id
    )
    SELECT
        agg.*,
        lq.last_receipt_qty,
        p.price_purchase AS purchase_price,
        COALESCE(ss.sold_since_receipt, 0) AS sold_since_receipt
    FROM agg
    LEFT JOIN last_qty AS lq ON lq.item_id = agg.item_id
    LEFT JOIN priced AS p ON p.item_id = agg.item_id AND p.rn = 1
    LEFT JOIN sold_since AS ss ON ss.item_id = agg.item_id
    """
    try:
        df = pd.read_sql(q, ENGINE)
    except Exception as exc:
        print(f"[turnover] stock_receipts недоступна: {exc}")
        return pd.DataFrame()

    if df.empty:
        return df

    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce").astype("Int64")
    for col in ("first_receipt_date", "last_receipt_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("receipts_docs", "receipt_qty_total", "receipt_qty_period",
                "last_receipt_qty", "purchase_price", "sold_since_receipt"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def _period_bounds(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    """start/end — концы месяцев."""
    p_start = pd.Timestamp(start).replace(day=1)
    p_end = min(pd.Timestamp(end), pd.Timestamp(date.today()))
    days = max((p_end - p_start).days + 1, 1)
    return p_start, p_end, days


def add_turnover_metrics(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    p_start, p_end, period_days = _period_bounds(start, end)
    today = pd.Timestamp(date.today())

    receipts = fetch_receipts_summary(
        p_start.strftime("%Y-%m-%d"),
        p_end.strftime("%Y-%m-%d"),
    )

    out["item_id"] = pd.to_numeric(out["item_id"], errors="coerce").astype("Int64")
    out = out.drop(columns=[c for c in TURNOVER_COLUMNS if c in out.columns], errors="ignore")

    if not receipts.empty:
        out = out.merge(receipts, on="item_id", how="left", validate="many_to_one")
    else:
        for col in ("first_receipt_date", "last_receipt_date"):
            out[col] = pd.NaT
        for col in ("receipts_docs", "receipt_qty_total", "receipt_qty_period",
                    "last_receipt_qty", "purchase_price", "sold_since_receipt"):
            out[col] = np.nan

    stock = pd.to_numeric(out.get("stock_available", 0), errors="coerce").fillna(0.0)
    sold = pd.to_numeric(out.get("quant", 0), errors="coerce").fillna(0.0).clip(lower=0)

    out["period_days"] = period_days
    out["avg_daily_sales"] = sold / period_days

    with np.errstate(divide="ignore", invalid="ignore"):
        turnover = np.where(
            (out["avg_daily_sales"] > 0) & (stock > 0),
            stock / out["avg_daily_sales"],
            np.nan,
        )
    out["turnover_days"] = np.round(turnover, 0)
    out["turns_per_year"] = np.where(
        out["turnover_days"] > 0, 365.0 / out["turnover_days"], np.nan
    )

    out["days_since_receipt"] = (today - out["last_receipt_date"]).dt.days

    sold_since = pd.to_numeric(out["sold_since_receipt"], errors="coerce")
    denom = sold_since.clip(lower=0) + stock
    out["sell_through"] = np.where(
        out["last_receipt_date"].notna() & (denom > 0),
        sold_since.clip(lower=0) / denom,
        np.nan,
    )

    price = pd.to_numeric(out["purchase_price"], errors="coerce")
    out["stock_value_purchase"] = np.where(price.notna(), price * stock, np.nan)

    no_stock = stock <= 0
    no_sales = sold <= 0
    is_new = out["days_since_receipt"].le(NEW_RECEIPT_DAYS).fillna(False)
    td = out["turnover_days"]

    out["turnover_status"] = np.select(
        [
            no_stock,
            no_sales & is_new,
            no_sales,
            td <= FAST_DAYS,
            td <= NORMAL_DAYS,
            td <= SLOW_DAYS,
        ],
        [
            STATUS_NO_STOCK,
            STATUS_NEW,
            STATUS_DEAD,
            STATUS_FAST,
            STATUS_NORMAL,
            STATUS_SLOW,
        ],
        default=STATUS_VERY_SLOW,
    )

    for col in ("first_receipt_date", "last_receipt_date"):
        out[col] = out[col].dt.strftime("%d.%m.%Y")

    out["turnover_period"] = f"{p_start:%d.%m.%Y} – {p_end:%d.%m.%Y}"
    return out


# ---------------------------------------------------------------------------
# Сводки, выводы, рекомендации
# ---------------------------------------------------------------------------

def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def turnover_kpis(df: pd.DataFrame) -> dict:
    if df is None or df.empty or "turnover_status" not in df.columns:
        return {"has_data": False}

    stock = _num(df, "stock_available").fillna(0)
    daily = _num(df, "avg_daily_sales").fillna(0)
    status = df["turnover_status"].fillna("")
    value = _num(df, "stock_value_purchase")

    with_stock = stock > 0
    total_stock = float(stock[with_stock].sum())
    total_daily = float(daily[with_stock].sum())

    dead = status == STATUS_DEAD
    very_slow = status == STATUS_VERY_SLOW
    slow = status == STATUS_SLOW

    has_money = bool(value.notna().any())

    def money(mask):
        return float(value[mask].fillna(0).sum()) if has_money else None

    return {
        "has_data": True,
        "has_receipts": bool(df.get("last_receipt_date", pd.Series(dtype=object)).notna().any()),
        "has_money": has_money,
        "period": df["turnover_period"].iloc[0] if "turnover_period" in df.columns else "",
        "sku_with_stock": int(with_stock.sum()),
        "stock_units": total_stock,
        "company_turnover_days": (total_stock / total_daily) if total_daily > 0 else None,
        "dead_sku": int(dead.sum()),
        "dead_units": float(stock[dead].sum()),
        "dead_share": float(stock[dead].sum() / total_stock) if total_stock else 0.0,
        "very_slow_sku": int(very_slow.sum()),
        "very_slow_units": float(stock[very_slow].sum()),
        "slow_sku": int(slow.sum()),
        "slow_units": float(stock[slow].sum()),
        "frozen_share": float(stock[dead | very_slow].sum() / total_stock) if total_stock else 0.0,
        "new_sku": int((status == STATUS_NEW).sum()),
        "fast_sku": int((status == STATUS_FAST).sum()),
        "stock_value": money(with_stock),
        "dead_value": money(dead),
        "frozen_value": money(dead | very_slow),
    }


def turnover_by_status(df: pd.DataFrame) -> pd.DataFrame:
    stock = _num(df, "stock_available").fillna(0)
    value = _num(df, "stock_value_purchase")
    g = (
        pd.DataFrame({
            "status": df["turnover_status"],
            "sku": 1,
            "stock": stock,
            "value": value,
        })
        .groupby("status", as_index=False)
        .agg(sku=("sku", "sum"), stock=("stock", "sum"), value=("value", lambda s: s.sum(min_count=1)))
    )
    g["order"] = g["status"].map({s: i for i, s in enumerate(STATUS_ORDER)}).fillna(99)
    total = g["stock"].sum()
    g["stock_share"] = g["stock"] / total if total else 0.0
    return g.sort_values("order").drop(columns="order")


def turnover_by_category(df: pd.DataFrame) -> pd.DataFrame:
    stock = _num(df, "stock_available").fillna(0)
    daily = _num(df, "avg_daily_sales").fillna(0)
    status = df["turnover_status"].fillna("")
    work = pd.DataFrame({
        "cat_name": df.get("cat_name", pd.Series("Без категории", index=df.index)).fillna("Без категории"),
        "sku": (stock > 0).astype(int),
        "stock": stock,
        "daily": np.where(stock > 0, daily, 0.0),
        "dead": np.where(status == STATUS_DEAD, stock, 0.0),
        "frozen": np.where(status.isin([STATUS_DEAD, STATUS_VERY_SLOW]), stock, 0.0),
        "value": _num(df, "stock_value_purchase"),
    })
    g = work.groupby("cat_name", as_index=False).agg(
        sku=("sku", "sum"), stock=("stock", "sum"), daily=("daily", "sum"),
        dead=("dead", "sum"), frozen=("frozen", "sum"),
        value=("value", lambda s: s.sum(min_count=1)),
    )
    g = g[g["stock"] > 0].copy()
    g["turnover_days"] = np.where(g["daily"] > 0, g["stock"] / g["daily"], np.nan)
    g["frozen_share"] = np.where(g["stock"] > 0, g["frozen"] / g["stock"], 0.0)
    return g.sort_values("stock", ascending=False)


def _fmt(n) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—"
    return f"{n:,.0f}".replace(",", " ")


def build_turnover_insights(df: pd.DataFrame) -> dict:
    """{"findings": [(level, text)], "actions": [(title, text)]}"""
    k = turnover_kpis(df)
    if not k.get("has_data"):
        return {"findings": [], "actions": []}

    findings: list[tuple[str, str]] = []
    actions: list[tuple[str, str]] = []

    stock = _num(df, "stock_available").fillna(0)
    status = df["turnover_status"].fillna("")

    if k["company_turnover_days"] is not None:
        d = k["company_turnover_days"]
        level = "ok" if d <= NORMAL_DAYS else ("warn" if d <= SLOW_DAYS else "bad")
        findings.append((level,
            f"Средняя оборачиваемость запаса — {_fmt(d)} дн.: при текущем темпе продаж "
            f"доступного остатка ({_fmt(k['stock_units'])} шт., {k['sku_with_stock']} SKU) хватит "
            f"примерно на {_fmt(d / 30)} мес."))

    if k["dead_sku"]:
        money = f", {_fmt(k['dead_value'])} ₽ по закупке" if k["has_money"] else ""
        findings.append(("bad",
            f"Неликвид: {k['dead_sku']} SKU без единой продажи за период держат "
            f"{_fmt(k['dead_units'])} шт. ({k['dead_share']:.0%} остатка){money}."))
        actions.append(("Неликвид",
            "Разобрать список «Неликвид» по производителям: возврат/обмен у поставщика, "
            "перемещение в магазин с лучшими продажами категории, уценка или комплект с "
            "ходовым товаром. Остановить дозаказ этих позиций."))

    if k["very_slow_sku"]:
        findings.append(("warn",
            f"Очень медленные позиции: {k['very_slow_sku']} SKU, {_fmt(k['very_slow_units'])} шт. — "
            f"остатка хватит больше чем на год."))
        actions.append(("Очень медленная оборачиваемость",
            "Не дозаказывать до снижения запаса ниже ROP; проверить выкладку и цену "
            "относительно аналогов; для сезонных позиций — запланировать акцию в сезон."))

    if k["frozen_share"] >= 0.3:
        findings.append(("bad",
            f"{k['frozen_share']:.0%} всего остатка в штуках приходится на неликвид и очень медленные "
            f"позиции — значительная часть запаса «заморожена»."))

    # Поставки, которые не продаются
    dsr = _num(df, "days_since_receipt")
    st = _num(df, "sell_through")
    stale = (dsr > STALE_RECEIPT_DAYS) & (st < LOW_SELL_THROUGH) & (stock > 0)
    if stale.any():
        top = df.loc[stale].assign(_s=stock[stale]).sort_values("_s", ascending=False)["fullname"].head(5)
        findings.append(("warn",
            f"{int(stale.sum())} SKU: с последнего прихода прошло больше {STALE_RECEIPT_DAYS} дн., "
            f"а продано меньше {LOW_SELL_THROUGH:.0%} партии. Крупнейшие: " + "; ".join(top) + "."))
        actions.append(("Партии без реализации",
            "Для позиций с низкой реализацией партии пересмотреть объём следующих закупок "
            "(брать меньшими партиями) и зафиксировать с менеджером причины: цена, выкладка, спрос."))

    if k["new_sku"]:
        findings.append(("info",
            f"Новые приходы без продаж: {k['new_sku']} SKU (партии моложе {NEW_RECEIPT_DAYS} дн.) — "
            f"оценивать их оборачиваемость пока рано."))
        actions.append(("Новые поступления",
            f"Проконтролировать новые поступления через {NEW_RECEIPT_DAYS} дн. после прихода: "
            "выставлены ли в магазинах, есть ли на сайте, первые продажи."))

    fast = (status == STATUS_FAST)
    if "stock_status" in df.columns:
        risk = fast & df["stock_status"].isin(["Дефицит", "Ниже ROP, заказ закрывает"])
        if risk.any():
            findings.append(("warn",
                f"{int(risk.sum())} быстрых позиций уже ниже точки заказа (ROP) — риск потерять продажи."))
            actions.append(("Ходовой товар",
                "Быстрые позиции ниже ROP — в приоритет закупки (см. колонку «Нужно заказать»)."))

    cats = turnover_by_category(df)
    cats_known = cats.dropna(subset=["turnover_days"])
    if len(cats_known) >= 2:
        worst = cats_known.sort_values("turnover_days", ascending=False).iloc[0]
        best = cats_known.sort_values("turnover_days").iloc[0]
        findings.append(("info",
            f"Самая медленная категория — «{worst['cat_name']}» ({_fmt(worst['turnover_days'])} дн.), "
            f"самая быстрая — «{best['cat_name']}» ({_fmt(best['turnover_days'])} дн.)."))

    findings.extend(abc_turnover_findings(df))

    if not k["has_receipts"]:
        findings.append(("info",
            "Приходы ещё не загружены — даты партий и реализация партии не посчитаны. "
            "Загрузите файл приходов в cosmo: «Остатки и приходы»."))
    if not k["has_money"]:
        findings.append(("info",
            "Закупочных цен пока нет — анализ в штуках. Суммы по закупке появятся "
            "автоматически, когда в выгрузке приходов будет колонка «Цена закупки без НДС»."))

    if not actions:
        actions.append(("Поддерживать", "Критичных отклонений нет — поддерживать текущие правила закупки."))

    return {"findings": findings, "actions": actions}


ABC_ORDER = ["A", "B", "C", "Новый", "Редкий", "Без продаж"]
MATRIX_STATUSES = [STATUS_FAST, STATUS_NORMAL, STATUS_SLOW, STATUS_VERY_SLOW, STATUS_DEAD, STATUS_NEW]


def abc_turnover_matrix(df: pd.DataFrame) -> dict:
    """ABC × статус оборачиваемости: штуки и SKU (только позиции с остатком)."""
    if df is None or df.empty or "turnover_status" not in df.columns or "abc" not in df.columns:
        return {"rows": [], "cols": [], "stock": [], "sku": []}
    stock = _num(df, "stock_available").fillna(0)
    work = pd.DataFrame({"abc": df["abc"].fillna("—"), "status": df["turnover_status"], "stock": stock})
    work = work[work["stock"] > 0]
    rows = [a for a in ABC_ORDER if a in set(work["abc"])] + sorted(set(work["abc"]) - set(ABC_ORDER))
    cols = [c for c in MATRIX_STATUSES if c in set(work["status"])]
    g = work.groupby(["abc", "status"]).agg(stock=("stock", "sum"), sku=("stock", "size"))
    stock_m = [[float(g["stock"].get((r, c), 0)) for c in cols] for r in rows]
    sku_m = [[int(g["sku"].get((r, c), 0)) for c in cols] for r in rows]
    return {"rows": rows, "cols": cols, "stock": stock_m, "sku": sku_m}


def abc_turnover_findings(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Выводы по матрице ABC × оборачиваемость."""
    if df is None or df.empty or "abc" not in df.columns or "turnover_status" not in df.columns:
        return []
    stock = _num(df, "stock_available").fillna(0)
    abc = df["abc"].fillna("")
    st = df["turnover_status"].fillna("")
    out = []
    a_slow = (abc.isin(["A", "B"])) & st.isin([STATUS_SLOW, STATUS_VERY_SLOW]) & (stock > 0)
    if a_slow.any():
        names = df.loc[a_slow].assign(_s=stock[a_slow]).sort_values("_s", ascending=False)["fullname"].head(3)
        out.append(("warn",
            f"Товары A/B с медленной оборачиваемостью: {int(a_slow.sum())} SKU, {_fmt(stock[a_slow].sum())} шт. "
            f"Продаются хорошо, но запас явно избыточный — сократить следующий заказ. Крупнейшие: "
            + "; ".join(names) + "."))
    c_fast = (abc == "C") & (st == STATUS_FAST)
    if c_fast.any():
        out.append(("info",
            f"Товары C с быстрой оборачиваемостью: {int(c_fast.sum())} SKU — продажи малые, но запас "
            f"тоже мал; проверить, не теряем ли продажи из-за недозаказа."))
    a_dead = (abc.isin(["A", "B"])) & (stock <= 0)
    if a_dead.any():
        out.append(("bad",
            f"Товары A/B без доступного остатка: {int(a_dead.sum())} SKU — прямые потери выручки, "
            f"в приоритет закупки."))
    return out
