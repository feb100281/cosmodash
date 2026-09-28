# pages/matrix/charts.py

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import plotly.graph_objects as go


# ============================================================================
# Общие helpers
# ============================================================================

def _safe_text(
    value: Any,
    default: str,
) -> str:
    """
    Безопасное преобразование значения в строку.
    """

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    return value if value else default


def _safe_number(
    value: Any,
) -> float:
    """
    Безопасное преобразование в число.
    """

    try:
        result = float(value)

        if math.isnan(result):
            return 0.0

        return result

    except (TypeError, ValueError):
        return 0.0


def empty_sunburst(
    message: str = "Нет данных для отображения",
) -> go.Figure:
    """
    Пустой график.
    """

    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={
            "size": 15,
            "color": "#6B7280",
        },
    )

    fig.update_layout(
        height=620,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20,
        ),
    )

    return fig


# ============================================================================
# Подготовка данных
# ============================================================================

def _prepare_stock_items(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Подготавливает одну строку на один SKU.

    Это важно:
    даже если исходный DataFrame по какой-то причине содержит
    повторяющийся item_id, остаток нельзя суммировать несколько раз.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()

    # ------------------------------------------------------------------
    # Проверяем обязательные поля
    # ------------------------------------------------------------------

    required_columns = [
        "fullname",
        "cat_name",
        "sc_name",
    ]

    for column in required_columns:
        if column not in work.columns:
            work[column] = None

    # ------------------------------------------------------------------
    # Числовые поля
    # ------------------------------------------------------------------

    numeric_columns = [
        "stock_available",
        "stock_ordered",
        "stock_total",
    ]

    for column in numeric_columns:

        if column not in work.columns:
            work[column] = 0.0

        work[column] = (
            pd.to_numeric(
                work[column],
                errors="coerce",
            )
            .fillna(0.0)
        )

    # ------------------------------------------------------------------
    # Текст
    # ------------------------------------------------------------------

    work["cat_name"] = work["cat_name"].apply(
        lambda x: _safe_text(
            x,
            "Без категории",
        )
    )

    work["sc_name"] = work["sc_name"].apply(
        lambda x: _safe_text(
            x,
            "Без подкатегории",
        )
    )

    work["fullname"] = work["fullname"].apply(
        lambda x: _safe_text(
            x,
            "Без наименования",
        )
    )

    if "manu" not in work.columns:
        work["manu"] = "Нет производителя"

    work["manu"] = work["manu"].apply(
        lambda x: _safe_text(
            x,
            "Нет производителя",
        )
    )

    if "abc" not in work.columns:
        work["abc"] = ""

    if "xyz" not in work.columns:
        work["xyz"] = ""

    if "stock_status" not in work.columns:
        work["stock_status"] = ""

    # ------------------------------------------------------------------
    # Уникальный ключ SKU
    # ------------------------------------------------------------------

    if "item_id" in work.columns:

        item_id = pd.to_numeric(
            work["item_id"],
            errors="coerce",
        )

        work["_item_key"] = item_id.apply(
            lambda value: (
                f"id:{int(value)}"
                if pd.notna(value)
                else None
            )
        )

    else:

        work["_item_key"] = None

    # fallback на fullname
    missing_key = work["_item_key"].isna()

    work.loc[
        missing_key,
        "_item_key",
    ] = (
        "name:"
        + work.loc[
            missing_key,
            "fullname",
        ].astype(str)
    )

    # ------------------------------------------------------------------
    # Одна строка на SKU
    #
    # stock_* берём MAX, а не SUM:
    # если строка случайно задублировалась, остаток не удвоится.
    # ------------------------------------------------------------------

    result = (
        work
        .groupby(
            "_item_key",
            as_index=False,
            dropna=False,
        )
        .agg(
            item_id=(
                "item_id",
                "first",
            )
            if "item_id" in work.columns
            else (
                "_item_key",
                "first",
            ),

            fullname=(
                "fullname",
                "first",
            ),

            cat_name=(
                "cat_name",
                "first",
            ),

            sc_name=(
                "sc_name",
                "first",
            ),

            manu=(
                "manu",
                "first",
            ),

            abc=(
                "abc",
                "first",
            ),

            xyz=(
                "xyz",
                "first",
            ),

            stock_status=(
                "stock_status",
                "first",
            ),

            stock_available=(
                "stock_available",
                "max",
            ),

            stock_ordered=(
                "stock_ordered",
                "max",
            ),

            stock_total=(
                "stock_total",
                "max",
            ),
        )
    )

    return result


# ============================================================================
# Sunburst
# ============================================================================

STOCK_PALETTE = [
    "#2F6656", "#9A6100", "#2F75B5", "#7B4437", "#5B9A83", "#8A6F9E",
    "#B08A1E", "#5C7C8A", "#1F5E4E", "#C2703D", "#6C8E3E", "#4A4A4A",
]

STOCK_METRIC_TITLES = {
    "stock_available": "Доступный остаток",
    "stock_ordered": "Заказано",
    "stock_total": "Остаток с заказами",
}


def _lighten(hex_color: str, factor: float) -> str:
    v = hex_color.lstrip("#")
    r, g, b = (int(v[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (255 - c) * factor) for c in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


def _stock_items(df: pd.DataFrame, value_field: str) -> pd.DataFrame:
    items = _prepare_stock_items(df)
    if items.empty:
        return items
    if value_field not in STOCK_METRIC_TITLES:
        value_field = "stock_available"
    items[value_field] = pd.to_numeric(items[value_field], errors="coerce").fillna(0)
    return items[items[value_field] > 0].copy()


def stock_category_summary(df: pd.DataFrame, value_field: str = "stock_available") -> pd.DataFrame:
    """Категории по убыванию метрики: cat_name, value, sku, share, color."""
    if value_field not in STOCK_METRIC_TITLES:
        value_field = "stock_available"
    items = _stock_items(df, value_field)
    if items.empty:
        return pd.DataFrame(columns=["cat_name", "value", "sku", "share", "color"])
    cats = (
        items.groupby("cat_name", as_index=False)
        .agg(value=(value_field, "sum"), sku=("_item_key", "nunique"))
        .sort_values("value", ascending=False)
        .reset_index(drop=True)
    )
    total = float(cats["value"].sum()) or 1.0
    cats["share"] = cats["value"] / total
    cats["color"] = [STOCK_PALETTE[i % len(STOCK_PALETTE)] for i in range(len(cats))]
    return cats


def build_stock_sunburst(
    df: pd.DataFrame,
    value_field: str = "stock_available",
) -> go.Figure:
    """Категория → подкатегория; в центре — итог по метрике."""
    if value_field not in STOCK_METRIC_TITLES:
        value_field = "stock_available"
    items = _stock_items(df, value_field)
    if items.empty:
        return empty_sunburst({
            "stock_available": "Нет товаров с доступным остатком",
            "stock_ordered": "Нет товаров в заказах",
            "stock_total": "Нет товаров в остатках и заказах",
        }[value_field])

    metric_title = STOCK_METRIC_TITLES[value_field]
    cats = stock_category_summary(df, value_field)
    total_value = float(items[value_field].sum())
    total_sku = int(items["_item_key"].nunique())

    ids, labels, parents, values, colors, custom = [], [], [], [], [], []

    ids.append("ROOT")
    labels.append(f"<b>{total_value:,.0f}</b><br>шт.".replace(",", " "))
    parents.append("")
    values.append(total_value)
    colors.append("rgba(0,0,0,0)")
    custom.append([metric_title, total_sku, 1.0])

    for cat in cats.itertuples():
        cat_id = f"CAT::{cat.cat_name}"
        ids.append(cat_id)
        labels.append(str(cat.cat_name))
        parents.append("ROOT")
        values.append(float(cat.value))
        colors.append(cat.color)
        custom.append([str(cat.cat_name), int(cat.sku), float(cat.share)])

        subs = (
            items[items["cat_name"] == cat.cat_name]
            .groupby("sc_name", as_index=False)
            .agg(value=(value_field, "sum"), sku=("_item_key", "nunique"))
            .sort_values("value", ascending=False)
            .reset_index(drop=True)
        )
        for j, sub in enumerate(subs.itertuples()):
            ids.append(f"{cat_id}::SC::{sub.sc_name}")
            labels.append(str(sub.sc_name))
            parents.append(cat_id)
            values.append(float(sub.value))
            colors.append(_lighten(cat.color, min(0.25 + 0.08 * j, 0.6)))
            custom.append([str(cat.cat_name), int(sub.sku), float(sub.value) / total_value])

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            customdata=custom,
            maxdepth=3,
            sort=False,
            insidetextorientation="radial",
            texttemplate="%{label}",
            hovertemplate=(
                "<b>%{label}</b><br>"
                f"{metric_title}: " "<b>%{value:,.0f} шт.</b><br>"
                "Доля в общем запасе: <b>%{percentRoot:.1%}</b><br>"
                "Доля в родителе: <b>%{percentParent:.1%}</b><br>"
                "SKU: <b>%{customdata[1]:,.0f}</b>"
                "<extra></extra>"
            ),
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=1.5)),
            leaf=dict(opacity=1),
            hoverlabel=dict(
                bgcolor="#FFFFFF",
                bordercolor="#D9D9D9",
                font=dict(family="Roboto, Arial, sans-serif", size=12, color="#1F1F1F"),
            ),
        )
    )
    fig.update_layout(
        height=520,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Roboto, Arial, sans-serif", size=12),
        uniformtext=dict(minsize=10, mode="hide"),
        separators=", ",
    )
    return fig
