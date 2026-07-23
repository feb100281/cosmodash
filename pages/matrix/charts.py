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

def build_stock_sunburst(
    df: pd.DataFrame,
    value_field: str = "stock_available",
) -> go.Figure:
    """
    Структура текущих запасов:

        Все остатки
            ↓
        Категория
            ↓
        Подкатегория
            ↓
        Номенклатура

    В секторах показывается КОЛИЧЕСТВО, а не процент.
    Процент родительского уровня остаётся только в hover.
    """

    items = _prepare_stock_items(df)

    if items.empty:
        return empty_sunburst()

    allowed_fields = {
        "stock_available",
        "stock_ordered",
        "stock_total",
    }

    if value_field not in allowed_fields:
        value_field = "stock_available"

    # ================================================================
    # Метрика
    # ================================================================

    items[value_field] = (
        pd.to_numeric(
            items[value_field],
            errors="coerce",
        )
        .fillna(0)
    )

    items = items[
        items[value_field] > 0
    ].copy()

    if items.empty:

        labels = {
            "stock_available": "Нет товаров с доступным остатком",
            "stock_ordered": "Нет товаров в заказах",
            "stock_total": "Нет товаров в остатках и заказах",
        }

        return empty_sunburst(
            labels[value_field]
        )

    metric_titles = {
        "stock_available": "Доступный остаток",
        "stock_ordered": "Заказано",
        "stock_total": "Остаток с заказами",
    }

    metric_title = metric_titles[value_field]

    # ================================================================
    # Контрольные итоги
    # ================================================================

    total_value = float(
        items[value_field].sum()
    )

    total_available = float(
        items["stock_available"].sum()
    )

    total_ordered = float(
        items["stock_ordered"].sum()
    )

    total_stock = float(
        items["stock_total"].sum()
    )

    total_sku = int(
        len(items)
    )

    # ================================================================
    # Палитра
    #
    # Сдержанные цвета для бизнес-дашборда.
    # Категория получает свой основной цвет.
    # Подкатегории и SKU наследуют оттенок своей категории.
    # ================================================================

    category_palette = [
        "#2563EB",  # blue
        "#0F766E",  # teal
        "#D97706",  # amber
        "#7C3AED",  # violet
        "#DC2626",  # red
        "#0891B2",  # cyan
        "#4F46E5",  # indigo
        "#65A30D",  # lime
        "#BE185D",  # pink
        "#475569",  # slate
        "#0284C7",
        "#A16207",
    ]

    def hex_to_rgba(
        hex_color: str,
        alpha: float,
    ) -> str:
        """
        HEX -> rgba().
        """

        value = hex_color.lstrip("#")

        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)

        return f"rgba({r}, {g}, {b}, {alpha})"

    # ================================================================
    # Nodes
    # ================================================================

    ids: list[str] = []
    labels: list[str] = []
    parents: list[str] = []
    values: list[float] = []
    colors: list[str] = []

    customdata: list[list[Any]] = []

    # ================================================================
    # ROOT
    # ================================================================

    root_id = "ROOT"

    ids.append(root_id)
    labels.append("Все остатки")
    parents.append("")
    values.append(total_value)

    # Тёмный нейтральный центр
    colors.append("#F8FAFC")

    customdata.append(
        [
            total_sku,
            total_available,
            total_ordered,
            total_stock,
            "",
            "",
            "",
        ]
    )

    # ================================================================
    # CATEGORY
    # ================================================================

    categories = (
        items
        .groupby(
            "cat_name",
            as_index=False,
        )
        .agg(
            value=(
                value_field,
                "sum",
            ),
            available=(
                "stock_available",
                "sum",
            ),
            ordered=(
                "stock_ordered",
                "sum",
            ),
            total=(
                "stock_total",
                "sum",
            ),
            sku=(
                "_item_key",
                "nunique",
            ),
        )
        .sort_values(
            "value",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    for category_index, (_, category) in enumerate(
        categories.iterrows()
    ):

        category_name = category["cat_name"]

        category_id = (
            f"CAT::{category_name}"
        )

        base_color = category_palette[
            category_index % len(category_palette)
        ]

        # ------------------------------------------------------------
        # Категория
        # ------------------------------------------------------------

        ids.append(category_id)
        labels.append(category_name)
        parents.append(root_id)

        values.append(
            float(category["value"])
        )

        colors.append(base_color)

        customdata.append(
            [
                int(category["sku"]),
                float(category["available"]),
                float(category["ordered"]),
                float(category["total"]),
                "",
                "",
                "",
            ]
        )

        category_items = items[
            items["cat_name"] == category_name
        ]

        # ============================================================
        # SUBCATEGORY
        # ============================================================

        subcategories = (
            category_items
            .groupby(
                "sc_name",
                as_index=False,
            )
            .agg(
                value=(
                    value_field,
                    "sum",
                ),
                available=(
                    "stock_available",
                    "sum",
                ),
                ordered=(
                    "stock_ordered",
                    "sum",
                ),
                total=(
                    "stock_total",
                    "sum",
                ),
                sku=(
                    "_item_key",
                    "nunique",
                ),
            )
            .sort_values(
                "value",
                ascending=False,
            )
        )

        for _, subcategory in subcategories.iterrows():

            subcategory_name = subcategory["sc_name"]

            subcategory_id = (
                f"{category_id}"
                f"::SC::{subcategory_name}"
            )

            ids.append(subcategory_id)
            labels.append(subcategory_name)
            parents.append(category_id)

            values.append(
                float(subcategory["value"])
            )

            # Чуть светлее категории
            colors.append(
                hex_to_rgba(
                    base_color,
                    0.72,
                )
            )

            customdata.append(
                [
                    int(subcategory["sku"]),
                    float(subcategory["available"]),
                    float(subcategory["ordered"]),
                    float(subcategory["total"]),
                    "",
                    "",
                    "",
                ]
            )

            # ========================================================
            # SKU
            # ========================================================

            sku_items = (
                category_items[
                    category_items["sc_name"]
                    == subcategory_name
                ]
                .sort_values(
                    value_field,
                    ascending=False,
                )
            )

            for _, item in sku_items.iterrows():

                item_key = item["_item_key"]

                item_node_id = (
                    f"{subcategory_id}"
                    f"::ITEM::{item_key}"
                )

                ids.append(item_node_id)

                labels.append(
                    item["fullname"]
                )

                parents.append(
                    subcategory_id
                )

                values.append(
                    float(
                        item[value_field]
                    )
                )

                # Самый лёгкий оттенок
                colors.append(
                    hex_to_rgba(
                        base_color,
                        0.48,
                    )
                )

                customdata.append(
                    [
                        1,
                        float(
                            item["stock_available"]
                        ),
                        float(
                            item["stock_ordered"]
                        ),
                        float(
                            item["stock_total"]
                        ),
                        _safe_text(
                            item["abc"],
                            "—",
                        ),
                        _safe_text(
                            item["xyz"],
                            "—",
                        ),
                        _safe_text(
                            item["stock_status"],
                            "—",
                        ),
                    ]
                )

    # ================================================================
    # Figure
    # ================================================================

    fig = go.Figure()

    fig.add_trace(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,

            branchvalues="total",

            # Категория + подкатегория.
            # До товара проваливаемся кликом.
            maxdepth=2,

            # --------------------------------------------------------
            # Подписи
            #
            # Было:
            #   label + percent parent
            #
            # Теперь:
            #   название
            #   11 153 ед.
            # --------------------------------------------------------

            texttemplate=(
                "<b>%{label}</b>"
                "<br>"
                "%{value:,.0f} ед."
            ),

            insidetextorientation="auto",

            customdata=customdata,

            # --------------------------------------------------------
            # Hover
            # --------------------------------------------------------

            hovertemplate=(
                "<b>%{label}</b>"
                "<br>"
                "────────────────────"
                "<br>"
                f"{metric_title}: "
                "<b>%{value:,.0f} ед.</b>"
                "<br>"
                "Доля уровня: "
                "<b>%{percentParent:.1%}</b>"
                "<br><br>"
                "SKU: "
                "<b>%{customdata[0]:,.0f}</b>"
                "<br>"
                "Доступно: "
                "<b>%{customdata[1]:,.0f} ед.</b>"
                "<br>"
                "Заказано: "
                "<b>%{customdata[2]:,.0f} ед.</b>"
                "<br>"
                "Всего: "
                "<b>%{customdata[3]:,.0f} ед.</b>"
                "<br><br>"
               
                "<extra></extra>"
            ),

            marker=dict(
                colors=colors,
                line=dict(
                    color="rgba(255,255,255,0.90)",
                    width=2,
                ),
            ),

            hoverlabel=dict(
                bgcolor="white",
                bordercolor="#CBD5E1",
                font=dict(
                    family="Arial, sans-serif",
                    size=13,
                    color="#1E293B",
                ),
            ),
        )
    )

    # ================================================================
    # Layout
    # ================================================================

    fig.update_layout(
        height=680,

        paper_bgcolor="white",
        plot_bgcolor="white",

        margin=dict(
            l=20,
            r=20,
            t=70,
            b=20,
        ),

        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#1E293B",
        ),

        title=dict(
            text=(
                "<b>Структура текущих запасов</b>"
                "<br>"
                "<span style='font-size:12px;color:#64748B'>"
                f"{metric_title}"
                f"  ·  {total_value:,.0f} ед."
                f"  ·  {total_sku:,} SKU"
                "</span>"
            ),
            x=0.01,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),

        uniformtext=dict(
            minsize=10,
            mode="hide",
        ),
    )

    return fig