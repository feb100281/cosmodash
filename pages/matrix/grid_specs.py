# pages/matrix/grid_specs.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def _qty_col(
    header: str,
    field: str,
    *,
    width: int = 120,
    column_group_show: Optional[str] = None,
) -> Dict[str, Any]:
    col: Dict[str, Any] = {
        "headerName": header,
        "field": field,
        "width": width,
        "type": "rightAligned",
        "valueFormatter": {"function": "IntOrDash(params.value)"},
    }

    if column_group_show:
        col["columnGroupShow"] = column_group_show

    return col


def _warehouse_columns(
    df: Optional[pd.DataFrame],
    prefix: str,
    *,
    suffix: str = "",
) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    columns = sorted(
        [
            str(column)
            for column in df.columns
            if str(column).startswith(prefix)
        ],
        key=str.lower,
    )

    result: List[Dict[str, Any]] = []

    for column in columns:
        warehouse = column.split("::", 1)[1]
        header = f"{warehouse}{suffix}"

        result.append(
            _qty_col(
                header=header,
                field=column,
                width=150,
                column_group_show="open",
            )
        )

    return result


def _num(header, field, fn="TwoDecimal", width=110, *, hidden=False, tooltip=None, **extra):
    col: Dict[str, Any] = {
        "headerName": header,
        "field": field,
        "width": width,
        "type": "rightAligned",
        "valueFormatter": {"function": f"{fn}(params.value)"},
    }
    if hidden:
        col["columnGroupShow"] = "open"
    if tooltip:
        col["headerTooltip"] = tooltip
    col.update(extra)
    return col


def _txt(header, field, width=160, *, hidden=False, tooltip=None, **extra):
    col: Dict[str, Any] = {"headerName": header, "field": field, "minWidth": width, "width": width}
    if hidden:
        col["columnGroupShow"] = "open"
    if tooltip:
        col["headerTooltip"] = tooltip
    col.update(extra)
    return col


def _group(header, group_id, children):
    return {
        "headerName": header,
        "groupId": group_id,
        "marryChildren": True,
        "openByDefault": False,
        "headerClass": "ag-center-header mx-group-header",
        "children": children,
    }


_BADGE_ABC = {
    "styleConditions": [
        {"condition": "params.value === 'A'", "style": {"color": "#1F5E4E", "fontWeight": 700}},
        {"condition": "params.value === 'B'", "style": {"color": "#3D7A67", "fontWeight": 600}},
        {"condition": "params.value === 'C'", "style": {"color": "#9A6100", "fontWeight": 600}},
        {"condition": "params.value === 'X'", "style": {"color": "#1F5E4E", "fontWeight": 700}},
        {"condition": "params.value === 'Y'", "style": {"color": "#3D7A67", "fontWeight": 600}},
        {"condition": "params.value === 'Z'", "style": {"color": "#9A6100", "fontWeight": 600}},
    ],
    "defaultStyle": {"color": "var(--mantine-color-dimmed)"},
}

_STOCK_STATUS_STYLE = {
    "styleConditions": [
        {"condition": "params.value === 'Дефицит' || params.value === 'Нет остатка'",
         "style": {"color": "#B4442E", "fontWeight": 600}},
        {"condition": "params.value && params.value.startsWith('Избыток')",
         "style": {"color": "#9A6100", "fontWeight": 600}},
        {"condition": "params.value && params.value.startsWith('Без продаж')",
         "style": {"color": "#7B4437", "fontWeight": 600}},
        {"condition": "params.value === 'Достаточно'", "style": {"color": "#3D7A67"}},
    ],
}

_TURNOVER_STATUS_STYLE = {
    "styleConditions": [
        {"condition": "params.value && params.value.startsWith('Неликвид')",
         "style": {"color": "#7B4437", "fontWeight": 600}},
        {"condition": "params.value && params.value.startsWith('Очень')",
         "style": {"color": "#9A6100", "fontWeight": 600}},
        {"condition": "params.value && params.value.startsWith('Медленная')",
         "style": {"color": "#B08A1E"}},
        {"condition": "params.value && params.value.startsWith('Быстрая')",
         "style": {"color": "#1F5E4E", "fontWeight": 600}},
        {"condition": "params.value && params.value.startsWith('Новый')",
         "style": {"color": "#2F75B5"}},
    ],
}


def get_matrix_column_defs(
    df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """
    По умолчанию видны ключевые колонки; остальные раскрываются
    стрелкой в заголовке группы.
    """
    warehouses = _warehouse_columns(df, "stock_wh::") + _warehouse_columns(df, "ordered_wh::", suffix=" — заказ")

    return [
        {"headerName": "item_id", "field": "item_id", "hide": True},
        {
            "headerName": "ABC",
            "field": "abc",
            "width": 72,
            "pinned": "left",
            "cellStyle": _BADGE_ABC,
            "headerTooltip": "Вклад в выручку",
        },
        {
            "headerName": "XYZ",
            "field": "xyz",
            "width": 72,
            "pinned": "left",
            "cellStyle": _BADGE_ABC,
            "headerTooltip": "Стабильность спроса",
        },
        {
            "headerName": "Номенклатура",
            "field": "fullname",
            "minWidth": 260,
            "width": 300,
            "pinned": "left",
            "tooltipField": "fullname",
            "cellStyle": {"fontWeight": 500},
        },
        _group("Товар", "product", [
            _txt("Производитель", "manu", 170),
            _txt("Артикул", "article", 130, hidden=True),
            _txt("Категория", "cat_name", 170, hidden=True),
            _txt("Подкатегория", "sc_name", 170, hidden=True),
            _txt("Штрихкоды", "barcode", 190, hidden=True, tooltipField="barcode"),
        ]),
        _group("Продажи за период", "stats", [
            _num("Выручка", "amount", "RUB", 125),
            _num("Кол-во", "quant", "IntOrDash", 90),
            _num("Доля выручки", "share", "FormatPercent", 110),
            _num("Ср. выручка / мес.", "mean_amount", "RUB", 130, hidden=True),
            _num("Доля в ср. выручке", "share_mean", "FormatPercent", 130, hidden=True),
            _num("Ср. μ, ед./мес.", "mean_month", "TwoDecimal", 115, hidden=True),
            _num("σ", "std_month", "TwoDecimal", 90, hidden=True, tooltip="Стандартное отклонение продаж в месяц"),
            _num("CV", "cv", "TwoDecimal", 80, hidden=True, tooltip="Коэффициент вариации σ/μ"),
            _num("Макс., ед.", "max_month", "TwoDecimal", 95, hidden=True),
            _num("Мин., ед.", "min_month", "TwoDecimal", 95, hidden=True),
            _txt("Первая продажа", "min_date", 120, hidden=True),
            _txt("Последняя продажа", "max_date", 130, hidden=True),
            _num("Мес. в продаже", "sales_period_months", "IntOrDash", 115, hidden=True),
            _num("Мес. без продаж", "missing_months", "IntOrDash", 115, hidden=True),
            _num("Мес. с продажами", "month_count", "IntOrDash", 120, hidden=True),
        ]),
        _group("Запас", "current_stock", [
            _num("Доступно", "stock_available", "IntOrDash", 100),
            _num("Покрытие, мес.", "stock_cover_months", "TwoDecimal", 115,
                 tooltip="На сколько месяцев хватит доступного остатка при среднем спросе"),
            _txt("Статус запаса", "stock_status", 180, cellStyle=_STOCK_STATUS_STYLE),
            _num("Нужно заказать", "order_need", "IntOrDash", 120),
            _txt("Дата остатков", "stock_date", 115, hidden=True),
            _num("Заказано", "stock_ordered", "IntOrDash", 100, hidden=True),
            _num("С заказами, мес.", "stock_cover_months_total", "TwoDecimal", 125, hidden=True),
            _num("SS, ед.", "ss", "IntOrDash", 90, hidden=True, tooltip="Страховой запас"),
            _num("ROP, ед.", "rop", "IntOrDash", 90, hidden=True, tooltip="Точка заказа"),
            _num("Δ к ROP", "stock_vs_rop", "TwoDecimal", 100, hidden=True),
            {
                "headerName": "Остатки по штрихкодам",
                "field": "barcode_stocks_display",
                "width": 300,
                "wrapText": True,
                "autoHeight": True,
                "columnGroupShow": "open",
                "cellStyle": {"whiteSpace": "pre-line", "lineHeight": "18px", "paddingTop": "6px", "paddingBottom": "6px"},
            },
            *warehouses,
        ]),
        _group("Оборачиваемость", "turnover", [
            _txt("Статус оборачиваемости", "turnover_status", 200, cellStyle=_TURNOVER_STATUS_STYLE),
            _num("Оборач., дн.", "turnover_days", "IntOrDash", 105,
                 tooltip="Доступный остаток / среднедневные продажи"),
            _num("Реализация партии", "sell_through", "PctOrDash", 130, hidden=True,
                 tooltip="Продано с последнего прихода / (продано с прихода + остаток)"),
            _num("Оборотов в год", "turns_per_year", "TwoDecimal", 115, hidden=True),
            _num("Продажи в день", "avg_daily_sales", "TwoDecimal", 115, hidden=True),
            _txt("Посл. приход", "last_receipt_date", 110, hidden=True),
            _num("Дней с прихода", "days_since_receipt", "IntOrDash", 115, hidden=True),
            _num("Посл. партия, шт.", "last_receipt_qty", "IntOrDash", 125, hidden=True),
            _num("Продано с прихода", "sold_since_receipt", "IntOrDash", 130, hidden=True),
            _num("Пришло за период", "receipt_qty_period", "IntOrDash", 130, hidden=True),
            _txt("Первый приход", "first_receipt_date", 115, hidden=True),
            _num("Цена закупки", "purchase_price", "RUBOrDash", 115, hidden=True),
            _num("Остаток по закупке", "stock_value_purchase", "RUBOrDash", 140, hidden=True),
        ]),
    ]


def get_matrix_grid_options() -> Dict[str, Any]:
    """dashGridOptions для таблицы матрицы."""
    return {
        "rowSelection": "single",
        "pagination": True,
        "paginationPageSize": 25,
        "paginationPageSizeSelector": [25, 50, 100],
        "rowHeight": 34,
        "headerHeight": 38,
        "groupHeaderHeight": 32,
        "tooltipShowDelay": 400,
        "suppressDragLeaveHidesColumns": True,
        "suppressRowClickSelection": False,
        "rowClass": "clickable-row",
        "ensureDomOrder": True,
        "animateRows": False,
    }
