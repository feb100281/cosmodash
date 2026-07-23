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
        "valueFormatter": {"function": "TwoDecimal(params.value)"},
        "cellStyle": {"textAlign": "center"},
        "headerClass": "ag-center-header",
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


def get_matrix_column_defs(
    df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """
    ColumnDefs для таблицы ассортиментной матрицы.

    df передаётся для динамического формирования колонок по складам.
    """
    column_defs: List[Dict[str, Any]] = [
        {
            "headerName": "item_id",
            "field": "item_id",
            "hide": True,
        },
        {
            "headerName": "Рейтинги",
            "groupId": "ratings",
            "minWidth": 50,
            "marryChildren": True,
            "headerClass": "ag-center-header",
            "children": [
                {
                    "headerName": "ABC",
                    "field": "abc",
                    "width": 90,
                    "type": "leftAligned",
                    "cellClass": "ag-firstcol-bg",
                    "headerClass": "ag-center-header",
                    "pinned": "left",
                },
                {
                    "headerName": "XYZ",
                    "field": "xyz",
                    "width": 90,
                    "type": "leftAligned",
                    "cellClass": "ag-firstcol-bg",
                    "headerClass": "ag-center-header",
                    "pinned": "left",
                },
            ],
        },
        {
            "headerName": "Номенклатура",
            "groupId": "product",
            "marryChildren": True,
            "headerClass": "ag-center-header",
            "openByDefault": False,
            "children": [
                {
                    "headerName": "Номенклатура",
                    "field": "fullname",
                    "minWidth": 235,
                    "type": "leftAligned",
                    "cellClass": "ag-firstcol-bg",
                    "headerClass": "ag-center-header",
                    "pinned": "left",
                },
                {
                    "headerName": "Артикул",
                    "field": "article",
                    "minWidth": 135,
                    "type": "leftAligned",
                },
                {
                    "headerName": "Производитель",
                    "field": "manu",
                    "minWidth": 160,
                    "filter": True,
                    "type": "leftAligned",
                },
                {
                    "headerName": "Штрихкоды",
                    "field": "barcode",
                    "minWidth": 190,
                    "type": "leftAligned",
                    "columnGroupShow": "open",
                },
                {
                    "headerName": "Категория",
                    "field": "cat_name",
                    "minWidth": 170,
                    "type": "leftAligned",
                    "columnGroupShow": "open",
                },
                {
                    "headerName": "Подкатегория",
                    "field": "sc_name",
                    "minWidth": 170,
                    "type": "leftAligned",
                    "columnGroupShow": "open",
                },
            ],
        },
        {
            "headerName": "Статистика продаж",
            "groupId": "stats",
            "marryChildren": True,
            "headerClass": "ag-center-header",
            "children": [
                {
                    "headerName": "Выручка",
                    "field": "amount",
                    "valueFormatter": {"function": "RUB(params.value)"},
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                },
                {
                    "headerName": "Кол-во",
                    "field": "quant",
                    "valueFormatter": {"function": "TwoDecimal(params.value)"},
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                },
                {
                    "headerName": "Доля выручки",
                    "field": "share",
                    "valueFormatter": {"function": "FormatPercent(params.value)"},
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                    "width": 110,
                },
                {
                    "headerName": "Ср. выручка",
                    "field": "mean_amount",
                    "valueFormatter": {"function": "RUB(params.value)"},
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                },
                {
                    "headerName": "Доля в ср. выручке",
                    "field": "share_mean",
                    "valueFormatter": {"function": "FormatPercent(params.value)"},
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                    "width": 125,
                    "columnGroupShow": "open",
                },
                {
                    "headerName": "Ср. μ (ед)",
                    "field": "mean_month",
                    "width": 120,
                    "cellStyle": {"textAlign": "center"},
                    "valueFormatter": {"function": "TwoDecimal(params.value)"},
                    "headerClass": "ag-center-header",
                },
                {
                    "headerName": "Ст. откл. σ",
                    "field": "std_month",
                    "width": 120,
                    "cellStyle": {"textAlign": "center"},
                    "valueFormatter": {"function": "TwoDecimal(params.value)"},
                    "headerClass": "ag-center-header",
                },
                {
                    "headerName": "CV",
                    "field": "cv",
                    "width": 100,
                    "cellStyle": {"textAlign": "center"},
                    "valueFormatter": {"function": "TwoDecimal(params.value)"},
                    "headerClass": "ag-center-header",
                    "columnGroupShow": "open",
                },
                {
                    "headerName": "Макс. (ед)",
                    "field": "max_month",
                    "width": 110,
                    "cellStyle": {"textAlign": "center"},
                    "valueFormatter": {"function": "TwoDecimal(params.value)"},
                    "headerClass": "ag-center-header",
                    "columnGroupShow": "open",
                },
                {
                    "headerName": "Мин. (ед)",
                    "field": "min_month",
                    "width": 110,
                    "cellStyle": {"textAlign": "center"},
                    "valueFormatter": {"function": "TwoDecimal(params.value)"},
                    "headerClass": "ag-center-header",
                    "columnGroupShow": "open",
                },
            ],
        },
        {
            "headerName": "Период продаж",
            "groupId": "dates",
            "marryChildren": True,
            "headerClass": "ag-center-header",
            "openByDefault": False,
            "children": [
                {
                    "headerName": "Нач. период",
                    "field": "min_date",
                    "width": 130,
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                },
                {
                    "headerName": "Конеч. период",
                    "field": "max_date",
                    "width": 130,
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                },
                {
                    "headerName": "Qпер. (мес)",
                    "field": "sales_period_months",
                    "width": 110,
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                },
                {
                    "headerName": "Нулевые периоды",
                    "field": "missing_months",
                    "width": 130,
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                    "columnGroupShow": "open",
                },
                {
                    "headerName": "Периоды с продажами",
                    "field": "month_count",
                    "width": 145,
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                    "columnGroupShow": "open",
                },
            ],
        },
        {
            "headerName": "Параметры запаса",
            "groupId": "stock_params",
            "marryChildren": True,
            "headerClass": "ag-center-header",
            "children": [
                {
                    "headerName": "SS (ед)",
                    "field": "ss",
                    "width": 100,
                    "valueFormatter": {"function": "TwoDecimal(params.value)"},
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                },
                {
                    "headerName": "ROP (ед)",
                    "field": "rop",
                    "width": 100,
                    "valueFormatter": {"function": "TwoDecimal(params.value)"},
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                },
            ],
        },
    ]

    current_stock_children: List[Dict[str, Any]] = [
        {
            "headerName": "Дата остатков",
            "field": "stock_date",
            "width": 125,
            "cellStyle": {"textAlign": "center"},
            "headerClass": "ag-center-header",
        },
        _qty_col("Доступно", "stock_available", width=110),
        _qty_col("Заказано", "stock_ordered", width=110),

        {
            "headerName": "Покрытие, мес.",
            "field": "stock_cover_months",
            "width": 125,
            "valueFormatter": {"function": "TwoDecimal(params.value)"},
            "cellStyle": {"textAlign": "center"},
            "headerClass": "ag-center-header",
        },
        {
            "headerName": "С заказами, мес.",
            "field": "stock_cover_months_total",
            "width": 135,
            "valueFormatter": {"function": "TwoDecimal(params.value)"},
            "cellStyle": {"textAlign": "center"},
            "headerClass": "ag-center-header",
        },
        _qty_col("Δ доступно к ROP", "stock_vs_rop", width=135),
        _qty_col("Нужно заказать", "order_need", width=125),
        {
            "headerName": "Статус",
            "field": "stock_status",
            "minWidth": 190,
            "type": "leftAligned",
            "headerClass": "ag-center-header",
        },
    ]

    # Складские колонки скрыты до раскрытия группы "+"
    current_stock_children.extend(
        _warehouse_columns(
            df,
            "stock_wh::",
        )
    )

    current_stock_children.extend(
        _warehouse_columns(
            df,
            "ordered_wh::",
            suffix=" — заказ",
        )
    )

    column_defs.append(
        {
            "headerName": "Текущие остатки",
            "groupId": "current_stock",
            "marryChildren": True,
            "headerClass": "ag-center-header",
            "openByDefault": False,
            "children": current_stock_children,
        }
    )

    return column_defs


def get_matrix_grid_options() -> Dict[str, Any]:
    """dashGridOptions для таблицы матрицы."""
    return {
        "rowSelection": "single",
        "pagination": True,
        "paginationPageSize": 25,
        "paginationPageSizeSelector": [25, 50, 100],
        "rowHeight": 34,
        "headerHeight": 36,
        "groupHeaderHeight": 34,
        "suppressRowClickSelection": False,
        "rowClass": "clickable-row",
        "ensureDomOrder": True,
        "animateRows": False,
    }
