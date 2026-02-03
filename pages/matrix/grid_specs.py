# matrix/grid_specs.py
from __future__ import annotations

from typing import Any, Dict, List


def get_matrix_column_defs() -> List[Dict[str, Any]]:
    """ColumnDefs для таблицы матрицы (Dash AgGrid)."""
    return [
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
                    "minWidth": 240,
                    "type": "leftAligned",
                    "cellClass": "ag-firstcol-bg",
                    "headerClass": "ag-center-header",
                    "pinned": "left",
                },
                {
                    "headerName": "Артикль",
                    "field": "article",
                    "minWidth": 240,
                    "type": "leftAligned",
                },
                
                {
                    "headerName": "Производитель",
                    "field": "manu",
                    "minWidth": 200,
                    "filter": True,
                    "type": "leftAligned",
        
                },

                {
                    "headerName": "Штрихкода",
                    "field": "barcode",
                    "minWidth": 240,
                    "type": "leftAligned",
                     "columnGroupShow": "open",
                },
                {
                    "headerName": "Категория",
                    "field": "cat_name",
                    "minWidth": 220,
                    "type": "leftAligned",
                     "columnGroupShow": "open",
                },
                {
                    "headerName": "Подкатегория",
                    "field": "sc_name",
                    "minWidth": 220,
                    "type": "leftAligned",
          
                },
            ],
        },
        {
            "headerName": "Статистика",
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
                    "headerName": "Доля выручка",
                    "field": "share",
                    "valueFormatter": {"function": "FormatPercent(params.value)"},
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                    "width": 100,
                },
                {
                    "headerName": "Ср. выручка",
                    "field": "mean_amount",
                    "valueFormatter": {"function": "RUB(params.value)"},
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                },
                {
                    "headerName": "Доля в ср выручке",
                    "field": "share_mean",
                    "valueFormatter": {"function": "FormatPercent(params.value)"},
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                    "width": 100,
                     "columnGroupShow": "open",
                },
      
                {
                    "headerName": "Ср. μ (ед)",
                    "field": "mean_month",
                    "width": 140,
                    "cellStyle": {"textAlign": "center"},
                    "valueFormatter": {"function": "TwoDecimal(params.value)"},
                    "headerClass": "ag-center-header",
                },
                {
                    "headerName": "Ст откл. σ ",
                    "field": "std_month",
                    "width": 140,
                    "cellStyle": {"textAlign": "center"},
                    "valueFormatter": {"function": "TwoDecimal(params.value)"},
                    "headerClass": "ag-center-header",
                },
                {
                    "headerName": "CV Квар.",
                    "field": "cv",
                    "width": 140,
                    "cellStyle": {"textAlign": "center"},
                    "valueFormatter": {"function": "TwoDecimal(params.value)"},
                    "headerClass": "ag-center-header",
                     "columnGroupShow": "open",
                },
                {
                    "headerName": "Макс. (ед)",
                    "field": "max_month",
                    "width": 140,
                    "cellStyle": {"textAlign": "center"},
                    "valueFormatter": {"function": "TwoDecimal(params.value)"},
                    "headerClass": "ag-center-header",
                },
                {
                    "headerName": "Мин. (ед)",
                    "field": "min_month",
                    "width": 140,
                    "cellStyle": {"textAlign": "center"},
                    "valueFormatter": {"function": "TwoDecimal(params.value)"},
                    "headerClass": "ag-center-header",
                },
            ],
        },
        {
            "headerName": "Даты",
            "groupId": "dates",
            "marryChildren": True,
            "headerClass": "ag-center-header",
            "children": [
                {
                    "headerName": "Нач. период",
                    "field": "min_date",
                    "width": 140,
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                },
                {
                    "headerName": "Конеч. период",
                    "field": "max_date",
                    "width": 140,
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                },
                {
                    "headerName": "Qпер. (мес)",
                    "field": "sales_period_months",
                    "width": 140,
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                },
                {
                    "headerName": "Нулевые периоды (мес)",
                    "field": "missing_months",
                    "width": 140,
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                },
                {
                    "headerName": "Периоды с продажами (мес)",
                    "field": "month_count",
                    "minWidth": 100,
                    "width": 140,
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                },
            ],
        },
        {
            "headerName": "Запасы и стоки (SS и ROP)",
            "groupId": "stock",
            "marryChildren": True,
            "headerClass": "ag-center-header",
            "children": [
                {
                    "headerName": "Страх. запас (ед) (SS)",
                    "field": "ss",
                    "valueFormatter": {"function": "TwoDecimal(params.value)"},
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                },
                {
                    "headerName": "ROP (ед)",
                    "field": "rop",
                    "valueFormatter": {"function": "TwoDecimal(params.value)"},
                    "cellStyle": {"textAlign": "center"},
                    "headerClass": "ag-center-header",
                },
            ],
        },
    ]


def get_matrix_grid_options() -> Dict[str, Any]:
    """dashGridOptions для таблицы матрицы."""
    return {
        "rowSelection": "single",
        "pagination": True,
        "paginationPageSize": 20,
        "suppressRowClickSelection": False,
        "rowClass": "clickable-row",
        "ensureDomOrder": True,
    }
