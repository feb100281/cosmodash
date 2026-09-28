# pages/matrix/turnover_ui.py
"""Вкладка «Оборачиваемость» ассортиментной матрицы."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash import dcc
from dash_iconify import DashIconify

from .turnover import (
    abc_turnover_matrix,
    STATUS_COLORS,
    STATUS_DEAD,
    STATUS_VERY_SLOW,
    build_turnover_insights,
    turnover_by_category,
    turnover_by_status,
    turnover_kpis,
)

TURNOVER_CONTENT_ID = "matrix-turnover-content"
TURNOVER_DOWNLOAD_BTN_ID = "matrix-turnover-download-btn"
TURNOVER_DOWNLOAD_ID = "matrix-turnover-download"
TURNOVER_STATUS_MS_ID = "matrix-turnover-status-ms"
TURNOVER_GRAPH_ID = "matrix-turnover-status-graph"
TURNOVER_RESET_ID = "matrix-turnover-reset"
ABC_MS_ID = "matrix-abc-ms"
HEATMAP_ID = "matrix-abc-turnover-heatmap"
HEATMAP_RESET_ID = "matrix-abc-turnover-reset"
HEATMAP_NOTE_ID = "matrix-abc-turnover-note"
ITEMS_GRID_ID = {"type": "mx-grid", "index": "turnover-items"}
ITEMS_TITLE_ID = "matrix-turnover-items-title"
CHART_HEIGHT = 340


def themed_figure(fig: go.Figure, dark: bool = False) -> go.Figure:
    text = "#E9ECEF" if dark else "#1F1F1F"
    muted = "#909296" if dark else "#8A8A8A"
    grid = "#373A40" if dark else "#E6EBF0"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text),
        title_font_color=text,
        legend_font_color=text,
    )
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid, tickfont_color=muted, title_font_color=muted)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid, tickfont_color=text)
    for trace in fig.data:
        if trace.type == "sunburst":
            trace.marker.line = dict(color="#1A1B1E" if dark else "#FFFFFF", width=1)
            trace.outsidetextfont = dict(color=text)
    if fig.layout.title and fig.layout.title.text:
        fig.layout.title.text = fig.layout.title.text.replace("color:#64748B", f"color:{muted}")
    return fig


def _fmt(v, suffix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:,.0f}".replace(",", " ") + suffix


def _card(label, value, unit, note, icon, color, bg):
    return dmc.Paper(
        withBorder=True, radius=0, px=14, py=10,
        style={"minHeight": "78px", "borderColor": "var(--mantine-color-default-border)", "borderTop": "2px solid #2F6656"},
        children=dmc.Group(
            gap="sm", wrap="nowrap", align="center",
            children=[
                dmc.Center(w=34, h=34, style={"backgroundColor": bg, "flex": "0 0 34px", "borderRadius": "4px"},
                           children=DashIconify(icon=icon, width=21, color=color)),
                dmc.Stack(gap=1, children=[
                    dmc.Text(label, size="xs", c="dimmed", fw=500),
                    dmc.Group(gap=5, align="baseline", children=[
                        dmc.Text(value, size="lg", fw=700, lh=1.05, c="teal.8"),
                        dmc.Text(unit, size="xs", c="dimmed"),
                    ]),
                    dmc.Text(note, size="xs", c="dimmed"),
                ]),
            ],
        ),
    )


def _status_chart(df: pd.DataFrame, dark: bool = False, selected=None) -> go.Figure:
    st = turnover_by_status(df)
    st = st[st["stock"] > 0]
    selected = set(selected or [])
    opacity = [1.0 if not selected or s in selected else 0.3 for s in st["status"]]
    fig = go.Figure(
        go.Bar(
            x=st["stock"],
            y=st["status"],
            orientation="h",
            customdata=st["status"],
            marker=dict(
                color=[STATUS_COLORS.get(s, "#8A8A8A") for s in st["status"]],
                opacity=opacity,
            ),
            text=[f"{_fmt(v)} шт. · {sh:.0%} · {int(n)} SKU"
                  for v, sh, n in zip(st["stock"], st["stock_share"], st["sku"])],
            textposition="outside",
            textangle=0,
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>%{x:,.0f} шт.<br>Клик — показать позиции<extra></extra>",
        )
    )
    max_x = float(st["stock"].max()) if not st.empty else 1.0
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=CHART_HEIGHT,
        bargap=0.35,
        yaxis=dict(autorange="reversed", title=None, automargin=True),
        xaxis=dict(title=None, separatethousands=True, range=[0, max_x * 1.45], showgrid=True),
        font=dict(family="Roboto, Inter, sans-serif", size=12),
        clickmode="event",
        separators=", ",
    )
    return themed_figure(fig, dark)


def abc_heatmap(df: pd.DataFrame, dark: bool = False, cell=None) -> go.Figure:
    """cell = (abc, короткое название статуса, полное название статуса)."""
    return _abc_heatmap(df, dark, [cell[0]] if cell else None, [cell[2]] if cell else None)


def _abc_heatmap(df: pd.DataFrame, dark: bool = False, sel_abc=None, sel_status=None) -> go.Figure:
    m = abc_turnover_matrix(df)
    if not m["rows"]:
        return themed_figure(go.Figure().update_layout(height=260), dark)
    short = [c.split(" (")[0].replace(", продаж ещё нет", "") for c in m["cols"]]
    custom = [[[m["rows"][i], m["cols"][j], m["sku"][i][j]] for j in range(len(m["cols"]))]
              for i in range(len(m["rows"]))]
    text = [[f"{_fmt(v)} шт.<br>{n} SKU" if n else "" for v, n in zip(rv, rn)]
            for rv, rn in zip(m["stock"], m["sku"])]
    fig = go.Figure(go.Heatmap(
        z=m["stock"],
        x=short,
        y=m["rows"],
        text=text,
        texttemplate="%{text}",
        customdata=custom,
        colorscale=[[0, "#F3F8F6"], [0.35, "#9CC7B6"], [1, "#1F5E4E"]],
        showscale=False,
        xgap=3,
        ygap=3,
        hovertemplate="ABC: <b>%{y}</b><br>%{x}<br>%{z:,.0f} шт. · %{customdata[2]} SKU"
                      "<br>Клик — показать позиции<extra></extra>",
    ))
    sel_abc = set(sel_abc or [])
    sel_status = set(sel_status or [])
    for i, a in enumerate(m["rows"]):
        for j, c in enumerate(m["cols"]):
            if sel_abc and sel_status and a in sel_abc and c in sel_status:
                fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                              line=dict(color="#C2703D", width=3), fillcolor="rgba(0,0,0,0)")
    fig.update_layout(
        height=60 + 52 * len(m["rows"]),
        margin=dict(l=10, r=10, t=10, b=10),
        clickmode="event",
        xaxis=dict(side="top", title=None, showgrid=False),
        yaxis=dict(autorange="reversed", title=None, showgrid=False),
        font=dict(family="Roboto, Inter, sans-serif", size=12),
        separators=", ",
    )
    return themed_figure(fig, dark)


_LEVEL = {
    "bad": ("tabler:alert-octagon", "#B4442E", "var(--mantine-color-red-light)"),
    "warn": ("tabler:alert-triangle", "#C08A1E", "var(--mantine-color-yellow-light)"),
    "ok": ("tabler:circle-check", "#2F8F6F", "var(--mantine-color-teal-light)"),
    "info": ("tabler:info-circle", "#2F75B5", "var(--mantine-color-blue-light)"),
}


def _insights_block(df, dark=False):
    ins = build_turnover_insights(df)
    findings = [
        dmc.Paper(
            radius=0, px=12, py=8,
            style={"backgroundColor": _LEVEL[level][2], "borderLeft": f"3px solid {_LEVEL[level][1]}"},
            children=dmc.Group(gap="xs", wrap="nowrap", align="flex-start", children=[
                DashIconify(icon=_LEVEL[level][0], width=18, color=_LEVEL[level][1]),
                dmc.Text(text, size="sm"),
            ]),
        )
        for level, text in ins["findings"]
    ]
    actions = [
        dmc.ListItem([dmc.Text(title, fw=700, size="sm", span=True, c="#1F5E4E"), dmc.Text(" — " + text, size="sm", span=True)])
        for title, text in ins["actions"]
    ]
    return dmc.SimpleGrid(cols=2, spacing="md", children=[
        dmc.Paper(withBorder=True, radius=0, p="md", children=[
            dmc.Text("Выводы", fw=700, mb="xs"),
            dmc.Stack(gap=6, children=findings or [dmc.Text("Нет данных", c="dimmed", size="sm")]),
        ]),
        dmc.Paper(withBorder=True, radius=0, p="md", children=[
            dmc.Text("Рекомендации", fw=700, mb="xs"),
            dmc.List(actions, spacing="xs", size="sm"),
        ]),
    ])


def _category_grid(df, class_name):
    cat = turnover_by_category(df)
    cols = [
        {"headerName": "Категория", "field": "cat_name", "minWidth": 220, "pinned": "left"},
        {"headerName": "SKU с остатком", "field": "sku", "valueFormatter": {"function": "IntOrDash(params.value)"}},
        {"headerName": "Остаток, шт.", "field": "stock", "valueFormatter": {"function": "IntOrDash(params.value)"}},
        {"headerName": "Продажи в день", "field": "daily", "valueFormatter": {"function": "TwoDecimal(params.value)"}},
        {"headerName": "Оборач., дн.", "field": "turnover_days", "valueFormatter": {"function": "IntOrDash(params.value)"}, "sort": "desc"},
        {"headerName": "Неликвид, шт.", "field": "dead", "valueFormatter": {"function": "IntOrDash(params.value)"}},
        {"headerName": "Заморожено", "field": "frozen_share", "valueFormatter": {"function": "PctOrDash(params.value)"}},
        {"headerName": "По закупке", "field": "value", "valueFormatter": {"function": "RUBOrDash(params.value)"}},
    ]
    return dag.AgGrid(
        id={"type": "mx-grid", "index": "turnover-cat"},
        rowData=cat.replace({np.nan: None}).to_dict("records"),
        columnDefs=cols,
        defaultColDef={"sortable": True, "filter": True, "resizable": True, "flex": 1, "minWidth": 110},
        dashGridOptions={"rowHeight": 32, "headerHeight": 36},
        className=class_name,
        style={"width": "100%", "height": f"{CHART_HEIGHT}px", "--ag-font-size": "12px"},
    )


ITEMS_FIELDS = ["fullname", "manu", "abc", "turnover_status", "stock_available", "turnover_days",
                "last_receipt_date", "sell_through", "stock_value_purchase"]


def items_rows(df: pd.DataFrame, full: bool = False) -> list:
    """full=True — все позиции выборки с остатком; иначе топ-30 неликвида и очень медленных."""
    if df is None or df.empty:
        return []
    if full:
        top = df[pd.to_numeric(df["stock_available"], errors="coerce").fillna(0) > 0]
    else:
        top = df[df["turnover_status"].isin([STATUS_DEAD, STATUS_VERY_SLOW])]
    top = top.sort_values("stock_available", ascending=False)
    if not full:
        top = top.head(30)
    keep = [c for c in ITEMS_FIELDS if c in top.columns]
    return top[keep].replace({np.nan: None}).to_dict("records")


def items_title(cell=None, n=None) -> str:
    if cell:
        return f"Позиции: {cell[0]} · {cell[1]} · {n} SKU"
    return "Топ-30 позиций, которые держат запас (неликвид и > 365 дн.)"


def cell_note(cell=None) -> str:
    if cell:
        return f"Выбрано: {cell[0]} · {cell[1]} — позиции в таблице ниже"
    return "Доступный остаток, шт. · клик по ячейке — её позиции в таблице ниже"


def _items_grid(df, class_name, rows=None):
    cols = [
        {"headerName": "Номенклатура", "field": "fullname", "minWidth": 260, "pinned": "left"},
        {"headerName": "Производитель", "field": "manu", "minWidth": 150},
        {"headerName": "ABC", "field": "abc", "minWidth": 80, "maxWidth": 100},
        {"headerName": "Статус", "field": "turnover_status", "minWidth": 190},
        {"headerName": "Остаток, шт.", "field": "stock_available", "valueFormatter": {"function": "IntOrDash(params.value)"}},
        {"headerName": "Оборач., дн.", "field": "turnover_days", "valueFormatter": {"function": "IntOrDash(params.value)"}},
        {"headerName": "Посл. приход", "field": "last_receipt_date", "valueFormatter": {"function": "String(params.value)"}},
        {"headerName": "Реализация партии", "field": "sell_through", "valueFormatter": {"function": "PctOrDash(params.value)"}},
        {"headerName": "По закупке", "field": "stock_value_purchase", "valueFormatter": {"function": "RUBOrDash(params.value)"}},
    ]
    return dag.AgGrid(
        id=ITEMS_GRID_ID,
        rowData=rows if rows is not None else items_rows(df),
        columnDefs=cols,
        defaultColDef={"sortable": True, "filter": True, "resizable": True, "flex": 1, "minWidth": 110},
        dashGridOptions={"rowHeight": 32, "headerHeight": 36, "pagination": True,
                         "paginationPageSize": 20, "paginationPageSizeSelector": [20, 50, 100]},
        className=class_name,
        style={"width": "100%", "height": "480px", "--ag-font-size": "12px"},
    )


def build_turnover_panel(
    df: pd.DataFrame,
    class_name: str = "ag-theme-alpine",
    dark: bool | None = None,
    chart_df: pd.DataFrame | None = None,
    selected=None,
    heatmap_df: pd.DataFrame | None = None,
    selected_abc=None,
):
    """chart_df — данные без фильтра по оборачиваемости (график показывает все статусы)."""
    if dark is None:
        dark = class_name.endswith("dark")
    if chart_df is None:
        chart_df = df
    if heatmap_df is None:
        heatmap_df = chart_df
    if df is None or df.empty or "turnover_status" not in df.columns:
        return dmc.Alert("Нет данных для анализа оборачиваемости", color="gray", radius=0)

    k = turnover_kpis(df)
    money_card = (
        _card("Остаток по закупке", _fmt(k["stock_value"]), "₽", "по цене последнего прихода",
              "tabler:currency-rubel", "#1F5E4E", "#E7F1ED")
        if k["has_money"] else
        _card("Заморожено запаса", f"{k['frozen_share']:.0%}", "", "неликвид + > 365 дн.",
              "tabler:snowflake", "#2F75B5", "#EAF2FB")
    )
    cards = dmc.SimpleGrid(cols=5, spacing="sm", children=[
        _card("Оборачиваемость запаса", _fmt(k["company_turnover_days"]), "дн.",
              f"период: {k['period']}", "tabler:refresh", "#2F6656", "#E7F1ED"),
        _card("Доступный остаток", _fmt(k["stock_units"]), "шт.", f"{k['sku_with_stock']} SKU",
              "tabler:box", "#3B82F6", "#EFF6FF"),
        _card("Неликвид", _fmt(k["dead_units"]), "шт.", f"{k['dead_sku']} SKU · {k['dead_share']:.0%} запаса",
              "tabler:archive-off", "#7B4437", "#F6E9E4"),
        _card("Очень медленные", _fmt(k["very_slow_units"]), "шт.", f"{k['very_slow_sku']} SKU · > 365 дн.",
              "tabler:hourglass-low", "#9A6100", "#FDF3DE"),
        money_card,
    ])

    return dmc.Stack(gap="md", children=[
        cards,
        _insights_block(df, dark),
        dmc.SimpleGrid(cols=2, spacing="md", children=[
            dmc.Paper(withBorder=True, radius=0, p="sm", children=[
                dmc.Group(justify="space-between", align="center", children=[
                    dmc.Stack(gap=0, children=[
                        dmc.Text("Запас по скорости оборачиваемости", fw=700, size="sm"),
                        dmc.Text(
                            f"Выбрано: {', '.join(selected)}" if selected else "Клик по столбику — показать его позиции",
                            size="xs", c="teal" if selected else "dimmed", fw=600 if selected else 400,
                        ),
                    ]),
                    dmc.Button(
                        "Показать все",
                        id=TURNOVER_RESET_ID,
                        size="xs",
                        radius=0,
                        variant="light" if selected else "subtle",
                        color="teal",
                        disabled=not selected,
                        n_clicks=0,
                        leftSection=DashIconify(icon="tabler:filter-off", width=14),
                    ),
                ]),
                dcc.Graph(
                    id=TURNOVER_GRAPH_ID,
                    figure=_status_chart(chart_df, dark, selected),
                    config={"displaylogo": False, "displayModeBar": False, "responsive": True},
                    style={"height": f"{CHART_HEIGHT}px", "width": "100%"},
                ),
            ]),
            dmc.Paper(withBorder=True, radius=0, p="sm", children=[
                dmc.Text("Оборачиваемость по категориям", fw=700, size="sm", mb="xs"),
                _category_grid(df, class_name),
            ]),
        ]),
        dmc.Paper(withBorder=True, radius=0, p="sm", children=[
            dmc.Group(justify="space-between", align="center", children=[
                dmc.Stack(gap=0, children=[
                    dmc.Text("ABC × оборачиваемость", fw=700, size="sm"),
                    dmc.Text(cell_note(), id=HEATMAP_NOTE_ID, size="xs", c="dimmed"),
                ]),
                dmc.Button(
                    "Показать все",
                    id=HEATMAP_RESET_ID,
                    size="xs",
                    radius=0,
                    variant="light",
                    color="teal",
                    disabled=True,
                    n_clicks=0,
                    leftSection=DashIconify(icon="tabler:filter-off", width=14),
                ),
            ]),
            dcc.Graph(
                id=HEATMAP_ID,
                figure=_abc_heatmap(heatmap_df, dark),
                config={"displaylogo": False, "displayModeBar": False, "responsive": True},
                style={"width": "100%"},
            ),
            dmc.Text(
                "Правый верхний угол (A/B + медленная, очень медленная) — лишний запас в ходовом товаре: "
                "сократить заказ. Левый нижний (C + быстрая) — возможен недозаказ.",
                size="xs", c="dimmed",
            ),
        ]),
        dmc.Paper(withBorder=True, radius=0, p="sm", children=[
            dmc.Text(items_title(), id=ITEMS_TITLE_ID, fw=700, size="sm", mb="xs"),
            _items_grid(df, class_name, items_rows(df, full=bool(selected))),
        ]),
        dmc.Space(h=30),
    ])


def build_turnover_tab(df: pd.DataFrame, class_name: str = "ag-theme-alpine"):
    header = dmc.Stack(gap=0, children=[
        dmc.Text("Анализ оборачиваемости", fw=700),
        dmc.Text("Оборачиваемость = доступный остаток / среднедневные продажи за выбранный период. "
                 "Фильтры сверху применяются и здесь. Выгрузка — меню «Экспорт» → «Анализ оборачиваемости».",
                 size="xs", c="dimmed"),
    ])
    return dmc.Stack(gap="md", children=[
        header,
        dmc.Container(id=TURNOVER_CONTENT_ID, fluid=True, px=0,
                      children=build_turnover_panel(df, class_name)),
    ])
