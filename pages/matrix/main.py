# pages/matrix/main.py
import math
import locale
from io import StringIO

import pandas as pd
import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from dash import dcc, Input, Output, State, no_update, ctx

from components import NoData, MonthSlider, DATES  # noqa: F401  (NoData может пригодиться)
from .barcode_details import fetch_barcode_breakdown, render_barcode_panel
from .data import (
    ENGINE,
    fletch_cats,
    matrix_calculation,
    fetch_current_stocks,
    fetch_items_metadata,
)

from .grid_specs import get_matrix_column_defs, get_matrix_grid_options
from .help_texts import ABC_HELP_MD, XYZ_HELP_MD, ROP_HELP_MD, FILTER_HELP_MD, MATRIX_ZONES_HELP_MD
from .ui_builders import build_help
from .ids import MatrixIds, MatrixRightIds
from .empty_state import render_matrix_empty_state
from .export_excel import build_matrix_excel_bytes
from .stock_export import build_stock_excel_bytes
from .charts import build_stock_sunburst, stock_category_summary
from .turnover import fetch_item_receipts
from .turnover import ABC_ORDER, STATUS_ORDER as TURNOVER_STATUS_ORDER, add_turnover_metrics
from .turnover_ui import (
    TURNOVER_CONTENT_ID,
    TURNOVER_STATUS_MS_ID,
    TURNOVER_GRAPH_ID,
    TURNOVER_RESET_ID,
    ABC_MS_ID,
    HEATMAP_ID,
    HEATMAP_RESET_ID,
    HEATMAP_NOTE_ID,
    ITEMS_GRID_ID,
    ITEMS_TITLE_ID,
    RECEIPTS_DRAWER_ID,
    RECEIPTS_DRAWER_BODY_ID,
    render_receipts_panel,
    abc_heatmap,
    cell_note,
    items_rows,
    items_title,
    themed_figure,
    TURNOVER_DOWNLOAD_BTN_ID,
    TURNOVER_DOWNLOAD_ID,
    build_turnover_panel,
    build_turnover_tab,
)
from .turnover_export import build_turnover_excel_bytes
from datetime import datetime

locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")


MATRIX_GRID_ID = {"type": "mx-grid", "index": "matrix"}
SUNBURST_ID = {"type": "mx-graph", "index": "sunburst"}
THEME_MIRROR_ID = "matrix-theme-mirror"
SEARCH_ID = "matrix-search"
FILTERS_RESET_ID = "matrix-filters-reset"
CONTEXT_STORE_ID = "matrix-context-store"

_MONTHS_RU = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]


def month_label(ts) -> str:
    return f"{_MONTHS_RU[ts.month - 1]} {ts.year}"
STOCK_LEGEND_ID = "matrix-stock-legend"
STOCK_SUMMARY_ID = "matrix-stock-summary"


def scope_text(ctx_data) -> str:
    if not ctx_data:
        return ""
    groups = ", ".join(ctx_data.get("groups") or []) or "все группы"
    cats = ", ".join(ctx_data.get("cats") or []) or "все категории"
    return f"Группы: {groups} · категории: {cats}"


def search_items(df, query):
    """Поиск по номенклатуре: все слова запроса, без учёта регистра; число — ещё и по item_id."""
    words = str(query or "").casefold().split()
    if not words or df.empty:
        return df
    cols = [c for c in ("fullname", "name") if c in df.columns]
    text = df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.casefold() if cols else pd.Series("", index=df.index)
    mask = pd.Series(True, index=df.index)
    for w in words:
        mask &= text.str.contains(w, regex=False)
    q = str(query).strip()
    if q.isdigit() and "item_id" in df.columns:
        mask |= df["item_id"].astype(str) == q
    return df[mask]


def apply_matrix_filters(df, manu_values=None, stock_values=None, turnover_values=None, abc_values=None,
                         search=None):
    df = search_items(df, search)
    if abc_values and "abc" in df.columns:
        df = df[df["abc"].fillna("").astype(str).isin(abc_values)]
    if manu_values and "manu" in df.columns:
        df = df[df["manu"].fillna("Нет производителя").astype(str).isin(manu_values)]
    if stock_values and "stock_status" in df.columns:
        df = df[df["stock_status"].fillna("").astype(str).isin(stock_values)]
    if turnover_values and "turnover_status" in df.columns:
        df = df[df["turnover_status"].fillna("").astype(str).isin(turnover_values)]
    return df


def id_to_months(start, end):
    return DATES[start].strftime("%Y-%m-%d"), DATES[end].strftime("%Y-%m-%d")


# --------------------------
# Left section (controls)
# --------------------------
class LeftSection:
    def __init__(self):
        self.ids = MatrixIds()

        # --- HELP MODALS ---
        abc_help_legend, abc_help_modal = build_help(
            open_btn_id=self.ids.abc_help_open,
            modal_id=self.ids.abc_help_modal,
            icon_text="💡",
            legend_text="Параметры ABC",
            modal_title="Параметры ABC — ранжирование по выручке",
            markdown_text=ABC_HELP_MD,
        )

        xyz_help_legend, xyz_help_modal = build_help(
            open_btn_id=self.ids.xyz_help_open,
            modal_id=self.ids.xyz_help_modal,
            icon_text="💡",
            legend_text="Параметры XYZ",
            modal_title="Параметры XYZ — ранжирование по спросу",
            markdown_text=XYZ_HELP_MD,
        )

        rop_help_legend, rop_help_modal = build_help(
            open_btn_id=self.ids.rop_help_open,
            modal_id=self.ids.rop_help_modal,
            icon_text="💡",
            legend_text="Параметры ROP и SS",
            modal_title="ROP и Safety Stock — параметры расчёта",
            markdown_text=ROP_HELP_MD,
        )

        filter_help_legend, filter_help_modal = build_help(
            open_btn_id=self.ids.filter_help_open,
            modal_id=self.ids.filter_help_modal,
            icon_text="⚙️",
            legend_text="Фильтр",
            modal_title="Фильтр по группам и категориям",
            markdown_text=FILTER_HELP_MD,
        )
        
        zones_help_legend, zones_help_modal = build_help(
            open_btn_id=self.ids.zones_help_open,
            modal_id=self.ids.zones_help_modal,
            icon_text="🎯",
            legend_text="Как читать зоны",
            modal_title="Матрица ABC-XYZ — смысл зон",
            markdown_text=MATRIX_ZONES_HELP_MD,
        )


        # --------------------------
        # Controls
        # --------------------------

        # --- ABC ---
        a_score = dmc.NumberInput(
            value=50,
            min=35,
            max=98,
            step=1,
            allowDecimal=False,
            suffix="%",
            leftSection=DashIconify(icon="mynaui:letter-a-waves-solid", color="red", width=24),
            w=80,
            size="xs",
            id=self.ids.a_score,
        )
        b_score = dmc.NumberInput(
            value=25,
            min=1,
            max=64,
            step=1,
            allowDecimal=False,
            suffix="%",
            leftSection=DashIconify(icon="mynaui:letter-b-waves-solid", color="blue", width=24),
            w=75,
            size="xs",
            id=self.ids.b_score,
        )
        c_score = dmc.NumberInput(
            value=25,
            min=1,
            max=64,
            step=1,
            allowDecimal=False,
            disabled=True,
            suffix="%",
            leftSection=DashIconify(icon="mynaui:letter-c-waves-solid", color="gray", width=24),
            w=80,
            size="xs",
            id=self.ids.c_score,
        )

        abc_fieldset = dmc.Fieldset(
            children=[
                dmc.SimpleGrid(cols=3, spacing="xs", children=[a_score, b_score, c_score]),
                abc_help_modal,
            ],
            radius="sm",
            legend=abc_help_legend,
        )

        # --- XYZ ---
        x_score = dmc.NumberInput(
            value=0.5,
            min=0.1,
            max=0.8,
            step=0.1,
            allowDecimal=True,
            prefix="≤",
            leftSection=DashIconify(icon="mynaui:letter-x-diamond-solid", color="red", width=24),
            w=80,
            size="xs",
            id=self.ids.x_score,
        )
        y_score = dmc.NumberInput(
            value=1,
            min=0.5,
            max=1.5,
            step=0.1,
            allowDecimal=True,
            leftSection=DashIconify(icon="mynaui:letter-y-diamond-solid", color="teal", width=24),
            w=75,
            size="xs",
            id=self.ids.y_score,
        )
        z_score = dmc.NumberInput(
            value=1,
            min=0.5,
            max=100,
            step=0.1,
            allowDecimal=True,
            prefix=">",
            leftSection=DashIconify(icon="mynaui:letter-z-diamond-solid", color="gray", width=24),
            w=80,
            size="xs",
            id=self.ids.z_score,
            disabled=True,
        )

        xyz_fieldset = dmc.Fieldset(
            children=[
                dmc.SimpleGrid(cols=3, spacing="xs", children=[x_score, y_score, z_score]),
                xyz_help_modal,
                zones_help_modal,
            ],
            radius="sm",
            legend=xyz_help_legend,
        )

        # --- Filters (groups / categories) ---
        self.cats_df = fletch_cats()

        gr_data = (
            self.cats_df[["gr_id", "gr_name"]]
            .dropna(subset=["gr_id", "gr_name"])
            .drop_duplicates()
            .sort_values("gr_name", key=lambda s: s.str.lower())
            .assign(gr_id=lambda x: x["gr_id"].astype(str))
            .rename(columns={"gr_id": "value", "gr_name": "label"})
            .to_dict(orient="records")
        )

        gr_ms = dmc.MultiSelect(
            id=self.ids.gr_ms,
            label="Группы",
            placeholder="Выберите группу",
            data=gr_data,
            w="100%",
            radius=0,
            clearable=True,
            searchable=True,
            leftSection=DashIconify(icon="tabler:folders"),
        )

        cat_ms = dmc.MultiSelect(
            id=self.ids.cat_ms,
            label="Категория",
            placeholder="Выберите категорию",
            data=[],
            w="100%",
            radius=0,
            clearable=True,
            searchable=True,
            leftSection=DashIconify(icon="tabler:tag"),
        )

        cats_ms_fieldset = dmc.Fieldset(
            children=[gr_ms, cat_ms, filter_help_modal],
            radius="sm",
            legend=filter_help_legend,
        )
        
        zones_help_block = dmc.Group(
            gap="xs",
            align="center",
            children=[
                # dmc.Text("Как читать зоны", fw=600),
                zones_help_legend, 
            ],
        )



        # --- Groupby switch (пока не используешь — но пусть будет) ---
        groupby_sc_switch = dmc.Switch(
            onLabel="ON",
            offLabel="OFF",
            radius="sm",
            labelPosition="right",
            label="Групировать по подкатегориям",
            checked=False,
            id=self.ids.groupby_sc,
        )
        groupby_sc_fieldset = dmc.Fieldset(
            children=[groupby_sc_switch],
            radius="sm",
            legend="Групировки номенклатур",
        )

        # --- ROP / SS ---
        lead_time = dmc.NumberInput(
            value=2,
            min=0.5,
            max=24,
            step=1,
            allowDecimal=True,
            suffix=" мес.",
            leftSection=DashIconify(icon="mdi:tool-time", color="red", width=24),
            w=120,
            size="xs",
            id=self.ids.lead_time,
        )
        service_ratio = dmc.NumberInput(
            value=95,
            min=70,
            max=99,
            step=1,
            allowDecimal=False,
            suffix="%",
            leftSection=DashIconify(icon="medical-icon:interpreter-services", color="blue", width=24),
            w=120,
            size="xs",
            id=self.ids.service_ratio,
        )

        rop_fieldset = dmc.Fieldset(
            children=[
                dmc.SimpleGrid(cols=2, spacing="md", children=[lead_time, service_ratio]),
                rop_help_modal,
            ],
            radius="sm",
            legend=rop_help_legend,
        )

        # --- Launch button ---
        launch_btn = dmc.Button(
            "Рассчитать",
            id=self.ids.launch,
            leftSection=DashIconify(icon="mynaui:rocket-solid", width=24),
            fullWidth=True,
        )

        # --- Final layout ---
        # Панель используется внутри Drawer, поэтому без отдельного заголовка
        # и без лишних больших вертикальных отступов.
        self.left_section_layout = dmc.Stack(
            gap="md",
            children=[
                abc_fieldset,
                xyz_fieldset,
                zones_help_block,
                rop_fieldset,
                cats_ms_fieldset,
                # groupby_sc_fieldset,  # включишь, когда реально понадобится
                dmc.Space(h=4),
                launch_btn,
            ],
        )

    def register_callbacks(self, app):
        # фильтр категорий при выбранной группе
        @app.callback(
            Output(self.ids.cat_ms, "data"),
            Input(self.ids.gr_ms, "value"),
            prevent_initial_call=True,
        )
        def filter_cat_ms(gr_list):
            if not gr_list:
                return []
            gr_list_int = [int(x) for x in gr_list]
            df = self.cats_df[self.cats_df["gr_id"].isin(gr_list_int)]

            return (
                df[["cat_id", "cat_name"]]
                .dropna(subset=["cat_id", "cat_name"])
                .drop_duplicates()
                .sort_values("cat_name", key=lambda s: s.str.lower())
                .assign(cat_id=lambda x: x["cat_id"].astype(str))
                .rename(columns={"cat_id": "value", "cat_name": "label"})
                .to_dict(orient="records")
            )

        # автопересчет abc
        @app.callback(
            Output(self.ids.b_score, "value"),
            Output(self.ids.c_score, "value"),
            Output(self.ids.b_score, "max"),
            Output(self.ids.c_score, "max"),
            Input(self.ids.a_score, "value"),
            prevent_initial_call=True,
        )
        def split_bc(a_val):
            r = 100 - a_val
            b = math.ceil(r / 2)
            c = 100 - b - a_val
            return b, c, r - 1, r - 1

        @app.callback(
            Output(self.ids.c_score, "value", allow_duplicate=True),
            Input(self.ids.b_score, "value"),
            State(self.ids.a_score, "value"),
            prevent_initial_call=True,
        )
        def adjust_c(b_val, a_val):
            return 100 - b_val - a_val

        # автопересчет xyz
        @app.callback(
            Output(self.ids.y_score, "value"),
            Output(self.ids.y_score, "min"),
            Output(self.ids.z_score, "value"),
            Input(self.ids.x_score, "value"),
            State(self.ids.y_score, "value"),
            prevent_initial_call=True,
        )
        def set_yz(x_val, y_val):
            y_min = x_val + 0.5
            z = y_val if (y_val is not None and y_val > y_min) else y_min
            return z, y_min, z

        @app.callback(
            Output(self.ids.z_score, "value", allow_duplicate=True),
            Input(self.ids.y_score, "value"),
            prevent_initial_call=True,
        )
        def set_z(y_val):
            return y_val

        # --- open/close modals ---
        @app.callback(
            Output(self.ids.abc_help_modal, "opened"),
            Input(self.ids.abc_help_open, "n_clicks"),
            State(self.ids.abc_help_modal, "opened"),
            prevent_initial_call=True,
        )
        def toggle_abc_help(n, opened):
            return not opened

        @app.callback(
            Output(self.ids.xyz_help_modal, "opened"),
            Input(self.ids.xyz_help_open, "n_clicks"),
            State(self.ids.xyz_help_modal, "opened"),
            prevent_initial_call=True,
        )
        def toggle_xyz_help(n, opened):
            return not opened

        @app.callback(
            Output(self.ids.rop_help_modal, "opened"),
            Input(self.ids.rop_help_open, "n_clicks"),
            State(self.ids.rop_help_modal, "opened"),
            prevent_initial_call=True,
        )
        def toggle_rop_help(n, opened):
            return not opened

        @app.callback(
            Output(self.ids.filter_help_modal, "opened"),
            Input(self.ids.filter_help_open, "n_clicks"),
            State(self.ids.filter_help_modal, "opened"),
            prevent_initial_call=True,
        )
        def toggle_filter_help(n, opened):
            return not opened
        
        @app.callback(
            Output(self.ids.zones_help_modal, "opened"),
            Input(self.ids.zones_help_open, "n_clicks"),
            State(self.ids.zones_help_modal, "opened"),
            prevent_initial_call=True,
        )
        def toggle_zones_help(n, opened):
            return not opened



# --------------------------
# Right section (matrix grid + drawer)
# --------------------------
class RightSection:
    def __init__(self):
        self.ids = MatrixRightIds()

        
        self.layout = dmc.Container(
            children=[
                # ✅ 1) тут будет header (пока пусто)
                dmc.Container(id=self.ids.header, fluid=True, px=0),

                dmc.Space(h=16),

                # ✅ 2) дальше как было — content / loading
                dcc.Loading(
                    id=self.ids.loading,
                    type="graph",
                    fullscreen=False,
                    children=dmc.Container(
                        id=self.ids.content,
                        fluid=True,
                        px=0,
                        children=render_matrix_empty_state(),
                    ),
                ),

                dcc.Download(id=self.ids.download),
                dcc.Download(id=self.ids.download_csv),      # быстрый CSV
                dcc.Download(id=self.ids.stocks_download),   # отдельная выгрузка остатков
                dcc.Download(id=TURNOVER_DOWNLOAD_ID),

                dmc.Drawer(
                    id=self.ids.barcode_drawer,
                    title=None,
                    opened=False,
                    position="right",
                    size=520,
                    overlayProps={"opacity": 0.45, "blur": 2},
                    children=dmc.Container(id=self.ids.barcode_drawer_body, fluid=True),
                ),
            ],
            id=self.ids.right_container,
            fluid=True,
            px=0,
        )



    def get_matrix(self, start, end, cat, threholds, lt, sr) -> pd.DataFrame:
        return matrix_calculation(start, end, cat, threholds, lt, sr)

    def matrix_ag_grid(self, df: pd.DataFrame, rrgrid_className: str):
        column_defs = get_matrix_column_defs(df)
        grid_opts = get_matrix_grid_options()
        row_data = df.to_dict("records")

        return dag.AgGrid(
            id=MATRIX_GRID_ID,
            rowData=row_data,
            columnDefs=column_defs,
            defaultColDef={"sortable": True, "filter": True, "resizable": True},
            dashGridOptions=grid_opts,
            style={
                "height": "calc(100vh - 300px)",
                "minHeight": "560px",
                "width": "100%",
                "--ag-font-size": "12px",
                "--ag-row-height": "34px",
                "--ag-header-height": "36px",
                "--ag-list-item-height": "30px",
                "--ag-grid-size": "5px",
                "--ag-cell-horizontal-padding": "10px",
                "--ag-header-column-separator-display": "block",
                "--ag-header-column-separator-height": "45%",

            },
            className=rrgrid_className,
            dangerously_allow_code=True,
        )

    @staticmethod
    def _fmt_qty(value) -> str:
        try:
            return f"{float(value):,.0f}".replace(",", " ")
        except (TypeError, ValueError):
            return "0"

    def build_stock_summary(self, df: pd.DataFrame):
        """
        Компактное summary по текущему набору строк.

        Важно:
        сюда входят и товары без продаж, если они есть в текущих остатках.
        """
        if df is None or df.empty:
            values = {
                "stock": 0,
                "ordered": 0,
                "no_stock": 0,
                "deficit": 0,
                "excess": 0,
                "no_sales_stock": 0,
            }
        else:
            stock_available = pd.to_numeric(
                df.get("stock_available", 0),
                errors="coerce",
            ).fillna(0)

            stock_ordered = pd.to_numeric(
                df.get("stock_ordered", 0),
                errors="coerce",
            ).fillna(0)

            statuses = df.get(
                "stock_status",
                pd.Series("", index=df.index),
            ).fillna("").astype(str)

            values = {
                "stock": float(stock_available.sum()),
                "ordered": float(stock_ordered.sum()),
                "no_stock": int((stock_available <= 0).sum()),
                "deficit": int((statuses == "Дефицит").sum()),
                "excess": int((statuses == "Избыток > 6 мес.").sum()),
                "no_sales_stock": int(
                    (statuses == "Без продаж, есть запас").sum()
                ),
            }

        cards = [
            {
                "label": "Доступный остаток",
                "value": self._fmt_qty(values["stock"]),
                "unit": "ед.",
                "icon": "tabler:box",
                "icon_color": "#3B82F6",
                "icon_bg": "#EFF6FF",
            },
            {
                "label": "Заказано",
                "value": self._fmt_qty(values["ordered"]),
                "unit": "ед.",
                "icon": "tabler:truck-delivery",
                "icon_color": "#7C3AED",
                "icon_bg": "#F5F3FF",
            },
            {
                "label": "Нет в наличии",
                "value": self._fmt_qty(values["no_stock"]),
                "unit": "SKU",
                "icon": "tabler:circle-minus",
                "icon_color": "#EA580C",
                "icon_bg": "#FFF7ED",
            },
            {
                "label": "Дефицит ниже ROP",
                "value": self._fmt_qty(values["deficit"]),
                "unit": "SKU",
                "icon": "tabler:alert-triangle",
                "icon_color": "#DC2626",
                "icon_bg": "#FEF2F2",
            },
            {
                "label": "Запас > 6 мес.",
                "value": self._fmt_qty(values["excess"]),
                "unit": "SKU",
                "icon": "tabler:chart-line",
                "icon_color": "#16A34A",
                "icon_bg": "#F0FDF4",
            },
            {
                "label": "Без продаж, есть запас",
                "value": self._fmt_qty(values["no_sales_stock"]),
                "unit": "SKU",
                "icon": "tabler:archive",
                "icon_color": "#059669",
                "icon_bg": "#ECFDF5",
            },
        ]

        return dmc.SimpleGrid(
            cols=6,
            spacing="sm",
            children=[
                dmc.Paper(
                    withBorder=True,
                    radius=0,
                    px=14,
                    py=10,
                    style={
                        "minHeight": "70px",
                        "borderColor": "var(--mantine-color-default-border)",
                    },
                    children=[
                        dmc.Group(
                            gap="sm",
                            wrap="nowrap",
                            align="center",
                            children=[
                                dmc.Center(
                                    w=34,
                                    h=34,
                                    style={
                                        "backgroundColor": card["icon_bg"],
                                        "borderRadius": "4px",
                                        "flex": "0 0 34px",
                                    },
                                    children=DashIconify(
                                        icon=card["icon"],
                                        width=21,
                                        color=card["icon_color"],
                                    ),
                                ),
                                dmc.Stack(
                                    gap=1,
                                    children=[
                                        dmc.Text(
                                            card["label"],
                                            size="xs",
                                            c="dimmed",
                                            fw=500,
                                        ),
                                        dmc.Group(
                                            gap=5,
                                            align="baseline",
                                            children=[
                                                dmc.Text(
                                                    card["value"],
                                                    size="lg",
                                                    fw=700,
                                                    lh=1.05,
                                                ),
                                                dmc.Text(
                                                    card["unit"],
                                                    size="xs",
                                                    c="dimmed",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                )
                for card in cards
            ],
        )

    @staticmethod
    def build_stock_legend(df: pd.DataFrame, value_field: str = "stock_available"):
        cats = stock_category_summary(df, value_field)
        if cats.empty:
            return dmc.Text("Нет данных", size="sm", c="dimmed")

        rows = []
        for cat in cats.head(12).itertuples():
            rows.append(
                dmc.Stack(gap=3, children=[
                    dmc.Group(justify="space-between", wrap="nowrap", gap="xs", children=[
                        dmc.Group(gap=8, wrap="nowrap", children=[
                            dmc.Box(w=10, h=10, style={"backgroundColor": cat.color, "borderRadius": "2px",
                                                        "flex": "0 0 10px"}),
                            dmc.Text(cat.cat_name, size="sm", lineClamp=1),
                        ]),
                        dmc.Group(gap=6, wrap="nowrap", children=[
                            dmc.Text(f"{cat.value:,.0f}".replace(",", " "), size="sm", fw=600),
                            dmc.Text(f"{cat.share:.0%}", size="xs", c="dimmed", w=36, ta="right"),
                        ]),
                    ]),
                    dmc.Progress(value=float(cat.share) * 100, color=cat.color, size="xs", radius=0),
                ])
            )
        if len(cats) > 12:
            rest = cats.iloc[12:]
            rows.append(dmc.Text(
                f"Ещё {len(rest)} категорий · {rest['value'].sum():,.0f} шт. · {rest['share'].sum():.0%}".replace(",", " "),
                size="xs", c="dimmed",
            ))
        return dmc.Stack(gap="sm", children=rows)

    @staticmethod
    def stock_structure_summary(df: pd.DataFrame, value_field: str = "stock_available") -> str:
        cats = stock_category_summary(df, value_field)
        if cats.empty:
            return "Структура запасов · нет данных"
        top = cats.iloc[0]
        total = f"{cats['value'].sum():,.0f}".replace(",", " ")
        return (
            f"Структура запасов · {total} шт. в {len(cats)} категориях · "
            f"крупнейшая — {top['cat_name']} ({top['share']:.0%})"
        )

    def build_stock_sunburst_block(
        self,
        df: pd.DataFrame,
        value_field: str = "stock_available",
    ):
        figure = themed_figure(build_stock_sunburst(df, value_field=value_field), dark=self._dark)

        return dmc.Stack(gap="sm", children=[
            dmc.Group(justify="space-between", align="center", children=[
                dmc.Text("Категория → подкатегория. Клик по сектору — приблизить, клик по центру — назад.",
                         size="xs", c="dimmed"),
                dmc.SegmentedControl(
                    id="matrix-stock-metric",
                    value=value_field,
                    size="xs",
                    radius=0,
                    color="teal",
                    data=[
                        {"label": "Доступно", "value": "stock_available"},
                        {"label": "С заказами", "value": "stock_total"},
                        {"label": "Заказано", "value": "stock_ordered"},
                    ],
                ),
            ]),
            dmc.Grid(gutter="lg", align="center", children=[
                dmc.GridCol(span={"base": 12, "md": 7}, children=dcc.Graph(
                    id=SUNBURST_ID,
                    figure=figure,
                    config={"displaylogo": False, "responsive": True, "displayModeBar": False},
                    style={"width": "100%", "height": "520px"},
                )),
                dmc.GridCol(span={"base": 12, "md": 5}, children=dmc.Box(
                    id=STOCK_LEGEND_ID,
                    children=self.build_stock_legend(df, value_field),
                )),
            ]),
        ])

    _dark = False

    def maxrix_layout(
        self,
        df: pd.DataFrame,
        rrgrid_className: str,
    ):
        self._dark = rrgrid_className.endswith("dark")
        matrix_dag = self.matrix_ag_grid(df, rrgrid_className)
        sunburst = self.build_stock_sunburst_block(df, value_field="stock_available")

        matrix_tab = dmc.Stack(
            gap="md",
            children=[
                dmc.Container(
                    id=self.ids.summary,
                    fluid=True,
                    px=0,
                    children=self.build_stock_summary(df),
                ),
                dmc.Group(
                    gap=6,
                    children=[
                        DashIconify(icon="tabler:layout-columns", width=16, color="gray"),
                        dmc.Text(
                            "Показаны ключевые колонки. Остальные — по стрелке «›» в заголовке группы "
                            "(Товар, Продажи, Запас, Оборачиваемость). Клик по строке — детализация по штрихкодам.",
                            size="xs",
                            c="dimmed",
                        ),
                    ],
                ),
                matrix_dag,
                dmc.Accordion(
                    variant="contained",
                    radius=0,
                    chevronPosition="left",
                    children=[
                        dmc.AccordionItem(
                            value="sunburst",
                            children=[
                                dmc.AccordionControl(
                                    dmc.Text(
                                        self.stock_structure_summary(df),
                                        id=STOCK_SUMMARY_ID,
                                        size="sm",
                                        fw=500,
                                    ),
                                    icon=DashIconify(icon="tabler:chart-donut-4", width=18, color="#2F6656"),
                                ),
                                dmc.AccordionPanel(sunburst),
                            ],
                        )
                    ],
                ),
                dmc.Space(h=30),
            ],
        )

        return dmc.Tabs(
            value="matrix",
            radius=0,
            keepMounted=True,
            color="teal",
            children=[
                dmc.TabsList([
                    dmc.TabsTab(
                        "Матрица",
                        value="matrix",
                        leftSection=DashIconify(icon="tabler:table", width=16),
                    ),
                    dmc.TabsTab(
                        "Оборачиваемость",
                        value="turnover",
                        leftSection=DashIconify(icon="tabler:refresh", width=16),
                    ),
                ]),
                dmc.TabsPanel(
                    dmc.Box(matrix_tab, pt="md"),
                    value="matrix",
                ),
                dmc.TabsPanel(
                    dmc.Box(build_turnover_tab(df, rrgrid_className), pt="md"),
                    value="turnover",
                ),
            ],
        )

    @staticmethod
    def _chip_group(label, icon, values, all_text, max_show=3):
        """Значения выбора в виде бейджей; длинный список сворачивается в «+N» с подсказкой."""
        if not values:
            chips = [dmc.Badge(all_text, variant="outline", color="gray", radius=0, size="md", tt="none")]
        else:
            shown = values[:max_show]
            chips = [dmc.Badge(v, variant="light", color="teal", radius=0, size="md", tt="none") for v in shown]
            rest = values[max_show:]
            if rest:
                chips.append(
                    dmc.Tooltip(
                        label=", ".join(rest),
                        multiline=True,
                        w=320,
                        withArrow=True,
                        children=dmc.Badge(f"+{len(rest)}", variant="filled", color="teal", radius=0, size="md"),
                    )
                )
        return dmc.Group(gap=6, wrap="wrap", align="center", children=[
            DashIconify(icon=icon, width=16, color="#2F6656"),
            dmc.Text(label, size="xs", c="dimmed", fw=600),
            *chips,
        ])

    def build_context(self, ctx_data: dict):
        params = (
            f"ABC {ctx_data['a']}/{ctx_data['b']}/{ctx_data['c']} · "
            f"XYZ {ctx_data['x']}/{ctx_data['y']} · LT {ctx_data['lt']} мес. · сервис {ctx_data['sr']}%"
        )
        return dmc.Group(
            gap="lg",
            wrap="wrap",
            align="center",
            children=[
                self._chip_group("Период", "tabler:calendar-month", [ctx_data["period"]], ""),
                self._chip_group("Группы", "tabler:folders", ctx_data["groups"], "Все группы"),
                self._chip_group(
                    "Категории", "tabler:category", ctx_data["cats"],
                    "Все категории выбранных групп" if ctx_data["groups"] else "Все категории",
                ),
                dmc.Tooltip(
                    label="Параметры расчёта: пороги ABC и XYZ, срок поставки и уровень сервиса",
                    withArrow=True,
                    children=dmc.Group(gap=6, children=[
                        DashIconify(icon="tabler:adjustments-horizontal", width=16, color="#2F6656"),
                        dmc.Text(params, size="xs", c="dimmed"),
                    ]),
                ),
            ],
        )

    def build_header(self, ctx_data: dict | None = None):
        """Фильтры над вкладками; применяются к матрице, оборачиваемости и выгрузкам."""
        def _ms(id_, label, placeholder, icon, w=280):
            return dmc.MultiSelect(
                id=id_,
                label=label,
                placeholder=placeholder,
                data=[],
                value=[],
                clearable=True,
                searchable=True,
                w=w,
                size="sm",
                radius=0,
                leftSection=DashIconify(icon=icon, width=18),
            )

        filters = dmc.Group(
                justify="space-between",
                align="flex-end",
                wrap="wrap",
                gap="sm",
                children=[
                    dmc.Group(
                        gap="sm",
                        align="flex-end",
                        wrap="wrap",
                        children=[
                            _ms(self.ids.manu_ms, "Производитель", "Все производители", "tabler:building-factory-2", 300),
                            _ms(self.ids.stock_status_ms, "Статус запаса", "Все статусы", "tabler:packages"),
                            _ms(TURNOVER_STATUS_MS_ID, "Оборачиваемость", "Все статусы", "tabler:refresh"),
                            _ms(ABC_MS_ID, "ABC", "Все классы", "tabler:chart-bar", 200),
                            dmc.TextInput(
                                id=SEARCH_ID,
                                label="Номенклатура",
                                placeholder="Поиск по названию или item_id",
                                leftSection=DashIconify(icon="tabler:search", width=16),
                                debounce=400,
                                radius=0,
                                w=280,
                            ),
                            dmc.Button(
                                "Сбросить",
                                id=FILTERS_RESET_ID,
                                variant="subtle",
                                color="gray",
                                radius=0,
                                n_clicks=0,
                                leftSection=DashIconify(icon="tabler:filter-off", width=16),
                            ),
                        ],
                    ),
                    dmc.Badge(
                        "0 SKU",
                        id=self.ids.manu_badge,
                        variant="light",
                        color="teal",
                        radius=0,
                        size="lg",
                    ),
                ],
            )

        children = []
        if ctx_data:
            children += [
                dmc.Group(gap=6, children=[
                    dmc.Text("Анализируем", size="sm", fw=700),
                    dmc.Text("· изменить — кнопка «Настройки»", size="xs", c="dimmed"),
                ]),
                self.build_context(ctx_data),
                dmc.Divider(my=4),
            ]
        children.append(filters)

        return dmc.Paper(
            withBorder=True,
            radius=0,
            px="md",
            py="sm",
            style={"borderTop": "3px solid #2F6656"},
            children=dmc.Stack(gap="xs", children=children),
        )




# --------------------------
# Main window (compose + callbacks)
# --------------------------
class MainWindow:
    def __init__(self):
        self.ls = LeftSection()
        self.rs = RightSection()
        self.mslider_id = "mslider-id-for-matrix-calculations"
        self.mslider = MonthSlider(id=self.mslider_id)

    def export_menu(self):
        def _item(label, id_, icon, note, disabled=True):
            return dmc.MenuItem(
                dmc.Stack(gap=0, children=[
                    dmc.Text(label, size="sm", fw=500),
                    dmc.Text(note, size="xs", c="dimmed"),
                ]),
                id=id_,
                leftSection=DashIconify(icon=icon, width=18, color="#2F6656"),
                disabled=disabled,
                n_clicks=0,
            )

        return dmc.Menu(
            position="bottom-end",
            shadow="md",
            width=320,
            radius=0,
            children=[
                dmc.MenuTarget(
                    dmc.Button(
                        "Экспорт",
                        color="teal",
                        radius=0,
                        leftSection=DashIconify(icon="tabler:download", width=18),
                        rightSection=DashIconify(icon="tabler:chevron-down", width=16),
                    )
                ),
                dmc.MenuDropdown([
                    dmc.MenuLabel("Excel · с учётом фильтров"),
                    _item("Матрица — полный отчёт", self.rs.ids.download_btn,
                          "mdi:file-excel-outline", "Матрица, остатки, штрихкоды, производители"),
                    _item("Анализ оборачиваемости", TURNOVER_DOWNLOAD_BTN_ID,
                          "tabler:refresh", "Выводы, статусы, категории, неликвид"),
                    dmc.MenuDivider(),
                    dmc.MenuLabel("Без расчёта матрицы"),
                    _item("Текущие остатки", self.rs.ids.stocks_download_btn,
                          "tabler:package-export", "По SKU, магазинам и штрихкодам", disabled=False),
                    dmc.MenuDivider(),
                    dmc.MenuLabel("Данные"),
                    _item("Матрица — CSV", self.rs.ids.download_csv_btn,
                          "mdi:file-delimited-outline", "Все колонки, разделитель «|»"),
                ]),
            ],
        )

    def layout(self):
        """
        Основная страница занимает всю доступную ширину.

        Настройки убраны в Drawer:
        - таблица больше не теряет 25% ширины;
        - после расчёта пользователь работает только с результатом;
        - настройки всегда доступны по кнопке сверху.
        """
        settings_drawer = dmc.Drawer(
            id=self.ls.ids.settings_drawer,
            title=dmc.Group(
                gap="xs",
                children=[
                    DashIconify(
                        icon="tabler:adjustments-horizontal",
                        width=22,
                    ),
                    dmc.Text(
                        "Настройки матрицы",
                        fw=700,
                        size="lg",
                    ),
                ],
            ),
            opened=False,
            position="left",
            size=430,
            padding="lg",
            overlayProps={
                "opacity": 0.30,
                "blur": 1,
            },
            children=self.ls.left_section_layout,
        )

        page_header = dmc.Group(
            justify="space-between",
            align="center",
            wrap="wrap",
            gap="md",
            children=[
                dmc.Stack(
                    gap=2,
                    children=[
                        dmc.Title(
                            "Ассортиментная матрица",
                            order=2,
                        ),
                        dmc.Text(
                            "ABC/XYZ-анализ, спрос, ROP и актуальные остатки",
                            size="sm",
                            c="dimmed",
                        ),
                    ],
                ),
                dmc.Group(
                    gap="xs",
                    align="center",
                    children=[
                        self.export_menu(),
                        dmc.Button(
                            "Настройки",
                            id=self.ls.ids.settings_open,
                            radius=0,
                            variant="outline",
                            color="teal",
                            leftSection=DashIconify(icon="tabler:adjustments-horizontal", width=18),
                        ),
                    ],
                ),
            ],
        )

        period_block = dmc.Paper(
            withBorder=True,
            radius=0,
            px="md",
            py="xs",
            children=[
                dmc.Group(
                    gap="xs",
                    align="center",
                    mb=4,
                    children=[
                        DashIconify(
                            icon="tabler:calendar-month",
                            width=18,
                        ),
                        dmc.Text(
                            "Период анализа",
                            fw=600,
                            size="sm",
                        ),
                    ],
                ),
                self.mslider,
            ],
        )

        return dmc.Container(
            fluid=True,
            px=24,
            py=16,
            children=[
                settings_drawer,
                dcc.Store(id=self.rs.ids.store),

                page_header,
                dmc.Space(h=14),

                period_block,
                dmc.Space(h=14),

                self.rs.layout,
            ],
        )

    def register_callbacks(self, app):
        self.ls.register_callbacks(app)

        # --------------------------------------------------------------
        # Настройки: открыть кнопкой, закрыть автоматически после расчёта
        # --------------------------------------------------------------
        @app.callback(
            Output(self.ls.ids.settings_drawer, "opened"),
            Input(self.ls.ids.settings_open, "n_clicks"),
            Input(self.ls.ids.launch, "n_clicks"),
            State(self.ls.ids.settings_drawer, "opened"),
            prevent_initial_call=True,
        )
        def toggle_settings_drawer(open_clicks, launch_clicks, opened):
            trigger = ctx.triggered_id

            if trigger == self.ls.ids.settings_open:
                return not opened

            if trigger == self.ls.ids.launch:
                return False

            return opened

        @app.callback(
            Output(self.rs.ids.header, "children"),
            Output(self.rs.ids.content, "children"),
            Output(self.rs.ids.store, "data"),
            Input(self.ls.ids.launch, "n_clicks"),
            State(self.ls.ids.a_score, "value"),
            State(self.ls.ids.b_score, "value"),
            State(self.ls.ids.c_score, "value"),
            State(self.ls.ids.x_score, "value"),
            State(self.ls.ids.y_score, "value"),
            State(self.ls.ids.z_score, "value"),
            State(self.ls.ids.gr_ms, "value"),
            State(self.ls.ids.cat_ms, "value"),
            State(self.mslider_id, "value"),
            State(self.ls.ids.lead_time, "value"),
            State(self.ls.ids.service_ratio, "value"),
            State("theme_switch", "checked"),
            prevent_initial_call=True,
        )
        def get_matrix(nclicks, a, b, c, x, y, z, grs, cats, ms, lt, sr, theme):
            if not nclicks:
                return no_update, no_update, no_update

            def find_cats_if_gr():
                gr_list_int = [int(v) for v in (grs or [])]
                df = self.ls.cats_df[self.ls.cats_df["gr_id"].isin(gr_list_int)]
                return df["cat_id"].to_list()

            threholds = {"a": a, "b": b, "c": c, "x": x, "y": y, "z": z}
            start, end = id_to_months(ms[0], ms[1])

            gr = None if not grs else ",".join(grs)
            cat = None if not cats else ",".join(cats)
            if gr and not cat:
                cat = ",".join(map(str, find_cats_if_gr()))

            rrgrid_className = "ag-theme-alpine-dark" if theme else "ag-theme-alpine"

            df_matrix = matrix_calculation(start, end, cat, threholds, lt, sr)
            df_matrix = add_turnover_metrics(df_matrix, start, end)
            store_json = df_matrix.to_json(date_format="iso", orient="records")

            gr_ids = [int(v) for v in (grs or [])]
            cat_ids = [int(v) for v in (cats or [])]
            cdf = self.ls.cats_df
            ctx_data = {
                "period": f"{month_label(DATES[ms[0]])} — {month_label(DATES[ms[1]])}",
                "groups": cdf[cdf["gr_id"].isin(gr_ids)]["gr_name"].drop_duplicates().astype(str).tolist(),
                "cats": cdf[cdf["cat_id"].isin(cat_ids)]["cat_name"].drop_duplicates().astype(str).tolist(),
                "a": a, "b": b, "c": c, "x": x, "y": y, "lt": lt, "sr": sr,
            }

            header = [
                self.rs.build_header(ctx_data),
                dcc.Store(id=THEME_MIRROR_ID, data=bool(theme)),
                dcc.Store(id=CONTEXT_STORE_ID, data=ctx_data),
            ]
            content = self.rs.maxrix_layout(df_matrix, rrgrid_className)

            return header, content, store_json

        
        @app.callback(
            Output(MATRIX_GRID_ID, "rowData"),
            Output(MATRIX_GRID_ID, "className"),
            Output(self.rs.ids.manu_badge, "children"),
            Output(self.rs.ids.download_btn, "disabled"),
            Output(self.rs.ids.download_csv_btn, "disabled"),
            Output(TURNOVER_DOWNLOAD_BTN_ID, "disabled"),
            Output(self.rs.ids.summary, "children"),
            Output(SUNBURST_ID, "figure"),
            Output(TURNOVER_CONTENT_ID, "children"),
            Output(STOCK_LEGEND_ID, "children"),
            Output(STOCK_SUMMARY_ID, "children"),
            Input(self.rs.ids.manu_ms, "value"),
            Input(self.rs.ids.stock_status_ms, "value"),
            Input(TURNOVER_STATUS_MS_ID, "value"),
            Input(ABC_MS_ID, "value"),
            Input("matrix-stock-metric", "value"),
            Input(THEME_MIRROR_ID, "data"),
            Input(SEARCH_ID, "value"),
            State(self.rs.ids.store, "data"),
        )
        def filter_matrix(manu_values, stock_status_values, turnover_values, abc_values, stock_metric, theme,
                          search, store_json):
            if not store_json:
                return (no_update,) * 11

            df_all = pd.read_json(StringIO(store_json), orient="records")
            df_base = apply_matrix_filters(df_all, manu_values, stock_status_values, search=search)
            df_bar = apply_matrix_filters(df_base, abc_values=abc_values)
            df = apply_matrix_filters(df_bar, turnover_values=turnover_values)
            metric = stock_metric or "stock_available"
            grid_class = "ag-theme-alpine-dark" if theme else "ag-theme-alpine"

            badge = f"{len(df):,}/{len(df_all):,} SKU".replace(",", " ")
            sunburst_figure = themed_figure(
                build_stock_sunburst(df, value_field=metric),
                dark=bool(theme),
            )

            return (
                df.to_dict("records"),
                grid_class,
                badge,
                False,
                False,
                False,
                self.rs.build_stock_summary(df),
                sunburst_figure,
                build_turnover_panel(
                    df, grid_class, dark=bool(theme),
                    chart_df=df_bar, selected=turnover_values,
                    heatmap_df=df_bar,
                ),
                self.rs.build_stock_legend(df, metric),
                self.rs.stock_structure_summary(df, metric),
            )

        @app.callback(
            Output(TURNOVER_STATUS_MS_ID, "value"),
            Input(TURNOVER_GRAPH_ID, "clickData"),
            Input(TURNOVER_RESET_ID, "n_clicks"),
            State(TURNOVER_STATUS_MS_ID, "value"),
            prevent_initial_call=True,
        )
        def turnover_bar_click(bar_click, bar_reset, cur_status):
            if ctx.triggered_id == TURNOVER_RESET_ID:
                return [] if bar_reset else no_update
            if not bar_click or not bar_click.get("points"):
                return no_update
            p = bar_click["points"][0]
            status = p.get("customdata") or p.get("y")
            if not status:
                return no_update
            return [] if list(cur_status or []) == [status] else [status]

        # Один колбэк на открытие и закрытие: при закрытии выбор сбрасывается,
        # поэтому повторный клик по той же строке снова открывает шторку.
        @app.callback(
            Output(RECEIPTS_DRAWER_ID, "opened"),
            Output(RECEIPTS_DRAWER_BODY_ID, "children"),
            Output(ITEMS_GRID_ID, "selectedRows"),
            Input(ITEMS_GRID_ID, "selectedRows"),
            Input(RECEIPTS_DRAWER_ID, "opened"),
            prevent_initial_call=True,
        )
        def open_receipts_drawer(rows, opened):
            if ctx.triggered_id == RECEIPTS_DRAWER_ID:
                return no_update, no_update, (no_update if opened else [])
            if not rows or rows[0].get("item_id") is None:
                return no_update, no_update, no_update
            row = rows[0]
            return True, render_receipts_panel(row, fetch_item_receipts(int(row["item_id"]))), no_update

        @app.callback(
            Output(ITEMS_GRID_ID, "rowData"),
            Output(ITEMS_TITLE_ID, "children"),
            Output(HEATMAP_ID, "figure"),
            Output(HEATMAP_NOTE_ID, "children"),
            Output(HEATMAP_RESET_ID, "disabled"),
            Input(HEATMAP_ID, "clickData"),
            Input(HEATMAP_RESET_ID, "n_clicks"),
            State(self.rs.ids.store, "data"),
            State(self.rs.ids.manu_ms, "value"),
            State(self.rs.ids.stock_status_ms, "value"),
            State(TURNOVER_STATUS_MS_ID, "value"),
            State(ABC_MS_ID, "value"),
            State(THEME_MIRROR_ID, "data"),
            State(SEARCH_ID, "value"),
            prevent_initial_call=True,
        )
        def heatmap_cell_click(cell_click, reset, store_json, manu_values, stock_values,
                               turnover_values, abc_values, theme, search):
            """Клик по ячейке фильтрует только таблицу позиций под картой."""
            if not store_json:
                return (no_update,) * 5
            df_all = pd.read_json(StringIO(store_json), orient="records")
            df_bar = apply_matrix_filters(df_all, manu_values, stock_values, abc_values=abc_values, search=search)
            df = apply_matrix_filters(df_bar, turnover_values=turnover_values)

            cell = None
            if ctx.triggered_id == HEATMAP_ID and cell_click and cell_click.get("points"):
                p = cell_click["points"][0]
                cd = p.get("customdata") or []
                if len(cd) >= 2:
                    cell = (cd[0], str(p.get("x") or cd[1]), cd[1])

            if cell is None:
                return (items_rows(df, full=bool(turnover_values)), items_title(),
                        abc_heatmap(df_bar, bool(theme)), cell_note(), True)

            sel = df_bar[(df_bar["abc"].fillna("") == cell[0]) & (df_bar["turnover_status"] == cell[2])]
            rows = items_rows(sel, full=True)
            return (rows, items_title(cell[:2], len(rows)), abc_heatmap(df_bar, bool(theme), cell),
                    cell_note(cell[:2]), False)

        @app.callback(
            Output(self.rs.ids.manu_ms, "data"),
            Output(self.rs.ids.stock_status_ms, "data"),
            Output(TURNOVER_STATUS_MS_ID, "data"),
            Output(ABC_MS_ID, "data"),
            Input(self.rs.ids.store, "data"),
            prevent_initial_call=True,
        )
        def fill_right_filters(store_json):
            if not store_json:
                return [], [], [], []

            df = pd.read_json(StringIO(store_json), orient="records")

            manu_list = (
                df["manu"]
                .fillna("Нет производителя")
                .astype(str)
                .sort_values(key=lambda s: s.str.lower())
                .unique()
                .tolist()
            )

            def ordered(column, order):
                existing = set(df.get(column, pd.Series(dtype="object")).dropna().astype(str))
                result = [v for v in order if v in existing]
                result += sorted(existing.difference(result), key=str.lower)
                return [{"value": v, "label": v} for v in result]

            stock_order = [
                "Без продаж, есть запас",
                "Дефицит",
                "Нет остатка",
                "Нет остатка, товар заказан",
                "Ниже ROP, заказ закрывает",
                "Достаточно",
                "Избыток > 6 мес.",
            ]

            return (
                [{"value": m, "label": m} for m in manu_list],
                ordered("stock_status", stock_order),
                ordered("turnover_status", TURNOVER_STATUS_ORDER),
                ordered("abc", ABC_ORDER),
            )

        @app.callback(
            Output(THEME_MIRROR_ID, "data"),
            Input("theme_switch", "checked"),
            prevent_initial_call=True,
        )
        def mirror_theme(theme):
            return bool(theme)

        @app.callback(
            Output(self.rs.ids.manu_ms, "value", allow_duplicate=True),
            Output(self.rs.ids.stock_status_ms, "value", allow_duplicate=True),
            Output(TURNOVER_STATUS_MS_ID, "value", allow_duplicate=True),
            Output(ABC_MS_ID, "value", allow_duplicate=True),
            Output(SEARCH_ID, "value", allow_duplicate=True),
            Input(FILTERS_RESET_ID, "n_clicks"),
            prevent_initial_call=True,
        )
        def reset_filters(n):
            if not n:
                return (no_update,) * 5
            return [], [], [], [], ""

        filter_states = [
            State(CONTEXT_STORE_ID, "data"),
            State(self.rs.ids.store, "data"),
            State(self.rs.ids.manu_ms, "value"),
            State(self.rs.ids.stock_status_ms, "value"),
            State(TURNOVER_STATUS_MS_ID, "value"),
            State(ABC_MS_ID, "value"),
            State(self.mslider_id, "value"),
            State(SEARCH_ID, "value"),
        ]

        def _filtered(store_json, manu_values, stock_values, turnover_values, abc_values=None, search=None):
            df = pd.read_json(StringIO(store_json), orient="records")
            return apply_matrix_filters(df, manu_values, stock_values, turnover_values, abc_values, search)

        @app.callback(
            Output(self.rs.ids.download, "data"),
            Input(self.rs.ids.download_btn, "n_clicks"),
            *filter_states,
            prevent_initial_call=True,
        )
        def download_excel(n, ctx_data, store_json, manu_values, stock_values, turnover_values, abc_values, ms, search=None):
            if not n or not store_json:
                return no_update

            df_matrix = _filtered(store_json, manu_values, stock_values, turnover_values, abc_values, search)
            start, end = id_to_months(ms[0], ms[1])
            xlsx_bytes = build_matrix_excel_bytes(
                ENGINE, df_matrix=df_matrix, start=start, end=end, scope=scope_text(ctx_data),
            )

            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            return dcc.send_bytes(lambda buffer: buffer.write(xlsx_bytes), f"matrix_{start}_{end}_{stamp}.xlsx")

        @app.callback(
            Output(TURNOVER_DOWNLOAD_ID, "data"),
            Input(TURNOVER_DOWNLOAD_BTN_ID, "n_clicks"),
            *filter_states,
            prevent_initial_call=True,
        )
        def download_turnover_excel(n, ctx_data, store_json, manu_values, stock_values, turnover_values, abc_values, ms, search=None):
            if not n or not store_json:
                return no_update

            df = _filtered(store_json, manu_values, stock_values, turnover_values, abc_values, search)
            start, end = id_to_months(ms[0], ms[1])
            xlsx_bytes = build_turnover_excel_bytes(
                df, period_label=f"{start} – {end}", scope=scope_text(ctx_data),
            )

            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            return dcc.send_bytes(lambda buffer: buffer.write(xlsx_bytes), f"turnover_{start}_{end}_{stamp}.xlsx")

        @app.callback(
            Output(self.rs.ids.stocks_download, "data"),
            Input(self.rs.ids.stocks_download_btn, "n_clicks"),
            prevent_initial_call=True,
        )
        def download_current_stocks(n):
            if not n:
                return no_update

            # ----------------------------------------------------------
            # Остатки берём напрямую из stocks_data.
            # Расчёт ABC/XYZ для этого не нужен.
            # ----------------------------------------------------------
            stocks = fetch_current_stocks()

            if stocks.empty:
                return no_update

            # ----------------------------------------------------------
            # Добавляем товарные метаданные:
            # категория, подкатегория, производитель, артикул и т.д.
            # ----------------------------------------------------------
            item_ids = (
                pd.to_numeric(
                    stocks["item_id"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

            metadata = fetch_items_metadata(item_ids)

            stocks = stocks.copy()
            stocks["item_id"] = pd.to_numeric(
                stocks["item_id"],
                errors="coerce",
            ).astype("Int64")

            if metadata is not None and not metadata.empty:
                metadata = metadata.copy()
                metadata["item_id"] = pd.to_numeric(
                    metadata["item_id"],
                    errors="coerce",
                ).astype("Int64")

                df = metadata.merge(
                    stocks,
                    on="item_id",
                    how="right",
                    validate="one_to_one",
                )
            else:
                df = stocks

            # Поля, используемые stock_export.py, но не обязательные
            # для самостоятельной выгрузки остатков.
            if "stock_status" not in df.columns:
                df["stock_status"] = ""

            if "barcode_stocks_display" not in df.columns:
                df["barcode_stocks_display"] = ""

            xlsx_bytes = build_stock_excel_bytes(df)

            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"stocks_{stamp}.xlsx"

            return dcc.send_bytes(
                lambda buffer: buffer.write(xlsx_bytes),
                filename,
            )


        @app.callback(
            Output(self.rs.ids.download_csv, "data"),
            Input(self.rs.ids.download_csv_btn, "n_clicks"),
            *filter_states,
            prevent_initial_call=True,
        )
        def download_csv(n, ctx_data, store_json, manu_values, stock_values, turnover_values, abc_values, ms, search=None):
            if not n or not store_json:
                return no_update

            df = _filtered(store_json, manu_values, stock_values, turnover_values, abc_values, search)

            # пользовательские названия для CSV
            csv_rename = {
                "fullname": "Номенклатура",
                "article": "Артикул",
                "manu": "Производитель",
                "abc": "ABC",
                "xyz": "XYZ",
                "cat_name": "Категория",
                "sc_name": "Подкатегория",
                "amount": "Выручка",
                "quant": "Кол-во",
                "share": "Доля выручки",
                "mean_amount": "Ср. выручка",
                "share_mean": "Доля в ср выручке",
                "mean_month": "Ср. продажи, ед/мес.",
                "std_month": "Ст. отклонение",
                "cv": "CV",
                "ss": "SS, ед.",
                "rop": "ROP, ед.",
                "stock_date": "Дата остатков",
                "stock_available": "Остаток доступно",
                "stock_ordered": "Заказано",
                "stock_total": "Остаток итого",
                "stock_cover_months": "Покрытие, мес.",
                "stock_cover_months_total": "Покрытие с заказами, мес.",
                "stock_vs_rop": "Отклонение доступного от ROP",
                "stock_total_vs_rop": "Отклонение итого от ROP",
                "order_need": "Нужно заказать",
                "stock_status": "Статус остатка",
                "turnover_status": "Статус оборачиваемости",
                "turnover_days": "Оборачиваемость, дн.",
                "turns_per_year": "Оборотов в год",
                "avg_daily_sales": "Продажи в день, шт.",
                "first_receipt_date": "Первый приход",
                "last_receipt_date": "Последний приход",
                "days_since_receipt": "Дней с прихода",
                "receipts_docs": "Документов прихода",
                "receipt_qty_total": "Пришло всего, шт.",
                "receipt_qty_period": "Пришло за период, шт.",
                "last_receipt_qty": "Последняя партия, шт.",
                "sold_since_receipt": "Продано с прихода, шт.",
                "sell_through": "Реализация партии",
                "purchase_price": "Цена закупки",
                "stock_value_purchase": "Остаток по закупке",
            }

            dynamic_rename = {}
            for column in df.columns:
                if str(column).startswith("stock_wh::"):
                    dynamic_rename[column] = (
                        "Остаток | "
                        + str(column).split("::", 1)[1]
                    )
                elif str(column).startswith("ordered_wh::"):
                    dynamic_rename[column] = (
                        "Заказано | "
                        + str(column).split("::", 1)[1]
                    )

            df = df.rename(
                columns={
                    **csv_rename,
                    **dynamic_rename,
                }
            )

            # служебные колонки
            df = df.drop(
                columns=[
                    "date_json",
                    "quant_json",
                    "ls_quant",
                    "ls_date",
                    "is_quant",
                    "is_date",
                    "cum_share",
                    "_amount",
                    "_share",
                ],
                errors="ignore",
            )

            start, end = id_to_months(ms[0], ms[1])
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"matrix_{start}_{end}_{stamp}.csv"

            return dcc.send_data_frame(df.to_csv, filename, index=False, sep="|", encoding="utf-8-sig")





        @app.callback(
            Output(self.rs.ids.barcode_drawer, "opened"),
            Output(self.rs.ids.barcode_drawer_body, "children"),
            Output(MATRIX_GRID_ID, "selectedRows"),
            Input(MATRIX_GRID_ID, "selectedRows"),
            Input(self.rs.ids.barcode_drawer, "opened"),
            State(self.mslider_id, "value"),
            prevent_initial_call=True,
        )
        def open_barcode_details(selected_rows, opened, ms):
            if ctx.triggered_id == self.rs.ids.barcode_drawer:
                return no_update, no_update, (no_update if opened else [])
            if not selected_rows:
                return no_update, no_update, no_update

            row = selected_rows[0]
            item_id = int(row["item_id"])
            fullname = row.get("fullname", "")

            start, end = id_to_months(ms[0], ms[1])

            df_bc = fetch_barcode_breakdown(ENGINE, item_id=item_id, start=start, end=end)
            panel = render_barcode_panel(
                        df_bc,
                        title_name=fullname,
                        subtitle=f"item_id = {item_id}",
                    )
            panel = dmc.Stack(gap="md", children=[
                panel,
                dmc.Divider(label="Оборачиваемость и приходы", labelPosition="left"),
                render_receipts_panel(row, fetch_item_receipts(item_id), show_title=False),
            ])

            return True, panel, no_update