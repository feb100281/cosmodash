# # pages/matrix/main.py
# import math
# import locale
# from io import StringIO

# import pandas as pd
# import dash_ag_grid as dag
# import dash_mantine_components as dmc
# from dash_iconify import DashIconify
# from dash import dcc, Input, Output, State, no_update, ctx

# from components import NoData, MonthSlider, DATES  # noqa: F401  (NoData может пригодиться)
# from .barcode_details import fetch_barcode_breakdown, render_barcode_panel
# from .data import ENGINE, fletch_cats, matrix_calculation

# from .grid_specs import get_matrix_column_defs, get_matrix_grid_options
# from .help_texts import ABC_HELP_MD, XYZ_HELP_MD, ROP_HELP_MD, FILTER_HELP_MD, MATRIX_ZONES_HELP_MD
# from .ui_builders import build_help
# from .ids import MatrixIds, MatrixRightIds
# from .empty_state import render_matrix_empty_state
# from .export_excel import build_matrix_excel_bytes
# from datetime import datetime

# locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")


# def id_to_months(start, end):
#     return DATES[start].strftime("%Y-%m-%d"), DATES[end].strftime("%Y-%m-%d")


# # --------------------------
# # Left section (controls)
# # --------------------------
# class LeftSection:
#     def __init__(self):
#         self.ids = MatrixIds()

#         # --- HELP MODALS ---
#         abc_help_legend, abc_help_modal = build_help(
#             open_btn_id=self.ids.abc_help_open,
#             modal_id=self.ids.abc_help_modal,
#             icon_text="💡",
#             legend_text="Параметры ABC",
#             modal_title="Параметры ABC — ранжирование по выручке",
#             markdown_text=ABC_HELP_MD,
#         )

#         xyz_help_legend, xyz_help_modal = build_help(
#             open_btn_id=self.ids.xyz_help_open,
#             modal_id=self.ids.xyz_help_modal,
#             icon_text="💡",
#             legend_text="Параметры XYZ",
#             modal_title="Параметры XYZ — ранжирование по спросу",
#             markdown_text=XYZ_HELP_MD,
#         )

#         rop_help_legend, rop_help_modal = build_help(
#             open_btn_id=self.ids.rop_help_open,
#             modal_id=self.ids.rop_help_modal,
#             icon_text="💡",
#             legend_text="Параметры ROP и SS",
#             modal_title="ROP и Safety Stock — параметры расчёта",
#             markdown_text=ROP_HELP_MD,
#         )

#         filter_help_legend, filter_help_modal = build_help(
#             open_btn_id=self.ids.filter_help_open,
#             modal_id=self.ids.filter_help_modal,
#             icon_text="⚙️",
#             legend_text="Фильтр",
#             modal_title="Фильтр по группам и категориям",
#             markdown_text=FILTER_HELP_MD,
#         )
        
#         zones_help_legend, zones_help_modal = build_help(
#             open_btn_id=self.ids.zones_help_open,
#             modal_id=self.ids.zones_help_modal,
#             icon_text="🎯",
#             legend_text="Как читать зоны",
#             modal_title="Матрица ABC-XYZ — смысл зон",
#             markdown_text=MATRIX_ZONES_HELP_MD,
#         )


#         # --------------------------
#         # Controls
#         # --------------------------

#         # --- ABC ---
#         a_score = dmc.NumberInput(
#             value=50,
#             min=35,
#             max=98,
#             step=1,
#             allowDecimal=False,
#             suffix="%",
#             leftSection=DashIconify(icon="mynaui:letter-a-waves-solid", color="red", width=24),
#             w=80,
#             size="xs",
#             id=self.ids.a_score,
#         )
#         b_score = dmc.NumberInput(
#             value=25,
#             min=1,
#             max=64,
#             step=1,
#             allowDecimal=False,
#             suffix="%",
#             leftSection=DashIconify(icon="mynaui:letter-b-waves-solid", color="blue", width=24),
#             w=75,
#             size="xs",
#             id=self.ids.b_score,
#         )
#         c_score = dmc.NumberInput(
#             value=25,
#             min=1,
#             max=64,
#             step=1,
#             allowDecimal=False,
#             disabled=True,
#             suffix="%",
#             leftSection=DashIconify(icon="mynaui:letter-c-waves-solid", color="gray", width=24),
#             w=80,
#             size="xs",
#             id=self.ids.c_score,
#         )

#         abc_fieldset = dmc.Fieldset(
#             children=[
#                 dmc.SimpleGrid(cols=3, spacing="xs", children=[a_score, b_score, c_score]),
#                 abc_help_modal,
#             ],
#             radius="sm",
#             legend=abc_help_legend,
#         )

#         # --- XYZ ---
#         x_score = dmc.NumberInput(
#             value=0.5,
#             min=0.1,
#             max=0.8,
#             step=0.1,
#             allowDecimal=True,
#             prefix="≤",
#             leftSection=DashIconify(icon="mynaui:letter-x-diamond-solid", color="red", width=24),
#             w=80,
#             size="xs",
#             id=self.ids.x_score,
#         )
#         y_score = dmc.NumberInput(
#             value=1,
#             min=0.5,
#             max=1.5,
#             step=0.1,
#             allowDecimal=True,
#             leftSection=DashIconify(icon="mynaui:letter-y-diamond-solid", color="teal", width=24),
#             w=75,
#             size="xs",
#             id=self.ids.y_score,
#         )
#         z_score = dmc.NumberInput(
#             value=1,
#             min=0.5,
#             max=100,
#             step=0.1,
#             allowDecimal=True,
#             prefix=">",
#             leftSection=DashIconify(icon="mynaui:letter-z-diamond-solid", color="gray", width=24),
#             w=80,
#             size="xs",
#             id=self.ids.z_score,
#             disabled=True,
#         )

#         xyz_fieldset = dmc.Fieldset(
#             children=[
#                 dmc.SimpleGrid(cols=3, spacing="xs", children=[x_score, y_score, z_score]),
#                 xyz_help_modal,
#                 zones_help_modal,
#             ],
#             radius="sm",
#             legend=xyz_help_legend,
#         )

#         # --- Filters (groups / categories) ---
#         self.cats_df = fletch_cats()

#         gr_data = (
#             self.cats_df[["gr_id", "gr_name"]]
#             .dropna(subset=["gr_id", "gr_name"])
#             .drop_duplicates()
#             .sort_values("gr_name", key=lambda s: s.str.lower())
#             .assign(gr_id=lambda x: x["gr_id"].astype(str))
#             .rename(columns={"gr_id": "value", "gr_name": "label"})
#             .to_dict(orient="records")
#         )

#         gr_ms = dmc.MultiSelect(
#             id=self.ids.gr_ms,
#             label="Группы",
#             placeholder="Выберите группу",
#             data=gr_data,
#             w="100%",
#             radius=0,
#             clearable=True,
#             searchable=True,
#             leftSection=DashIconify(icon="tabler:folders"),
#         )

#         cat_ms = dmc.MultiSelect(
#             id=self.ids.cat_ms,
#             label="Категория",
#             placeholder="Выберите категорию",
#             data=[],
#             w="100%",
#             radius=0,
#             clearable=True,
#             searchable=True,
#             leftSection=DashIconify(icon="tabler:tag"),
#         )

#         cats_ms_fieldset = dmc.Fieldset(
#             children=[gr_ms, cat_ms, filter_help_modal],
#             radius="sm",
#             legend=filter_help_legend,
#         )
        
#         zones_help_block = dmc.Group(
#             gap="xs",
#             align="center",
#             children=[
#                 # dmc.Text("Как читать зоны", fw=600),
#                 zones_help_legend, 
#             ],
#         )



#         # --- Groupby switch (пока не используешь — но пусть будет) ---
#         groupby_sc_switch = dmc.Switch(
#             onLabel="ON",
#             offLabel="OFF",
#             radius="sm",
#             labelPosition="right",
#             label="Групировать по подкатегориям",
#             checked=False,
#             id=self.ids.groupby_sc,
#         )
#         groupby_sc_fieldset = dmc.Fieldset(
#             children=[groupby_sc_switch],
#             radius="sm",
#             legend="Групировки номенклатур",
#         )

#         # --- ROP / SS ---
#         lead_time = dmc.NumberInput(
#             value=2,
#             min=0.5,
#             max=24,
#             step=1,
#             allowDecimal=True,
#             suffix=" мес.",
#             leftSection=DashIconify(icon="mdi:tool-time", color="red", width=24),
#             w=120,
#             size="xs",
#             id=self.ids.lead_time,
#         )
#         service_ratio = dmc.NumberInput(
#             value=95,
#             min=70,
#             max=99,
#             step=1,
#             allowDecimal=False,
#             suffix="%",
#             leftSection=DashIconify(icon="medical-icon:interpreter-services", color="blue", width=24),
#             w=120,
#             size="xs",
#             id=self.ids.service_ratio,
#         )

#         rop_fieldset = dmc.Fieldset(
#             children=[
#                 dmc.SimpleGrid(cols=2, spacing="md", children=[lead_time, service_ratio]),
#                 rop_help_modal,
#             ],
#             radius="sm",
#             legend=rop_help_legend,
#         )

#         # --- Launch button ---
#         launch_btn = dmc.Button(
#             "Рассчитать",
#             id=self.ids.launch,
#             leftSection=DashIconify(icon="mynaui:rocket-solid", width=24),
#             fullWidth=True,
#         )

#         # --- Final layout ---
#         # Панель используется внутри Drawer, поэтому без отдельного заголовка
#         # и без лишних больших вертикальных отступов.
#         self.left_section_layout = dmc.Stack(
#             gap="md",
#             children=[
#                 abc_fieldset,
#                 xyz_fieldset,
#                 zones_help_block,
#                 rop_fieldset,
#                 cats_ms_fieldset,
#                 # groupby_sc_fieldset,  # включишь, когда реально понадобится
#                 dmc.Space(h=4),
#                 launch_btn,
#             ],
#         )

#     def register_callbacks(self, app):
#         # фильтр категорий при выбранной группе
#         @app.callback(
#             Output(self.ids.cat_ms, "data"),
#             Input(self.ids.gr_ms, "value"),
#             prevent_initial_call=True,
#         )
#         def filter_cat_ms(gr_list):
#             if not gr_list:
#                 return []
#             gr_list_int = [int(x) for x in gr_list]
#             df = self.cats_df[self.cats_df["gr_id"].isin(gr_list_int)]

#             return (
#                 df[["cat_id", "cat_name"]]
#                 .dropna(subset=["cat_id", "cat_name"])
#                 .drop_duplicates()
#                 .sort_values("cat_name", key=lambda s: s.str.lower())
#                 .assign(cat_id=lambda x: x["cat_id"].astype(str))
#                 .rename(columns={"cat_id": "value", "cat_name": "label"})
#                 .to_dict(orient="records")
#             )

#         # автопересчет abc
#         @app.callback(
#             Output(self.ids.b_score, "value"),
#             Output(self.ids.c_score, "value"),
#             Output(self.ids.b_score, "max"),
#             Output(self.ids.c_score, "max"),
#             Input(self.ids.a_score, "value"),
#             prevent_initial_call=True,
#         )
#         def split_bc(a_val):
#             r = 100 - a_val
#             b = math.ceil(r / 2)
#             c = 100 - b - a_val
#             return b, c, r - 1, r - 1

#         @app.callback(
#             Output(self.ids.c_score, "value", allow_duplicate=True),
#             Input(self.ids.b_score, "value"),
#             State(self.ids.a_score, "value"),
#             prevent_initial_call=True,
#         )
#         def adjust_c(b_val, a_val):
#             return 100 - b_val - a_val

#         # автопересчет xyz
#         @app.callback(
#             Output(self.ids.y_score, "value"),
#             Output(self.ids.y_score, "min"),
#             Output(self.ids.z_score, "value"),
#             Input(self.ids.x_score, "value"),
#             State(self.ids.y_score, "value"),
#             prevent_initial_call=True,
#         )
#         def set_yz(x_val, y_val):
#             y_min = x_val + 0.5
#             z = y_val if (y_val is not None and y_val > y_min) else y_min
#             return z, y_min, z

#         @app.callback(
#             Output(self.ids.z_score, "value", allow_duplicate=True),
#             Input(self.ids.y_score, "value"),
#             prevent_initial_call=True,
#         )
#         def set_z(y_val):
#             return y_val

#         # --- open/close modals ---
#         @app.callback(
#             Output(self.ids.abc_help_modal, "opened"),
#             Input(self.ids.abc_help_open, "n_clicks"),
#             State(self.ids.abc_help_modal, "opened"),
#             prevent_initial_call=True,
#         )
#         def toggle_abc_help(n, opened):
#             return not opened

#         @app.callback(
#             Output(self.ids.xyz_help_modal, "opened"),
#             Input(self.ids.xyz_help_open, "n_clicks"),
#             State(self.ids.xyz_help_modal, "opened"),
#             prevent_initial_call=True,
#         )
#         def toggle_xyz_help(n, opened):
#             return not opened

#         @app.callback(
#             Output(self.ids.rop_help_modal, "opened"),
#             Input(self.ids.rop_help_open, "n_clicks"),
#             State(self.ids.rop_help_modal, "opened"),
#             prevent_initial_call=True,
#         )
#         def toggle_rop_help(n, opened):
#             return not opened

#         @app.callback(
#             Output(self.ids.filter_help_modal, "opened"),
#             Input(self.ids.filter_help_open, "n_clicks"),
#             State(self.ids.filter_help_modal, "opened"),
#             prevent_initial_call=True,
#         )
#         def toggle_filter_help(n, opened):
#             return not opened
        
#         @app.callback(
#             Output(self.ids.zones_help_modal, "opened"),
#             Input(self.ids.zones_help_open, "n_clicks"),
#             State(self.ids.zones_help_modal, "opened"),
#             prevent_initial_call=True,
#         )
#         def toggle_zones_help(n, opened):
#             return not opened



# # --------------------------
# # Right section (matrix grid + drawer)
# # --------------------------
# class RightSection:
#     def __init__(self):
#         self.ids = MatrixRightIds()

        
#         self.layout = dmc.Container(
#             children=[
#                 # ✅ 1) тут будет header (пока пусто)
#                 dmc.Container(id=self.ids.header, fluid=True, px=0),

#                 dmc.Space(h=16),

#                 # ✅ 2) дальше как было — content / loading
#                 dcc.Loading(
#                     id=self.ids.loading,
#                     type="graph",
#                     fullscreen=False,
#                     children=dmc.Container(
#                         id=self.ids.content,
#                         fluid=True,
#                         px=0,
#                         children=render_matrix_empty_state(),
#                     ),
#                 ),

#                 dcc.Download(id=self.ids.download),
#                 dcc.Download(id=self.ids.download_csv),      # быстрый CSV

#                 dmc.Drawer(
#                     id=self.ids.barcode_drawer,
#                     title=None,
#                     opened=False,
#                     position="right",
#                     size=520,
#                     overlayProps={"opacity": 0.45, "blur": 2},
#                     children=dmc.Container(id=self.ids.barcode_drawer_body, fluid=True),
#                 ),
#             ],
#             id=self.ids.right_container,
#             fluid=True,
#             px=0,
#         )



#     def get_matrix(self, start, end, cat, threholds, lt, sr) -> pd.DataFrame:
#         return matrix_calculation(start, end, cat, threholds, lt, sr)

#     def matrix_ag_grid(self, df: pd.DataFrame, rrgrid_className: str):
#         column_defs = get_matrix_column_defs(df)
#         grid_opts = get_matrix_grid_options()
#         row_data = df.to_dict("records")

#         return dag.AgGrid(
#             id=self.ids.matrix_grid,
#             rowData=row_data,
#             columnDefs=column_defs,
#             defaultColDef={"sortable": True, "filter": True, "resizable": True},
#             dashGridOptions=grid_opts,
#             style={
#                 "height": "calc(100vh - 330px)",
#                 "minHeight": "600px",
#                 "width": "100%",
#                 "--ag-font-size": "12px",
#                 "--ag-row-height": "34px",
#                 "--ag-header-height": "36px",
#                 "--ag-list-item-height": "30px",
#                 "--ag-grid-size": "5px",
#                 "--ag-cell-horizontal-padding": "10px",
#                 "--ag-header-column-separator-display": "block",
#                 "--ag-header-column-separator-height": "45%",
#                 "--ag-border-color": "#DCE3EA",
#                 "--ag-row-border-color": "#E6EBF0",
#             },
#             className=rrgrid_className,
#             dangerously_allow_code=True,
#         )

#     @staticmethod
#     def _fmt_qty(value) -> str:
#         try:
#             return f"{float(value):,.0f}".replace(",", " ")
#         except (TypeError, ValueError):
#             return "0"

#     def build_stock_summary(self, df: pd.DataFrame):
#         """
#         Компактное summary по текущему набору строк.

#         Важно:
#         сюда входят и товары без продаж, если они есть в текущих остатках.
#         """
#         if df is None or df.empty:
#             values = {
#                 "stock": 0,
#                 "ordered": 0,
#                 "no_stock": 0,
#                 "deficit": 0,
#                 "excess": 0,
#                 "no_sales_stock": 0,
#             }
#         else:
#             stock_available = pd.to_numeric(
#                 df.get("stock_available", 0),
#                 errors="coerce",
#             ).fillna(0)

#             stock_ordered = pd.to_numeric(
#                 df.get("stock_ordered", 0),
#                 errors="coerce",
#             ).fillna(0)

#             statuses = df.get(
#                 "stock_status",
#                 pd.Series("", index=df.index),
#             ).fillna("").astype(str)

#             values = {
#                 "stock": float(stock_available.sum()),
#                 "ordered": float(stock_ordered.sum()),
#                 "no_stock": int((stock_available <= 0).sum()),
#                 "deficit": int((statuses == "Дефицит").sum()),
#                 "excess": int((statuses == "Избыток > 6 мес.").sum()),
#                 "no_sales_stock": int(
#                     (statuses == "Без продаж, есть запас").sum()
#                 ),
#             }

#         cards = [
#             {
#                 "label": "Доступный остаток",
#                 "value": self._fmt_qty(values["stock"]),
#                 "unit": "ед.",
#                 "icon": "tabler:box",
#                 "icon_color": "#3B82F6",
#                 "icon_bg": "#EFF6FF",
#             },
#             {
#                 "label": "Заказано",
#                 "value": self._fmt_qty(values["ordered"]),
#                 "unit": "ед.",
#                 "icon": "tabler:truck-delivery",
#                 "icon_color": "#7C3AED",
#                 "icon_bg": "#F5F3FF",
#             },
#             {
#                 "label": "Нет в наличии",
#                 "value": self._fmt_qty(values["no_stock"]),
#                 "unit": "SKU",
#                 "icon": "tabler:circle-minus",
#                 "icon_color": "#EA580C",
#                 "icon_bg": "#FFF7ED",
#             },
#             {
#                 "label": "Дефицит ниже ROP",
#                 "value": self._fmt_qty(values["deficit"]),
#                 "unit": "SKU",
#                 "icon": "tabler:alert-triangle",
#                 "icon_color": "#DC2626",
#                 "icon_bg": "#FEF2F2",
#             },
#             {
#                 "label": "Запас > 6 мес.",
#                 "value": self._fmt_qty(values["excess"]),
#                 "unit": "SKU",
#                 "icon": "tabler:chart-line",
#                 "icon_color": "#16A34A",
#                 "icon_bg": "#F0FDF4",
#             },
#             {
#                 "label": "Без продаж, есть запас",
#                 "value": self._fmt_qty(values["no_sales_stock"]),
#                 "unit": "SKU",
#                 "icon": "tabler:archive",
#                 "icon_color": "#059669",
#                 "icon_bg": "#ECFDF5",
#             },
#         ]

#         return dmc.SimpleGrid(
#             cols=6,
#             spacing="sm",
#             children=[
#                 dmc.Paper(
#                     withBorder=True,
#                     radius=0,
#                     px=14,
#                     py=10,
#                     style={
#                         "minHeight": "70px",
#                         "borderColor": "#DCE3EA",
#                         "backgroundColor": "#FFFFFF",
#                     },
#                     children=[
#                         dmc.Group(
#                             gap="sm",
#                             wrap="nowrap",
#                             align="center",
#                             children=[
#                                 dmc.Center(
#                                     w=34,
#                                     h=34,
#                                     style={
#                                         "backgroundColor": card["icon_bg"],
#                                         "flex": "0 0 34px",
#                                     },
#                                     children=DashIconify(
#                                         icon=card["icon"],
#                                         width=21,
#                                         color=card["icon_color"],
#                                     ),
#                                 ),
#                                 dmc.Stack(
#                                     gap=1,
#                                     children=[
#                                         dmc.Text(
#                                             card["label"],
#                                             size="xs",
#                                             c="dimmed",
#                                             fw=500,
#                                         ),
#                                         dmc.Group(
#                                             gap=5,
#                                             align="baseline",
#                                             children=[
#                                                 dmc.Text(
#                                                     card["value"],
#                                                     size="lg",
#                                                     fw=700,
#                                                     lh=1.05,
#                                                 ),
#                                                 dmc.Text(
#                                                     card["unit"],
#                                                     size="xs",
#                                                     c="dimmed",
#                                                 ),
#                                             ],
#                                         ),
#                                     ],
#                                 ),
#                             ],
#                         ),
#                     ],
#                 )
#                 for card in cards
#             ],
#         )

#     def maxrix_layout(self, df: pd.DataFrame, rrgrid_className: str) -> dmc.Container:
#         matrix_dag = self.matrix_ag_grid(df, rrgrid_className)

#         return dmc.Container(
#             [
#                 dmc.Container(
#                     id=self.ids.summary,
#                     fluid=True,
#                     px=0,
#                     children=self.build_stock_summary(df),
#                 ),
#                 dmc.Space(h=12),
#                 matrix_dag,
#                 dmc.Space(h=40),
#             ],
#             fluid=True,
#         )

#     def build_header(self):
#         """
#         Панель над таблицей.

#         Заголовок страницы находится выше, поэтому здесь оставляем только
#         рабочие фильтры и кнопки выгрузки. Это экономит вертикальное место.
#         """
#         return dmc.Group(
#             justify="space-between",
#             align="flex-end",
#             wrap="wrap",
#             gap="sm",
#             children=[
#                 dmc.Group(
#                     gap="sm",
#                     align="flex-end",
#                     wrap="wrap",
#                     children=[
#                         dmc.MultiSelect(
#                             id=self.ids.manu_ms,
#                             label="Производитель",
#                             placeholder="Все производители",
#                             data=[],
#                             value=[],
#                             clearable=True,
#                             searchable=True,
#                             w=300,
#                             size="sm",
#                             radius=0,
#                             leftSection=DashIconify(
#                                 icon="tabler:building-factory-2",
#                                 width=18,
#                             ),
#                         ),
#                         dmc.MultiSelect(
#                             id=self.ids.stock_status_ms,
#                             label="Статус запаса",
#                             placeholder="Все статусы",
#                             data=[],
#                             value=[],
#                             clearable=True,
#                             searchable=True,
#                             w=300,
#                             size="sm",
#                             radius=0,
#                             leftSection=DashIconify(
#                                 icon="tabler:packages",
#                                 width=18,
#                             ),
#                         ),
#                     ],
#                 ),
#                 dmc.Group(
#                     gap="xs",
#                     align="center",
#                     children=[
#                         dmc.Badge(
#                             "0 SKU",
#                             id=self.ids.manu_badge,
#                             variant="light",
#                             radius=0,
#                             size="lg",
#                         ),
#                         dmc.Tooltip(
#                             label="Скачать Excel",
#                             withArrow=True,
#                             children=dmc.ActionIcon(
#                                 DashIconify(
#                                     icon="mdi:file-excel-outline",
#                                     width=19,
#                                 ),
#                                 id=self.ids.download_btn,
#                                 variant="light",
#                                 color="green",
#                                 radius=0,
#                                 size="lg",
#                                 disabled=True,
#                             ),
#                         ),
#                         dmc.Tooltip(
#                             label="Скачать CSV",
#                             withArrow=True,
#                             children=dmc.ActionIcon(
#                                 DashIconify(
#                                     icon="mdi:file-delimited-outline",
#                                     width=19,
#                                 ),
#                                 id=self.ids.download_csv_btn,
#                                 variant="light",
#                                 color="blue",
#                                 radius=0,
#                                 size="lg",
#                                 disabled=True,
#                             ),
#                         ),
#                     ],
#                 ),
#             ],
#         )




# # --------------------------
# # Main window (compose + callbacks)
# # --------------------------
# class MainWindow:
#     def __init__(self):
#         self.ls = LeftSection()
#         self.rs = RightSection()
#         self.mslider_id = "mslider-id-for-matrix-calculations"
#         self.mslider = MonthSlider(id=self.mslider_id)

#     def layout(self):
#         """
#         Основная страница занимает всю доступную ширину.

#         Настройки убраны в Drawer:
#         - таблица больше не теряет 25% ширины;
#         - после расчёта пользователь работает только с результатом;
#         - настройки всегда доступны по кнопке сверху.
#         """
#         settings_drawer = dmc.Drawer(
#             id=self.ls.ids.settings_drawer,
#             title=dmc.Group(
#                 gap="xs",
#                 children=[
#                     DashIconify(
#                         icon="tabler:adjustments-horizontal",
#                         width=22,
#                     ),
#                     dmc.Text(
#                         "Настройки матрицы",
#                         fw=700,
#                         size="lg",
#                     ),
#                 ],
#             ),
#             opened=False,
#             position="left",
#             size=430,
#             padding="lg",
#             overlayProps={
#                 "opacity": 0.30,
#                 "blur": 1,
#             },
#             children=self.ls.left_section_layout,
#         )

#         page_header = dmc.Group(
#             justify="space-between",
#             align="center",
#             wrap="wrap",
#             gap="md",
#             children=[
#                 dmc.Stack(
#                     gap=2,
#                     children=[
#                         dmc.Title(
#                             "Ассортиментная матрица",
#                             order=2,
#                         ),
#                         dmc.Text(
#                             "ABC/XYZ-анализ, спрос, ROP и актуальные остатки",
#                             size="sm",
#                             c="dimmed",
#                         ),
#                     ],
#                 ),
#                 dmc.Button(
#                     "Настройки",
#                     id=self.ls.ids.settings_open,
#                     variant="outline",
#                     radius=0,
#                     size="sm",
#                     leftSection=DashIconify(
#                         icon="tabler:adjustments-horizontal",
#                         width=18,
#                     ),
#                 ),
#             ],
#         )

#         period_block = dmc.Paper(
#             withBorder=True,
#             radius=0,
#             px="md",
#             py="xs",
#             children=[
#                 dmc.Group(
#                     gap="xs",
#                     align="center",
#                     mb=4,
#                     children=[
#                         DashIconify(
#                             icon="tabler:calendar-month",
#                             width=18,
#                         ),
#                         dmc.Text(
#                             "Период анализа",
#                             fw=600,
#                             size="sm",
#                         ),
#                     ],
#                 ),
#                 self.mslider,
#             ],
#         )

#         return dmc.Container(
#             fluid=True,
#             px=24,
#             py=16,
#             children=[
#                 settings_drawer,
#                 dcc.Store(id=self.rs.ids.store),

#                 page_header,
#                 dmc.Space(h=14),

#                 period_block,
#                 dmc.Space(h=14),

#                 self.rs.layout,
#             ],
#         )

#     def register_callbacks(self, app):
#         self.ls.register_callbacks(app)

#         # --------------------------------------------------------------
#         # Настройки: открыть кнопкой, закрыть автоматически после расчёта
#         # --------------------------------------------------------------
#         @app.callback(
#             Output(self.ls.ids.settings_drawer, "opened"),
#             Input(self.ls.ids.settings_open, "n_clicks"),
#             Input(self.ls.ids.launch, "n_clicks"),
#             State(self.ls.ids.settings_drawer, "opened"),
#             prevent_initial_call=True,
#         )
#         def toggle_settings_drawer(open_clicks, launch_clicks, opened):
#             trigger = ctx.triggered_id

#             if trigger == self.ls.ids.settings_open:
#                 return not opened

#             if trigger == self.ls.ids.launch:
#                 return False

#             return opened

#         @app.callback(
#             Output(self.rs.ids.header, "children"),
#             Output(self.rs.ids.content, "children"),
#             Output(self.rs.ids.store, "data"),
#             Input(self.ls.ids.launch, "n_clicks"),
#             State(self.ls.ids.a_score, "value"),
#             State(self.ls.ids.b_score, "value"),
#             State(self.ls.ids.c_score, "value"),
#             State(self.ls.ids.x_score, "value"),
#             State(self.ls.ids.y_score, "value"),
#             State(self.ls.ids.z_score, "value"),
#             State(self.ls.ids.gr_ms, "value"),
#             State(self.ls.ids.cat_ms, "value"),
#             State(self.mslider_id, "value"),
#             State(self.ls.ids.lead_time, "value"),
#             State(self.ls.ids.service_ratio, "value"),
#             State("theme_switch", "checked"),
#             prevent_initial_call=True,
#         )
#         def get_matrix(nclicks, a, b, c, x, y, z, grs, cats, ms, lt, sr, theme):
#             if not nclicks:
#                 return no_update, no_update, no_update

#             def find_cats_if_gr():
#                 gr_list_int = [int(v) for v in (grs or [])]
#                 df = self.ls.cats_df[self.ls.cats_df["gr_id"].isin(gr_list_int)]
#                 return df["cat_id"].to_list()

#             threholds = {"a": a, "b": b, "c": c, "x": x, "y": y, "z": z}
#             start, end = id_to_months(ms[0], ms[1])

#             gr = None if not grs else ",".join(grs)
#             cat = None if not cats else ",".join(cats)
#             if gr and not cat:
#                 cat = ",".join(map(str, find_cats_if_gr()))

#             rrgrid_className = "ag-theme-alpine-dark" if theme else "ag-theme-alpine"

#             df_matrix = matrix_calculation(start, end, cat, threholds, lt, sr)
#             store_json = df_matrix.to_json(date_format="iso", orient="records")

#             # header создаём (он уже содержит manu_ms, badge, download_btn)
#             header = self.rs.build_header()

#             # таблица
#             content = self.rs.maxrix_layout(df_matrix, rrgrid_className)

#             return header, content, store_json

        
#         @app.callback(
#             Output(self.rs.ids.matrix_grid, "rowData"),
#             Output(self.rs.ids.manu_badge, "children"),
#             Output(self.rs.ids.download_btn, "disabled"),
#             Output(self.rs.ids.download_csv_btn, "disabled"),
#             Output(self.rs.ids.summary, "children"),
#             Input(self.rs.ids.manu_ms, "value"),
#             Input(self.rs.ids.stock_status_ms, "value"),
#             State(self.rs.ids.store, "data"),
#         )
#         def filter_matrix(manu_values, stock_status_values, store_json):
#             if not store_json:
#                 return no_update, no_update, no_update, no_update, no_update

#             df_all = pd.read_json(StringIO(store_json), orient="records")
#             df = df_all.copy()

#             if manu_values:
#                 df = df[
#                     df["manu"]
#                     .fillna("Нет производителя")
#                     .astype(str)
#                     .isin(manu_values)
#                 ]

#             if stock_status_values and "stock_status" in df.columns:
#                 df = df[
#                     df["stock_status"]
#                     .fillna("")
#                     .astype(str)
#                     .isin(stock_status_values)
#                 ]

#             badge = (
#                 f"{len(df):,}/{len(df_all):,} SKU"
#                 .replace(",", " ")
#             )

#             summary = self.rs.build_stock_summary(df)

#             return (
#                 df.to_dict("records"),
#                 badge,
#                 False,
#                 False,
#                 summary,
#             )


#         @app.callback(
#             Output(self.rs.ids.manu_ms, "data"),
#             Output(self.rs.ids.stock_status_ms, "data"),
#             Input(self.rs.ids.store, "data"),
#             prevent_initial_call=True,
#         )
#         def fill_right_filters(store_json):
#             if not store_json:
#                 return [], []

#             df = pd.read_json(StringIO(store_json), orient="records")

#             manu_list = (
#                 df["manu"]
#                 .fillna("Нет производителя")
#                 .astype(str)
#                 .sort_values(key=lambda s: s.str.lower())
#                 .unique()
#                 .tolist()
#             )

#             status_order = [
#                 "Без продаж, есть запас",
#                 "Дефицит",
#                 "Нет остатка",
#                 "Нет остатка, товар заказан",
#                 "Ниже ROP, заказ закрывает",
#                 "Достаточно",
#                 "Избыток > 6 мес.",
#             ]

#             existing_statuses = set(
#                 df.get(
#                     "stock_status",
#                     pd.Series(dtype="object"),
#                 )
#                 .dropna()
#                 .astype(str)
#                 .tolist()
#             )

#             statuses = [
#                 status
#                 for status in status_order
#                 if status in existing_statuses
#             ]

#             statuses.extend(
#                 sorted(
#                     existing_statuses.difference(statuses),
#                     key=str.lower,
#                 )
#             )

#             return (
#                 [{"value": m, "label": m} for m in manu_list],
#                 [{"value": s, "label": s} for s in statuses],
#             )


#         @app.callback(
#             Output(self.rs.ids.download, "data"),
#             Input(self.rs.ids.download_btn, "n_clicks"),
#             State(self.rs.ids.store, "data"),
#             State(self.rs.ids.manu_ms, "value"),
#             State(self.rs.ids.stock_status_ms, "value"),
#             State(self.mslider_id, "value"),
#             prevent_initial_call=True,
#         )
#         def download_excel(n, store_json, manu_values, stock_status_values, ms):
#             if not n or not store_json:
#                 return no_update

#             df_matrix = pd.read_json(StringIO(store_json), orient="records")


#             if manu_values:
#                 df_matrix = df_matrix[
#                     df_matrix["manu"]
#                     .fillna("Нет производителя")
#                     .astype(str)
#                     .isin(manu_values)
#                 ]

#             if stock_status_values and "stock_status" in df_matrix.columns:
#                 df_matrix = df_matrix[
#                     df_matrix["stock_status"]
#                     .fillna("")
#                     .astype(str)
#                     .isin(stock_status_values)
#                 ]

#             start, end = id_to_months(ms[0], ms[1])

#             xlsx_bytes = build_matrix_excel_bytes(ENGINE, df_matrix=df_matrix, start=start, end=end)

#             stamp = datetime.now().strftime("%Y%m%d_%H%M")
#             filename = f"matrix_{start}_{end}_{stamp}.xlsx"
#             # return dcc.send_bytes(xlsx_bytes, filename)
#             return dcc.send_bytes(lambda buffer: buffer.write(xlsx_bytes), filename)
        
        
#         @app.callback(
#             Output(self.rs.ids.download_csv, "data"),
#             Input(self.rs.ids.download_csv_btn, "n_clicks"),
#             State(self.rs.ids.store, "data"),
#             State(self.rs.ids.manu_ms, "value"),
#             State(self.rs.ids.stock_status_ms, "value"),
#             State(self.mslider_id, "value"),
#             prevent_initial_call=True,
#         )
#         def download_csv(n, store_json, manu_values, stock_status_values, ms):
#             if not n or not store_json:
#                 return no_update

#             df = pd.read_json(StringIO(store_json), orient="records")

#             if manu_values:
#                 df = df[
#                     df["manu"]
#                     .fillna("Нет производителя")
#                     .astype(str)
#                     .isin(manu_values)
#                 ]

#             if stock_status_values and "stock_status" in df.columns:
#                 df = df[
#                     df["stock_status"]
#                     .fillna("")
#                     .astype(str)
#                     .isin(stock_status_values)
#                 ]

#             # пользовательские названия для CSV
#             csv_rename = {
#                 "fullname": "Номенклатура",
#                 "article": "Артикул",
#                 "manu": "Производитель",
#                 "abc": "ABC",
#                 "xyz": "XYZ",
#                 "cat_name": "Категория",
#                 "sc_name": "Подкатегория",
#                 "amount": "Выручка",
#                 "quant": "Кол-во",
#                 "share": "Доля выручки",
#                 "mean_amount": "Ср. выручка",
#                 "share_mean": "Доля в ср выручке",
#                 "mean_month": "Ср. продажи, ед/мес.",
#                 "std_month": "Ст. отклонение",
#                 "cv": "CV",
#                 "ss": "SS, ед.",
#                 "rop": "ROP, ед.",
#                 "stock_date": "Дата остатков",
#                 "stock_available": "Остаток доступно",
#                 "stock_ordered": "Заказано",
#                 "stock_total": "Остаток итого",
#                 "stock_cover_months": "Покрытие, мес.",
#                 "stock_cover_months_total": "Покрытие с заказами, мес.",
#                 "stock_vs_rop": "Отклонение доступного от ROP",
#                 "stock_total_vs_rop": "Отклонение итого от ROP",
#                 "order_need": "Нужно заказать",
#                 "stock_status": "Статус остатка",
#             }

#             dynamic_rename = {}
#             for column in df.columns:
#                 if str(column).startswith("stock_wh::"):
#                     dynamic_rename[column] = (
#                         "Остаток | "
#                         + str(column).split("::", 1)[1]
#                     )
#                 elif str(column).startswith("ordered_wh::"):
#                     dynamic_rename[column] = (
#                         "Заказано | "
#                         + str(column).split("::", 1)[1]
#                     )

#             df = df.rename(
#                 columns={
#                     **csv_rename,
#                     **dynamic_rename,
#                 }
#             )

#             # служебные колонки
#             df = df.drop(
#                 columns=[
#                     "date_json",
#                     "quant_json",
#                     "ls_quant",
#                     "ls_date",
#                     "is_quant",
#                     "is_date",
#                     "cum_share",
#                     "_amount",
#                     "_share",
#                 ],
#                 errors="ignore",
#             )

#             start, end = id_to_months(ms[0], ms[1])
#             stamp = datetime.now().strftime("%Y%m%d_%H%M")
#             filename = f"matrix_{start}_{end}_{stamp}.csv"

#             return dcc.send_data_frame(df.to_csv, filename, index=False, sep="|", encoding="utf-8-sig")





#         @app.callback(
#             Output(self.rs.ids.barcode_drawer, "opened"),
#             Output(self.rs.ids.barcode_drawer_body, "children"),
#             Input(self.rs.ids.matrix_grid, "selectedRows"),
#             State(self.mslider_id, "value"),
#             prevent_initial_call=True,
#         )
#         def open_barcode_details(selected_rows, ms):
#             if not selected_rows:
#                 return False, no_update

#             row = selected_rows[0]
#             item_id = int(row["item_id"])
#             fullname = row.get("fullname", "")

#             start, end = id_to_months(ms[0], ms[1])

#             df_bc = fetch_barcode_breakdown(ENGINE, item_id=item_id, start=start, end=end)
#             panel = render_barcode_panel(
#                         df_bc,
#                         title_name=fullname,
#                         subtitle=f"item_id = {item_id}",
#                     )

#             # panel = render_barcode_panel(df_bc, title=f"{fullname} (item_id={item_id})")

#             return True, panel




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
from .data import ENGINE, fletch_cats, matrix_calculation

from .grid_specs import get_matrix_column_defs, get_matrix_grid_options
from .help_texts import ABC_HELP_MD, XYZ_HELP_MD, ROP_HELP_MD, FILTER_HELP_MD, MATRIX_ZONES_HELP_MD
from .ui_builders import build_help
from .ids import MatrixIds, MatrixRightIds
from .empty_state import render_matrix_empty_state
from .export_excel import build_matrix_excel_bytes
from .charts import build_stock_sunburst
from datetime import datetime

locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")


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
            id=self.ids.matrix_grid,
            rowData=row_data,
            columnDefs=column_defs,
            defaultColDef={"sortable": True, "filter": True, "resizable": True},
            dashGridOptions=grid_opts,
            style={
                "height": "calc(100vh - 330px)",
                "minHeight": "600px",
                "width": "100%",
                "--ag-font-size": "12px",
                "--ag-row-height": "34px",
                "--ag-header-height": "36px",
                "--ag-list-item-height": "30px",
                "--ag-grid-size": "5px",
                "--ag-cell-horizontal-padding": "10px",
                "--ag-header-column-separator-display": "block",
                "--ag-header-column-separator-height": "45%",
                "--ag-border-color": "#DCE3EA",
                "--ag-row-border-color": "#E6EBF0",
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
                        "borderColor": "#DCE3EA",
                        "backgroundColor": "#FFFFFF",
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

    def build_stock_sunburst_block(
        self,
        df: pd.DataFrame,
        value_field: str = "stock_available",
    ):
        """
        Большой интерактивный Sunburst по структуре запасов.

        Иерархия:
            Категория → Подкатегория → Номенклатура

        Переключатель позволяет смотреть:
            - доступный остаток;
            - доступный + заказанный;
            - только заказанный.
        """
        figure = build_stock_sunburst(
            df,
            value_field=value_field,
        )

        return dmc.Paper(
            withBorder=True,
            radius=0,
            p=0,
            style={
                "borderColor": "#DCE3EA",
                "backgroundColor": "#FFFFFF",
                "overflow": "hidden",
            },
            children=[
                dmc.Group(
                    justify="space-between",
                    align="center",
                    wrap="wrap",
                    gap="sm",
                    px="md",
                    py="sm",
                    style={
                        "borderBottom": "1px solid #E5E7EB",
                    },
                    children=[
                        dmc.Group(
                            gap="xs",
                            children=[
                                DashIconify(
                                    icon="tabler:chart-donut-4",
                                    width=20,
                                    color="#2563EB",
                                ),
                                dmc.Stack(
                                    gap=0,
                                    children=[
                                        dmc.Text(
                                            "Структура запасов",
                                            fw=700,
                                            size="sm",
                                        ),
                                        dmc.Text(
                                            "Категория → подкатегория → номенклатура",
                                            size="xs",
                                            c="dimmed",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dmc.SegmentedControl(
                            id="matrix-stock-metric",
                            value=value_field,
                            size="xs",
                            radius=0,
                            data=[
                                {
                                    "label": "Доступно",
                                    "value": "stock_available",
                                },
                                {
                                    "label": "С заказами",
                                    "value": "stock_total",
                                },
                                {
                                    "label": "Заказано",
                                    "value": "stock_ordered",
                                },
                            ],
                        ),
                    ],
                ),
                dcc.Graph(
                    id="matrix-stock-sunburst",
                    figure=figure,
                    config={
                        "displaylogo": False,
                        "responsive": True,
                        "scrollZoom": False,
                    },
                    style={
                        "width": "100%",
                        "height": "680px",
                    },
                ),
            ],
        )

    def maxrix_layout(
        self,
        df: pd.DataFrame,
        rrgrid_className: str,
    ) -> dmc.Container:
        matrix_dag = self.matrix_ag_grid(
            df,
            rrgrid_className,
        )

        sunburst = self.build_stock_sunburst_block(
            df,
            value_field="stock_available",
        )

        return dmc.Container(
            [
                # ------------------------------------------------------
                # Summary
                # ------------------------------------------------------
                dmc.Container(
                    id=self.ids.summary,
                    fluid=True,
                    px=0,
                    children=self.build_stock_summary(df),
                ),

                dmc.Space(h=14),

                # ------------------------------------------------------
                # Sunburst на всю ширину
                # ------------------------------------------------------
                sunburst,

                dmc.Space(h=18),

                # ------------------------------------------------------
                # Таблица
                # ------------------------------------------------------
                matrix_dag,

                dmc.Space(h=40),
            ],
            fluid=True,
            px=0,
        )

    def build_header(self):
        """
        Панель над таблицей.

        Заголовок страницы находится выше, поэтому здесь оставляем только
        рабочие фильтры и кнопки выгрузки. Это экономит вертикальное место.
        """
        return dmc.Group(
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
                        dmc.MultiSelect(
                            id=self.ids.manu_ms,
                            label="Производитель",
                            placeholder="Все производители",
                            data=[],
                            value=[],
                            clearable=True,
                            searchable=True,
                            w=300,
                            size="sm",
                            radius=0,
                            leftSection=DashIconify(
                                icon="tabler:building-factory-2",
                                width=18,
                            ),
                        ),
                        dmc.MultiSelect(
                            id=self.ids.stock_status_ms,
                            label="Статус запаса",
                            placeholder="Все статусы",
                            data=[],
                            value=[],
                            clearable=True,
                            searchable=True,
                            w=300,
                            size="sm",
                            radius=0,
                            leftSection=DashIconify(
                                icon="tabler:packages",
                                width=18,
                            ),
                        ),
                    ],
                ),
                dmc.Group(
                    gap="xs",
                    align="center",
                    children=[
                        dmc.Badge(
                            "0 SKU",
                            id=self.ids.manu_badge,
                            variant="light",
                            radius=0,
                            size="lg",
                        ),
                        dmc.Tooltip(
                            label="Скачать Excel",
                            withArrow=True,
                            children=dmc.ActionIcon(
                                DashIconify(
                                    icon="mdi:file-excel-outline",
                                    width=19,
                                ),
                                id=self.ids.download_btn,
                                variant="light",
                                color="green",
                                radius=0,
                                size="lg",
                                disabled=True,
                            ),
                        ),
                        dmc.Tooltip(
                            label="Скачать CSV",
                            withArrow=True,
                            children=dmc.ActionIcon(
                                DashIconify(
                                    icon="mdi:file-delimited-outline",
                                    width=19,
                                ),
                                id=self.ids.download_csv_btn,
                                variant="light",
                                color="blue",
                                radius=0,
                                size="lg",
                                disabled=True,
                            ),
                        ),
                    ],
                ),
            ],
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
                dmc.Button(
                    "Настройки",
                    id=self.ls.ids.settings_open,
                    variant="outline",
                    radius=0,
                    size="sm",
                    leftSection=DashIconify(
                        icon="tabler:adjustments-horizontal",
                        width=18,
                    ),
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
            store_json = df_matrix.to_json(date_format="iso", orient="records")

            # header создаём (он уже содержит manu_ms, badge, download_btn)
            header = self.rs.build_header()

            # таблица
            content = self.rs.maxrix_layout(df_matrix, rrgrid_className)

            return header, content, store_json

        
        @app.callback(
            Output(self.rs.ids.matrix_grid, "rowData"),
            Output(self.rs.ids.manu_badge, "children"),
            Output(self.rs.ids.download_btn, "disabled"),
            Output(self.rs.ids.download_csv_btn, "disabled"),
            Output(self.rs.ids.summary, "children"),
            Output("matrix-stock-sunburst", "figure"),
            Input(self.rs.ids.manu_ms, "value"),
            Input(self.rs.ids.stock_status_ms, "value"),
            Input("matrix-stock-metric", "value"),
            State(self.rs.ids.store, "data"),
        )
        def filter_matrix(
            manu_values,
            stock_status_values,
            stock_metric,
            store_json,
        ):
            if not store_json:
                return (
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                )

            df_all = pd.read_json(StringIO(store_json), orient="records")
            df = df_all.copy()

            if manu_values:
                df = df[
                    df["manu"]
                    .fillna("Нет производителя")
                    .astype(str)
                    .isin(manu_values)
                ]

            if stock_status_values and "stock_status" in df.columns:
                df = df[
                    df["stock_status"]
                    .fillna("")
                    .astype(str)
                    .isin(stock_status_values)
                ]

            badge = (
                f"{len(df):,}/{len(df_all):,} SKU"
                .replace(",", " ")
            )

            summary = self.rs.build_stock_summary(df)

            sunburst_figure = build_stock_sunburst(
                df,
                value_field=(
                    stock_metric
                    or "stock_available"
                ),
            )

            return (
                df.to_dict("records"),
                badge,
                False,
                False,
                summary,
                sunburst_figure,
            )


        @app.callback(
            Output(self.rs.ids.manu_ms, "data"),
            Output(self.rs.ids.stock_status_ms, "data"),
            Input(self.rs.ids.store, "data"),
            prevent_initial_call=True,
        )
        def fill_right_filters(store_json):
            if not store_json:
                return [], []

            df = pd.read_json(StringIO(store_json), orient="records")

            manu_list = (
                df["manu"]
                .fillna("Нет производителя")
                .astype(str)
                .sort_values(key=lambda s: s.str.lower())
                .unique()
                .tolist()
            )

            status_order = [
                "Без продаж, есть запас",
                "Дефицит",
                "Нет остатка",
                "Нет остатка, товар заказан",
                "Ниже ROP, заказ закрывает",
                "Достаточно",
                "Избыток > 6 мес.",
            ]

            existing_statuses = set(
                df.get(
                    "stock_status",
                    pd.Series(dtype="object"),
                )
                .dropna()
                .astype(str)
                .tolist()
            )

            statuses = [
                status
                for status in status_order
                if status in existing_statuses
            ]

            statuses.extend(
                sorted(
                    existing_statuses.difference(statuses),
                    key=str.lower,
                )
            )

            return (
                [{"value": m, "label": m} for m in manu_list],
                [{"value": s, "label": s} for s in statuses],
            )


        @app.callback(
            Output(self.rs.ids.download, "data"),
            Input(self.rs.ids.download_btn, "n_clicks"),
            State(self.rs.ids.store, "data"),
            State(self.rs.ids.manu_ms, "value"),
            State(self.rs.ids.stock_status_ms, "value"),
            State(self.mslider_id, "value"),
            prevent_initial_call=True,
        )
        def download_excel(n, store_json, manu_values, stock_status_values, ms):
            if not n or not store_json:
                return no_update

            df_matrix = pd.read_json(StringIO(store_json), orient="records")


            if manu_values:
                df_matrix = df_matrix[
                    df_matrix["manu"]
                    .fillna("Нет производителя")
                    .astype(str)
                    .isin(manu_values)
                ]

            if stock_status_values and "stock_status" in df_matrix.columns:
                df_matrix = df_matrix[
                    df_matrix["stock_status"]
                    .fillna("")
                    .astype(str)
                    .isin(stock_status_values)
                ]

            start, end = id_to_months(ms[0], ms[1])

            xlsx_bytes = build_matrix_excel_bytes(ENGINE, df_matrix=df_matrix, start=start, end=end)

            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"matrix_{start}_{end}_{stamp}.xlsx"
            # return dcc.send_bytes(xlsx_bytes, filename)
            return dcc.send_bytes(lambda buffer: buffer.write(xlsx_bytes), filename)
        
        
        @app.callback(
            Output(self.rs.ids.download_csv, "data"),
            Input(self.rs.ids.download_csv_btn, "n_clicks"),
            State(self.rs.ids.store, "data"),
            State(self.rs.ids.manu_ms, "value"),
            State(self.rs.ids.stock_status_ms, "value"),
            State(self.mslider_id, "value"),
            prevent_initial_call=True,
        )
        def download_csv(n, store_json, manu_values, stock_status_values, ms):
            if not n or not store_json:
                return no_update

            df = pd.read_json(StringIO(store_json), orient="records")

            if manu_values:
                df = df[
                    df["manu"]
                    .fillna("Нет производителя")
                    .astype(str)
                    .isin(manu_values)
                ]

            if stock_status_values and "stock_status" in df.columns:
                df = df[
                    df["stock_status"]
                    .fillna("")
                    .astype(str)
                    .isin(stock_status_values)
                ]

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
            Input(self.rs.ids.matrix_grid, "selectedRows"),
            State(self.mslider_id, "value"),
            prevent_initial_call=True,
        )
        def open_barcode_details(selected_rows, ms):
            if not selected_rows:
                return False, no_update

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

            # panel = render_barcode_panel(df_bc, title=f"{fullname} (item_id={item_id})")

            return True, panel