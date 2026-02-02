# pages/matrix/main.py
import math
import locale
from io import StringIO

import pandas as pd
import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from dash import dcc, Input, Output, State, no_update

from components import NoData, MonthSlider, DATES  # noqa: F401  (NoData может пригодиться)
from .barcode_details import fetch_barcode_breakdown, render_barcode_panel
from .data import ENGINE, fletch_cats, matrix_calculation

from .grid_specs import get_matrix_column_defs, get_matrix_grid_options
from .help_texts import ABC_HELP_MD, XYZ_HELP_MD, ROP_HELP_MD, FILTER_HELP_MD
from .ui_builders import build_help
from .ids import MatrixIds, MatrixRightIds
from .empty_state import render_matrix_empty_state
from .export_excel import build_matrix_excel_bytes
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
        self.left_section_layout = dmc.Container(
            children=[
                dmc.Title("Настройки матрицы", order=4),
                dmc.Space(h=20),
                abc_fieldset,
                dmc.Space(h=20),
                xyz_fieldset,
                dmc.Space(h=20),
                rop_fieldset,
                dmc.Space(h=20),
                cats_ms_fieldset,
                # dmc.Space(h=20),
                # groupby_sc_fieldset,  # включишь, когда реально начнёшь использовать
                dmc.Space(h=20),
                launch_btn,
            ],
            fluid=True,
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


# --------------------------
# Right section (matrix grid + drawer)
# --------------------------
class RightSection:
    def __init__(self):
        self.ids = MatrixRightIds()

        
        self.layout = dmc.Container(
            children=[
                # ✅ 1) тут будет header (пока пусто)
                dmc.Container(id="matrix_header_container", fluid=True),

                dmc.Space(h=16),

                # ✅ 2) дальше как было — content / loading
                dcc.Loading(
                    id=self.ids.loading,
                    type="graph",
                    fullscreen=False,
                    children=dmc.Container(
                        id=self.ids.content,
                        fluid=True,
                        children=render_matrix_empty_state(),
                    ),
                ),

                dcc.Download(id=self.ids.download),

                dmc.Drawer(
                    id=self.ids.barcode_drawer,
                    title="Детализация по штрихкодам",
                    opened=False,
                    position="right",
                    size=520,
                    overlayProps={"opacity": 0.45, "blur": 2},
                    children=dmc.Container(id=self.ids.barcode_drawer_body, fluid=True),
                ),
            ],
            id=self.ids.right_container,
            fluid=True,
        )



    def get_matrix(self, start, end, cat, threholds, lt, sr) -> pd.DataFrame:
        return matrix_calculation(start, end, cat, threholds, lt, sr)

    def matrix_ag_grid(self, df: pd.DataFrame, rrgrid_className: str):
        column_defs = get_matrix_column_defs()
        grid_opts = get_matrix_grid_options()
        row_data = df.to_dict("records")

        return dag.AgGrid(
            id=self.ids.matrix_grid,
            rowData=row_data,
            columnDefs=column_defs,
            defaultColDef={"sortable": True, "filter": True, "resizable": True},
            dashGridOptions=grid_opts,
            style={"height": "600px", "width": "100%"},
            className=rrgrid_className,
            dangerously_allow_code=True,
        )

    def maxrix_layout(self, df: pd.DataFrame, rrgrid_className: str) -> dmc.Container:
        matrix_dag = self.matrix_ag_grid(df, rrgrid_className)
        return dmc.Container([matrix_dag, dmc.Space(h=40)], fluid=True)
    
    def build_header(self):
        return dmc.Group(
            justify="space-between",
            align="center",
            children=[
                dmc.Title("Расчет ассортиментной матрицы", order=3),
                dmc.Group(
                    gap="sm",
                    align="center",
                    children=[
                        dmc.MultiSelect(
                            id=self.ids.manu_ms,
                            placeholder="Все производители",
                            data=[],
                            value=[],
                            clearable=True,
                            searchable=True,
                            w=260,
                            size="sm",
                            radius="md",
                            leftSection=DashIconify(icon="tabler:building-factory-2", width=18),
                        ),
                        dmc.Badge("0/0", id=self.ids.manu_badge, variant="light", radius="sm"),
                        dmc.Tooltip(
                            label="Скачать Excel",
                            withArrow=True,
                            children=dmc.ActionIcon(
                                DashIconify(icon="mdi:file-excel-outline", width=18),
                                id=self.ids.download_btn,
                                variant="light",
                                color="green",
                                radius="md",
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
        return dmc.Container(
            children=[
                dmc.Title("Создание и анализ ассортиментой матрицы", order=1, c="blue"),
                dmc.Text("В данном разделе можно создавать и анализировать ассортиментные матрицы", size="xs"),
                dmc.Space(h=40),
                self.mslider,
                dcc.Store(id=self.rs.ids.store),
                dmc.Space(h=20),
                dmc.Grid(
                    [
                        dmc.GridCol([self.ls.left_section_layout], span=3),
                        dmc.GridCol([self.rs.layout], span=9),
                    ]
                ),
            ],
            fluid=True,
        )

    def register_callbacks(self, app):
        self.ls.register_callbacks(app)

        @app.callback(
            Output("matrix_header_container", "children"),
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
            Input(self.rs.ids.manu_ms, "value"),
            State(self.rs.ids.store, "data"),
        )
        def filter_by_manu(manu_values, store_json):
            if not store_json:
                return no_update, no_update, no_update

            df = pd.read_json(StringIO(store_json), orient="records")

            manu_all = (
                df["manu"].fillna("Нет производителя").astype(str)
                .sort_values(key=lambda s: s.str.lower())
                .unique()
                .tolist()
            )
            total = len(manu_all)

            if manu_values:
                df = df[df["manu"].fillna("Нет производителя").astype(str).isin(manu_values)]
                badge = f"{len(manu_values)}/{total}"
            else:
                badge = f"Все/{total}"

            return df.to_dict("records"), badge, False
        
        
        
        @app.callback(
            Output(self.rs.ids.manu_ms, "data"),
            Input(self.rs.ids.store, "data"),
            prevent_initial_call=True,
        )
        def fill_manu_options(store_json):
            if not store_json:
                return []

            df = pd.read_json(StringIO(store_json), orient="records")
            manu_list = (
                df["manu"].fillna("Нет производителя").astype(str)
                .sort_values(key=lambda s: s.str.lower())
                .unique()
                .tolist()
            )
            return [{"value": m, "label": m} for m in manu_list]







        @app.callback(
            Output(self.rs.ids.download, "data"),
            Input(self.rs.ids.download_btn, "n_clicks"),
            State(self.rs.ids.store, "data"),
            State(self.rs.ids.manu_ms, "value"),
            State(self.mslider_id, "value"),
            prevent_initial_call=True,
        )
        def download_excel(n, store_json, manu_values, ms):
            if not n or not store_json:
                return no_update

            df_matrix = pd.read_json(StringIO(store_json), orient="records")


            if manu_values:
                df_matrix = df_matrix[df_matrix["manu"].fillna("Нет производителя").astype(str).isin(manu_values)]

            start, end = id_to_months(ms[0], ms[1])

            xlsx_bytes = build_matrix_excel_bytes(ENGINE, df_matrix=df_matrix, start=start, end=end)

            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"matrix_{start}_{end}_{stamp}.xlsx"
            return dcc.send_bytes(xlsx_bytes, filename)



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
            panel = render_barcode_panel(df_bc, title=f"{fullname} (item_id={item_id})")

            return True, panel
