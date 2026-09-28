import pandas as pd
import numpy as np
import dash_mantine_components as dmc
from dash import dcc, Input, Output, State, no_update, MATCH
import locale
locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")
from components import(
    MonthSlider,
    DATES,
    COLORS_BY_COLOR,
    COLORS_BY_SHADE,
    LoadingScreen,
    ValuesRadioGroups
    
)
from datetime import datetime

from .queries import fletch_cats, cats_report, VALS_DICT, OPTIONS_SWITCHS
from .analysis import build_narrative, fmt_delta, fmt_pct, fmt_value, level_table, load_compare
from .cats_export import build_cats_excel_bytes, build_cats_reference_pdf
from dash_iconify import DashIconify
# from data import load_df_from_redis, delete_df_from_redis, save_df_to_redis


def id_to_months(start, end):
    return DATES[start].strftime("%Y-%m-%d"), DATES[end].strftime("%Y-%m-%d")








class CatsMainWindow:
    def __init__(self):
        self.mslider_id = "cats_monthslider"
        self.mslider = MonthSlider(id=self.mslider_id)
        self.df_store_id = "cats_df_store"
        self.df_store = dcc.Store(id=self.df_store_id, storage_type="session")
        self.last_update_lb_id = "cats_last_update_lb"
        self.summary_data_conteiner_id = "summary_data_conteiner_id_for_cats"
        self.value_controll_id = "value_controll_id_cats"
        self.option_controll_id = "option_controll_id_cats"
        self.excel_btn_id = "cats_export_excel_btn"
        self.pdf_btn_id = "cats_export_pdf_btn"
        self.excel_dl_id = "cats_export_excel"
        self.pdf_dl_id = "cats_export_pdf"

        self.value_controlls = dmc.SegmentedControl(
            id=self.value_controll_id,
            value="amount",
            data=[{"value": k, "label": v} for k, v in VALS_DICT.items()],
            color="teal",
            radius=0,
            size="sm",
        )
        self.option_controll = dmc.SegmentedControl(
            id=self.option_controll_id,
            value="cat",
            data=[{"value": "cat", "label": "Категории"}, {"value": "store_gr_name", "label": "Магазины"}],
            color="teal",
            radius=0,
            size="sm",
        )

    @staticmethod
    def period_bounds(slider_value):
        start, end = id_to_months(slider_value[0], slider_value[1])
        start_dt = pd.to_datetime(start) + pd.offsets.MonthBegin(-1)
        return start_dt.strftime("%Y-%m-%d"), end

    def make_tree(self):
        df = fletch_cats()
        tree = []

        def find_or_create(lst, value, label):
            for node in lst:
                if node["value"] == str(value):
                    return node
            node = {"value": str(value), "label": str(label), "children": [], "_count": 0}
            lst.append(node)
            return node

        for _, row in df.iterrows():
            pid, pname = int(row["parent_id"]), row["parent"]
            cid, cname = int(row["cat_id"]), row["cat"]
            sid, sname = row["subcat_id"], row["subcat"]
            parent_node = find_or_create(tree, f"{pid}", pname)
            cid_key = f"{pid}-{cid}"
            cat_node = find_or_create(parent_node["children"], cid_key, cname)
            if sid is not None:
                find_or_create(cat_node["children"], f"{cid_key}-{sid}", sname)
        return tree

    def export_menu(self):
        def _item(label, id_, icon, note):
            return dmc.MenuItem(
                dmc.Stack(gap=0, children=[
                    dmc.Text(label, size="sm", fw=500),
                    dmc.Text(note, size="xs", c="dimmed"),
                ]),
                id=id_,
                n_clicks=0,
                leftSection=DashIconify(icon=icon, width=18, color="#2F6656"),
            )

        return dmc.Menu(
            position="bottom-end",
            shadow="md",
            width=320,
            radius=0,
            children=[
                dmc.MenuTarget(dmc.Button(
                    "Экспорт",
                    color="teal",
                    radius=0,
                    leftSection=DashIconify(icon="tabler:download", width=18),
                    rightSection=DashIconify(icon="tabler:chevron-down", width=16),
                )),
                dmc.MenuDropdown([
                    dmc.MenuLabel("По выбранным месяцам и метрике"),
                    _item("Отчёт Excel", self.excel_btn_id, "mdi:file-excel-outline",
                          "Справка, группы, категории, подкатегории"),
                    _item("Справка PDF", self.pdf_btn_id, "tabler:file-text",
                          "Выводы обычным языком на 1–2 страницы"),
                ]),
            ],
        )

    def layout(self):
        header = dmc.Group(
            justify="space-between",
            align="center",
            wrap="wrap",
            children=[
                dmc.Stack(gap=2, children=[
                    dmc.Title("Анализ категорий", order=2),
                    dmc.Text("Сравнение последнего месяца периода с первым: группы, категории, подкатегории",
                             size="sm", c="dimmed"),
                ]),
                self.export_menu(),
            ],
        )

        controls = dmc.Paper(
            withBorder=True,
            radius=0,
            px="md",
            py="sm",
            style={"borderTop": "3px solid #2F6656"},
            children=dmc.Stack(gap="sm", children=[
                dmc.Group(gap="xs", align="center", children=[
                    DashIconify(icon="tabler:calendar-month", width=18, color="#2F6656"),
                    dmc.Text("Период", fw=600, size="sm"),
                    dcc.Loading(dmc.Badge(id=self.last_update_lb_id, variant="light", color="teal",
                                          radius=0, size="lg", tt="none")),
                ]),
                self.mslider,
                dmc.Divider(),
                dmc.Group(gap="xl", wrap="wrap", align="flex-end", children=[
                    dmc.Stack(gap=4, children=[dmc.Text("Метрика", size="xs", c="dimmed", fw=600),
                                               self.value_controlls]),
                    dmc.Stack(gap=4, children=[dmc.Text("Разрез в графиках групп", size="xs", c="dimmed", fw=600),
                                               self.option_controll]),
                ]),
            ]),
        )

        return dmc.Container(
            fluid=True,
            px=24,
            py=16,
            children=[
                header,
                dmc.Space(h=14),
                controls,
                dmc.Space(h=14),
                dcc.Loading(
                    type="graph",
                    children=dmc.Container(id=self.summary_data_conteiner_id, fluid=True, px=0),
                ),
                dcc.Store(id="dummy_store_for_cat_trigger"),
                dcc.Download(id=self.excel_dl_id),
                dcc.Download(id=self.pdf_dl_id),
                dmc.Space(h=40),
            ],
        )

    # ------------------------------------------------------------------
    # Обзор
    # ------------------------------------------------------------------
    @staticmethod
    def _movers_table(frame, val, title, cur_label, ref_label, name_col="cat"):
        rows = []
        for r in frame.itertuples():
            neg = getattr(r, f"{val}_var") < 0
            color = "#B4442E" if neg else "#2F8F6F"
            if val == "cr":
                color = "#2F8F6F" if neg else "#B4442E"
            rows.append(dmc.TableTr([
                dmc.TableTd(dmc.Text(getattr(r, name_col), size="sm")),
                dmc.TableTd(dmc.Text(r.parent, size="xs", c="dimmed")),
                dmc.TableTd(fmt_value(getattr(r, f"{val}_cur"), val), style={"textAlign": "right"}),
                dmc.TableTd(fmt_value(getattr(r, f"{val}_ref"), val), style={"textAlign": "right"}),
                dmc.TableTd(dmc.Text(fmt_delta(getattr(r, f"{val}_var"), val), size="sm", fw=600, c=color),
                            style={"textAlign": "right"}),
                dmc.TableTd(dmc.Text(fmt_pct(getattr(r, f"{val}_pct")), size="sm", c=color),
                            style={"textAlign": "right"}),
            ]))
        return dmc.Paper(withBorder=True, radius=0, p="sm", children=[
            dmc.Text(title, fw=700, size="sm", mb="xs"),
            dmc.Table(
                [
                    dmc.TableThead(dmc.TableTr([
                        dmc.TableTh("Категория" if name_col == "cat" else "Подкатегория"),
                        dmc.TableTh("Группа"),
                        dmc.TableTh(cur_label, style={"textAlign": "right"}),
                        dmc.TableTh(ref_label, style={"textAlign": "right"}),
                        dmc.TableTh("Δ", style={"textAlign": "right"}),
                        dmc.TableTh("Δ%", style={"textAlign": "right"}),
                    ])),
                    dmc.TableTbody(rows or [dmc.TableTr([dmc.TableTd("Нет изменений")])]),
                ],
                striped=True,
                highlightOnHover=True,
                verticalSpacing="xs",
                fz="sm",
            ),
        ])

    def overview(self, data, val):
        sections = build_narrative(data, val)
        icons = {
            "Общая картина": "tabler:eye",
            "Группы": "tabler:folders",
            "Категории": "tabler:category",
            "Подкатегории": "tabler:list-tree",
            "Возвраты": "tabler:arrow-back-up",
            "Что стоит сделать": "tabler:checklist",
        }
        cards = [
            dmc.Paper(withBorder=True, radius=0, p="md", children=[
                dmc.Group(gap=8, mb=6, children=[
                    DashIconify(icon=icons.get(sec["title"], "tabler:point"), width=18, color="#2F6656"),
                    dmc.Text(sec["title"], fw=700),
                ]),
                dmc.List([dmc.ListItem(dmc.Text(i, size="sm")) for i in sec["items"]], spacing=6, size="sm"),
            ])
            for sec in sections
        ]
        blocks = [
            dmc.Group(justify="space-between", align="center", children=[
                dmc.Stack(gap=0, children=[
                    dmc.Text("Справка", fw=700, size="lg"),
                    dmc.Text("Коротко и обычным языком. Скачать — «Экспорт» → «Справка PDF».",
                             size="xs", c="dimmed"),
                ]),
            ]),
            dmc.SimpleGrid(cols={"base": 1, "md": 2}, spacing="md", children=cards),
        ]
        if not data["df"].empty:
            cats = level_table(data["df"], "cat")
            subs = level_table(data["df"], "subcat")
            g = cats[cats[f"{val}_var"] > 0].sort_values(f"{val}_var", ascending=False).head(10)
            f = cats[cats[f"{val}_var"] < 0].sort_values(f"{val}_var").head(10)
            sg = subs[subs[f"{val}_var"] > 0].sort_values(f"{val}_var", ascending=False).head(10)
            sf = subs[subs[f"{val}_var"] < 0].sort_values(f"{val}_var").head(10)
            cur, ref = data["cur_label"], data["ref_label"]
            blocks += [
                dmc.Space(h=6),
                dmc.SimpleGrid(cols={"base": 1, "lg": 2}, spacing="md", children=[
                    self._movers_table(g, val, "Категории: наибольший рост", cur, ref),
                    self._movers_table(f, val, "Категории: наибольшее снижение", cur, ref),
                    self._movers_table(sg, val, "Подкатегории: наибольший рост", cur, ref, "subcat"),
                    self._movers_table(sf, val, "Подкатегории: наибольшее снижение", cur, ref, "subcat"),
                ]),
            ]
        return dmc.Stack(gap="md", children=blocks)

    def registered_callbacks(self, app):

        @app.callback(
            Output(self.last_update_lb_id, "children"),
            Output(self.summary_data_conteiner_id, "children"),
            Input(self.mslider_id, "value"),
            Input("dummy_store_for_cat_trigger", "id"),
            Input(self.option_controll_id, "value"),
            Input(self.value_controll_id, "value"),
            prevent_initial_call=False,
        )
        def save_to_cash(slider_value, dummy, opt, val):
            start, end = self.period_bounds(slider_value)
            data = load_compare(start, end)
            badge = f"Сравниваем: {data['cur_label']} против {data['ref_gen']}"

            content = dmc.Tabs(
                value="groups",
                radius=0,
                color="teal",
                keepMounted=False,
                children=[
                    dmc.TabsList([
                        dmc.TabsTab("По группам", value="groups",
                                    leftSection=DashIconify(icon="tabler:chart-bar", width=16)),
                        dmc.TabsTab("Обзор и справка", value="overview",
                                    leftSection=DashIconify(icon="tabler:file-text", width=16)),
                    ]),
                    dmc.TabsPanel(dmc.Box(cats_report(start, end, opt, val), pt="md"), value="groups"),
                    dmc.TabsPanel(dmc.Box(self.overview(data, val), pt="md"), value="overview"),
                ],
            )
            return badge, content

        @app.callback(
            Output(self.excel_dl_id, "data"),
            Input(self.excel_btn_id, "n_clicks"),
            State(self.mslider_id, "value"),
            State(self.value_controll_id, "value"),
            prevent_initial_call=True,
        )
        def download_excel(n, slider_value, val):
            if not n:
                return no_update
            start, end = self.period_bounds(slider_value)
            data = load_compare(start, end)
            content = build_cats_excel_bytes(data, val or "amount")
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            return dcc.send_bytes(lambda b: b.write(content), f"categories_{start}_{end}_{stamp}.xlsx")

        @app.callback(
            Output(self.pdf_dl_id, "data"),
            Input(self.pdf_btn_id, "n_clicks"),
            State(self.mslider_id, "value"),
            State(self.value_controll_id, "value"),
            prevent_initial_call=True,
        )
        def download_pdf(n, slider_value, val):
            if not n:
                return no_update
            start, end = self.period_bounds(slider_value)
            data = load_compare(start, end)
            content = build_cats_reference_pdf(data, val or "amount")
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            return dcc.send_bytes(lambda b: b.write(content), f"categories_reference_{stamp}.pdf")

        @app.callback(
            Output({"type": "cat_chart", "index": MATCH}, "withBarValueLabel"),
            Input({"type": "val_switch", "index": MATCH}, "checked"),
        )
        def toggle_values(show_values):
            return bool(show_values)
