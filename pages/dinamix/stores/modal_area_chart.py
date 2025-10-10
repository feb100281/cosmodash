# Файл для модалки для AreaChart по магазинам

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
import dash_mantine_components as dmc
import dash_ag_grid as dag
from dash_iconify import DashIconify
import calendar

# импортируй как у тебя принято – оставляю, даже если тут не используются
from components import (
    ValuesRadioGroups,
    DATES,
    NoData,
    BASE_COLORS,
    COLORS_BY_COLOR,
    COLORS_BY_SHADE,
    InDevNotice,
)
from data import (
    load_columns_df,
    save_df_to_redis,
    load_df_from_redis,
    load_columns_dates,
    COLS_DICT,
)


class StoreAreaChartModal:
    def __init__(self, clickdata=None, clickSeriesName=None):
        self.clickdata = clickdata or {}
        bad = (None, "", "—", "Все магазины")
        # None / "" / "—" => отчёт по всем магазинам
        # self.clickSeriesName = clickSeriesName if clickSeriesName not in (None, "", "—") else None
        self.clickSeriesName = None if clickSeriesName in bad else clickSeriesName


        # IDs
        self.modal_id = {"type": "store_area_chart_modal", "index": "1"}
        self.container_id = {"type": "store_area_chart_modal_container", "index": "1"}
        self.inner_container_id = {"type": "store_area_chart_modal_inner", "index": "1"}
        self.metric_id = {"type": "store_area_chart_metric", "index": "1"}

    # ---------- Публичный API ----------
    def create_components(self):
        return dmc.Modal(
            id=self.modal_id,
            size="90%",
            centered=True,
            withCloseButton=True,
            overlayProps={"opacity": 0.45, "blur": 2},
            children=[dmc.Container(id=self.container_id, fluid=True, px="md", py="md")],
        )

    def modal_children(self):
        content = self._build_content()
        return dmc.Box(id=self.inner_container_id, children=content)

    def update_modal(self):
        return self.modal_children()

    def registered_callbacks(self, app):

        return

    # ---------- Внутренняя логика ----------
    def _build_content(
        self,
        metric: str = "amount",
        period_mode: str = "month",
        date_range: tuple | None = None,
    ):
        # 1️⃣ Базовые периоды из клика
        month, last_month, last_year = self._derive_periods(self.clickdata)
        df_current = self._load_base_df([month, last_month, last_year])

        # Если данных нет вообще
        if df_current.empty:
            return self._empty_state()

        # Если месяц не передан — берём последний доступный
        if pd.isna(month) and not df_current.empty:
            month = df_current["eom"].max()
            last_month = month + pd.offsets.MonthEnd(-1)
            last_year = (month - pd.DateOffset(years=1)) + pd.offsets.MonthEnd(0)

        # 2️⃣ Фильтр по магазину (если не выбран — все магазины)
        if self.clickSeriesName:
            df_scope = df_current.loc[df_current["store_gr_name"] == self.clickSeriesName].copy()
            if df_scope.empty:
                return self._empty_state(f"Нет данных по магазину: {self.clickSeriesName}")
            scope_title = self.clickSeriesName
        else:
            df_scope = df_current.copy()
            scope_title = "Все магазины"

        # 3️⃣ Применяем фильтр по периоду, если задан (date_range передаётся в виде кортежа (start, end))
        if date_range and all(date_range):
            try:
                start_date = pd.to_datetime(date_range[0]).normalize()
                end_date = pd.to_datetime(date_range[1]).normalize()
                if end_date < start_date:
                    start_date, end_date = end_date, start_date
                df_scope = df_scope.loc[(df_scope["date"] >= start_date) & (df_scope["date"] <= end_date)]
                period_label = f"{start_date.strftime('%d.%m.%Y')} — {end_date.strftime('%d.%m.%Y')}"
            except Exception:
                period_label = self._period_label(month)
        else:
            # По умолчанию берём выбранный месяц
            start_date = (month - pd.offsets.MonthBegin(1)).normalize()
            end_date = month.normalize()
            df_scope = df_scope.loc[(df_scope["date"] >= start_date) & (df_scope["date"] <= end_date)]
            period_label = self._period_label(month)

        if df_scope.empty:
            return self._empty_state("Нет данных за выбранный период")

        # 4️⃣ Формируем итоговые таблицы и визуализации
        daily_stats, daily_cum = self._prepare_daily_series(df_scope, metric=metric)
        slices = self._make_slices(df_scope)

        header = self._header(scope_title, period_label, metric, month)
        tabs = self._tabs(daily_stats, daily_cum, slices, df_scope, month, metric)

        return dmc.Stack([header, tabs], gap="md")


    # ---------- Data ----------
    @staticmethod
    def _derive_periods(clickdata):
        month = pd.to_datetime(clickdata.get("eom"), errors="coerce")
        last_month = month + pd.offsets.MonthEnd(-1) if pd.notna(month) else pd.NaT
        last_year = ((month - pd.DateOffset(years=1)) + pd.offsets.MonthEnd(0)) if pd.notna(month) else pd.NaT
        return month, last_month, last_year

    @staticmethod
    def _load_base_df(dates):
        COLS = [
            "eom", "date", "dt", "cr", "amount", "store_gr_name", "chanel", "manager",
            "cat", "subcat", "client_order", "quant", "client_order_number",
            "store_gr_name_amount_ytd", "manu", "brend", 'fullname'
        ]
        df = load_columns_dates(COLS, dates)
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=COLS)

        df["orders_type"] = np.where(
            df["client_order"].eq("<Продажи без заказа>"),
            "Продажи без заказа",
            "Заказы клиента",
        )
        df["eom"] = pd.to_datetime(df["eom"], errors="coerce") + pd.offsets.MonthEnd(0)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["day"] = df["date"].dt.day
        return df

    @staticmethod
    def _prepare_daily_series(df_scope: pd.DataFrame, metric: str = "amount"):
        col = "amount" if metric in ("amount", "aov") else "quant"
        daily = (
            df_scope.pivot_table(index="day", columns="eom", values=col, aggfunc="sum")
            .fillna(0)
            .sort_index()
        )
        return daily.reset_index(), daily.cumsum().reset_index()

    @staticmethod
    def _make_slices(df_scope: pd.DataFrame):
        def _agg(col):
            out = (
                df_scope.groupby(col, dropna=False)["amount"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            out["amount"] = out["amount"].astype(float)
            return out

        return {
            "orders_type": _agg("orders_type"),
            "cat": _agg("cat").head(15),
            "subcat": _agg("subcat").head(15),
            "manager": _agg("manager").head(15),
            "chanel": _agg("chanel").head(15),
        }

    # ---------- UI ----------
    def _header(self, scope_title: str, month_name: str | None, metric: str, month):
        subtitle = month_name or self._period_label(month)
        metric_label = {"amount": "Выручка", "quant": "Количество", "aov": "Средний чек"}.get(metric, "Выручка")

        return dmc.Group(
            [
                dmc.Stack(
                    [
                        dmc.Group(
                            [DashIconify(icon="ph:storefront-duotone", width=22), dmc.Title(scope_title, order=2)],
                            gap="xs",
                        ),
                        dmc.Group(
                            [
                                dmc.Badge(subtitle, size="lg", variant="light",
                                          leftSection=DashIconify(icon="tabler:calendar", width=16)),
                                dmc.Badge(metric_label, size="lg", color="pink", variant="outline",
                                          leftSection=DashIconify(icon="mdi:chart-line", width=16)),
                            ],
                            gap="xs",
                        ),
                    ],
                    gap=4,
                ),
            ],
            justify="space-between",
            align="center",
        )

    def _tabs(self, daily_stats, _daily_cum_unused, slices, df_scope, month, metric):
        return dmc.Tabs(
            value="overview",
            children=[
                dmc.TabsList(
                    [
                        dmc.TabsTab("Обзор", value="overview"),
                        dmc.TabsTab("Разрезы", value="slices"),
                        dmc.TabsTab("Тёпловая карта", value="heat"),
                        dmc.TabsTab("Все операции", value="table"),
                    ]
                ),
                dmc.TabsPanel(self._overview_block(df_scope, month), value="overview"),
                dmc.TabsPanel(self._slices_block(slices), value="slices"),
                dmc.TabsPanel(self._heatmap_block(df_scope, month, metric), value="heat"),
                dmc.TabsPanel(self._table_block(df_scope), value="table"),
            ],
        )

    # ---------- Обзор (текст + бейджи) ----------
    def _overview_block(self, df_scope: pd.DataFrame, month):
        period_label = self._period_label(month)
        cur = df_scope.loc[df_scope["eom"].eq(month)].copy()
        if cur.empty:
            return dmc.Stack(
                [
                    dmc.Text(f"Обзор за {period_label}", size="sm", c="dimmed"),
                    self._empty_state("Нет данных за выбранный период"),
                ],
                gap="sm",
            )

        last_month = month + pd.offsets.MonthEnd(-1)
        last_year = (month - pd.DateOffset(years=1)) + pd.offsets.MonthEnd(0)

        def _sum_for(eom, col):
            src = df_scope.loc[df_scope["eom"].eq(eom)]
            return float(src[col].sum()) if col in src.columns else 0.0

        has_dt = "dt" in cur.columns
        has_cr = "cr" in cur.columns
        has_amount = "amount" in cur.columns

        sales_cur = _sum_for(month, "dt") if has_dt else float(cur.loc[cur["amount"] > 0, "amount"].sum())
        returns_cur = _sum_for(month, "cr") if has_cr else float(abs(cur.loc[cur["amount"] < 0, "amount"].sum()))
        net_cur = _sum_for(month, "amount") if has_amount else sales_cur - returns_cur

        sales_pm = _sum_for(last_month, "dt") if has_dt else float(df_scope.loc[(df_scope["eom"].eq(last_month)) & (df_scope["amount"] > 0), "amount"].sum())
        returns_pm = _sum_for(last_month, "cr") if has_cr else float(abs(df_scope.loc[(df_scope["eom"].eq(last_month)) & (df_scope["amount"] < 0), "amount"].sum()))
        net_pm = _sum_for(last_month, "amount") if has_amount else sales_pm - returns_pm

        sales_py = _sum_for(last_year, "dt") if has_dt else float(df_scope.loc[(df_scope["eom"].eq(last_year)) & (df_scope["amount"] > 0), "amount"].sum())
        returns_py = _sum_for(last_year, "cr") if has_cr else float(abs(df_scope.loc[(df_scope["eom"].eq(last_year)) & (df_scope["amount"] < 0), "amount"].sum()))
        net_py = _sum_for(last_year, "amount") if has_amount else sales_py - returns_py

        def _pct(a, b):
            if b == 0:
                return None
            return (a - b) / b

        orders = cur.loc[cur["client_order_number"].notna()].copy()
        if orders.empty:
            aov = median_check = max_check = 0.0
            orders_cnt = 0
        else:
            base_col = "dt" if has_dt else "amount"
            order_checks = orders.groupby("client_order_number")[base_col].sum().astype(float)
            order_checks = order_checks[order_checks > 0]
            orders_cnt = int(order_checks.shape[0])
            aov = float(order_checks.mean()) if orders_cnt else 0.0
            median_check = float(order_checks.median()) if orders_cnt else 0.0
            max_check = float(order_checks.max()) if orders_cnt else 0.0

        returns_share = (returns_cur / sales_cur) if sales_cur else 0.0
        no_order_share = 0.0
        if "orders_type" in cur.columns and has_amount:
            no_order_amt = float(cur.loc[cur["orders_type"].eq("Продажи без заказа"), "amount"].clip(lower=0).sum())
            no_order_share = (no_order_amt / sales_cur) if sales_cur else 0.0

        line_icon = lambda i: dmc.ThemeIcon(DashIconify(icon=i, width=16), radius="xl", variant="light")

        overview = dmc.Paper(
            withBorder=True, radius="md", p="md",
            children=dmc.Stack(
                [
                    dmc.Group(
                        [
                            line_icon("mdi:cash-multiple"),
                            dmc.Text("Чистая выручка: "),
                            dmc.Badge(self._fmt_cur(net_cur), color="pink", size="lg"),
                            dmc.Text("Динамика:"),
                            self._pct_badge(_pct(net_cur, net_pm), "MoM"),
                            self._pct_badge(_pct(net_cur, net_py), "YoY"),
                        ],
                        gap=8, align="center", wrap=True
                    ),
                    dmc.Group(
                        [
                            line_icon("mdi:swap-horizontal-bold"),
                            dmc.Text("Продажи / Возвраты:"),
                            dmc.Badge(self._fmt_cur(sales_cur), color="green", variant="light"),
                            dmc.Text("—"),
                            dmc.Badge(self._fmt_cur(returns_cur), color="red", variant="light"),
                            dmc.Text("Доля возвратов:"),
                            dmc.Badge(self._fmt_pct(returns_share), color="red", variant="outline"),
                        ],
                        gap=8, align="center", wrap=True
                    ),
                    dmc.Group(
                        [
                            line_icon("mdi:cart-outline"),
                            dmc.Text("Заказы:"),
                            dmc.Badge(f"{orders_cnt:,}".replace(",", " "), variant="filled"),
                            dmc.Text("Средний чек:"),
                            dmc.Badge(self._fmt_cur(aov), variant="outline"),
                            dmc.Text("Медианный / Максимальный:"),
                            dmc.Badge(self._fmt_cur(median_check), variant="light"),
                            dmc.Badge(self._fmt_cur(max_check), variant="light"),
                        ],
                        gap=8, align="center", wrap=True
                    ),
                    dmc.Group(
                        [
                            line_icon("mdi:account-off-outline"),
                            dmc.Text("Продажи без заказа:"),
                            dmc.Badge(self._fmt_pct(no_order_share), variant="dot", color="grape"),
                            dmc.Badge(self._period_label(month), color="gray", variant="light",
                                      leftSection=DashIconify(icon="tabler:calendar", width=14)),
                        ],
                        gap=8, align="center", wrap=True
                    ),
                ],
                gap="sm",
            ),
        )

        returns_tabs = self._returns_tabs(cur, returns_cur)

        return dmc.Stack(
            [
                dmc.Title("Обзор", order=3),
                overview,
                dmc.Divider(variant="dashed"),
                dmc.Group([dmc.Title("Возвраты", order=4)], justify="space-between"),
                returns_tabs,
            ],
            gap="md",
        )

    def _returns_tabs(self, cur_month_df: pd.DataFrame, returns_cur: float):
        has_cr = "cr" in cur_month_df.columns
        return_df = cur_month_df.copy()
        if has_cr:
            return_df["return_value"] = return_df["cr"].astype(float)
        else:
            return_df["return_value"] = return_df["amount"].where(return_df["amount"] < 0, 0).abs().astype(float)

        def _make_returns_table(col: str, title: str, top_n: int = 15):
            if col not in return_df.columns:
                return dmc.Alert(f"Нет данных по: {title}", color="gray", variant="light")

            grp = (
                return_df.groupby(col, dropna=False)["return_value"]
                .sum()
                .sort_values(ascending=False)
            )
            grp = grp[grp > 0]
            if grp.empty:
                return dmc.Alert(f"— по «{title}» нет возвратов", color="gray", variant="light")

            grp_top = grp.head(top_n)
            total_returns = float(returns_cur) if returns_cur else 0.0

            thead = dmc.TableThead(
                dmc.TableTr([
                    dmc.TableTh("№"),
                    dmc.TableTh(title),
                    dmc.TableTh("Возвраты, ₽", ta="right"),
                    dmc.TableTh("Доля", ta="right"),
                ])
            )

            rows = []
            for i, (k, v) in enumerate(grp_top.items(), start=1):
                name = str(k) if pd.notna(k) else "—"
                share = (v / total_returns) if total_returns else 0.0
                rows.append(
                    dmc.TableTr([
                        dmc.TableTd(f"{i}"),
                        dmc.TableTd(name),
                        dmc.TableTd(self._fmt_cur(float(v)), ta="right"),
                        dmc.TableTd(self._fmt_pct(float(share)), ta="right"),
                    ])
                )

            total_in_table = float(grp_top.sum())
            share_in_table = (total_in_table / total_returns) if total_returns else 0.0
            tfoot = dmc.TableTfoot(
                dmc.TableTr([
                    dmc.TableTh("ИТОГО"),
                    dmc.TableTh("-"),
                    dmc.TableTh(self._fmt_cur(total_in_table), ta="right"),
                    dmc.TableTh(self._fmt_pct(share_in_table), ta="right"),
                ])
            )

            return dmc.Table(
                children=[thead, dmc.TableTbody(rows), tfoot],
                striped=True,
                highlightOnHover=True,
                horizontalSpacing="sm",
                verticalSpacing="xs",
                withTableBorder=True,
                withColumnBorders=False,
            )

        return dmc.Tabs(
            [
                dmc.TabsList([
                    dmc.TabsTab("Категории", value="cat"),
                    dmc.TabsTab("Производители", value="manu"),
                    dmc.TabsTab("Бренды", value="brand"),
                ]),
                dmc.TabsPanel(_make_returns_table("cat", "Категория"), value="cat"),
                dmc.TabsPanel(_make_returns_table("manu", "Производитель"), value="manu"),
                dmc.TabsPanel(
                    _make_returns_table("brand", "Бренд") if "brand" in cur_month_df.columns
                    else _make_returns_table("brend", "Бренд"),
                    value="brand",
                ),
            ],
            value="cat",
            color="pink",
            variant="pills",
            keepMounted=False,
        )

   
    # ---------- Разрезы ----------
    
    def _slices_block(self, slices: dict):
        # фиксированный порядок разделов
        config = [
            ("cat", "Топ категорий"),
            ("subcat", "Топ подкатегорий"),
            ("manager", "Топ менеджеров"),
            ("chanel", "Каналы продаж"),
        ]

        slides = []
        for key, title in config:
            chart = self._dmc_bar_from_slice(slices[key], x=key, title=title, top_n=10)
            slides.append(
                dmc.CarouselSlide(
                    dmc.Paper(
                        withBorder=True, radius="md", p="md",
                        children=chart
                    )
                )
            )

        # Карусель с индикаторами и стрелками
        return dmc.Carousel(
            children=slides,
            slideSize="50%",            # 2 карточки на экран; хочешь 1 — поставь "100%"
            slideGap="md",
            withIndicators=True,
            controlsOffset="xs",
            controlSize=28,
            height=340,
            styles={"indicator": {"width": 8, "height": 8}}
        )


    @staticmethod
    def _dmc_bar_from_slice(df_slice: pd.DataFrame, x: str, title: str = "", top_n: int | None = None):
        src = (df_slice[[x, "amount"]]
            .dropna()
            .sort_values("amount", ascending=False))
        if top_n:
            src = src.head(top_n)

        src["amount_fmt"] = src["amount"].apply(lambda v: f"{v:,.0f}".replace(",", " ") + " ₽")

        data = [{"label": str(r[x]), "amount": float(r["amount"]), "amount_fmt": r["amount_fmt"]}
                for _, r in src.iterrows()]

        return dmc.Stack(
            [
                dmc.Text(title, fw=600, size="sm", mb=4),
                dmc.BarChart(
                    data=data,
                    dataKey="label",
                    series=[{"name": "amount"}],
                    orientation="horizontal",
                    withLegend=False,
                    h=280,
                    barProps={"radius": 6},
                    valueFormatter = {"function": "formatNumberIntl"},
                ),
            ],
            gap="xs",
        )
    
    
    
    
    
    







    
    
    # ---------- Таблица ----------
    def _table_block(self, df_scope: pd.DataFrame):
        eom = df_scope["eom"].max()
        tx = df_scope.loc[
            df_scope["eom"].eq(eom),
            [
                "date","client_order_number","orders_type",
                "cat","subcat","manager","chanel","quant","amount","fullname",
            ],
        ].copy()
        tx["date"] = pd.to_datetime(tx["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        # === база колонок для ТРАНЗАКЦИЙ (одинакова в layout при первом рендере и в callbacks) ===
        def _base_tx_columns():
            return [
                {"headerName": "Дата", "field": "date", "colId": "date",
                "sortable": True, "enableRowGroup": True,
                "cellClass": "ag-firstcol-bg", "pinned": "left"},

                {
                    "headerName": "Заказ",
                    "groupId": "order",
                    "marryChildren": True,
                    "headerClass": "ag-center-header",
                    "children": [
                        {"headerName": "№ заказа", "field": "client_order_number", "colId": "client_order_number",
                        "sortable": True, "minWidth": 150, "type": "leftAligned",
                        "headerClass": "ag-center-header"},
                        {"headerName": "Тип", "field": "orders_type", "colId": "orders_type",
                        "sortable": True, "enableRowGroup": True, "columnGroupShow": "open",
                        "headerClass": "ag-center-header"},
                    ]
                },

                {
                    "headerName": "Номенклатура",
                    "groupId": "product",
                    "marryChildren": True,
                    "headerClass": "ag-center-header",
                    "children": [
                        {"headerName": "Номенклатура", "field": "fullname", "colId": "fullname",
                        "sortable": True, "enableRowGroup": True, "minWidth": 220, "type": "leftAligned",
                        "cellClass": "ag-firstcol-bg", "headerClass": "ag-center-header"},
                        {"headerName": "Категория", "field": "cat", "colId": "cat",
                        "sortable": True, "enableRowGroup": True, "minWidth": 160, "type": "leftAligned",
                        "columnGroupShow": "open", "headerClass": "ag-center-header"},
                        {"headerName": "Подкатегория", "field": "subcat", "colId": "subcat",
                        "sortable": True, "enableRowGroup": True, "minWidth": 180, "type": "leftAligned",
                        "columnGroupShow": "open", "headerClass": "ag-center-header"},
                    ]
                },

                {"headerName": "Кол-во", "field": "quant", "colId": "quant",       
                 "cellClassRules": {
                        "neg-cell": "params.value < 0"
                    },
                "type": "rightAligned", "aggFunc": "sum", "minWidth": 100,    "valueFormatter": {"function": "params.value != null ? d3.format(',.0f')(params.value).replace(/,/g, '\\u202F') : ''"},},

                {"headerName": "Сумма", "field": "amount", "colId": "amount",
                "type": "rightAligned",
                "valueFormatter": {"function": "params.value ? '₽'+ d3.format(',.2f')(params.value).replace(/,/g, '\\u202F') : ''"},
                "aggFunc": "sum",     
                # если значение < 0 — навешиваем CSS-класс neg-cell
                    "cellClassRules": {
                        "neg-cell": "params.value < 0"
                    },},

                {"headerName": "Менеджер", "field": "manager", "colId": "manager",
                "sortable": True, "enableRowGroup": True, "pinned": "right"},
            ]

        # порядок/пины, которые нужно жёстко удерживать в ТРАНЗАКЦИЯХ
        def _tx_state():
            return [
                {"colId": "date", "pinned": "left"},
                {"colId": "client_order_number"},
                {"colId": "orders_type"},   # появится при раскрытии группы (columnGroupShow управляет видимостью)
                {"colId": "fullname"},
                {"colId": "cat"},
                {"colId": "subcat"},
                {"colId": "quant"},
                {"colId": "amount"},
                {"colId": "manager", "pinned": "right"},
            ]

        column_defs_tx = _base_tx_columns()
        column_state_tx = _tx_state()

        # элементы управления
        mode_id  = {"type": "store_area_table_mode",  "index": "1"}
        quick_id = {"type": "store_area_table_quick", "index": "1"}
        grid_id  = {"type": "store_area_chart_table","index": "1"}
        store_id = {"type": "store_area_table_store","index": "1"}

        return dmc.Paper(
            withBorder=True, radius="md", p="sm",
            children=dmc.Stack([
                dmc.Group([
                    dmc.SegmentedControl(
                        id=mode_id,
                        value="tx",
                        data=[
                            {"label": "Транзакции", "value": "tx"},
                            {"label": "По дням", "value": "by_day"},
                            {"label": "По категориям", "value": "by_cat"},
                            {"label": "По менеджерам", "value": "by_manager"},
                        ],
                        size="sm", color="blue", radius="xs",
                    ),
                    dmc.TextInput(
                        id=quick_id,
                        placeholder="Быстрый поиск…",
                        leftSection=DashIconify(icon="tabler:search", width=18),
                        size="sm",
                        style={"minWidth": 260},
               
                    ),
                ], justify="space-between"),

                dcc.Store(id=store_id, data=tx.to_dict("records")),

                dag.AgGrid(
                    id=grid_id,
                    rowData=tx.to_dict("records"),
                    columnDefs=column_defs_tx,
                    columnState=column_state_tx,  # <- стартовое состояние (порядок/пины)
                    defaultColDef={
                        "resizable": True,
                        "filter": True,
                        "sortable": True,
                        "headerClass": "ag-center-header",
                        "suppressMovableColumns": True,  # фиксируем порядок от пользователя
                    },
                    dashGridOptions={
                        "animateRows": True,
                        "rowSelection": "single",
                        "groupDisplayType": "groupRows",
                        "groupIncludeFooter": True,
                        "groupIncludeTotalFooter": True,
                        "sideBar": True,
                    },
                    style={"height": "520px", "width": "100%"},
                    className="ag-theme-alpine",
                ),
            ], gap="sm"),
        )


    def registered_callbacks(self, app):
        from dash import Input, Output, State, no_update
        import pandas as pd

        grid_id  = {"type": "store_area_chart_table","index": "1"}
        mode_id  = {"type": "store_area_table_mode",  "index": "1"}
        quick_id = {"type": "store_area_table_quick", "index": "1"}
        store_id = {"type": "store_area_table_store","index": "1"}

        # одинаковая база для ТРАНЗАКЦИЙ
        def _base_tx_columns():
            return [
                {"headerName": "Дата", "field": "date", "colId": "date",
                "sortable": True, "enableRowGroup": True,  "cellClass": "ag-firstcol-bg",},

                {"headerName": "Заказ", "groupId": "order", "marryChildren": True,
                "headerClass": "ag-center-header",
                "children": [
                    {"headerName": "№ заказа", "field": "client_order_number", "colId": "client_order_number",
                    "sortable": True, "minWidth": 150, "type": "leftAligned",
                    "headerClass": "ag-center-header"},
                    {"headerName": "Тип", "field": "orders_type", "colId": "orders_type",
                    "sortable": True, "enableRowGroup": True, "columnGroupShow": "open",
                    "headerClass": "ag-center-header"},
                ]},

                {"headerName": "Номенклатура", "groupId": "product", "marryChildren": True,
                "headerClass": "ag-center-header",
                "children": [
                    {"headerName": "Номенклатура", "field": "fullname", "colId": "fullname",
                    "sortable": True, "enableRowGroup": True, "minWidth": 220, "type": "leftAligned",
                    "cellClass": "ag-firstcol-bg", "headerClass": "ag-center-header"},
                    {"headerName": "Категория", "field": "cat", "colId": "cat",
                    "sortable": True, "enableRowGroup": True, "minWidth": 160, "type": "leftAligned",
                    "columnGroupShow": "open", "headerClass": "ag-center-header"},
                    {"headerName": "Подкатегория", "field": "subcat", "colId": "subcat",
                    "sortable": True, "enableRowGroup": True, "minWidth": 180, "type": "leftAligned",
                    "columnGroupShow": "open", "headerClass": "ag-center-header"},
                ]},

                {"headerName": "Кол-во", "field": "quant", "colId": "quant", "type": "rightAligned",  
                 "cellClassRules": {
        "neg-cell": "params.value < 0"}, "valueFormatter": {"function": "params.value != null ? d3.format(',.0f')(params.value).replace(/,/g, '\\u202F') : ''"},},
                {"headerName": "Сумма", "field": "amount", "colId": "amount", "type": "rightAligned",
                  "cellClassRules": {
        "neg-cell": "params.value < 0"
    },
                "valueFormatter": {"function": "params.value ? '₽'+ d3.format(',.2f')(params.value).replace(/,/g, '\\u202F') : ''"}, "cellClass": "ag-firstcol-bg",},
                {"headerName": "Менеджер", "field": "manager", "colId": "manager",
                "sortable": True, "enableRowGroup": True},
            ]

        def _tx_state():
            return [
                {"colId": "date", "pinned": "left"},
                {"colId": "client_order_number"},
                {"colId": "orders_type"},
                {"colId": "fullname"},
                {"colId": "cat"},
                {"colId": "subcat"},
                {"colId": "quant"},
                {"colId": "amount"},
                {"colId": "manager", "pinned": "right"},
            ]

        # наборы колонок/состояний для остальных режимов
        def _by_day_columns():
            return [
                {"headerName": "Дата", "field": "date", "colId": "date", "sortable": True, "cellClass": "ag-firstcol-bg"},
                {"headerName": "Сумма за день", "field": "amount", "colId": "amount", 
                  "cellClassRules": {
                        "neg-cell": "params.value < 0"
                    },
                "type": "rightAligned", "cellClass": "ag-firstcol-bg",
                "valueFormatter": {"function": "params.value ? '₽'+ d3.format(',.2f')(params.value).replace(/,/g, '\\u202F') : ''"}},
                {"headerName": "Кол-во позиций", "field": "n_items", "colId": "n_items", 
                 "cellClassRules": {
                        "neg-cell": "params.value < 0"
                    },
                "type": "rightAligned", "valueFormatter": {"function": "d3.format(',.0f')(value)"},  "valueFormatter": {"function": "params.value != null ? d3.format(',.0f')(params.value).replace(/,/g, '\\u202F') : ''"},},
                {"headerName": "Кол-во менеджеров", "field": "n_managers", "colId": "n_managers",
                "type": "rightAligned", "valueFormatter": {"function": "d3.format(',.0f')(value)"}},
                {"headerName": "Кол-во шт", "field": "quant", "colId": "quant",
                  "cellClassRules": {
                            "neg-cell": "params.value < 0"
                        },
                "type": "rightAligned",  "valueFormatter": {"function": "params.value != null ? d3.format(',.0f')(params.value).replace(/,/g, '\\u202F') : ''"},},
            ]
        def _by_day_state():
            return [
                {"colId": "date"},
                {"colId": "amount"},
                {"colId": "n_items"},
                {"colId": "n_managers"},
                {"colId": "quant"},
            ]

        def _by_cat_columns():
            return [
                {"headerName": "Категория", "field": "cat", "colId": "cat", "sortable": True,  "cellClass": "ag-firstcol-bg",},
                {"headerName": "Сумма", "field": "amount", "colId": "amount",
                  "cellClassRules": {
                            "neg-cell": "params.value < 0"
                        },
                "type": "rightAligned",
                "valueFormatter": {"function": "params.value ? '₽'+ d3.format(',.2f')(params.value).replace(/,/g, '\\u202F') : ''"}},
                {"headerName": "Кол-во позиций", "field": "n_items", "colId": "n_items",
                "type": "rightAligned",   "valueFormatter": {"function": "params.value != null ? d3.format(',.0f')(params.value).replace(/,/g, '\\u202F') : ''"},},
            ]
        def _by_cat_state():
            return [
                {"colId": "cat"},
                {"colId": "amount"},
                {"colId": "n_items"},
            ]

        def _by_manager_columns():
            return [
                {"headerName": "Менеджер", "field": "manager", "colId": "manager", "sortable": True,  "cellClass": "ag-firstcol-bg",},
                {"headerName": "Сумма", "field": "amount", "colId": "amount",
                                "cellClassRules": {
                        "neg-cell": "params.value < 0"
                    },
                "type": "rightAligned",
                "valueFormatter": {"function": "params.value ? '₽'+ d3.format(',.2f')(params.value).replace(/,/g, '\\u202F') : ''"}},
                {"headerName": "Кол-во позиций", "field": "n_items", "colId": "n_items",
                "type": "rightAligned", "valueFormatter": {"function": "d3.format(',.0f')(value)"}},
            ]
        def _by_manager_state():
            return [
                {"colId": "manager"},
                {"colId": "amount"},
                {"colId": "n_items"},
            ]

        # --- переключатель: возвращаем и columnDefs, и rowData, и columnState ---
        @app.callback(
            Output(grid_id, "columnDefs"),
            Output(grid_id, "rowData"),
            Output(grid_id, "columnState"),
            Input(mode_id, "value"),
            State(store_id, "data"),
            prevent_initial_call=True,
        )
        def _switch_table_view(mode, data):
            if not data:
                return no_update, no_update, no_update

            df = pd.DataFrame(data)
            # типы
            for c in ("amount","quant"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

            mode = mode or "tx"

            if mode == "by_day":
                agg = (df.groupby("date", as_index=False)
                        .agg(amount=("amount","sum"),
                            quant=("quant","sum"),
                            n_items=("fullname", lambda s: s.dropna().nunique()),
                            n_managers=("manager", lambda s: s.dropna().nunique()))
                        .sort_values("date"))
                return _by_day_columns(), agg.to_dict("records"), _by_day_state()

            if mode == "by_cat":
                agg = (df.groupby("cat", as_index=False)
                        .agg(amount=("amount","sum"),
                            n_items=("fullname", lambda s: s.dropna().nunique()))
                        .sort_values(["amount"], ascending=False))
                agg["cat"] = agg["cat"].fillna("—")
                return _by_cat_columns(), agg.to_dict("records"), _by_cat_state()

            if mode == "by_manager":
                agg = (df.groupby("manager", as_index=False)
                        .agg(amount=("amount","sum"),
                            n_items=("fullname", lambda s: s.dropna().nunique()))
                        .sort_values(["amount"], ascending=False))
                agg["manager"] = agg["manager"].fillna("—")
                return _by_manager_columns(), agg.to_dict("records"), _by_manager_state()

            # default: транзакции
            return _base_tx_columns(), df.to_dict("records"), _tx_state()

        # быстрый поиск — одинаков для всех режимов
        @app.callback(
            Output(grid_id, "quickFilterText"),
            Input(quick_id, "value"),
            prevent_initial_call=False,
        )
        def _quick_filter(text):
            return text or ""
        
        ### изменение тем у таблиц
        from dash import MATCH
        @app.callback(
            Output({"type":"store_area_chart_table","index": MATCH}, "className"),
            Input("theme_switch", "checked"),
        )
        def toggle_each_modal_grid_theme(checked):
            return "ag-theme-alpine-dark" if checked else "ag-theme-alpine"






    # ---------- Figures ----------
    @staticmethod
    def _daily_area(daily_df: pd.DataFrame, title: str = ""):
        fig = go.Figure()
        cols = [c for c in daily_df.columns if c != "day"]
        for col in cols:
            name = str(pd.to_datetime(col).strftime("%b %y")) if pd.notna(col) else str(col)
            fig.add_trace(go.Scatter(x=daily_df["day"], y=daily_df[col], mode="lines", name=name))
        fig.update_layout(
            title=title,
            margin=dict(l=8, r=8, t=40, b=8),
            xaxis_title="День",
            yaxis_title="Сумма",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        return fig

    @staticmethod
    def _bars_from_slice(df_slice: pd.DataFrame, x: str, title: str = "", top_n: int | None = None):
        src = df_slice.copy()
        if top_n:
            src = src.head(top_n)
        fig = go.Figure(go.Bar(x=src[x].astype(str), y=src["amount"]))
        fig.update_layout(title=title, margin=dict(l=8, r=8, t=40, b=8),
                          xaxis_title="", yaxis_title="Сумма")
        return fig

    # ---------- Helpers ----------
    @staticmethod
    def _fmt_cur(x: float | None):
        if x is None:
            return "—"
        return f"{x:,.0f}".replace(",", " ")

    @staticmethod
    def _fmt_pct(x: float | None):
        if x is None:
            return "—"
        return f"{x*100:.1f}%"

    @staticmethod
    def _pct_badge(x: float | None, label: str = ""):
        color = "gray"; text = "—"
        if x is not None:
            if x > 0:
                color = "green"; text = f"▲ {x*100:.1f}% {label}".strip()
            elif x < 0:
                color = "red"; text = f"▼ {abs(x)*100:.1f}% {label}".strip()
            else:
                text = f"0.0% {label}".strip()
        return dmc.Badge(text, color=color, variant="light", radius="xs")

    @staticmethod
    def _empty_state(text: str = "Данные отсутствуют"):
        return dmc.Center(
            dmc.Stack(
                [
                    DashIconify(icon="mdi:database-off", width=48),
                    dmc.Text(text, size="sm", c="dimmed"),
                ],
                align="center",
            ),
            mih=240,
        )

    @staticmethod
    def _ensure_aov_series(daily_stats: pd.DataFrame, month, metric: str):
        # для AOV пока оставляем amount как прокси на графике
        return daily_stats

    @staticmethod
    def _period_label(month):
        if pd.isna(month):
            return "выбранный месяц"
        start = (month - pd.offsets.MonthBegin(1)).date()
        end = month.date()
        months = ["января","февраля","марта","апреля","мая","июня","июля","августа",
                  "сентября","октября","ноября","декабря"]
        def _fmt(d): return f"{d.day} {months[d.month-1]} {d.year}"
        if start.month == end.month and start.year == end.year:
            return f"{start.day}–{end.day} {months[end.month-1]} {end.year}"
        return f"{_fmt(start)} — {_fmt(end)}"


    
    
    # ---------- Тепловая карта с датами ----------

    @staticmethod
    def _heatmap_block(df_scope: pd.DataFrame, month, metric: str):


        if pd.isna(month):
            return dmc.Stack([dmc.Text("Нет данных для тепловой карты", size="sm", c="dimmed")])

        start = (month - pd.offsets.MonthBegin(1)).normalize()
        end   = month.normalize()
        cur = df_scope.loc[(df_scope["date"] >= start) & (df_scope["date"] <= end)].copy()
        if cur.empty:
            return dmc.Stack([dmc.Text("Нет данных для тепловой карты", size="sm", c="dimmed")])

        val_col = "amount" if metric in ("amount", "aov") else "quant"

        # агрегат по конкретной дате
        day_sum = cur.groupby(cur["date"].dt.date)[val_col].sum().astype(float)

        # календарная сетка Пн..Вс
        y = int(month.year)
        m = int(month.month)
        calendar.setfirstweekday(calendar.MONDAY)
        weeks = calendar.monthcalendar(y, m)  # 0 = пустая клетка

        # матрицы: значения, подписи-дат, маска реальных дат
        z, custom, mask = [], [], []
        for wk in weeks:
            row_vals, row_custom, row_mask = [], [], []
            for d in wk:
                if d == 0:
                    row_vals.append(np.nan)
                    row_custom.append("")
                    row_mask.append(False)
                else:
                    the_date = pd.Timestamp(y, m, d).date()
                    v = float(day_sum.get(the_date, 0.0))
                    row_vals.append(v)
                    row_custom.append(the_date.strftime("%d.%m"))
                    row_mask.append(True)
            z.append(row_vals); custom.append(row_custom); mask.append(row_mask)

        z_arr = np.array(z, dtype=float)

        # единицы
        z_max = float(np.nanmax(z_arr)) if np.isfinite(z_arr).any() else 0.0
        if   z_max >= 1_000_000: div, suffix = 1_000_000.0, " млн"
        elif z_max >=   100_000: div, suffix = 1_000.0,     " тыс"
        else:                    div, suffix = 1.0,         " ₽"

        def fmt(v: float) -> str:
            if not np.isfinite(v): return "—"
            if div == 1_000_000.0: return f"{v/div:.1f}{suffix}"
            if div == 1_000.0:     return f"{v/div:.1f}{suffix}"
            return f"{v:,.0f} ₽".replace(",", " ")

        weekday_names = ["Пн","Вт","Ср","Чт","Пт","Сб","Вс"]
        x_labels = weekday_names

        # суммы по неделям (строкам) для подписи "Неделя N — сумма"
        row_sums = np.nansum(z_arr, axis=1) if np.isfinite(z_arr).any() else np.zeros(len(weeks))
        y_labels = [f"Неделя {i+1} — {fmt(row_sums[i])}" for i in range(len(weeks))]

        # граф
        fig = go.Figure(
            data=go.Heatmap(
                z=z_arr,
                x=x_labels,
                y=y_labels,
                coloraxis="coloraxis",
                hovertemplate="<b>%{customdata}</b><br>%{y} / %{x}<br>Значение: %{z:,.0f} ₽<extra></extra>",
                customdata=custom,
                zmin=0 if z_max > 0 else None,
            )
        )

        # контраст для текста
        finite_vals = z_arr[np.isfinite(z_arr)]
        z_min = float(np.nanmin(finite_vals)) if finite_vals.size else 0.0
        z_mid = (z_min + (z_max if z_max > 0 else 1.0)) / 2.0

        # аннотации в ячейках (дата + значение)
        for r_idx, y_lab in enumerate(y_labels):
            for c_idx, x_lab in enumerate(x_labels):
                if not mask[r_idx][c_idx]:
                    continue
                val = float(z_arr[r_idx, c_idx]) if np.isfinite(z_arr[r_idx, c_idx]) else 0.0
                date_txt = custom[r_idx][c_idx]
                txt = f"{date_txt}<br>{fmt(val)}"
                color = "white" if val > z_mid else "black"
                fig.add_annotation(
                    x=x_lab, y=y_lab, text=txt, showarrow=False,
                    font=dict(size=12, color=color),
                )

        fig.update_layout(
            # title={"text": "Тепловая карта продаж (с датами)", "x": 0.5, "xanchor": "center",
            #     "font": {"size": 18, "family": "Manrope"}},
            margin=dict(l=30, r=30, t=60, b=30),
            coloraxis=dict(colorscale="Blues", colorbar=dict(title="Выручка, ₽", tickformat=",.0f")),
            xaxis=dict(type="category"),
            yaxis=dict(type="category"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        # Top-3 дня (по абсолютным значениям)
        flat = []
        for r in range(len(weeks)):
            for c in range(7):
                v = z_arr[r, c]
                if np.isfinite(v):
                    flat.append((float(v), custom[r][c]))
        top3 = sorted(flat, key=lambda x: x[0], reverse=True)[:3]

        top3_block = dmc.Group(
            [
                dmc.Badge(f"{d} — {int(v):,}".replace(",", " "), color="blue", variant="light", radius="xs")
                for v, d in top3
            ] if top3 else [dmc.Badge("Нет данных", color="gray", variant="light")],
            gap="xs"
        )

        return dmc.Paper(
            withBorder=True, radius="md", p="md", shadow="sm",
            children=[
                dmc.Group(
                    [
                        dmc.Text("Календарь продаж по неделям", fw=600, size="md"),
                        dmc.Badge(
                            f"Подписи: {('млн' if z_max >= 1_000_000 else 'тыс' if z_max >= 100_000 else '₽')}",
                            color="gray", variant="light", radius='xs'
                        ),
                    ],
                    justify="space-between", align="center", mb="sm",
                ),
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
                dmc.Group([dmc.Text("Top-3 дня:", fw=600), top3_block], gap="sm", mt="sm"),
            ],
        )

