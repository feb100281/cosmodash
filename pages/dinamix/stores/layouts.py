# Файл основной разметки таба по магазинам

import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate
from decimal import Decimal, ROUND_HALF_UP

import dash
from dash import (
    dcc, html, Input, Output, State,
    MATCH, no_update, callback_context as ctx
)
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import locale

locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")

from components import NoData, BASE_COLORS, COLORS_BY_SHADE
from data import load_df_from_redis
from .modal_area_chart import StoreAreaChartModal





def seg_item(icon: str, text: str, value: str, color: str = None):
    return {
        "value": value,
        "label": dmc.Group(
            gap=6,
            wrap=False,
            align="center",
            children=[
                DashIconify(icon=icon, width=18, color=color),
                dmc.Text(text, size="sm"),
            ],
        ),
    }
    
def layout(df_id: str, period_label: str | None = None):
    return StoresComponents(df_id=df_id, period_label=period_label).tab_layout()


def fmt_abs(val, kind):
                # kind: 'mln' (деньги в млн ₽) | 'int' (целое) | 'rub' (в ₽ с разделителями)
                if kind == 'mln':
                    v = Decimal(val) / Decimal('1000000')
                    v = v.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                    return f"{v:,.2f} млн ₽"
                if kind == 'rub':
                    v = Decimal(val).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                    s = f"{v:,.2f}"
                    return s.replace(",", "X").replace(".", ",").replace("X", "\u202F") + " ₽"
                # int
                try:
                    return f"{int(round(val)):,}".replace(",", "\u202F")
                except:
                    return "0"

def delta_node(curr, prev, good_when_up=True, as_pct=True, w=110, ta="left", kind_abs='mln'):
    import math, pandas as pd
    if prev in (None, 0) or pd.isna(prev) or math.isclose(prev, 0.0):
        return dmc.Text("—", c="gray", ta=ta, w=w, ff="tabular-nums")

    diff = curr - prev
    is_up_good = (diff > 0) if good_when_up else (diff < 0)
    arrow_char = "▲" if diff > 0 else ("▼" if diff < 0 else "■")
    color = "green" if is_up_good else ("red" if diff != 0 else "gray")

    if as_pct:
        val = (diff / prev) * 100.0
        # округлим «по-человечески», без хвостов
        txt = f"{abs(Decimal(val).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))}%"
    else:
        # формат в зависимости от метрики
        txt = fmt_abs(abs(diff), kind_abs)

    return dmc.Text(f"{arrow_char} {txt}", c=color, ta=ta, w=w, ff="tabular-nums")



class StoresComponents:
    def __init__(self, df_id=None, period_label: str | None = None):
        self.df_id = df_id if df_id is not None else None
        self.period_label = period_label

        # IDs (pattern ids)
        self.chart_data_store_id   = {'type':'st_data_store','index':'1'}
        self.chart_series_store_id = {'type':'st_series_store','index':'1'}
        self.filters_data_store_id = {'type':'filter_store','index':'1'}
        self.raw_eom_store_id      = {'type':'st_raw_eom','index':'1'}

        self.chanel_multyselect_id = {'type':'chanel_multyselect','index':'1'}
        self.region_multyselect_id = {'type':'region_multyselect','index':'1'}
        self.store_multyselect_id  = {'type':'store_multyselect','index':'1'}
        self.metric_segment_id     = {'type':'st_metric','index':'1'}

        self.stores_area_chart_id  = {'type':'stores_area_chart','index':'1'}

        self.report_drawer_id      = {'type':'st_report_drawer','index':'1'}
        self.report_button_id      = {'type':'st_report_open','index':'1'}

    # ======================= COMPONENTS =======================

    def create_components(self):
        if not self.df_id:
            return
        
        df_data: pd.DataFrame = load_df_from_redis(self.df_id)
        if df_data is None or df_data.empty:
            return
        df_data['store_region'] = df_data['store_region'].fillna('Регион не указан')

        # гарантируем наличие store_region
        if 'store_region' not in df_data.columns:
            df_data['store_region'] = 'Все регионы'

        # ======= ЗАКАЗЫ: считаем строго как в av_check =========
        # берём только нужные поля и убираем NaN в ключевых колонках
        df_store_checks = df_data[['eom','store_gr_name','dt','client_order_number']].dropna(
            subset=['eom','store_gr_name','dt','client_order_number']
        )
        # 1) сумма каждого чека по магазину и месяцу
        df_stores_checks_agg = (
            df_store_checks
            .groupby(['eom','store_gr_name','client_order_number'], as_index=False)['dt']
            .sum()
        )
        # 2) убираем нулевые чеки
        df_stores_checks_agg = df_stores_checks_agg[df_stores_checks_agg['dt'] != 0]

        # 3) агрегаты по магазинам: число заказов и сумма "живых" чеков (может пригодиться)
        df_stores_checks_analytics = (
            df_stores_checks_agg
            .groupby(['eom','store_gr_name'], as_index=False)
            .agg(orders=('client_order_number','nunique'),
                orders_amount=('dt','sum'))
        )
        # при желании можно посчитать средний чек по магазину:
        df_stores_checks_analytics['av_check'] = np.where(
            df_stores_checks_analytics['orders'] > 0,
            df_stores_checks_analytics['orders_amount'] / df_stores_checks_analytics['orders'],
            0.0
        )

        # ======= БАЗОВЫЕ СУММЫ ПО МЕТРИКАМ =======
        def present(cols): 
            return [c for c in cols if c in df_data.columns]

        value_cols = present(['dt','cr','amount','quant','quant_dt','quant_cr'])
        df_eom = (
            df_data.pivot_table(
                index=['eom','store_gr_name','chanel','store_region'],
                values=value_cols,
                aggfunc='sum'
            )
            .fillna(0).reset_index().sort_values('eom')
        )

        # мержим корректные заказы и средний чек
        df_eom = df_eom.merge(
            df_stores_checks_analytics[['eom','store_gr_name','orders','av_check']],
            on=['eom','store_gr_name'], how='left'
        )
        df_eom['orders']   = df_eom['orders'].fillna(0).astype(int)
        df_eom['av_check'] = df_eom['av_check'].fillna(0.0)

        # фильтры
        df_filters = df_eom[['store_gr_name','chanel','store_region']].drop_duplicates()

        # ---- stores
        def store_filters():
            return dcc.Store(
                id=self.filters_data_store_id,
                data=df_filters.to_dict('records'),
                storage_type='memory'
            )

        def store_raw():
            wanted = ['eom','store_gr_name','chanel','store_region',
                    'amount','dt','cr','quant','quant_dt','quant_cr',
                    'orders','av_check']
            tmp = df_eom.copy()
            for c in wanted:
                if c not in tmp.columns:
                    tmp[c] = 0
            return dcc.Store(
                id=self.raw_eom_store_id,
                data=tmp[wanted].to_dict('records'),
                storage_type='memory'
            )

        # ---- right drawer (отчёт)
        def report_drawer():
            return dmc.Drawer(
                id=self.report_drawer_id,
                title="Отчёт",
                position="right",
                size=520,
                padding="md",
                opened=False,
                children=html.Div()  # наполняем колбэком
            )

        # ---- sidebar / controls (как у тебя было)
        def controls():
            return dmc.Card(
                withBorder=True, shadow="xs", radius=0, p="md",
                children=[
                    dmc.Group(
                        justify="space-between", align="center", mb=8,
                        children=[
                            dmc.Group(
                                gap="sm", align="center",
                                children=[
                                    dmc.ThemeIcon(radius=0, size="lg", variant="transparent",
                                                children=DashIconify(icon="tabler:adjustments-alt")),
                                    dmc.Text("Панель настроек", fw=700),
                                ],
                            ),
                            dmc.Group(
                                gap="xs",
                                children=[
                                    dmc.Tooltip(label="Сбросить фильтры", withArrow=True),
                                    dmc.ActionIcon(
                                        id={'type': 'st_controls_reset', 'index': '1'},
                                        radius=0, variant="light",
                                        children=DashIconify(icon="tabler:refresh"),
                                    ),
                                    dmc.Tooltip(label="Отчёт по текущему отбору", withArrow=True),
                                    dmc.ActionIcon(
                                        id=self.report_button_id,
                                        radius=0, variant="light",
                                        children=DashIconify(icon="tabler:report-analytics"),
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dmc.Divider(variant="dashed", my=10),
                    dmc.Stack(gap="sm", children=[
                        dmc.MultiSelect(
                            id=self.chanel_multyselect_id, label="Канал", placeholder="Выберите канал",
                            data=sorted(df_filters["chanel"].dropna().unique().tolist()),
                            w="100%", radius=0, clearable=True, searchable=True,
                            leftSection=DashIconify(icon="tabler:arrows-shuffle"),
                        ),
                        dmc.MultiSelect(
                            id=self.region_multyselect_id, label="Регион", placeholder="Выберите регион",
                            data=sorted(df_filters["store_region"].dropna().unique().tolist()),
                            w="100%", radius=0, clearable=True, searchable=True,
                            leftSection=DashIconify(icon="tabler:map-pin"),
                        ),
                        dmc.MultiSelect(
                            id=self.store_multyselect_id, label="Магазин", placeholder="Выберите магазин",
                            data=sorted(df_filters["store_gr_name"].dropna().unique().tolist()),
                            w="100%", radius=0, clearable=True, searchable=True,
                            leftSection=DashIconify(icon="tabler:building-store"),
                        ),
                        dmc.SegmentedControl(
                            id=self.metric_segment_id,
                            data=[
                                seg_item("tabler:chart-line",       "Выручка",            "amount"),   
                                seg_item("tabler:cash-banknote",    "Возвраты ₽",         "ret_amt_abs"),   
                                seg_item("tabler:box-seam",         "Возвраты шт",        "ret_qty_abs"),  
                                seg_item("tabler:arrows-left-right","Коэф. возвратов, %", "ret_coef"),   
                                seg_item("tabler:shopping-cart",    "Кол-во заказов",     "count_order"), 
                                seg_item("tabler:credit-card",      "Средний чек",        "avr_recept"),  
                            ],
                            value="amount",
                            radius="sm",
                            fullWidth=True,
                            orientation="vertical",
                            color="blue",
                        ),
                        dmc.Divider(variant="dashed", my=10),
                        dmc.Switch(
                            id={"type": "st_switch_legend", "index": "1"},
                            label="Показывать легенду", checked=True, radius="xs",
                        ),
                        dmc.Switch(
                            id={"type": "st_switch_points", "index": "1"},
                            label="Маркеры точек", checked=False, radius="xs",
                        ),
                    ]),
                ],
            )

        # ---- main chart (как у тебя было)
        def chart():
            df = df_eom.pivot_table(index='eom', columns='store_gr_name', values='amount', aggfunc='sum') \
                    .fillna(0).reset_index().sort_values('eom')
            df["eom"] = pd.to_datetime(df["eom"], errors="coerce")
            df["month_name"] = df["eom"].dt.strftime("%b\u202F%y").str.capitalize()
            store_cols = [c for c in df.columns if c not in ["eom","month_name"]]
            df["Все магазины"] = df[store_cols].sum(axis=1)
            data = df.to_dict("records")

            all_cols = ["Все магазины"] + store_cols
            series_full = []
            for i, col in enumerate(all_cols):
                color = BASE_COLORS[0] if col == "Все магазины" else COLORS_BY_SHADE[i % len(COLORS_BY_SHADE)]
                series_full.append({"name": col, "color": color})
            default_series = [s for s in series_full if s["name"] == "Все магазины"]

            return dmc.Stack([
                dmc.AreaChart(
                    id=self.stores_area_chart_id, h=600, dataKey='month_name',
                    data=data, series=default_series,
                    tooltipAnimationDuration=500,
                    areaProps={
                        "isAnimationActive": True, "animationDuration": 500,
                        "animationEasing": "ease-in-out", "animationBegin": 500,
                    },
                    withPointLabels=False,
                    valueFormatter={"function": "formatNumberIntl"},
                    withLegend=True, 
                    connectNulls=True,
                ),
                dcc.Store(id=self.chart_data_store_id, data=data, storage_type='memory'),
                dcc.Store(id=self.chart_series_store_id, data=series_full, storage_type='memory'),
            ])

       

        def memo():
            # ===== период вида "01 МАР 24 - 29 СЕН 25" =====
            MONTHS_RU_3 = ["ЯНВ","ФЕВ","МАР","АПР","МАЙ","ИЮН","ИЮЛ","АВГ","СЕН","ОКТ","НОЯ","ДЕК"]
            def fmt_dd_MMM_yy(d):
                if d is None or pd.isna(d): return ""
                d = pd.to_datetime(d)
                return f"{d.day:02d} {MONTHS_RU_3[d.month-1]} {d.strftime('%y')}"

            d_min = pd.to_datetime(df_data['date'].min(), errors='coerce')
            d_max = pd.to_datetime(df_data['date'].max(), errors='coerce')
            period_text = f"{fmt_dd_MMM_yy(d_min)} - {fmt_dd_MMM_yy(d_max)}"

            # --- агрегаты на стартовый рендер (base=period, delta=%) ---
            total_sales   = float(df_data['dt'].sum())
            total_net     = float(df_data['amount'].sum())
            total_returns = float(df_data['cr'].sum())
            total_orders  = int(df_eom['orders'].sum()) if 'orders' in df_eom.columns else 0

            eom_series = pd.to_datetime(df_eom['eom'], errors='coerce')
            first_eom, last_eom = eom_series.min(), eom_series.max()

            def _fmt_mln(x): 
                try: return f"{x/1_000_000:,.2f} млн ₽"
                except: return "0,00 млн ₽"
            def _fmt_int(x):
                try: return f"{int(x):,}".replace(",", "\u202F")
                except: return "0"

            def sum_at(col, eom):
                if col not in df_eom.columns or pd.isna(eom): return 0.0
                m = pd.to_datetime(df_eom['eom']) == eom
                return float(df_eom.loc[m, col].sum())

            first_amount, last_amount = sum_at('amount', first_eom), sum_at('amount', last_eom)
            first_dt,     last_dt     = sum_at('dt', first_eom),     sum_at('dt', last_eom)
            first_cr,     last_cr     = sum_at('cr', first_eom),     sum_at('cr', last_eom)
            first_ord,    last_ord    = sum_at('orders', first_eom), sum_at('orders', last_eom)

            
            def metric_row(label, value_text, delta):
                return dmc.Group(
                    gap="sm", align="center",
                    children=[
                        dmc.Text(f"{label}:", w=180, ta="left"),
                        dmc.Text(value_text, fw=700, w=140, ta="right", ff="tabular-nums"),
                        delta
                    ]
                )

            # стартовые строки (base=первый месяц, delta=%)
            init_list = html.Ul(
                style={"listStyleType":"disc","margin":0,"paddingLeft":"1.2rem"},
                children=[
                    html.Li(metric_row("Чистая выручка", _fmt_mln(total_net),
                                    delta_node(last_amount, first_amount, True, True))),
      
                    html.Li(metric_row("Общие продажи",  _fmt_mln(total_sales),
                                    delta_node(last_dt, first_dt, True, True))),
                    html.Li(metric_row("Возвраты",       _fmt_mln(total_returns),
                                    delta_node(last_cr, first_cr, good_when_up=False, as_pct=True))),
                    html.Li(metric_row("Кол-во заказов", _fmt_int(total_orders),
                                    delta_node(last_ord, first_ord, True, True))),
                ]
            )

            # ===== UI =====
            header = dmc.Group(
                gap="xs",
                children=[
                    dmc.Text(f"Краткий отчёт за период: {period_text}", fw=700),
            
                ],
            )

            # переключатели сравнения и формата дельты + понятная подпись
            controls = dmc.Group(
                gap=10, align="center",
                children=[
                    dmc.SegmentedControl(
                        id={'type': 'sum_base_mode', 'index': '1'},
                        data=[{"label":"за период","value":"period"},
                            {"label":"посл. месяц","value":"last_month"}],
                        value="period", size="xs", radius="sm", color="blue"
                    ),
                    dmc.SegmentedControl(
                        id={'type': 'sum_delta_mode', 'index': '1'},
                        data=[{"label":"Абс.","value":"abs"},
                            {"label":"%","value":"pct"}],
                        value="pct", size="xs", radius="sm", color="blue"
                    ),
                    dmc.Badge(id={'type':'sum_caption','index':'1'},
                   size="md", radius="xs", variant="outline",)
                ]
            )

            # блок лидеров (как было у тебя, без дублирования id)
            df_store_total = (
                df_eom.groupby('store_gr_name', as_index=False)['dt'].sum()
                .rename(columns={'dt': 'sum_dt'})
            )
            def store_at(eom):
                m = pd.to_datetime(df_eom['eom']) == eom
                return (df_eom.loc[m]
                        .groupby('store_gr_name', as_index=False)['dt'].sum()
                        .rename(columns={'dt': 'val'}))
            st_first = store_at(first_eom).rename(columns={'val':'first_dt'})
            st_last  = store_at(last_eom).rename(columns={'val':'last_dt'})
            st = (df_store_total.merge(st_first, on='store_gr_name', how='left')
                                .merge(st_last,  on='store_gr_name', how='left')
                                .fillna(0.0)
                                .sort_values('sum_dt', ascending=False))
            def store_row(rank, name, sum_dt, first_dt_s, last_dt_s):
                share_pct = float(sum_dt / total_sales * 100) if total_sales else 0.0
                return dmc.Group(
                    gap="sm", align="center",
                    children=[
                        dmc.Badge(str(rank), variant="filled", color="teal", w=40, ta="center"),
                        dmc.Text(str(name), w=220, ta="left"),
                        dmc.Text(_fmt_mln(sum_dt), fw=600, w=140, ta="right"),
                        
                        delta_node(last_dt_s, first_dt_s, True, True, w=90),
                        dmc.Group(align="center", gap=8, children=[
                            dmc.Progress(value=share_pct, w=180, size="sm", radius="xl"),
                            dmc.Text(f"{share_pct:.1f}%", w=44, ta="right", c="dimmed", ff="tabular-nums"),
                        ]),
                    ]
                )
            store_block = dmc.Stack(
                gap="xs",
                children=[store_row(i+1, r['store_gr_name'], r['sum_dt'], r['first_dt'], r['last_dt'])
                        for i, (_, r) in enumerate(st.iterrows())]
            )

            return dmc.Alert(
                title=None, color="blue", radius="md", variant="light",
                children=[
                    header,
                    dmc.Divider(variant="dashed", my=8),
                    controls,
                    dmc.Divider(variant="dashed", my=8),
                    dmc.Box(id={'type':'sum_rows','index':'1'}, children=[init_list]),  
                    dmc.Divider(variant="dashed", my=8),
                    dmc.Spoiler(
                       showLabel=dmc.Badge(
                            "Продажи по магазинам",
             	        variant="light", color="teal", radius="xs", size="md",
                          
                        ),
                        hideLabel=dmc.Badge(
                            "Скрыть",
                        variant="light", color="gray", radius="xs", size="md",
                            leftSection=DashIconify(icon="tabler:chevron-up")
                        ),
                        maxHeight=0, transitionDuration=200,
                        children=[
                            dmc.Paper(
                                withBorder=True, radius="md", p="sm", mt="xs",
                                children=dmc.Stack(id={'type':'store_block','index':'1'}, gap="xs")
                            )
                        ]
                    )
                ]
            )




            





        # ---- right drawer
        def report_drawer():
            return dmc.Drawer(
                id=self.report_drawer_id,
                title="Отчёт",
                position="right",
                size=520,
                padding="md",
                opened=False,
                children=html.Div()
            )

        # вернём все 6 элементов (как ожидает tab_layout)
        return (
            store_filters(),
            store_raw(),
            controls(),
            chart(),
            report_drawer(),
            memo(),
        )


    # ======================= LAYOUT =======================

    def tab_layout(self):
        if not self.df_id:
            return NoData().component
        filter_store, raw_store, controls, chart, report_drawer, memo = self.create_components()
        return dmc.Container(
            fluid=True,
            children=[
                dmc.Title('Динамика продаж по магазинам', order=3, c='blue'),
                dmc.Space(h=6),
                memo,
                dmc.Space(h=10),
                dmc.Grid(
                    gutter="lg", align="stretch",
                    children=[
                        dmc.GridCol(
                            span={"base": 12, "lg": 3},
                            children=dmc.Card(
                                withBorder=True, radius=0, shadow="sm", p="md",
                                style={"position": "sticky", "top": 70, "alignSelf": "start"},
                                children=controls,
                            ),
                        ),
                        dmc.GridCol(
                            span={"base": 12, "lg": 9},
                            children=[
                                dmc.Title('График продаж по магазинам', order=4),
                                dmc.Space(h=10),
                                chart,
                            ],
                        ),
                    ],
                ),
                dmc.Space(h=6),
                filter_store,
                raw_store,
                report_drawer,
                StoreAreaChartModal().create_components(),
            ],
        )
        



    # ======================= CALLBACKS =======================

    def register_callbacks(self, app):
        area_chart   = self.stores_area_chart_id['type']
        ch_filter    = self.chanel_multyselect_id['type']
        rg_filter    = self.region_multyselect_id['type']
        st_filter    = self.store_multyselect_id['type']
        metric_id    = self.metric_segment_id['type']
        series_store = self.chart_series_store_id['type']
        data_store   = self.chart_data_store_id['type']
        filter_data  = self.filters_data_store_id['type']
        modal        = StoreAreaChartModal().modal_id['type']
        modal_cont   = StoreAreaChartModal().conteiner_id['type']

        # модальные колбэки из твоего класса
        StoreAreaChartModal().registered_callbacks(app)

        @app.callback(
            Output({"type": modal, "index": MATCH}, "opened"),
            Output({"type": modal_cont, "index": MATCH}, "children"),
            Input({"type": area_chart, "index": MATCH}, "clickData"),
            Input({"type": area_chart, "index": MATCH}, "clickSeriesName"),
            State({"type": modal, "index": MATCH}, "opened"),
            prevent_initial_call=True,
        )
        def show_and_update_modal(clickData, clickSeriesName, opened):
            cont = StoreAreaChartModal(clickData, clickSeriesName).update_modal()
            return not opened, cont

        # === главный апдейт: фильтры + выбор метрики ===
        @app.callback(
            Output({'type': st_filter,  'index': MATCH}, 'data'),        # options "Магазин"
            Output({'type': area_chart, 'index': MATCH}, 'series'),      # серии
            Output({'type': area_chart, 'index': MATCH}, 'data'),        # данные графика
            Output({'type': data_store, 'index': MATCH}, 'data'),        # кэш данных
            Output({'type': area_chart, 'index': MATCH}, 'valueFormatter'),  # форматтер значений
            Input({'type': ch_filter,   'index': MATCH}, 'value'),
            Input({'type': rg_filter,   'index': MATCH}, 'value'),
            Input({'type': st_filter,   'index': MATCH}, 'value'),
            Input({'type': metric_id,   'index': MATCH}, 'value'),       # amount | ret_amt_abs | ret_qty_abs | ret_coef
            State({'type': filter_data, 'index': MATCH}, 'data'),
            State({'type': series_store,'index': MATCH}, 'data'),
            State({'type': 'st_raw_eom','index': MATCH}, 'data'),
            prevent_initial_call=True,
        )
        def update_chart(ch_val, rg_val, st_val, metric, filter_df, series_val, raw_eom):
            def L(x):
                if x is None: return []
                return x if isinstance(x, list) else [x]

            # --- фильтрация сырых данных
            rdf = pd.DataFrame(raw_eom)
            if L(ch_val): rdf = rdf[rdf['chanel'].isin(L(ch_val))]
            if L(rg_val): rdf = rdf[rdf['store_region'].isin(L(rg_val))]
            
            

            # --- ветки по метрике
            if metric == 'amount':
                formatter = {"function": "formatNumberIntl"}
                pivot = (
                    rdf.pivot_table(index='eom', columns='store_gr_name', values='amount', aggfunc='sum')
                    .fillna(0).sort_index().reset_index()
                )

            elif metric == 'ret_amt_abs':
                formatter = {"function": "formatNumberIntl"}
                pivot = (
                    rdf.pivot_table(index='eom', columns='store_gr_name', values='cr', aggfunc='sum')
                    .fillna(0).sort_index().reset_index()
                )

            elif metric == 'ret_qty_abs':
                formatter = {"function": "formatIntl"}   # целые значения
                pivot = (
                    rdf.pivot_table(index='eom', columns='store_gr_name', values='quant_cr', aggfunc='sum')
                    .fillna(0).sort_index().reset_index()
                )
            
            elif metric == 'count_order':
                formatter = {"function": "formatIntl"}  # целые
                pivot = (
                    rdf.pivot_table(
                        index='eom',
                        columns='store_gr_name',
                        values='orders',
                        aggfunc='first'   # <--- не суммируем повторно
                    )
                    .fillna(0).sort_index().reset_index()
                )
                # «Все магазины» = сумма по магазинам
                store_cols = [c for c in pivot.columns if c not in ['eom']]
                pivot['Все магазины'] = pivot[store_cols].sum(axis=1)


            elif metric == 'avr_recept':
                # Средний чек = сумма amount / сумма orders
                formatter = {"function": "formatNumberIntl"}
                by = (
                    rdf.groupby(['eom','store_gr_name'])
                    .agg(amount_sum=('dt','sum'), orders=('orders','sum'))
                    .reset_index()
                )
                by['value'] = np.where(by['orders'] > 0, by['amount_sum'] / by['orders'], 0.0)
                pivot = (
                    by.pivot_table(index='eom', columns='store_gr_name', values='value', aggfunc='mean')
                    .fillna(0).sort_index().reset_index()
                )
                
                tot = (
                    rdf.groupby('eom')
                    .agg(amount_sum=('dt','sum'), orders=('orders','sum'))
                    .reset_index()
                )
                tot['Все магазины'] = np.where(tot['orders'] > 0, tot['amount_sum'] / tot['orders'], 0.0)
                pivot = pivot.merge(tot[['eom','Все магазины']], on='eom', how='left')

            
            

            else:  # 'ret_coef' — коэффициент возвратов (₽), %
                formatter = {"function": "formatPercentIntl"}
                g = rdf.groupby(['eom','store_gr_name'])[['cr','dt']].sum().reset_index()
                g['value'] = np.where(g['dt'] > 0, 100.0 * g['cr'] / g['dt'], 0.0)
                pivot = (
                    g.pivot_table(index='eom', columns='store_gr_name', values='value', aggfunc='mean')
                    .fillna(0).sort_index().reset_index()
                )
                # pooled ratio для «Все магазины»
                tot = rdf.groupby('eom')[['cr','dt']].sum().reset_index()
                tot['Все магазины'] = np.where(tot['dt'] > 0, 100.0 * tot['cr'] / tot['dt'], 0.0)
                pivot = pivot.merge(tot[['eom','Все магазины']], on='eom', how='left')

            # --- сумма «Все магазины» для абсолютных метрик
            if metric in ('amount', 'ret_amt_abs', 'ret_qty_abs'):
                store_cols = [c for c in pivot.columns if c not in ['eom']]
                pivot['Все магазины'] = pivot[store_cols].sum(axis=1)

            # --- оформление дат
            pivot['eom'] = pd.to_datetime(pivot['eom'], errors='coerce')
            pivot = pivot.sort_values('eom').reset_index(drop=True)
            pivot['month_name'] = pivot['eom'].dt.strftime("%b\u202F%y").str.capitalize()

            # переносим month_name в конец
            cols_no_mn = [c for c in pivot.columns if c != 'month_name']
            pivot = pivot[cols_no_mn + ['month_name']]
            chart_data = pivot.to_dict('records')

            # --- options для "Магазин" после фильтров
            fdf = pd.DataFrame(filter_df)
            if L(ch_val): fdf = fdf[fdf['chanel'].isin(L(ch_val))]
            if L(rg_val): fdf = fdf[fdf['store_region'].isin(L(rg_val))]
            store_options = [{"value": s, "label": s} for s in sorted(fdf['store_gr_name'].unique().tolist())]

            available = {o["value"] for o in store_options}
            selected  = [s for s in L(st_val) if s in available]

            def series_by(names): return [s for s in series_val if s["name"] in names]
            def total_series():   return [s for s in series_val if s["name"] == "Все магазины"]

            out_series = series_by(selected) if selected else total_series()

            return store_options, out_series, chart_data, chart_data, formatter



        # переключатели легенды/поинтов
        @app.callback(
            Output({'type': area_chart, 'index': MATCH}, 'withLegend'),
            Output({'type': area_chart, 'index': MATCH}, 'withPointLabels'),
            Input({'type': 'st_switch_legend', 'index': MATCH}, 'checked'),
            Input({'type': 'st_switch_points', 'index': MATCH}, 'checked'),
            prevent_initial_call=True,
        )
        def update_chart_view(legend_on, points_on):
            return bool(legend_on), bool(points_on)

        # === отчёт в правом Drawer: открытие + генерация контента ===
        @app.callback(
            Output({'type': 'st_report_drawer', 'index': MATCH}, 'opened'),
            Output({'type': 'st_report_drawer', 'index': MATCH}, 'title'),
            Output({'type': 'st_report_drawer', 'index': MATCH}, 'children'),
            Input({'type': 'st_report_open', 'index': MATCH}, 'n_clicks'),
            State({'type': ch_filter, 'index': MATCH}, 'value'),
            State({'type': rg_filter, 'index': MATCH}, 'value'),
            State({'type': st_filter, 'index': MATCH}, 'value'),
            State({'type': metric_id,  'index': MATCH}, 'value'),
            State({'type': 'st_raw_eom', 'index': MATCH}, 'data'),
            prevent_initial_call=True,
        )
        def open_report(n, ch_val, rg_val, st_val, metric, raw_eom):
            def L(x): return [] if x is None else (x if isinstance(x, list) else [x])

            rdf = pd.DataFrame(raw_eom)
            if L(ch_val): rdf = rdf[rdf['chanel'].isin(L(ch_val))]
            if L(rg_val): rdf = rdf[rdf['store_region'].isin(L(rg_val))]
            scope = "Все магазины" if not L(st_val) else ", ".join(L(st_val))
            if L(st_val): rdf = rdf[rdf['store_gr_name'].isin(L(st_val))]

            # агрегаты для текста
            rdf['eom'] = pd.to_datetime(rdf['eom'])
            last_eom = rdf['eom'].max()
            uniq_sorted = sorted(rdf['eom'].dropna().unique())
            prev_eom = uniq_sorted[-2] if len(uniq_sorted) >= 2 else None

            def sum_col(c): return float(rdf[c].sum()) if c in rdf.columns else 0.0

            # ---- ветки по метрикам (значение "metric" совпадает с SegmentedControl) ----
            if metric == 'amount':
                metric_name = "Чистая выручка"
                y_series_name = "Выручка"
                total = sum_col('amount')
                last = float(rdf.loc[rdf['eom'] == last_eom, 'amount'].sum())
                prev = float(rdf.loc[rdf['eom'] == prev_eom, 'amount'].sum()) if prev_eom is not None else 0.0
                trend = 0 if prev == 0 else round(100 * (last - prev) / prev, 1)
                # мини-график: сумма amount по месяцам
                g = rdf.groupby('eom')['amount'].sum().reset_index().sort_values('eom')
                g['month_name'] = pd.to_datetime(g['eom']).dt.strftime("%b\u202F%y").str.capitalize()
                mini_data = g[['month_name', 'amount']].rename(columns={'amount': 'value'}).to_dict('records')
                value_suffix = " ₽"
                trend_suffix = " %"

            elif metric == 'ret_amt_abs':
                metric_name = "Возвраты (₽)"
                y_series_name = "CR (₽)"
                total = sum_col('cr')
                last = float(rdf.loc[rdf['eom'] == last_eom, 'cr'].sum())
                prev = float(rdf.loc[rdf['eom'] == prev_eom, 'cr'].sum()) if prev_eom is not None else 0.0
                trend = 0 if prev == 0 else round(100 * (last - prev) / prev, 1)
                # мини-график: сумма cr по месяцам
                g = rdf.groupby('eom')['cr'].sum().reset_index().sort_values('eom')
                g['month_name'] = pd.to_datetime(g['eom']).dt.strftime("%b\u202F%y").str.capitalize()
                mini_data = g[['month_name', 'cr']].rename(columns={'cr': 'value'}).to_dict('records')
                value_suffix = " ₽"
                trend_suffix = " %"

            elif metric == 'ret_qty_abs':
                metric_name = "Возвраты (шт.)"
                y_series_name = "QCR (шт.)"
                total = sum_col('quant_cr')
                last = float(rdf.loc[rdf['eom'] == last_eom, 'quant_cr'].sum())
                prev = float(rdf.loc[rdf['eom'] == prev_eom, 'quant_cr'].sum()) if prev_eom is not None else 0.0
                trend = 0 if prev == 0 else round(100 * (last - prev) / prev, 1)
                # мини-график: сумма quant_cr по месяцам
                g = rdf.groupby('eom')['quant_cr'].sum().reset_index().sort_values('eom')
                g['month_name'] = pd.to_datetime(g['eom']).dt.strftime("%b\u202F%y").str.capitalize()
                mini_data = g[['month_name', 'quant_cr']].rename(columns={'quant_cr': 'value'}).to_dict('records')
                value_suffix = ""
                trend_suffix = " %"

            else:  # 'ret_coef' — коэффициент возвратов (₽), %
                metric_name = "Коэф. возвратов (₽), %"
                y_series_name = "CR/DT, %"
                # текущие и прошлые значения как процент
                def ratio_at(eom):
                    part = rdf.loc[rdf['eom'] == eom]
                    dt_sum = float(part['dt'].sum())
                    cr_sum = float(part['cr'].sum())
                    return 0.0 if dt_sum == 0 else (100.0 * cr_sum / dt_sum)

                last = ratio_at(last_eom)
                prev = ratio_at(prev_eom) if prev_eom is not None else 0.0
                trend = round(last - prev, 1)
                total = sum_col('cr')  # просто чтобы было что показать в summary (не используется в тексте)
                # мини-график: процент по месяцам
                g = rdf.groupby('eom')[['cr', 'dt']].sum().reset_index().sort_values('eom')
                g['value'] = np.where(g['dt'] > 0, 100.0 * g['cr'] / g['dt'], 0.0)
                g['month_name'] = pd.to_datetime(g['eom']).dt.strftime("%b\u202F%y").str.capitalize()
                mini_data = g[['month_name', 'value']].to_dict('records')
                value_suffix = " %"
                trend_suffix = " п.п."

            title = f"Отчёт • {scope}"
            trend_txt = ("+" if trend > 0 else "") + f"{trend}{trend_suffix}"

            rows = [
                dmc.Group(justify="space-between", children=[
                    dmc.Text(metric_name, fw=700),
                    dmc.Badge(scope, radius=0),
                ]),
                dmc.Text(f"Последний период: {last_eom.strftime('%b %Y').capitalize()}"),
                dmc.Text(f"Тек. значение: {last:,.2f}{value_suffix}"),
                dmc.Text(f"Изменение к пред.: {trend_txt}"),
                dmc.Divider(variant="dashed", my=8),
                dmc.AreaChart(
                    h=220, dataKey='month_name',
                    data=mini_data,
                    series=[{"name": y_series_name, "color": BASE_COLORS[0], "valueKey": "value"}],
                    withLegend=False, withPointLabels=False, connectNulls=True,
                ),
            ]
            return True, title, dmc.Stack(children=rows, gap="xs")

        
        
        # КНОПКА СБРОСИТЬ
        @app.callback(
            Output({'type': ch_filter, 'index': MATCH}, 'value'),
            Output({'type': rg_filter, 'index': MATCH}, 'value'),
            Output({'type': st_filter, 'index': MATCH}, 'value'),
            Output({'type': 'st_metric', 'index': MATCH}, 'value'),
            Input({'type': 'st_controls_reset', 'index': MATCH}, 'n_clicks'),
            prevent_initial_call=True,
        )
        def reset_filters(n_clicks):
            return None, None, None, 'amount'
        
        
        
        
        #### ОБНОВЛЕНИЕ САММАРИ
        @app.callback(
            Output({'type':'sum_rows','index':MATCH}, 'children'),
            Output({'type':'sum_caption','index':MATCH}, 'children'),
            Input({'type':'sum_base_mode','index':MATCH}, 'value'),
            Input({'type':'sum_delta_mode','index':MATCH}, 'value'),
            State({'type':'st_raw_eom','index':MATCH}, 'data'),
            prevent_initial_call=False,
        )
        def update_summary_rows(base_mode, delta_mode, raw_eom):
            import pandas as pd, numpy as np, math
            rdf = pd.DataFrame(raw_eom or [])
            if rdf.empty:
                return dmc.Alert("Нет данных для расчёта KPI", color="gray", variant="light", radius="md"), ""

            # даты
            rdf['eom'] = pd.to_datetime(rdf['eom'], errors='coerce')
            rdf = rdf.dropna(subset=['eom'])
            eoms = np.sort(rdf['eom'].unique())
            last_eom = eoms[-1] if len(eoms) else None
            prev_eom = eoms[-2] if len(eoms) >= 2 else None
            first_eom = eoms[0] if len(eoms) else None

            # форматтеры
            def _fmt_mln(x): return f"{x/1_000_000:,.2f} млн ₽"
            def _fmt_int(x):  return f"{int(x):,}".replace(",", "\u202F")

            MONTHS_RU_3 = ["ЯНВ","ФЕВ","МАР","АПР","МАЙ","ИЮН","ИЮЛ","АВГ","СЕН","ОКТ","НОЯ","ДЕК"]
            def mon_yy(d):
                if d is None or pd.isna(d): return ""
                d = pd.to_datetime(d)
                return f"{MONTHS_RU_3[d.month-1]} {d.strftime('%y')}"

            # агрегаторы
            def total(col): return float(rdf[col].sum()) if col in rdf.columns else 0.0
            def at(col, eom):
                if col not in rdf.columns or eom is None: return 0.0
                return float(rdf.loc[rdf['eom']==eom, col].sum())

            
            pairs = [
                ("Чистая выручка", 'amount', True,  'mln'),  
                ("Общие продажи",  'dt',     True,  'mln'),
                ("Возвраты",       'cr',     False, 'mln'),
                ("Кол-во заказов", 'orders', True,  'int'),  
            ]
                        # база сравнения для стрелки
            base_eom = prev_eom if base_mode == 'last_month' else first_eom
            def display_value(col):
                if base_mode == 'last_month':
                    return at(col, last_eom)  
                else:
                    return total(col)        
            
            # собираем строки
            rows = []
            for label, col, good_up, kind_abs in pairs:
                val_center = display_value(col)
                if kind_abs == 'mln':
                    value_text = _fmt_mln(val_center)
                else:
                    value_text = _fmt_int(val_center)

                
                value_box_children = [dmc.Text(value_text, fw=700, w=140, ta="right", ff="tabular-nums")]
                
                curr_for_delta = at(col, last_eom)
                prev_for_delta = at(col, base_eom)
                delta_comp = delta_node(curr_for_delta, prev_for_delta,
                            good_when_up=good_up,
                            as_pct=(delta_mode=='pct'),
                            kind_abs=kind_abs)

                rows.append(
                    html.Li(
                        dmc.Group(
                            gap="sm", align="center",
                            children=[
                                dmc.Text(f"{label}:", w=180, ta="left"),
                                dmc.Group(gap=6, align="center", justify="end", w=200, children=value_box_children),
                                delta_comp,
                            ]
                        )
                    )
                )

            
            def fmt_vs(a, b):
                if a and b:
                    return f"{a} vs {b}"
                return a or b or "—"

            if base_mode == 'last_month':
                main = fmt_vs(mon_yy(last_eom), mon_yy(prev_eom))
            else:
                main = fmt_vs(mon_yy(last_eom), mon_yy(first_eom))

            delta_txt = "Δ %" if delta_mode == 'pct' else "Δ абс."
            cap_text = f"{main} • {delta_txt}"


            ul = html.Ul(style={"listStyleType":"disc","margin":0,"paddingLeft":"1.2rem"}, children=rows)
            return ul, cap_text
        
        
        
        
        #### ОБНОВЛЕНИЕ ПРОДАЖ ПО МАГАЗИНАМ
        @app.callback(
            Output({'type':'store_block','index':MATCH}, 'children'),
            Input({'type':'sum_base_mode','index':MATCH}, 'value'),
            Input({'type':'sum_delta_mode','index':MATCH}, 'value'),
            State({'type':'st_raw_eom','index':MATCH}, 'data'),
            prevent_initial_call=False,
        )
        def update_store_block(base_mode, delta_mode, raw_eom):
            import pandas as pd, numpy as np, math
            rdf = pd.DataFrame(raw_eom or [])
            if rdf.empty:
                raise PreventUpdate

            rdf['eom'] = pd.to_datetime(rdf['eom'], errors='coerce')
            rdf = rdf.dropna(subset=['eom'])

            # опорные месяцы
            eoms = np.sort(rdf['eom'].unique())
            last_eom = eoms[-1] if len(eoms) else None
            prev_eom = eoms[-2] if len(eoms) >= 2 else None
            first_eom = eoms[0] if len(eoms) else None

            # сумма за весь период по магазину
            sales_total = rdf.groupby('store_gr_name', as_index=False)['dt'].sum().rename(columns={'dt':'sum_dt'})

            # суммы по месяцам для дельт и "последнего месяца"
            def sum_at(eom, col='dt', alias='val'):
                if eom is None:  # безопасно, вернём пустое
                    out = rdf[['store_gr_name']].drop_duplicates().copy()
                    out[alias] = 0.0
                    return out
                part = rdf.loc[rdf['eom'] == eom]
                return (part.groupby('store_gr_name', as_index=False)[col].sum()
                            .rename(columns={col: alias}))

            last_sales  = sum_at(last_eom,  alias='last_dt')
            prev_sales  = sum_at(prev_eom,  alias='prev_dt')
            first_sales = sum_at(first_eom, alias='first_dt')

            # сводная таблица
            st = (sales_total
                .merge(last_sales,  on='store_gr_name', how='left')
                .merge(prev_sales,  on='store_gr_name', how='left')
                .merge(first_sales, on='store_gr_name', how='left')
                .fillna(0.0))

            # что показываем как "значение" и чем делим для процента доли
            if base_mode == 'last_month':
                st['value'] = st['last_dt']                # значение в строке
                denom = st['last_dt'].sum() or 0.0         # для прогресса
                prev_for_delta_col = 'prev_dt'             # стрелка: last vs prev
            else:
                st['value'] = st['sum_dt']                 # значение в строке
                denom = st['sum_dt'].sum() or 0.0          # для прогресса
                prev_for_delta_col = 'first_dt'            # стрелка: last vs first

            st['share'] = np.where(denom > 0, st['value'] / denom * 100.0, 0.0)

            # сортировка по значению (что логичнее визуально)
            st = st.sort_values('value', ascending=False).reset_index(drop=True)

            # форматтеры
            def _fmt_mln(x): return f"{x/1_000_000:,.2f} млн ₽"

            from decimal import Decimal, ROUND_HALF_UP
            def fmt_abs(val):
                v = (Decimal(val) / Decimal('1000000')).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                return f"{v:,.2f} млн ₽"

            def delta_node(curr, prev, good_when_up=True, as_pct=True, w=90, ta="left"):
                import pandas as pd
                if prev in (None, 0) or pd.isna(prev) or math.isclose(prev, 0.0):
                    return dmc.Text("—", c="gray", ta=ta, w=w, ff="tabular-nums")
                diff = curr - prev
                is_up_good = (diff > 0) if good_when_up else (diff < 0)
                arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "■")
                color = "green" if is_up_good else ("red" if diff != 0 else "gray")
                if as_pct:
                    val = Decimal(diff/prev*100).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
                    txt = f"{abs(val)}%"
                else:
                    txt = fmt_abs(abs(diff))
                return dmc.Text(f"{arrow} {txt}", c=color, ta=ta, w=w, ff="tabular-nums")

            
            def store_row(rank, r):
                return dmc.Group(
                    gap="sm", align="center",
                    children=[
                        dmc.Badge(str(rank), variant="filled", color="teal", w=40, ta="center"),
                        dmc.Text(str(r['store_gr_name']), w=220, ta="left"),
                        dmc.Text(_fmt_mln(r['value']), fw=600, w=140, ta="right"),
                        delta_node(r['last_dt'], r[prev_for_delta_col], True, as_pct=(delta_mode=='pct'), w=90),
                        dmc.Group(align="center", gap=8, children=[
                            dmc.Progress(value=float(r['share']), w=180, size="sm", radius="xl"),
                            dmc.Text(f"{r['share']:.1f}%", w=44, ta="right", c="dimmed", ff="tabular-nums"),
                        ]),
                    ]
                )

            rows = [store_row(i+1, r) for i, r in st.iterrows()]
            return rows



        
        
        
        
        




