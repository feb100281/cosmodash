# Файл основной разметки таба по магазинам

import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate

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


        # гарантируем наличие store_region
        if 'store_region' not in df_data.columns:
            df_data['store_region'] = 'Все регионы'

        # безопасная агрегация
        def present(cols): return [c for c in cols if c in df_data.columns]
        value_cols = present(['dt','cr','amount','client_order_number','quant','quant_dt','quant_cr'])
        agg = {c: 'sum' for c in value_cols}
        if 'client_order_number' in value_cols:
            agg['client_order_number'] = 'nunique'

        df_eom = (
            df_data.pivot_table(
                index=['eom','store_gr_name','chanel','store_region'],
                values=value_cols, aggfunc=agg
            )
            .fillna(0).reset_index().sort_values('eom')
        )
        

        df_filters = df_eom[['store_gr_name','chanel','store_region']].drop_duplicates()

        # ---- stores
        def store_filters():
            return dcc.Store(id=self.filters_data_store_id, data=df_filters.to_dict('records'), storage_type='memory')

        def store_raw():
            wanted = ['eom','store_gr_name','chanel','store_region','amount','dt','cr','quant','quant_dt','quant_cr']
            tmp = df_eom.copy()
            for c in wanted:
                if c not in tmp.columns:
                    tmp[c] = 0
            return dcc.Store(id=self.raw_eom_store_id, data=tmp[wanted].to_dict('records'), storage_type='memory')

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

        # ---- sidebar / controls
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
                                seg_item("tabler:chart-line",     "Выручка",            "amount",      "#4cafef"),   # голубой
                                seg_item("tabler:cash-banknote", "Возвраты ₽",         "ret_amt_abs", "#e53935"),   # красный
                                seg_item("tabler:package",       "Возвраты шт",        "ret_qty_abs", "#ff9800"),   # оранжевый
                                seg_item("tabler:arrow-back-up", "Коэф. возвратов, %", "ret_coef",    "#9c27b0"),   # фиолетовый
                            ],
                            value="amount",
                            radius="sm",
                            fullWidth=True,
                            orientation="vertical",
                            color = 'blue'
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

        # ---- main chart with default "Все магазины"
        def chart():
            # подготовим базовый dataset (выручка)
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

        # ---- memo
        def memo():
            if self.period_label:
                period_txt = self.period_label
            else:
                # fallback: считаем по данным вкладки (как было)
                min_date = pd.to_datetime(df_eom['eom'].min()).strftime('%d %b %Y')
                max_date = pd.to_datetime(df_eom['eom'].max()).strftime('%d %b %Y')
                period_txt = f"{min_date} — {max_date}"

            df_stores = (
                df_eom.pivot_table(index='store_gr_name', values='dt', aggfunc='sum')
                .fillna(0).reset_index().sort_values(by='dt', ascending=False)
            )
            l = ''.join([
                f"- {r['store_gr_name']}: {r['dt']/1_000_000:,.2f} млн рублей \n"
                for _, r in df_stores.iterrows()
            ])
            md = f"""## Краткий отчет за период {period_txt}\n\n{l}"""
            return dmc.Spoiler(
                children=dcc.Markdown(md, className='markdown-body'),
                maxHeight=50, hideLabel='Скрыть', showLabel='Читать далее'
            )
                
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

