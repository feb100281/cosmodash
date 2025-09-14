import pandas as pd
import numpy as np
from data import (
    load_columns_df,
    save_df_to_redis,
    load_df_from_redis,
    delete_df_from_redis,
)
from dash.exceptions import PreventUpdate
from datetime import datetime, date, timedelta
import io
import dash
from dash import (
    Dash,
    dcc,
    html,
    Input,
    Output,
    State,
    _dash_renderer,
    clientside_callback,
    MATCH,
    ALL,
    ctx,
    Patch,
    no_update,
)
import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import locale

locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")

from components import ValuesRadioGroups, MonthSlider, DATES
from data import (
    load_columns_df,
    save_df_to_redis,
    load_df_from_redis,
    COLORS,
    COLS_DICT,
)

COLS = [
    "date",
    "dt",
    "cr",
    "amount",
    "store",
    "eom",
    "chanel",
    "manager",
    "cat",
    "subcat",
    "client_order",
    "quant",
    "client_order_number",
]


def id_to_months(start, end):
    return DATES[start].strftime("%Y-%m-%d"), DATES[end].strftime("%Y-%m-%d")


class DataHendler:
    def __init__(self, df_id):
        self.df_id = df_id
        pass

    def update_on(self):
        pass


class TabGeneral:
    def __init__(self):

        # Лэйблы для групп графиков
        self.revenue_title_id = "revenue_title"
        self.revenue_title = dmc.Title(
            "Oбщая выручка", order=4, id=self.revenue_title_id
        )

        self.revenue_title_memo_id = "revenue_title_memo"
        self.revenue_title_memo = dmc.Spoiler(
            showLabel="Показать",
            hideLabel="Скрыть",
            maxHeight=50,
            id=self.revenue_title_memo_id,
            children="",
        )

        # Связанные чарты по выручке
        self.sales_chart_id = "sales_chart"
        self.sales_chart = dmc.CompositeChart(
            id=self.sales_chart_id,
            h=400,
            dataKey="eom",
            data=[{"eom": "2025-01-01", "value": 0}],
            tooltipAnimationDuration=500,
            areaProps={
                "isAnimationActive": True,
                "animationDuration": 500,
                "animationEasing": "ease-in-out",
                "animationBegin": 500,
            },
            withPointLabels=False,
            series=[
                {"name": "value", "dataKey": "value", "color": "red", "type": "line"}
            ],
            valueFormatter={"function": "formatNumberIntl"},
            # type="stacked",
            withLegend=True,
            legendProps={"verticalAlign": "bottom"},
            tooltipProps={"content": {"function": "chartTooltip"}},
            # type="default",
        )

        self.av_check_title = dmc.Title("Средний чек", order=4)
        # График для среднего чека
        self.av_check_chart_id = "av_check_chart"
        self.av_check_chart = dmc.AreaChart(
            id=self.av_check_chart_id,
            h=200,
            dataKey="eom",
            data=[{"eom": "2025-01-01", "value": 0}],
            tooltipAnimationDuration=500,
            areaProps={
                "isAnimationActive": True,
                "animationDuration": 500,
                "animationEasing": "ease-in-out",
                "animationBegin": 500,
            },
            withPointLabels=False,
            series=[{"name": "value", "dataKey": "value", "color": "red"}],
            valueFormatter={"function": "formatNumberIntl"},
            # type="stacked",
            withLegend=True,
            legendProps={"verticalAlign": "bottom"},
            connectNulls=True,
            # tooltipProps={"content":  {"function": "chartTooltip"}},
            # type="default",
        )

        
    
    def data(self, df_id):
        df_data: pd.DataFrame = load_df_from_redis(df_id)

        def update_area_chart():
            df: pd.DataFrame = (
                df_data.pivot_table(
                    index="eom", values=["dt", "cr", "amount"], aggfunc="sum"
                )
                .fillna(0)
                .reset_index()
                .sort_values("eom")
            )
            df.rename(columns=COLS_DICT, inplace=True)
            for col in ["Продажи", "Возвраты", "Чистая выручка"]:
                df[f"{col}_from_first"] = ((df[col] / df[col].iloc[0] - 1) * 100).round(
                    2
                )

            # отклонение от предыдущего значения (в %)
            for col in ["Продажи", "Возвраты", "Чистая выручка"]:
                df[f"{col}_from_prev"] = ((df[col] / df[col].shift(1) - 1) * 100).round(
                    2
                )

            df = df.fillna(0)

            df["eom"] = pd.to_datetime(df["eom"], errors="coerce")
            df["eom"] = df["eom"].dt.strftime("%b\u202F%y").str.capitalize()

            data = df.to_dict(orient="records")

            # columns = [col for col in df.columns if col not in ["eom"]]

            series = [
                {"name": "Чистая выручка", "color": "indigo.3", "type": "bar"},
                {"name": "Продажи", "color": "green.6", "type": "line"},
                {"name": "Возвраты", "color": "red.6", "type": "line"},
            ]

            return dmc.CompositeChart(
            id=self.sales_chart_id,
            h=400,
            dataKey="eom",
            data=data,
            tooltipAnimationDuration=500,
            areaProps={
                "isAnimationActive": True,
                "animationDuration": 500,
                "animationEasing": "ease-in-out",
                "animationBegin": 500,
            },
            withPointLabels=False,
            series= series,
            valueFormatter={"function": "formatNumberIntl"},
            # type="stacked",
            withLegend=True,
            legendProps={"verticalAlign": "bottom"},
            tooltipProps={"content": {"function": "chartTooltip"}},
            # type="default",
        )

        def update_av_check_chart():
            df: pd.DataFrame = df_data[["eom", "dt", "client_order_number"]]
            df = df.dropna()
            df_sum = (
                df.groupby(["eom", "client_order_number"], as_index=False)["dt"].sum()
            ).reset_index()
            df_sum = df_sum[df_sum["dt"] != 0]
            df = df_sum.pivot_table(
                index="eom",
                values=["dt", "client_order_number"],
                aggfunc={
                    "dt": ["sum", "median", "max", "min"],
                    "client_order_number": "nunique",
                },
            ).fillna(0)
            df.columns = ["_".join(col).strip() for col in df.columns.values]
            df = df.reset_index().sort_values(by="eom")
            df.rename(
                columns={"dt_median": "Медиана", "dt_max": "Макс", "dt_min": "Мин"},
                inplace=True,
            )
            
            
            df["Средний чек"] = df["dt_sum"] / df["client_order_number_nunique"]
            #df = df[["Средний чек", "eom", "Медиана"]]

            data = df.to_dict(orient="records")
            columns = [col for col in df.columns if col not in ["eom"]]

            series = [
                   {"name": "Средний чек", "color": "green.6"},
                   {"name": "Медиана", "color": "blue.6"},                   
               ]
            
            selector_vals = {            
                "1":"Средн / Медиана",
                "2":"Макс/Мин"
            }
            
                        
            return dmc.Stack(
                [
            ValuesRadioGroups(
            id_radio={"type":"selector","index":'1'}, 
            options_dict=selector_vals,
            val="1"),
                
            dmc.AreaChart(
            id={"type":"av_check_chart","index":'1'},
            h=200,
            dataKey="eom",
            data=data,
            tooltipAnimationDuration=500,
            areaProps={
                "isAnimationActive": True,
                "animationDuration": 500,
                "animationEasing": "ease-in-out",
                "animationBegin": 500,
            },
            withPointLabels=False,
            series=series,
            valueFormatter={"function": "formatNumberIntl"},
            # type="stacked",
            withLegend=True,            
            legendProps={"verticalAlign": "bottom"},
            connectNulls=True,
            
            # tooltipProps={"content":  {"function": "chartTooltip"}},
            # type="default",
                    )
                ]
            )
            
        def memo():
            
            return dmc.Container(
                [
                    dmc.Text([
                        "За рассматриваемый период общие продажи составили",
                        dmc.Mark(f"{df_data['dt'].sum()/1_000_000:,.2f}",'green'),
                        " млн рублей.  Возвраты -",
                        dmc.Mark(f"{df_data['cr'].sum()/1_000_000:,.2f}",'red'),
                        "млн рублей. ",
                        dmc.Text("Чистая выручка от реализации - ",fw=500),
                        dmc.Mark(f"{df_data['amount'].sum()/1_000_000:,.2f}",'indigo'),
                        " млн рублей."
                        
                    ],
                             size='sm'
                             )
                ],
                fluid=True
            )
            
            

        # Для мемо
        memo_text = f"""
        За рассматриваемый период общие продажи составили {df_data['dt'].sum()/1_000_000:,.2f} млн рублей. Возвраты - {df_data['cr'].sum()/1_000_000:,.2f} млн рублей.
        **Чистая выручка от реализации** - {df_data['amount'].sum()/1_000_000:,.2f} млн рублей.
        
        - общее количество проданных товаров - {df_data['quant'].sum():,.0f} шт.
        - Количество обработанных заказов - {df_data['client_order'].nunique():,.0f} шт.
                 
        """

        return update_area_chart(),update_av_check_chart(),memo()

    def tabconteiner(self):
       
        return dmc.Container(
            children=[
                dmc.Space(h=20),                           
            ],
            fluid=True,
            id = 'tab_general_layout'
        )


class Components:
    def __init__(self):

        # Общий титул и мемо
        self.title = dmc.Title("Анализ динамики продаж", order=1, c="blue")
        self.memo = dmc.Text("На графиках ниже ...", size="xs")

        # Общий слайдер
        self.mslider_id = "sd_monthslider"
        self.mslider = MonthSlider(id=self.mslider_id)

        # store для ханнения df по слайдеру
        self.df_store_id = "df_store"
        self.df_store = dcc.Store(id=self.df_store_id, storage_type="session")

        # lable для хранения дат c учетом последнего обновления
        self.last_update_lb_id = "last_update_lb"
        self.last_update_lb = dcc.Loading(
            dmc.Text(size="xs", id=self.last_update_lb_id)
        )

        

        # self.Tab_general_charts_data_store_id = "charts_data_store"
        # self.Tab_general_charts_data_store = dcc.Store(
        #     id=self.Tab_general_charts_data_store_id
        # )

        # Общие табы
        self.tabs = dmc.Container(
            children=[
                dmc.Tabs(
                    [
                        dmc.TabsList(
                            [
                                dmc.TabsTab("Общее", value="general"),
                                dmc.TabsTab("Магазины", value="stores"),
                                dmc.TabsTab("Категории", value="cats"),
                                dmc.TabsTab("Менеджеры", value="managers"),
                            ],
                            justify="right",
                        ),
                        dmc.TabsPanel(TabGeneral().tabconteiner(), value="general"),
                        # dmc.TabsPanel(self.areachart_contriner, value="stores"),
                        # dmc.TabsPanel("Settings tab content", value="managers"),
                        # dmc.TabsPanel("Settings tab content", value="matrix"),
                    ],
                    color="teal.3",
                    autoContrast=True,
                    variant="outline",
                    value="general",
                    orientation="horizontal",
                )
            ],
            fluid=True,
        )

    def make_layout(self):
        return dmc.Container(
            children=[
                self.title,
                self.memo,
                self.mslider,
                self.last_update_lb,
                self.tabs,
                dcc.Store(id="dummy_imputs_for_slider"),
                dcc.Store(id="dummy_imputs_for_render"),
                # self.df_store
            ],
            fluid=True,
        )

    def register_callbacks(self, app: Dash):
        # Востанавливаем значения слайдера при заходе не страницу
        @app.callback(
            Output(self.mslider_id, "value"),
            Input("dummy_imputs_for_slider", "data"),
            State(self.df_store_id, "data"),
        )
        def restore_slider(dummy, store_data):
            if store_data and "slider_val" in store_data:
                return store_data["slider_val"]
            return dash.no_update

        # Обновляем df и пешем в redis по ключу
        @app.callback(
            Output(self.df_store_id, "data"),
            Output(self.last_update_lb_id, "children"),
            Input(self.mslider_id, "value"),
            Input("dummy_imputs_for_render", "data"),
            State(self.df_store_id, "data"),
            prevent_initial_call=False,
        )
        def update_df(slider_value, dummy, store_data):
            start, end = id_to_months(slider_value[0], slider_value[1])

            # if ctx.triggered_id != self.mslider_id:
            #     return no_update, no_update
            if store_data and "df_id" in store_data:
                if store_data["start"] == start and store_data["end"] == end:
                    df = load_df_from_redis(store_data["df_id"])
                    if df is not None:  # ключ ещё живой в Redis
                        min_date = pd.to_datetime(df["date"].min())
                        max_date = pd.to_datetime(df["date"].max())
                        notification = f"{min_date.strftime('%d %b %y')} - {max_date.strftime('%d %b %y')}"
                        return no_update, notification

                delete_df_from_redis(store_data["df_id"])

            df = load_columns_df(columns=COLS, start_eom=start, end_eom=end)

            df_id = save_df_to_redis(df, expire_seconds=1200)

            store_dict = {
                "df_id": df_id,
                "start": start,
                "end": end,
                "slider_val": slider_value,
            }

            min_date = pd.to_datetime(df["date"].min())
            max_date = pd.to_datetime(df["date"].max())

            notificattion = (
                f"{min_date.strftime('%d %b %y')} - {max_date.strftime('%d %b %y')}"
            )

            return store_dict, notificattion

        # Обновляем табы
        @app.callback(            
            Output('tab_general_layout', "children"),
            Input(self.df_store_id, "data"),
        )
        def update_tabs(store_data):
            df_id = store_data["df_id"]
            sales_chart, av_check_chart, memo = TabGeneral().data(df_id=df_id)

            return (
                [
                    dmc.Space(h=20),
                    memo,
                    dmc.Space(h=10),
                    TabGeneral().revenue_title,
                    dmc.Space(h=10),
                    sales_chart,
                    dmc.Space(h=10),
                    TabGeneral().av_check_title,
                    dmc.Space(h=10),   
                    #TabGeneral().av_check_view_selector,                      
                    av_check_chart,
                    
                    
                ]
            )

        # Изменить представления средних чеков
        @app.callback(
            Output({"type": "av_check_chart", "index": MATCH}, "series"),
            Input({"type": "selector", "index": MATCH}, "value"),
            prevent_initial_call=True
        )
        def change_series(val):
            # print("Button clicked", n_clicks)
            if val  == '1':  # пример переключения
                return [
                    {"name": "Средний чек", "dataKey": "Средний чек", "color": "green.6"},
                    {"name": "Медиана", "dataKey": "Медиана", "color": "blue.6"},
                ]
            else:
               return [
                   {"name": "Макс", "dataKey": "Макс", "color": "green.6"},
                   {"name": "Мин", "dataKey": "Мин", "color": "blue.6"},                   
               ] 
            
            
            # if value == '1':
               
