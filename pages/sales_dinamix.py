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

from components import ValuesRadioGroups, MonthSlider, DATES, NoData, LoadingScreen

COLS = [
    "date",
    "dt",
    "cr",
    "amount",
    "store",
    "store_gr_name",
    "eom",
    "chanel",
    "manager",
    "agent",
    "cat",
    "subcat",
    "client_order",
    "quant",
    "client_order_number",
    'store_region',
    'quant_dt',
    'quant_cr',
    'fullname',
    'manu',
    'brend',
]


def id_to_months(start, end):
    return DATES[start].strftime("%Y-%m-%d"), DATES[end].strftime("%Y-%m-%d")


class Components:
    def __init__(self):
        self.title = dmc.Title("Анализ динамики продаж", order=1, c="blue")
        self.memo = dmc.Text("Данный раздел предоставляет аналитику по динамики изменения ключевых метрик.", size="xs")

        self.tab_store_id = 'sd_tab_store'
        self.tab_store = dcc.Store(id=self.tab_store_id, storage_type='memory')

        self.tab_conteiner_id = "sd_tab_conteiner"
        # ВАЖНО: dcc.Loading снаружи, а внутри — тот самый контейнер, чей children меняется колбэком
        self.tab_conteiner = dcc.Loading(
                id="tabs-loader",
                type="cube",
                delay_show=250,
                children=dmc.Container(
                    id=self.tab_conteiner_id,
                    fluid=True,
                    children=[LoadingScreen().component],
                ),
            )



        # Общий слайдер
        self.mslider_id = "sd_monthslider"
        self.mslider = MonthSlider(id=self.mslider_id)

        # store для ханнения df по слайдеру
        self.df_store_id = "df_store"
        self.df_store = dcc.Store(id=self.df_store_id, storage_type="session")

        # lable для хранения дат c учетом последнего обновления
        self.last_update_lb_id = "last_update_lb"
        self.last_update_lb = dcc.Loading(
            dmc.Badge(size="md", variant="light", radius="xs", 	color="red", id=self.last_update_lb_id)
        )

    def make_layout(self):
        return dmc.Container(
            children=[
                self.title,
                self.memo,
                self.mslider,
                self.last_update_lb,
                self.tab_conteiner,
                dcc.Store(id="dummy_imputs_for_slider"),
                dcc.Store(id="dummy_imputs_for_render"),
                self.tab_store,
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
            Output(self.tab_conteiner_id, "children"),
            Input(self.df_store_id, "data"),
            State(self.tab_store_id,'data')
        )
        def update_tabs(store_data,recent_tab):
            tab = 'general' if not recent_tab else recent_tab
            id = store_data["df_id"]
            from pages.dinamix.general.generaltab import layout as generaltab_layout
            from pages.dinamix.stores.main import layout as storetab_layout
            from pages.dinamix.cats.cattab import layout as cattab_layout
            from pages.dinamix.managers.managertab import layout as managertab_layout

            return dmc.Tabs(
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
                    dmc.TabsPanel(generaltab_layout(df_id=id), value="general"),
                    dmc.TabsPanel(storetab_layout(df_id=id), value="stores"),
                    dmc.TabsPanel(cattab_layout(), value="cats"),
                    dmc.TabsPanel(managertab_layout(), value="managers"),
                ],
                color="teal.3",
                autoContrast=True,
                variant="outline",
                value=tab,
                orientation="horizontal",
                id = 'sd_tabs'
            )
        
        @app.callback(
            Output(self.tab_store_id,'data'),
            Input('sd_tabs','value'),
            prevent_initial_call=True   
        )
        def update_tab_store(val):
            return val

        # Импортируем колбэки из страниц табов
        from pages.dinamix.general.generaltab import registed_callbacks as gt_callbacks
        from pages.dinamix.stores.main import callbacks  
        gt_callbacks(app)
        callbacks.register_callbacks(app)
        
