import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate
from datetime import datetime, date, timedelta
import io
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
import base64

from components import ValuesRadioGroups, MonthSlider


class StoreDynamix:
    def __init__(self):

        self.tab_title = dmc.Title(
            "Динамика продаж по торговым точкам и форматам", order=3, c="teal"
        )
        self.memo = dmc.Text(
            "Данный раздел показывает динаминку продаж по магазинам в соответсвии с выбраннами параметрами ....."
        )
        self.chanels_filter_id = "sd_chanels_filter"

        self.chanels_filter = dmc.MultiSelect(
            label="Канал",
            placeholder="Выберете канал продаж",
            data=[],
            id=self.chanels_filter_id,
            w="100%",
            mb=10,
            clearable=True,
            searchable=True,
            leftSection=DashIconify(icon="tabler:users"),
        )
        self.region_filter_id = "sd_region_filter"
        self.region_filter = dmc.MultiSelect(
            label="Регион",
            placeholder="Выберете регион",
            data=[],
            id=self.region_filter_id,
            w="100%",
            mb=10,
            clearable=True,
            searchable=True,
            leftSection=DashIconify(icon="tabler:users"),
        )
        self.store_filter_id = "sd_store_filter"
        self.store_filter = dmc.MultiSelect(
            label="Магазин",
            placeholder="Выберете магазин",
            data=[],
            id=self.store_filter_id,
            w="100%",
            mb=10,
            clearable=True,
            searchable=True,
            leftSection=DashIconify(icon="tabler:users"),
        )
        values_options = {
            "amount": dmc.Text("Выручка", c="dimmed", size="xs"),
            "av_check": dmc.Text("Средний чек", c="teal", size="xs"),
            "returns": dmc.Text("% возвратов", c="red", size="xs"),
        }

        self.sd_radio_id = "sd_vals_choised"
        self.sd_radio_choises = ValuesRadioGroups(
            id_radio=self.sd_radio_id, options_dict=values_options, grouped=True
        )
        
        


class SalesDynamix:
    def __init__(self):

        self.title = dmc.Title("Анализ динамики продаж", order=1, c="blue")
        self.memo = dmc.Text("На графиках ниже ...", size="xs")

        self.store_switch_id = "sd_store_swich"
        self.store_switch = dmc.Stack(
            [
                dmc.Tooltip(
                    label="Показать по магазинам",
                    position="top-start",
                    withArrow=True,
                    children=dmc.Switch(
                        id=self.store_switch_id,
                        onLabel="On",
                        offLabel="Off",
                        radius="sm",
                        size="lg",
                        checked=False,
                        color="blue",
                        thumbIcon=DashIconify(
                            icon="tabler:chart-pie",
                            width=16,
                            color=dmc.DEFAULT_THEME["colors"]["blue"][5],
                        ),
                    ),
                ),
                dmc.Text("Показать по машазинам", size="xs", c="dimmed", ml=4),
            ],
            gap="xs",
            mt=5,
        )

        self.areachart_id = "sd_areachart"
        self.areachart = dmc.AreaChart(
            id=self.areachart_id,
            h=550,
            dataKey="month_fmt",
            data=[],
            tooltipAnimationDuration=500,
            withPointLabels=False,
            areaProps={
                "isAnimationActive": True,
                "animationDuration": 500,
                "animationEasing": "ease-in-out",
                "animationBegin": 500,
            },
            series=[],
            valueFormatter={"function": "formatNumberIntl"},
            type="default",
            withLegend=True,
            legendProps={"verticalAlign": "bottom"},
            highlightHover=True,
            curveType="Monotone",
            referenceLines=[],
            py="xl",
            px="xl",
        )

        values_options = {
            "amount": dmc.Text("Выручка", c="dimmed", size="xs"),
            "av_check": dmc.Text("Средний чек", c="teal", size="xs"),
            "returns": dmc.Text("% возвратов", c="red", size="xs"),
        }

        self.sd_radio_id = "sd_vals_choised"
        self.sd_radio_choises = ValuesRadioGroups(
            id_radio=self.sd_radio_id, options_dict=values_options, grouped=True
        )
        self.mslider_id = "sd_mslider"
        self.mslider = MonthSlider(id=self.mslider_id)

        self.areachart_contriner = dmc.Container(
            children=[
                dmc.Flex(dmc.Title("Выручка по магазинам", order=4), justify="center"),
                self.store_switch,
                self.areachart,
                self.sd_radio_choises,
            ],
            fluid=True,
        )

        # Табы внутри таба

        self.tab_in_tab = dmc.Container(
            children=[
                dmc.Tabs(
                    [
                        dmc.TabsList(
                            [
                                dmc.TabsTab("Магазины", value="stores"),
                                dmc.TabsTab("Категории", value="cats"),
                                dmc.TabsTab("Менеджеры", value="managers"),
                            ],
                            justify="right",
                        ),
                        dmc.TabsPanel("Gallery tab content", value="summary"),
                        dmc.TabsPanel(self.areachart_contriner, value="stores"),
                        dmc.TabsPanel("Settings tab content", value="managers"),
                        dmc.TabsPanel("Settings tab content", value="matrix"),
                    ],
                    color="teal.3",
                    autoContrast=True,
                    variant="outline",
                    value="stores",
                    orientation="horizontal",
                )
            ],
            fluid=True,
        )

        self.tab_container = dmc.Container(
            children=[self.title, self.memo, self.mslider, self.tab_in_tab],
            fluid=True,
            px="xl",
            py="xl",
        )

    def sd_callbacks(self, app: Dash):

        @app.callback(
            Output(self.areachart_id, "data"),
            Output(self.areachart_id, "series"),
            Input(self.store_switch_id, "checked"),
        )
        def update_charts(store_sitch_check):
            from data import SalesDynamix, COLORS

            sd = SalesDynamix(
                store_filter=store_sitch_check
            )  # store_breadown -> store_filter?

            df = sd.data()[0]
            if df is None or df.empty:
                return dmc.Text("Нет данных для отображения", c="red")

            df = (
                df.pivot_table(
                    index=["month_id", "month_fmt"],
                    columns=["store_gr_name"],
                    values="amount",
                    aggfunc="sum",
                )
                .reset_index()
                .fillna(0)
                .sort_values(by="month_id")
            )

            # проверка на пустой фрейм
            if df.empty:
                return dmc.Text("Нет данных после агрегации", c="red")

            df = df.drop(columns="month_id")
            data = df.to_dict(orient="records")
            columns = [col for col in df.columns if col not in ["month_fmt"]]

            series = [
                {"name": col, "color": COLORS[i % len(COLORS)]}
                for i, col in enumerate(columns)
            ]

            return data, series
