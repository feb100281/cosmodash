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

from components import ValuesRadioGroups, DATES, NoData, month_str_to_date
from data import (    
    load_df_from_redis,
    COLS_DICT,
)


class AreaChartModal:
    def __init__(self,month=None):
        self.month = month if month is not None else None
        self.modal_id = {'type':'gt_area_chart_modal','index':'1'}
        
    def make_modal(self):
        month = self.month 
        if not month:
           return 
        
        


class Components:
    def __init__(self, df_id=None):

        self.df_id = df_id if df_id is not None else None
        self.area_chart_id = {"type": "gt_big_area_chart", "index": "1"}
        self.av_check_chart_id = {"type": "gt_av_check_chart", "index": "1"}
        self.check_selector_id = {"type": "selector", "index": "1"}
        self.memo_id = {"type": "memo", "index": "1"}

    def data(self):
        df_id = self.df_id

        if not df_id:
            return None

        df_data: pd.DataFrame = load_df_from_redis(df_id)

        if df_data.empty:
            return None

        df_eom = (
            df_data.pivot_table(
                index="eom", values=["dt", "cr", "amount"], aggfunc="sum"
            )
            .fillna(0)
            .reset_index()
            .sort_values("eom")
        )

        def update_area_chart():
            df: pd.DataFrame = df_eom.copy()
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

            series = [
                {"name": "Чистая выручка", "color": "indigo.3", "type": "bar"},
                {"name": "Продажи", "color": "green.6", "type": "line"},
                {"name": "Возвраты", "color": "red.6", "type": "line"},
            ]

            return dmc.CompositeChart(
                id=self.area_chart_id,
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
                series=series,
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
            if df.empty:
                return NoData().component
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

            data = df.to_dict(orient="records")
            columns = [col for col in df.columns if col not in ["eom"]]

            series = [
                {"name": "Средний чек", "color": "green.6"},
                {"name": "Медиана", "color": "blue.6"},
            ]

            selector_vals = {"1": "Средн / Медиана", "2": "Макс/Мин"}

            return dmc.Stack(
                [
                    ValuesRadioGroups(
                        id_radio=self.check_selector_id,
                        options_dict=selector_vals,
                        val="1",
                    ),
                    dmc.AreaChart(
                        id=self.av_check_chart_id,
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
                    ),
                ]
            )

        def memo():

            from_date = pd.to_datetime(
                df_data["date"].min(), errors="coerce"
            ).normalize()
            to_date = pd.to_datetime(df_data["date"].max(), errors="coerce").normalize()

            md_text = f"""
            ## Краткий отчет за период с {from_date.strftime('%d %B %Y')} по {to_date.strftime('%d %B %Y')} 
            
            За рассматриваемый период:
            
            - **чистая выручка от реализации** составила: {df_data['amount'].sum()/1_000_000:,.2f} млн рублей;
            - общие продажи: {df_data['dt'].sum()/1_000_000:,.2f} млн рублей;
            - возвраты {df_data['cr'].sum()/1_000_000:,.2f} млн рублей.
            """

            return dmc.Spoiler(
                children=[dcc.Markdown(md_text)],
                maxHeight=50,
                hideLabel="Скрыть",
                showLabel="Читать далее",
            )
        return update_area_chart(), update_av_check_chart(), memo()


def layout(df_id=None):
    df_id = df_id if df_id is not None else None
    comp = Components(df_id)
    try:
        area_chart, av_check_chart, memo = comp.data()

        return dmc.Container(
            children=[
                dmc.Space(h=10),
                memo,
                dmc.Space(h=20),
                dmc.Title("Динамика целевых показателей продаж", order=4, c="blue"),
                dmc.Space(h=10),
                area_chart,
                dmc.Space(h=20),
                dmc.Title("Динамика средних чеков по заказам", order=4, c="blue"),
                dmc.Space(h=10),
                av_check_chart,
            ]
        )
    except:
        return NoData().component

def registed_callbacks(app):
    c = Components()
    @app.callback(
        Output({"type": c.av_check_chart_id['type'], "index": MATCH}, "series"),
        Input({"type": c.check_selector_id['type'], "index": MATCH}, "value"),
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
            
