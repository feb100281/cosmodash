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

from components import ValuesRadioGroups, DATES, NoData, BASE_COLORS, COLORS_BY_COLOR, COLORS_BY_SHADE
from data import (
    load_columns_df,
    save_df_to_redis,
    load_df_from_redis,    
    COLS_DICT,
)


class Components:
    def __init__(self,df_id=None):
    
        self.df_id = df_id if df_id is not None else None
        self.chart_data_store_id = {'type':'data_store','index':'1'}
        self.filters_data_store_id = {'type':'filter_store','index':'1'}
        self.chanel_multyselect_id = {'type':'chanel_multyselect','index':'1'}
        self.store_multyselect_id = {'type':'store_multyselect','index':'1'}
        self.big_area_chart_id = {'type':'big_area_chart','index':'1'}
        
     
    def data(self):
        df_id = self.df_id
        
        if not df_id:
            return None
        
        df_data: pd.DataFrame = load_df_from_redis(df_id)
        
        if df_data.empty:
           return None
        
        
        df_eom = df_data[['eom','store','chanel','dt','cr','amount','quant','client_order_number']].copy().fillna(0)
        df_eom = df_eom.pivot_table(
                index=['eom','store','chanel'],
                values=['dt','cr','amount','client_order_number','quant'],
                aggfunc={
                    'dt':'sum',
                    'cr':'sum',
                    'amount':'sum',
                    'quant':'sum',
                    'client_order_number':'nunique'
                }
            ).fillna(0).reset_index().sort_values(by='eom')
        df_filters = df_eom[['store','chanel']].drop_duplicates()
        
        def data_store():            
            return dcc.Store(id=self.chart_data_store_id,data=df_eom.to_dict("records"),storage_type='memory')
            
        def filter_store():
            return dcc.Store(id=self.filters_data_store_id,data=df_filters.to_dict("records"),storage_type='memory')
        
        def charts_multyselects():
            return dmc.Group(
                children=[
                    dmc.MultiSelect(
                        id=self.chanel_multyselect_id,
                        label='Канал',
                        placeholder='Выберите канал',
                        data=df_filters['chanel'].unique(),
                        w='100%',
                        mb=10,
                        clearable=True,
                        searchable=True,
                        leftSection=DashIconify(icon="tabler:users")                        
                    ),
                    dmc.MultiSelect(
                        id=self.store_multyselect_id,
                        label='Магазин',
                        placeholder='Выберите магазин',
                        data=df_filters['store'].unique(),
                        w='100%',
                        mb=10,
                        clearable=True,
                        searchable=True,
                        leftSection=DashIconify(icon="tabler:users")                        
                    ),
                    
                ]
            )
        
        def big_area_chart():
            df = df_eom.copy()
            df = df.pivot_table(
                index='eom',
                columns='store',
                values='amount',
                aggfunc='sum'
            ).fillna(0).reset_index().sort_values(by='eom')
            
            df["eom"] = pd.to_datetime(df["eom"], errors="coerce")
            df["eom"] = df["eom"].dt.strftime("%b\u202F%y").str.capitalize()
            
            df.rename(columns=COLS_DICT)
            
            data = df.to_dict(orient="records")
            columns = [col for col in df.columns if col not in ["eom"]]
            
            series = [
            {"name": col, "color": COLORS_BY_SHADE[i % len(COLORS_BY_SHADE)]}
            for i, col in enumerate(columns)
            ]
            
            return dmc.AreaChart(
                id = self.big_area_chart_id,
                h=600,
                dataKey='eom',
                data=data,
                series=series,
                tooltipAnimationDuration=500,
                areaProps={
                "isAnimationActive": True,
                "animationDuration": 500,
                "animationEasing": "ease-in-out",
                "animationBegin": 500,
                },
                withPointLabels=False,
                valueFormatter={"function": "formatNumberIntl"},
                withLegend=True,
                legendProps={"verticalAlign": "bottom"},
                connectNulls=True, 
            )
        
        return (
            data_store(),
            filter_store(),
            charts_multyselects(),
            big_area_chart()            
        )
            
        
def layout(df_id=None):
    df_id = df_id if df_id is not None else None
    comp = Components(df_id)
    try:    
        data_store, filter_store, charts_multyselects, big_area_chart = comp.data() 
            
        return dmc.Container(
            children=[
                dmc.Title("Динамика по магазинам и каналам",order=4,c='blue'),
                data_store,
                filter_store,
                dmc.Space(h=10),
                charts_multyselects,
                big_area_chart           
                
            ]
        )
    except:
        return NoData().component

