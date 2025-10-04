import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate


import dash # вот здесь был пробел

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
    callback_context,
)
import dash_ag_grid as dag
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


from components import InDevNotice
def layout():
    
    return InDevNotice().in_dev_conteines


class CatComponents:
    def __init__(self,df_id=None):
        
        self.df_id = df_id
        
        self.tab_conteiner_id = {'type':'CatComponents_Conteiner','index':'1'}
        
        self.ag_grid_id = {'type':'Cat_ag_grid-id','index':'1'}
        
        self.area_chart_conteiner_id = {'type':'Cat_area_chart_conteiner_id','index':'1'}
        
    def create(self):
        pass
        