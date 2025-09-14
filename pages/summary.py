# Страница с суммарной информацией

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


from data import load_columns_df, COLORS, COLS_DICT
from components import InDevNotice


class SummaryComponents:    
    def __init__(self):
       self.layout = InDevNotice().in_dev_conteines     
    
    def register_callbacks(app: Dash):
        pass

