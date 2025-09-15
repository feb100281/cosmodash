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

