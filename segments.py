import pandas as pd
import numpy as np

# from datetime import date, datetime
# from io import BytesIO
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

class SegmentAnalisys:
    def __init__(self):
        from data import SegmentAnalisys
        sa = SegmentAnalisys()
        self.tree_id = 'sa_tree'
        self.tree = dmc.Tree(
            
            data=sa.data()[1],
            expandedIcon=DashIconify(icon="fa6-solid:arrow-down"),
            checkboxes=True,
            expandOnClick = True,
            selectOnClick =True,
            id = self.tree_id,
            expandOnSpace = True,
            className="my-tree"

        )
        self.details_id = 'sa_details'
        self.details = dmc.Text(children=['not selected'],id=self.details_id)
        self.tree_conteiner = dmc.Container(
            dmc.SimpleGrid(
                children=[
                self.tree,
                self.details
                ]
            )
        )
    
    def sa_callbacks(self, app: Dash):

        @app.callback(
            Output(self.details_id, "children"),            
            Input(self.tree_id, "checked"),
            Input(self.tree_id, "selected"),
        )
        def get_details(selecttd, vals):
            return f"Selected {selecttd} values {vals}"
