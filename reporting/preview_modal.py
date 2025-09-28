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
import base64

locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")

from components import ValuesRadioGroups, MonthSlider, DATES, NoData
from .report_generator import ReportGenerator

THEME_LIST = [
    "flatly",
    "lux",
    "superhero",
    "brite",
    "litera",
    "zephyr",
    "simplex",
    "journal",
    "sandstone",
    "united",
    "yeti",
    "vapor",
    "cerulian",
    "cosmo",
    "minty",
    "morph",
    "pulse",
    "quartz",
    "sketchy",
    "spacelab",
]


class PreviewModal:
    def __init__(self, report: ReportGenerator = None):
        self.theme_list = THEME_LIST
        self.download_content = report
        dl = self.download_content.for_dash_download(as_pdf=True)
        b64 = dl["content"]
        self.src = "data:application/pdf;base64," + b64
        self.modal_id = "PreviewModal"
        self.dnl_button_id = "PreviewModal_dnl_button_id"
        self.theme_selector_id = "PreviewModal_dnl_button_id"
        self.pdf_conteiner_id = "pdf_container"

        self.theme_select = dmc.Select(
            data=self.theme_list,
            value="spacelab",
            id=self.theme_selector_id,
            size="xs",
            searchable=True,
        )

        self.dnl_button = dmc.Button(
            "Загрузить", size="compact-xs", id=self.dnl_button_id
        )

        self.PreviewModal = dmc.Modal(
            id=self.modal_id,
            size="90%",
            children=[
                dmc.Group([self.theme_select, self.dnl_button]),
                dmc.Container(
                    id=self.pdf_conteiner_id,
                    fluid=True,
                    children=[
                        html.Embed(
                            id="pdf_file",
                            src=self.src,
                            type="application/pdf",
                            style={"width": "100%", "height": "80vh"},
                        )
                    ],
                ),
            ],
        )

    def preview_callbacks(self, app):
        pass
