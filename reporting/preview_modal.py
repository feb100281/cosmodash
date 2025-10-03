import dash
from dash import (
    dcc,
    html,
    Input,
    Output,
    State,    
    ctx,
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

from components import ValuesRadioGroups
from .report_generator import ReportGenerator, THEMES, SIZES
SIZES_FLIPPED = {v: k for k, v in SIZES.items()}

from data import load_report,save_report,delete_report

PREVIEW_MODAL = html.Div(
    dmc.Modal(
        id="preview_modal",
        size="90%",
        opened=False,
        children=[
            dmc.Space(h=10),
            dmc.Group(
                [
                    dmc.Fieldset(
                    [ValuesRadioGroups("PreviewModal_size_chose", SIZES_FLIPPED),],
                    legend=dmc.Text("Размер шрифта",size='md')  
                    ),
                    
                    dmc.Select(
                        id="PreviewModal_theme_selector",
                        data=list(THEMES.keys()),
                        size="md",
                        value=None,
                        label='Тема',
                        h=90
                    ),
                    
                    dmc.Button(
                        children=["Скачать"],
                        id="PreviewModal_dnl_button",
                        size="compact-sm",
                    ),
                ],
            ),
            dmc.Space(h=10),
            dcc.Loading(
            [dmc.Container(id="pdf_content", fluid=True),
             dcc.Download(id="PreviewModal_dcc_downloading")
             ],
            
            type='graph'
            )
        ],
    )
)


def preview_callbacks(app):
    @app.callback(
        Output('pdf_content','children', allow_duplicate=True),
        Output('pdf_download','data', allow_duplicate=True),
        Input('PreviewModal_theme_selector','value'),
        Input('PreviewModal_size_chose','value'),
        State('pdf_download','data'),
        prevent_initial_call=True,
    )
    def update_dnl_content(theme,fontsize,rid):
        report: ReportGenerator = load_report(rid)
        fs = SIZES_FLIPPED[fontsize]
        report.change_theme(theme)
        report.change_fontsize(fs)
        delete_report(rid)
        new_id = save_report(report)
        
        return report.return_iframe(),new_id
    
    @app.callback(
        Output('PreviewModal_dcc_downloading','data', allow_duplicate=True),
        Input('PreviewModal_dnl_button','n_clicks'),
        State('pdf_download','data'),
        prevent_initial_call=True,
    )
    def save_pdf(n_clicks,rid):
        if n_clicks:
           report: ReportGenerator = load_report(rid) 
           return report.for_dash_download(as_pdf=True)
           
        

