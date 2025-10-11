
import pandas as pd
import numpy as np
import dash
import jwt
from flask import request, session, redirect

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
    page_container
)
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import locale
locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")

from pages.summary import SummaryComponents
from pages.sales_dinamix import Components as sd_components
from pages.segment_analisis import SEGMENTS_LAYOUT, SEGMENT_DF_STORE, SEGMENTS_CALLBACKS
from pages.planing import PLANING
from components import NoData, InDevNotice, ClickOnNotice
from reporting.preview_modal import PREVIEW_MODAL, preview_callbacks

scripts = [
    "https://cdnjs.cloudflare.com/ajax/libs/dayjs/1.10.8/dayjs.min.js",     # dayjs  
    "https://cdnjs.cloudflare.com/ajax/libs/dayjs/1.10.8/locale/ru.min.js", # russian locale
]

class MainWnidow:
    def __init__(self):
        # Здесь описываем компонеты основного окна

        # Лого на сайте что бы не переписывать пути указал на static
        self.logo_dark = dmc.Image(
            src="/static/assets/logo.png", #!!!! БЕРЕМ ИЗ STAIC
            id="logo-img",
            h=25,
            w="auto",
            style={
                "filter": "invert(1)",
                "transition": "transform 0.3s ease",
                "cursor": "pointer",
            },
            
        )

        self.logo_light = dmc.Image(
            src="/static/assets/logo.png",
            id="logo-img",
            h=25,
            w="auto",
            style={
                "transition": "transform 0.3s ease",
                "cursor": "pointer",
            },
        )
        
        self.header_logo_id = 'header_logo'
        self.header_logo =  dmc.AspectRatio(
        ratio=10,
        children=self.logo_dark,
        id = self.header_logo_id,
        )

        # Переключатель тем
        self.theme_switch_id = "theme_switch"
        self.theme_switch = dmc.Switch(
            id=self.theme_switch_id,
            label="",
            checked=True,
            onLabel=DashIconify(
                icon="line-md:moon-filled-to-sunny-filled-loop-transition",
                color=dmc.DEFAULT_THEME["colors"]["yellow"][5],
                width=30,
            ),
            offLabel=DashIconify(
                icon="line-md:moon-loop",
                color=dmc.DEFAULT_THEME["colors"]["yellow"][5],
                width=30,
            ),
            size="lg",
            radius="sm",
            color="blue",
            style={
                "alignSelf": "center",
                "--switch-checked-bg": "#228be6",
                "--switch-thumb-color": "#ffffff",
            },
        )

        # Заголовок сайте header
        self.header_title = dmc.Center(
            children=[dmc.Title("ПАНЕЛЬ ПРОДАЖ", order=1, c="blue")]
        )
        
        #Бургер для меню
        self.burger_id = 'burger'
        self.burger = dmc.Burger(
                        id=self.burger_id, 
                        size="sm",                       
                        visibleFrom="sm",
                        opened=False,
                    )
        
        self.mobile_burger_id = 'mobile_burger'
        self.mobile_burger = dmc.Burger(
                        id=self.mobile_burger_id, 
                        size="sm",
                        hiddenFrom="sm",
                        opened=False,
                    )
                
        self.header_group = dmc.Group(
            children=[
                self.burger,
                self.mobile_burger,
                self.header_logo,
                self.header_title,
                dmc.Flex(children=self.theme_switch,ml="auto")
            ],
            h="100%",
            px="md",
        )
        
        self.layout = dmc.AppShell(
            [
            dmc.AppShellHeader(
                self.header_group
                                
            ),       
            dmc.AppShellNavbar(
                id="navbar",
                children=[
                    "Навигация",
                    dmc.NavLink(label="Резюме", 
                                description = 'Краткое резюме по продажам',
                                href="/summary", 
                                active='exact',
                                leftSection=DashIconify(icon='fluent-mdl2:total',width=16)
                                ),
                    dmc.NavLink(label="Динамика",
                                description = 'Анализ динамики продаж', 
                                href="/", 
                                active='exact',
                                leftSection=DashIconify(icon='vaadin:line-bar-chart',width=16)
                                ),
                    dmc.NavLink(label="Сегменты", 
                                href="/Segments", 
                                active='exact',
                                description = 'Сегментный анализ и аналитика', 
                                leftSection=DashIconify(icon='fluent-emoji-high-contrast:puzzle-piece',width=16)
                                ),
                    dmc.NavLink(label="Матрица", 
                                href="/Matrix", 
                                active='exact',
                                leftSection=DashIconify(icon='mdi:matrix',width=16),
                                description = 'Анализ ассортиментой матрицы', 
                                ),
                    dmc.NavLink(label="Планирование", 
                                href="/forecast", 
                                active='exact',
                                leftSection=DashIconify(icon='streamline-ultimate:presentation-projector-screen-budget-analytics',width=16),
                                description = 'Планирование продаж', 
                                ),
                    
                ],
                p="md",
  
            ),
            dmc.AppShellMain([page_container,
                              html.Div(id="dummy-theme-output", style={"display": "none"}),                              
                              sd_components().df_store,                              
                              dcc.Store(id='pdf_download', storage_type='memory'),
                              PREVIEW_MODAL,
                              SEGMENT_DF_STORE,
                                                         
                              ]),
        ],
            header={"height": 60},
            navbar={
                "width": 300,
                "breakpoint": "sm",
                "collapsed": {"mobile": True, "desktop": False},
            },
            padding="md",
            id="appshell",
        )
        

        self.initial_theme = "dark"

        # Делаем layout для страницы и колбэков
        self.page_layout = dmc.MantineProvider(
            id="mantine-provider",
            defaultColorScheme=self.initial_theme,
            children=[
                self.layout,
                dcc.Store(id="theme-init", storage_type="local"),                
            ],
        )

    # Делаем колбэки
    def main_page_callbacks(self, app: Dash):

       
        @app.callback(
            Output("appshell", "navbar"),
            Input(self.mobile_burger_id, "opened"),
            Input(self.burger_id, "opened"),
            State("appshell", "navbar"),
        )
        def toggle_navbar(mobile_opened, desktop_opened, navbar):
            navbar["collapsed"] = {
                "mobile": not mobile_opened,
                "desktop": not desktop_opened,
            }
            return navbar
        
        @app.callback(
            Output(self.header_logo_id, "children"),
            #Output({"type":"check_distiribution_ag",'index':MATCH}, "className"),
            # Output("ps_tenant_table", "className"),
            # Output("ps_vacant_table", "className"),
            Input(self.theme_switch_id, "checked"),
            prevent_initial_call=True,
        )
        def theme_switch_change(checked):
            logo = self.logo_dark if checked else self.logo_light
            #rrgrid_className = "ag-theme-alpine-dark" if checked else "ag-theme-alpine"
            
            return logo, #rrgrid_className

        # 🎯 Добавь фиктивный Output
        app.clientside_callback(
            """
            function(checked) {
                const theme = checked ? 'dark' : 'light';
                document.documentElement.setAttribute('data-mantine-color-scheme', theme);
                localStorage.setItem('dash_theme', theme);
                return '';
            }
            """,
            Output("dummy-theme-output", "children"),  # заменили проблемный Output
            Input(self.theme_switch_id, "checked"),
        )

        # Колбэк инициализации темы
        app.clientside_callback(
            """
            function() {
                const savedTheme = localStorage.getItem('dash_theme') || 'light';
                document.documentElement.setAttribute('data-mantine-color-scheme', savedTheme);
                return savedTheme === 'dark';
            }
            """,
            Output(self.theme_switch_id, "checked"),
            Input("theme-init", "modified_timestamp"),
        )


def main_app():
    app = Dash(
        use_pages=True, 
        pages_folder="",          
        title="Панель продаж",
        suppress_callback_exceptions=True,
        external_scripts=scripts
    )
    

    MainWnidow().main_page_callbacks(app)
    sd_components().register_callbacks(app)
    SEGMENTS_CALLBACKS.register_callbacks(app)
    PLANING.registered_callbacks(app)
    preview_callbacks(app)
    # print(app.callback_map.keys())

    dash.register_page("Резюме", path="/summary", layout=SummaryComponents().layout)
    dash.register_page("Динамика продаж", path="/", layout=sd_components().make_layout())
    dash.register_page("Сегментный анализ", path="/Segments", layout=SEGMENTS_LAYOUT)
    dash.register_page("Матрица", path="/Matrix", layout=InDevNotice().in_dev_conteines)
    dash.register_page("Планирование", path="/forecast", layout=PLANING.layout())

    app.layout = MainWnidow().page_layout
    
    

    return app

if __name__ == "__main__":
    main_app().run(debug=True, port=8050)

