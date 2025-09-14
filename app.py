
import pandas as pd
import numpy as np
import dash

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
                    "Navbar",
                    dmc.NavLink(label="Резюме", href="/", active='exact'),
                    dmc.NavLink(label="Динамика", href="/Sales_dimamix", active='exact'),
                    dmc.NavLink(label="Сегменты", href="/Segments", active='exact'),
                    dmc.NavLink(label="Матрица", href="/Matrix", active='exact'),
                    
                ],
                p="md",
  
            ),
            dmc.AppShellMain([page_container,
                              html.Div(id="dummy-theme-output", style={"display": "none"}),
                              # Глобальные сторы что бы не было лишних телодвижений
                              sd_components().df_store,
                                              
                              
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
            # Output("rrag_grid", "className"),
            # Output("ps_tenant_table", "className"),
            # Output("ps_vacant_table", "className"),
            Input(self.theme_switch_id, "checked"),
            prevent_initial_call=True,
        )
        def theme_switch_change(checked):
            logo = self.logo_dark if checked else self.logo_light
            # rrgrid_className = "ag-theme-alpine-dark" if checked else "ag-theme-alpine"
            return logo

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





app = Dash(use_pages=True, pages_folder="",title="CosmoDash")
MainWnidow().main_page_callbacks(app)
sd_components().register_callbacks(app)

dash.register_page("Резюме", path="/", layout=SummaryComponents().layout)
dash.register_page("Динамика продаж", path="/Sales_dimamix", layout=sd_components().make_layout())
dash.register_page("Сегментный анализ", path="/Segments", layout=html.Div("page 1 subject 1"))
dash.register_page("Матрица", path="/Matrix", layout=html.Div("page 1 subject 2"))




app.layout = MainWnidow().page_layout

if __name__ == "__main__":
    app.run(debug=True)




# class MainWnidow:
#     def __init__(self):
#         # Здесь описываем компонеты основного окна

#         # Лого на сайте что бы не переписывать пути указал на static
#         self.logo_dark = dmc.Image(
#             src="/static/assets/logo.png",
#             id="logo-img",
#             h=25,
#             w="auto",
#             style={
#                 "filter": "invert(1)",
#                 "transition": "transform 0.3s ease",
#                 "cursor": "pointer",
#             },
#             mt="sm",
#         )

#         self.logo_light = dmc.Image(
#             src="/static/assets/logo.png",
#             id="logo-img",
#             h=25,
#             w="auto",
#             style={
#                 "transition": "transform 0.3s ease",
#                 "cursor": "pointer",
#             },
#         )

#         # Переключатель тем
#         self.theme_switch_id = "theme_switch"
#         self.theme_switch = dmc.Switch(
#             id=self.theme_switch_id,
#             label="",
#             checked=True,
#             onLabel=DashIconify(
#                 icon="line-md:moon-filled-to-sunny-filled-loop-transition",
#                 color=dmc.DEFAULT_THEME["colors"]["yellow"][5],
#                 width=30,
#             ),
#             offLabel=DashIconify(
#                 icon="line-md:moon-loop",
#                 color=dmc.DEFAULT_THEME["colors"]["yellow"][5],
#                 width=30,
#             ),
#             size="lg",
#             radius="sm",
#             color="blue",
#             style={
#                 "alignSelf": "center",
#                 "--switch-checked-bg": "#228be6",
#                 "--switch-thumb-color": "#ffffff",
#             },
#         )

#         # Заголовок сайте header
#         self.header_title = dmc.Center(
#             children=[dmc.Title("ПАНЕЛЬ ПРОДАЖ", order=1, c="blue")]
#         )

#         # Колонки для заголоки страницы
#         self.logo_col_id = "logo_col"
#         self.header_columns = dmc.Grid(
#             children=[
#                 dmc.GridCol(
#                     dmc.Flex(
#                         children=self.logo_dark,
#                         id=self.logo_col_id,
#                         justify="flex-start",
#                         align="end",
#                     ),
#                     span=3,
#                 ),
#                 dmc.GridCol(
#                     dmc.Flex(
#                         children=self.header_title, align="baseline", justify="center"
#                     ),
#                     span=6,
#                 ),
#                 dmc.GridCol(
#                     dmc.Flex(
#                         children=self.theme_switch, justify="flex-end", align="baseline"
#                     ),
#                     span=3,
#                 ),
#             ],
#             gutter="xl",
#             px="xl",
#             pt="xl",
#         )

#         # Делаем струтуру страницы
#         self.main_coteiner = dmc.AppShell(
#             children=[
#                 dmc.AppShellHeader(
#                     self.header_columns,
#                     style={
#                         "position": "fixed",
#                         "top": 0,
#                         "left": 0,
#                         "right": 0,
#                         "zIndex": 700,
#                     },
#                 ),
#                 # 👇 Основной контент
#                 dmc.AppShellMain(
#                     [
#                         self.main_tabs(),
#                         html.Div(id="fake-output", style={"display": "none"}),
#                         html.Div(id="dummy-theme-output", style={"display": "none"}),
#                     ]
#                 ),
#             ],
#             header={"height": 100},
#             padding="sm",
#         )
#         # Oпределяем начальную тему
#         self.initial_theme = "dark"

#         # Делаем layout для страницы и колбэков
#         self.page_layout = dmc.MantineProvider(
#             id="mantine-provider",
#             defaultColorScheme=self.initial_theme,
#             children=[
#                 self.main_coteiner,
#                 dcc.Store(id="theme-init", storage_type="local"),
#             ],
#         )

#     # Делаем колбэки
#     def main_page_callbacks(self, app: Dash):

#         @app.callback(
#             Output(self.logo_col_id, "children"),
#             # Output("rrag_grid", "className"),
#             # Output("ps_tenant_table", "className"),
#             # Output("ps_vacant_table", "className"),
#             Input(self.theme_switch_id, "checked"),
#             prevent_initial_call=True,
#         )
#         def theme_switch_change(checked):
#             logo = self.logo_dark if checked else self.logo_light
#             # rrgrid_className = "ag-theme-alpine-dark" if checked else "ag-theme-alpine"
#             return logo

#         # 🎯 Добавь фиктивный Output
#         app.clientside_callback(
#             """
#             function(checked) {
#                 const theme = checked ? 'dark' : 'light';
#                 document.documentElement.setAttribute('data-mantine-color-scheme', theme);
#                 localStorage.setItem('dash_theme', theme);
#                 return '';
#             }
#             """,
#             Output("dummy-theme-output", "children"),  # заменили проблемный Output
#             Input(self.theme_switch_id, "checked"),
#         )

#         # Колбэк инициализации темы
#         app.clientside_callback(
#             """
#             function() {
#                 const savedTheme = localStorage.getItem('dash_theme') || 'light';
#                 document.documentElement.setAttribute('data-mantine-color-scheme', savedTheme);
#                 return savedTheme === 'dark';
#             }
#             """,
#             Output(self.theme_switch_id, "checked"),
#             Input("theme-init", "modified_timestamp"),
#         )

#     # Табуляция для main
#     def main_tabs(self):
#         from dinamix import SalesDynamix
#         from segments import SegmentAnalisys
#         sd = SalesDynamix()
#         sa = SegmentAnalisys()

#         return dmc.Container(
#             children=[
#                 dmc.Tabs(
#                     [
#                         dmc.TabsList(
#                             [
#                                 dmc.TabsTab("Краткое резюме", value="summary"),
#                                 dmc.TabsTab("Динамика продаж", value="dinamix"),
#                                 dmc.TabsTab("Структурный анализ", value="structure"),
#                                 dmc.TabsTab("Матрица продаж", value="matrix"),
#                                 dmc.TabsTab("Пронгоз продаж", value="forecast"),
                                
                                
#                             ]
#                         ),
#                         dmc.TabsPanel("Gallery tab content", value="summary"),
#                         dmc.TabsPanel(sd.tab_container, value="dinamix"),
#                         dmc.TabsPanel(sa.tree_conteiner, value="structure"),
#                         dmc.TabsPanel("Settings tab content", value="matrix"),
#                         dmc.TabsPanel("Settings tab content", value="forecast"),
#                     ],
#                     color="blue.3",
#                     autoContrast=True,
#                     variant="pills",
#                     value="summary",
#                 )
#             ],
#             fluid=True,
#         )


# if __name__ == "__main__":

#     main_page = MainWnidow()
#     app = Dash(
#         __name__,
#         #suppress_callback_exceptions=True,
#     )
#     from dinamix import SalesDynamix
#     SalesDynamix().sd_callbacks(app)
    
#     from segments import SegmentAnalisys
#     SegmentAnalisys().sa_callbacks(app)
    
#     app.layout = main_page.page_layout

#     main_page.main_page_callbacks(app)
    

#     app.run(debug=True)
