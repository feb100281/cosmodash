import pandas as pd
import numpy as np
import dash_mantine_components as dmc
from dash import dcc,Input,Output,State, no_update, MATCH
import locale
locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")
from components import(
    MonthSlider,
    DATES,
    COLORS_BY_COLOR,
    COLORS_BY_SHADE,
    LoadingScreen,
    ValuesRadioGroups
    
)
from .queries import fletch_cats, cats_report, VALS_DICT, OPTIONS_SWITCHS
from dash_iconify import DashIconify
# from data import load_df_from_redis, delete_df_from_redis, save_df_to_redis


def id_to_months(start, end):
    return DATES[start].strftime("%Y-%m-%d"), DATES[end].strftime("%Y-%m-%d")








class CatsMainWindow:
    def __init__(self):
        self.title = dmc.Title([DashIconify(icon='streamline-flex:search-category')," Анализ категорий"], 
                               order=1, c="blue")
        self.memo = dmc.Text("Данный раздел предоставляет аналитику по динамики изменения категорий.", size="xs")
        self.mslider_id = "cats_monthslider"
        self.mslider = MonthSlider(id=self.mslider_id)
        self.df_store_id = "cats_df_store"
        self.df_store = dcc.Store(id=self.df_store_id, storage_type="session")
        self.last_update_lb_id = "cats_last_update_lb"
        self.last_update_lb = dcc.Loading(
            dmc.Badge(size="md", variant="light", radius="xs", 	color="red", id=self.last_update_lb_id)
        )
        self.data_conteiner_id = 'cats_data_conteirer_id'
        self.tree_id = 'cattree_id_for_cat_analisys'
        self.cat_group_controll_id = 'cat_group_controll_id'
        self.cat_period_controll_id = 'cat_period_controll_id'
        self.summary_data_conteiner_id = 'summary_data_conteiner_id_for_cats'
        
        self.group_controll = dmc.SegmentedControl(
            data = [
                {"value": 0, "label": "Группа"},
                {"value": 1, "label": "Категория"},
                {"value": 2, "label": "Подкатегория"},
                ],
            value=1, 
            id = self.cat_group_controll_id,
            variant='filled',  
            color = "blue",
            withItemsBorders=False,
            radius="md"
                         
            
        )
        self.period_controll = dmc.SegmentedControl(
            data = [
                {"value": 0, "label": DashIconify(icon='iwwa:calendar',width=16)},
                {"value": 1, "label": DashIconify(icon='iwwa:week',width=16)},
                {"value": 2, "label": DashIconify(icon='iwwa:year',width=16)},
                ],
            value=2,  
            id = self.cat_period_controll_id,
            variant='filled',  
            color = "blue",
            withItemsBorders=False,
            radius="md"
            
        )
        self.value_controll_id = 'value_controll_id_cats'
        
        self.value_controlls = ValuesRadioGroups(id_radio=self.value_controll_id,options_dict=VALS_DICT)
        
        self.option_controll_id = 'option_controll_id_cats'
        self.option_controll = ValuesRadioGroups(id_radio=self.option_controll_id, options_dict = OPTIONS_SWITCHS)
        
        
        
    def make_tree(self):
        df = fletch_cats()
        tree = []

        def find_or_create(lst, value, label):
            """Находит или создаёт узел"""
            for node in lst:
                if node["value"] == str(value):
                    return node
            node = {
                "value": str(value),
                "label": str(label),
                "children": [],
                "_count": 0,
            }
            lst.append(node)
            return node

        for _, row in df.iterrows():
            pid, pname = int(row["parent_id"]), row["parent"]
            cid, cname = int(row["cat_id"]), row["cat"]
            sid, sname = row["subcat_id"], row["subcat"]

            # Уровень 1
            pid_key = f"{pid}"
            parent_node = find_or_create(tree, pid_key, pname)

            # Уровень 2
            if cid is not None:
                cid_key = f"{pid}-{cid}"
                cat_node = find_or_create(parent_node["children"], cid_key, cname)
            else:
                cat_node = parent_node
                cid_key = pid_key

            # Уровень 3
            if sid is not None:
                sid_key = f"{cid_key}-{sid}"
                subcat_node = find_or_create(cat_node["children"], sid_key, sname)
            else:
                subcat_node = cat_node
                sid_key = cid_key
             
        return tree
    
    def layout(self):
        return dmc.Container(
            [
                # Верхняя часть: заголовок, заметка, слайдер периода, дата обновления
                dmc.Stack(
                    [
                        self.title,
                        self.memo,
                        self.mslider,
                        dmc.Group(
                            [self.last_update_lb],
                            justify="space-between",
                            align="center",
                        ),
                    ],
                    gap="xs",
                ),

                dmc.Space(h=8),

                # Заголовок раздела
                dmc.Group(
                    [
                        dmc.Title("Отчет по категориям", order=3, c="blue"),
                        dmc.Badge("Аналитика по категориям", variant="outline", radius="sm"),
                    ],
                    justify="space-between",
                    align="center",
                ),

                dmc.Space(h=6),

                

                # Карточка с настройками (МЕТРИКА / ОПЦИИ)
                dmc.Card(
                    withBorder=True,
                    radius="md",
                    p="md",
                    style={"backdropFilter": "blur(4px)"},
                    children=[
                        dmc.Group(
                            [
                                # МЕТРИКА
                                dmc.Stack(
                                    [
                                        dmc.Group(
                                            [
                                                dmc.ThemeIcon(
                                                    radius="xl",
                                                    size="sm",
                                                    variant="light",
                                                    color="blue",
                                                    children=DashIconify(icon="lucide:bar-chart-2", width=16),
                                                ),
                                                dmc.Text("МЕТРИКА", tt="uppercase", fw=700, size="sm", c="dimmed"),
                                            ],
                                            gap=6,
                                        ),
                                        self.value_controlls,
                                    ],
                                    gap="xs",
                                    w="100%",
                                ),

                                # ОПЦИИ
                                dmc.Stack(
                                    [
                                        dmc.Group(
                                            [
                                                dmc.ThemeIcon(
                                                    radius="xl",
                                                    size="sm",
                                                    variant="light",
                                                    color="blue",
                                                    children=DashIconify(icon="lucide:settings", width=16),
                                                ),
                                                dmc.Text("ОПЦИИ", tt="uppercase", fw=700, size="sm", c="dimmed"),
                                            ],
                                            gap=6,
                                        ),
                                        self.option_controll,
                                    ],
                                    gap="xs",
                                    w="100%",
                                ),
                            ],
                            grow=True,
                            justify="space-between",
                            align="flex-start",
                            wrap="wrap",
                            gap="xl",
                        )
                    ],
                ),


                dmc.Space(h=10),

                # Контейнер с саммари/графиками
                dmc.Container(id=self.summary_data_conteiner_id, fluid=True),

                dmc.Space(h=10),
                dcc.Store(id="dummy_store_for_cat_trigger"),
                dmc.Space(h=40),
            ],
            fluid=True,
        )

    
    def registered_callbacks(self,app):
        
        @app.callback(
            # Output(self.df_store,'data'),
            Output(self.last_update_lb_id,'children'),
            Output(self.summary_data_conteiner_id,'children'),
            Input(self.mslider_id,'value'),
            Input('dummy_store_for_cat_trigger','id'),
            Input(self.option_controll_id,'value'),
            Input(self.value_controll_id,'value'),
            # State(self.df_store,'data'),
            prevent_initial_call=False,
        )
        def save_to_cash(slider_value,dummy,opt,val):
            start, end = id_to_months(slider_value[0], slider_value[1]) 
            end_dt = pd.to_datetime(end)
            start_dt = pd.to_datetime(start) + pd.offsets.MonthBegin(-1)
            start = start_dt.strftime('%Y-%m-%d')
                       
            

            notificattion = (
                f"{start_dt.strftime('%b %y')} - {end_dt.strftime('%b %y')}"
            )

            return notificattion, cats_report(start, end, opt, val)
        
     
        

        
        
        @app.callback(
            Output({'type': 'cat_chart', 'index': MATCH}, 'withBarValueLabel'),
            Input({'type': 'val_switch', 'index': MATCH}, 'checked')
        )
        def toggle_values(show_values):
            return bool(show_values)