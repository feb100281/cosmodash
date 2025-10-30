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
from .queries import fletch_cats, get_df, cats_report, VALS_DICT, OPTIONS_SWITCHS
from dash_iconify import DashIconify
from data import load_df_from_redis, delete_df_from_redis, save_df_to_redis


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
        self.df_store = dcc.Store(id=self.df_store_id, storage_type="session")
        self.tree_id = 'cattree_id_for_cat_analisys'
        self.last_update_lb_id = "cat_last_update_lb"
        self.last_update_lb = dcc.Loading(
            dmc.Badge(size="md", variant="light", radius="xs", 	color="red", id=self.last_update_lb_id)
        )
        self.cat_group_controll_id = 'cat_group_controll_id'
        self.cat_period_controll_id = 'cat_group_controll_id'
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
            color = "#6d28d9",
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
            color = "#6d28d9",
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
                self.title,
                self.memo,                
                self.mslider,                
                self.last_update_lb,                
                dmc.Space(h=10),
                dmc.Title('Отчет по категориям',c='blue',order=3),
                dmc.Group(
                    [
                        self.value_controlls,
                        self.option_controll,
                    ]
                    ),
                
                dmc.Container(
                    id = self.summary_data_conteiner_id,
                    fluid=True
                    ),
                dmc.Space(h=10),
                # dmc.Grid(
                #     [
                #         dmc.GridCol(
                #             [
                #                 dmc.Container(
                #                     [
                #                         dmc.Tree(
                #                             id = self.tree_id,
                #                             data=self.make_tree(),
                #                             expandedIcon=DashIconify(icon="line-md:chevron-right-circle", width=20),
                #                             collapsedIcon=DashIconify(icon="line-md:arrow-up-circle", width=20),
                #                             checkboxes=True,
                #                         )                                        
                #                     ],
                #                     fluid=True
                #                 )                                
                #             ],
                #             span=3.5
                #         ),
                #         dmc.GridCol(
                #             [
                #                 dmc.Container(
                #                     [
                                       
                #                         dmc.Group([self.group_controll,self.period_controll])
                                           
                #                     ],
                #                     id = self.data_conteiner_id,
                #                     fluid=True
                #                 )
                #             ],
                #             span=7.5
                #         )
                        
                #     ]
                    
                # ),
                # dmc.Text(id='cat_tex_temp',),
                dcc.Store(id='dummy_store_for_cat_trigger'),
                dmc.Space(h=50)
                
            ],
            fluid=True
        )
    
    def registered_callbacks(self,app):
        
        #Пишем df в cash
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
                       
            # if store_data and "df_id" in store_data:
            #     if store_data["start"] == start and store_data["end"] == end:
            #        return  no_update, f"{start_dt.strftime('%d %b %y')} - {end_dt.strftime('%d %b %y')}"

            #     delete_df_from_redis(store_data["df_id"])

            # df = get_df(start, end)

            # df_id = save_df_to_redis(df,expire_seconds=1200)

            # store_dict = {
            #     "df_id": df_id,
            #     "start": start,
            #     "end": end,
            #     "slider_val": slider_value,
            # }

            # min_date = pd.to_datetime(df["date"].min())
            # max_date = pd.to_datetime(df["date"].max())

            notificattion = (
                f"{start_dt.strftime('%b %y')} - {end_dt.strftime('%b %y')}"
            )

            return notificattion, cats_report(start, end, opt, val)
        
        # @app.callback(
        #     Output('cat_tex_temp','children'),
        #     Input(self.df_store,'data'),
        #     prevent_initial_call=True,
        # )
        # def showids(data):
            
        #     return data['df_id']
        
        
        
        
        @app.callback(
            Output({'type':'cat_chart','index':MATCH},'withBarValueLabel'),
            Input({'type':'val_switch','index':MATCH},'checked'),            
            prevent_initial_call=True,            
        )
        def show_data(val):
            
            return val
        
        
        
        