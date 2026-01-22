import pandas as pd
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import json
import dash_ag_grid as dag
import math

from dash import dcc, Input, Output, State, no_update

# from .forecast import SEASONS_OPTIONS, forecast
from components import NoData

import locale

locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")

from .data import fletch_cats

#Делам header и общие контролеры со слайдером
class GeneralControllers:
    pass

#Делаем контроллеры матрицы
class LeftSection:
    def __init__(self):
        
        self.a_score_id = "a_score_id"
        self.b_score_id = "b_score_id"
        self.c_score_id = "c_score_id"
        
        self.x_score_id = "x_score_id"
        self.y_score_id = "y_score_id"
        self.z_score_id = "z_score_id"
        
        self.gr_multyselect_id = "gr_multyselect_id_for_matrix"
        self.cat_multyselect_id = "cat_multyselect_id_for_matrix"
        
        self.groupby_sc_id = "groupby_sc_id_for_matrix"
        
        self.launch_batton_id = "launch_batton_id_for_matrix"
        
        #--------------------------
        # Прописываем компоненты
        #--------------------------
        
        #Кнопки управления ABC
        a_acore_number_imput = dmc.NumberInput(            
            value=50,
            min=35,
            max=98,
            step = 1,
            allowDecimal=False,
            suffix="%",
            leftSection=DashIconify(icon="mynaui:letter-a-waves-solid",color="red",width=24),
            w=80,
            size='xs',
            id=self.a_score_id
        )
        b_acore_number_imput = dmc.NumberInput(            
            value=25,
            min=1,
            max=64,
            step = 1,
            allowDecimal=False,
            suffix="%",
            leftSection=DashIconify(icon="mynaui:letter-b-waves-solid",color="blue",width=24),
            w=75,
            size='xs',
            id=self.b_score_id
        )
        c_acore_number_imput = dmc.NumberInput(            
            value=25,
            min=1,
            max=64,
            step = 1,
            allowDecimal=False,
            disabled=True,
            suffix="%",
            leftSection=DashIconify(icon="mynaui:letter-c-waves-solid",color="gray",width=24),
            w=80,
            size='xs',
            id=self.c_score_id
        )
        abc_fieldset = dmc.Fieldset(
            children=[
                dmc.Group([
                    a_acore_number_imput,
                    b_acore_number_imput,
                    c_acore_number_imput
                ]
                )
                ],
            
            radius="sm",
            legend="Параметры для ABC расчетов"
        )
        
        #Кнопки управления XYZ
        x_acore_number_imput = dmc.NumberInput(            
            value=0.25,
            min=0.1,
            max=1,
            step = 0.01,
            allowDecimal=True,
            prefix="≤",
            leftSection=DashIconify(icon="mynaui:letter-x-diamond-solid",color="red",width=24),
            w=80,
            size='xs',
            id=self.x_score_id
        )
        y_acore_number_imput = dmc.NumberInput(            
            value=0.5,
            min=0.25,
            max=2,
            step = 0.01,
            allowDecimal=True,            
            leftSection=DashIconify(icon="mynaui:letter-y-diamond-solid",color="teal",width=24),
            w=75,
            size='xs',
            id=self.y_score_id
        )
        z_acore_number_imput = dmc.NumberInput(            
            value=0.5,
            min=0.5,
            max=3,
            step = 0.01,
            allowDecimal=True,
            prefix=">",
            leftSection=DashIconify(icon="mynaui:letter-z-diamond-solid",color="gray",width=24),
            w=80,
            size='xs',
            id=self.z_score_id,
            disabled=True
        )
        xyz_fieldset = dmc.Fieldset(
            children=[
                dmc.Group([
                    x_acore_number_imput,
                    y_acore_number_imput,
                    z_acore_number_imput
                ]
                )
                ],
            
            radius="sm",
            legend="Параметры для XYZ расчетов"
        )
        
        #Мультиселекты по группам и категориям
        
        self.cats_df = fletch_cats()
        
        gr_data = (
            self.cats_df[["gr_id", "gr_name"]]
            .dropna(subset=["gr_id", "gr_name"])
            .drop_duplicates()
            .assign(gr_id=lambda x: x["gr_id"].astype(str))
            .rename(columns={"gr_id": "value", "gr_name": "label"})
            .to_dict(orient="records")
        )
        
        cat_data = (
            self.cats_df[["cat_id", "cat_name"]]
            .dropna(subset=["cat_id", "cat_name"])
            .drop_duplicates()
            .assign(cat_id=lambda x: x["cat_id"].astype(str))
            .rename(columns={"cat_id": "value", "cat_name": "label"})
            .to_dict(orient="records")
        )
        
        gr_mulyselect = dmc.MultiSelect(
                            id=self.gr_multyselect_id, 
                            label="Группы", 
                            placeholder="Выберите группу",
                            data=gr_data,
                            w="100%", 
                            radius=0, 
                            clearable=True, 
                            searchable=True,
                            leftSection=DashIconify(icon="tabler:building-store"),
                        )
        
        cat_mulyselect = dmc.MultiSelect(
                            id=self.cat_multyselect_id, 
                            label="Магазин", 
                            placeholder="Выберите категорию",
                            data=cat_data,
                            w="100%", 
                            radius=0, 
                            clearable=True, 
                            searchable=True,
                            leftSection=DashIconify(icon="tabler:building-store"),
                        )
        
        cats_ms_fieldset = dmc.Fieldset(
            children=[
                
                    gr_mulyselect,
                    cat_mulyselect
            ],
                
                
            
            radius="sm",
            legend="Фильтр групп и категорий"
        )
        
        #Групировки 
        
        sc_groupby_switch = dmc.Switch(
            onLabel="ON", 
            offLabel="OFF", 
            radius="sm",
            labelPosition="right",
	        label="Групировать по подкатегориям",
            checked=False,
            id=self.groupby_sc_id        
            )
        
        groupby_sc_fieldset = dmc.Fieldset(
            children=[
                
                    sc_groupby_switch,
                   
            ],
                
                
            
            radius="sm",
            legend="Групировки номенклатур"
        )
        
        
        # Кнопка запуска
        
        launch_btn = dmc.Button(
            "Рассчитать",
            id=self.launch_batton_id,
            leftSection=DashIconify(icon="mynaui:rocket-solid",width=24),
        )
        
        
        
        #--------------------------
        # Финальный Layout
        #--------------------------
        
        self.left_section_layout = dmc.Container(
            children=[
                dmc.Title("Настройки матрицы",order=4),
                dmc.Space(h=20),
                abc_fieldset,
                dmc.Space(h=20),
                xyz_fieldset,
                dmc.Space(h=20),
                cats_ms_fieldset,
                dmc.Space(h=20),
                groupby_sc_fieldset,
                dmc.Space(h=20),
                launch_btn,
            ],
            fluid=True
        )
        
    def register_callbacks(self, app):
        
        #фильтр категорий при выбранной группе
        @app.callback(
            Output(self.cat_multyselect_id,"data"),
            Input(self.gr_multyselect_id,"value"),
            prevent_initial_call=True            
        )
        def filter_cat_ms(gr_list):
            gr_list_int = [int(x) for x in gr_list]
            df = self.cats_df[
                self.cats_df["gr_id"].isin(gr_list_int)
            ]
            
            return (
            df[["cat_id", "cat_name"]]
            .dropna(subset=["cat_id", "cat_name"])
            .drop_duplicates()
            .assign(cat_id=lambda x: x["cat_id"].astype(str))
            .rename(columns={"cat_id": "value", "cat_name": "label"})
            .to_dict(orient="records")
        )
        
        #автопересчет abc
        @app.callback(
            Output(self.b_score_id,'value'),
            Output(self.c_score_id,'value'),
            Output(self.b_score_id,'max'),
            Output(self.c_score_id,'max'),
            Input(self.a_score_id,'value'),
            prevent_initial_call=True      
        )
        def split_bc(a_val):            
            r = 100 - a_val
            b = math.ceil(r/2)  
            c = 100-b-a_val
            return b,c,r-1,r-1
        
        @app.callback(
            Output(self.c_score_id,'value',allow_duplicate=True),            
            Input(self.b_score_id,'value'),
            State(self.a_score_id,'value'),
            prevent_initial_call=True      
        )
        def adjust_c(b_val,a_val):    
            c = 100-b_val-a_val
            return c
            
        #автопересчет xyz
        @app.callback(
            Output(self.y_score_id,'value'),
            Output(self.y_score_id,'min'),
            Output(self.z_score_id,'value'),
            Input(self.x_score_id,'value'),
            State(self.y_score_id,'value'),
            prevent_initial_call=True   
        )
        def set_yz(x_val,y_val):
            y_min = x_val+0.25
            z = 0 
            if y_val > y_min:
                z = y_val
            else:
                z = y_min
            
            return z, y_min, z 
            
        @app.callback(
            Output(self.z_score_id,'value',allow_duplicate=True),
            Input(self.y_score_id,'value'),
            prevent_initial_call=True    
        )
        def set_z(y_val):
             return y_val
            
             
        
        

#Делаем саму матрицу
class RightSection:
    pass

#Соединяем все вместе в единый layout
class MainWindow:
    
    def __init__(self):
        self.left_section = LeftSection()
    
    def layout(self):
       
        
        return dmc.Container(
            children=[
                dmc.Title("Создание и анализ ассортиментой матрицы", order=1, c="indigo"),
                dmc.Text(
                    "В данном разделе можно создавать и анализировать ассортиментные матрицы",
                    size="xs",
                ),
                dmc.Space(h=40),
                dmc.Grid(
                    [                    
                    dmc.GridCol([self.left_section.left_section_layout],span=3),
                    dmc.GridCol([],span=9)
                    ]
                    )
                ],
            fluid=True                    
                )
            
        
    def register_callbacks(self, app):
        self.left_section.register_callbacks(app)
        


