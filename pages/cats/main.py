import pandas as pd
import numpy as np
import dash_mantine_components as dmc
from dash import dcc
import locale
locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")
from components import(
    MonthSlider,
    DATES,
    COLORS_BY_COLOR,
    COLORS_BY_SHADE,
    LoadingScreen
)


class CatsMainWindow:
    def __init__(self):
        self.title = dmc.Title("Анализ категорий", order=1, c="blue")
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
        
        