# Файл для модалки для AreaChart по магазинам

import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate
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
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import locale

locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")

from components import ValuesRadioGroups, DATES, NoData, BASE_COLORS, COLORS_BY_COLOR, COLORS_BY_SHADE, InDevNotice
from data import (
    load_columns_df,
    save_df_to_redis,
    load_df_from_redis,  
    load_columns_dates,  
    COLS_DICT,
)

# Делаем класс с модалкой

class StoreAreaChartModal:
    def __init__(self, clickdata=None,clickSeriesName=None): # clickdata - это возвращенный словарь при клике на areachart
        
        self.clickdata = clickdata
        self.clickSeriesName = clickSeriesName
        # Также прописываем id компонентов        
        self.modal_id = {'type':'store_area_chart_modal','index':'1'} #ID самого модала
        self.conteiner_id = {'type':'store_area_chart_modal_conteiner','index':'1'} #ID контейнера модала куда будем все складывать
    
    # Делаем первоначальный модал для layout    
    def create_components(self):        
        return  dmc.Modal(
                    children=[dmc.Container(id=self.conteiner_id)],
                    id=self.modal_id,
                    size="90%",
                )
    
    # Делаем метод для апдэйта модала при нажатие на график
    
    def update_modal(self):
        
        df = pd.DataFrame()
        
        #Загрудаем данные из редиски за текущий и предыдущий месяц
        
        if not self.clickdata:
           return 
        month = self.clickdata['eom']
        month = pd.to_datetime(month, errors="coerce")
        last_month = month + pd.offsets.MonthEnd(-1)
        last_year = month - pd.DateOffset(years=1)
        last_year = last_year + pd.offsets.MonthEnd(0)
        

        COLS = [
            "eom",
            "date",
            "dt",
            "cr",
            "amount",
            "store_gr_name",
            "eom",
            "chanel",
            "manager",
            "cat",
            "subcat",
            "client_order",
            "quant",
            "client_order_number",
            "store_gr_name_amount_ytd",
            
        ]
        dates = [month, last_month, last_year]
        df_current = load_columns_dates(COLS, dates)
        df_current["orders_type"] = np.where(
            df_current["client_order"] == "<Продажи без заказа>",
            "Прочие",
            "Заказы клиента",
        ) 
        
        # просто для примера
        
        df_selected_store:pd.DataFrame =  df_current[df_current['store_gr_name']==self.clickSeriesName].copy()
        df_selected_store['date']  = pd.to_datetime(df_selected_store['date'],errors='coerce')
        df_selected_store['day'] = df_selected_store['date'].dt.day 
        df_selected_store = df_selected_store.pivot_table(
            index = 'day',
            columns='eom',
            values=['amount'],
            aggfunc='sum'
        ).reset_index().sort_values(by='day').fillna(0)
        
        if isinstance(df_selected_store.columns, pd.MultiIndex):
            df_selected_store.columns = [col[1] if col[0] == 'amount' else col[0] for col in df_selected_store.columns]
            
        
        
        
        # Делаем отчет (пока такой)
        
        def memo():
            md_text = f"""
### Это clickdata 

```
{self.clickdata}
```

можно ее использовать что бы загружать данные из редиски      

### Это  clickSeriesName

```
{self.clickSeriesName}
```

можно использовать для филтрации данных по выбраному магазину

например 

```
df = pd.DataFrame()
        
#Загрудаем данные из редиски за текущий и предыдущий месяц

if not self.clickdata:
    return 
month = self.clickdata['eom']
month = pd.to_datetime(month, errors="coerce")
last_month = month + pd.offsets.MonthEnd(-1)
last_year = month - pd.DateOffset(years=1)
last_year = last_year + pd.offsets.MonthEnd(0)


COLS = [
    "eom",
    "date",
    "dt",
    "cr",
    "amount",
    "store_gr_name",
    "eom",
    "chanel",
    "manager",
    "cat",
    "subcat",
    "client_order",
    "quant",
    "client_order_number",
    "store_gr_name_amount_ytd",
    
]
dates = [month, last_month, last_year]
df_current = load_columns_dates(COLS, dates)
df_current["orders_type"] = np.where(
    df_current["client_order"] == "<Продажи без заказа>",
    "Прочие",
    "Заказы клиента",
)

# просто для примера

df_selected_store:pd.DataFrame =  df_current[df_current['store_gr_name']==self.clickSeriesName].copy()
df_selected_store['date']  = pd.to_datetime(df_selected_store['date'],errors='coerce')
df_selected_store['day'] = df_selected_store['date'].dt.day 
df_selected_store = df_selected_store.pivot_table(
    index = 'day',
    columns='eom',
    values=['amount'],
    aggfunc='sum'
).reset_index().sort_values(by='day')

if isinstance(df_selected_store.columns, pd.MultiIndex):
    df_selected_store.columns = [col[1] if col[0] == 'amount' else col[0] for col in df_selected_store.columns]
```


Вот df-ка получается какая

## Данные по магазину {self.clickSeriesName} за {self.clickdata['month_name']}

{df_selected_store.to_markdown(index=False)}

            
            
            """
            
            return dcc.Markdown(md_text,className='markdown-body')
        
        # Делаем дфку в md для пробы пера
        
        def temp_grid():
            return InDevNotice().in_dev_conteines
        
        # Основной return

        return memo(), temp_grid()
    
    def modal_children(self):
        
        if not self.clickdata:
           return NoData().component
        
        
        memo, temp_grid = self.update_modal()
        
        return dmc.Stack(
            [
                dmc.Title('Пробный модал по клику на график'),
                memo,
                temp_grid
            ]
        )
    
    def registered_callbacks(self,app): # Сюда колбэки добавляем которые с модалкой связанны будут 
        pass
        

            
        
         

            
        