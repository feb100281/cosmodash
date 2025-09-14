# Тут функции для создания компонентов что бы все были одинаковые
import pandas as pd
import numpy as np

from dash import (    
    dcc,
    html,   
)
import dash_ag_grid as dag
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import locale
locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")

DATES = pd.date_range("2021-01-31", periods=30*12, freq="ME")

class MonthSlider(dmc.RangeSlider):
    def __init__(self,id, min_date='2022-01-31', max_date=None, defaul_period=12, **kwargs):
        dates = DATES
        if max_date is None:
            today = pd.Timestamp.today() 
            max_date = today + pd.offsets.MonthEnd(0)
            max_date = max_date.strftime('%Y-%m-%d')
        else:
            max_date = pd.to_datetime(max_date) + pd.offsets.MonthEnd(0)
            max_date = max_date.strftime('%Y-%m-%d')
        
        # собираем DataFrame
        df = pd.DataFrame({
            "month_id": range(len(dates)),
            "date": dates            
        })
        df['month_name'] = df['date'].dt.strftime("%b").str.capitalize() + "\u202f" + df['date'].dt.strftime("%y")
        df['month_name'].str.capitalize()
               
        min_id = df.loc[df['date'] == pd.Timestamp(min_date), 'month_id'].iloc[0]
        max_id = df.loc[df['date'] == pd.Timestamp(max_date), 'month_id'].iloc[0]        
        
        df = df[df['month_id']<=max_id]
        
        month_marks = (
        df[["month_id", "month_name"]]          
        .sort_values("month_id") 
        .assign(
            value=lambda x: x["month_id"], label=lambda x: x["month_name"]
        )[  
            ["value", "label"]
        ]  
        .to_dict("records")  
        )
        short_marks = [mark.copy() for mark in month_marks]  # Копируем month_marks
        diff = max_id - min_id
        keep_indices = []
        if diff <= 12:
            keep_indices = [min_id, max_id]
        elif diff <= 24:
            # 3 точки: начало, середина, конец
            keep_indices = np.linspace(min_id, max_id, 3, dtype=int).tolist()
        elif diff <= 36:
            # 4 точки: начало, 1/3, 2/3, конец
            keep_indices = np.linspace(min_id, max_id, 4, dtype=int).tolist()
        elif diff <= 42:
            # 5 точек для больших диапазонов
            keep_indices = np.linspace(min_id, max_id, 5, dtype=int).tolist()
        else:
            keep_indices = np.linspace(min_id, max_id, 6, dtype=int).tolist()        
       
        for i in range(max_id):
            if i not in keep_indices:
                short_marks[i]["label"] = ""
        
        
        super().__init__(
            id=id,
            value=[max_id - defaul_period, max_id],
            marks=short_marks,
            mb=35,
            min=min_id,
            max=max_id,
            minRange=1,
            labelAlwaysOn=True,
            size=10,
            mt="xl",
            styles={"thumb": {"borderWidth": 2, "padding": 3}},
            color="red",
            thumbSize=42,
            thumbChildren=[
                DashIconify(icon="mdi:arrow-right-circle", width=22), # mdi:heart
                DashIconify(icon="mdi:arrow-left-circle", width=22),
            ],
            label={
                "function": "formatMonthLabel",
                "options": {
                    "monthDict": {month["value"]: month["label"] for month in month_marks}
                },
            },
            **kwargs,
        )

class ValuesRadioGroups(dmc.RadioGroup):
    def __init__(self,id_radio,options_dict:dict,grouped=True,val=None,**kwargs):
        
        if val is None and options_dict:
            val = next(iter(options_dict.keys()))
        
        container = dmc.Group if grouped else dmc.Stack

        
        children = container([
            dmc.Radio(
                label=v,
                value=k,
                color="blue"
            )
            for k, v in options_dict.items()
        ])

        super().__init__(
            children=children,
            my=5,           
            value=val,
            size="xs",
            mt=5,
            id=id_radio,
            **kwargs,
        )

class InDevNotice:
    def __init__(self):
        self.in_dev_conteines = dmc.Container(
            dmc.Center(
                dmc.Stack(
                    [
                        dmc.Space(h=20),
                        dmc.Center(
                            DashIconify(icon="lucide:hammer", width=80, color="gray"),
                        ),
                        dmc.Center(
                            dmc.Title("Раздел в разработке", order=3, c="gray"),
                        ),
                        dmc.Center(
                            dmc.Text(
                                "Я активно работаю над этим разделом. Скоро здесь появятся новые возможности.",
                                size="md",
                                c="dimmed",
                            ),
                        ),
                        dmc.Space(h=20),
                    ]
                )
            ),
            fluid=True,
            px="xl",
        )
        