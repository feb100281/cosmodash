import pandas as pd
import numpy as np
from dash_iconify import DashIconify
import dash_mantine_components as dmc
from .db_queries import fletch_item_details
from dash import dcc, Input, Output, State, no_update, MATCH

import locale
locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')


class AGModal:
    def __init__(self):
        self.modal_id = {'type': 'segments_ag_modal', 'index': '1'}
        self.modal_conteiner_id = {'type': 'segments_ag_modal_container', 'index': '1'}
    
    def layout(self):
        return dmc.Modal(
                id=self.modal_id,
                size="90%",
                opened=False,
                children=[                    
                    dmc.Container(id=self.modal_conteiner_id, fluid=True),                   
                ]
            )
    def update_modal(self, d, start, end):
        
        item_id = int(d['item_id'])
        fullname = d['fullname']
        init_date = d['init_date']
        article = d['article']
        manu = d['manu']
        brend = d['brend']
       
        
        first_date = pd.to_datetime(init_date)
        last_date = pd.to_datetime(end)
        
        # Делаем таймлайн по месяцам будем использовать для сводной таблицы и графиков
        timeline = pd.date_range(
            start=first_date + pd.offsets.MonthEnd(0),
            end=last_date + pd.offsets.MonthEnd(0),
            freq='ME'
        )
        
        timeline = pd.DataFrame({
            'eom': timeline,
        })
        timeline['month_id'] = timeline['eom'].dt.strftime('%Y-%m')
        
        # Получаем данные по продажам товара
        sales_data = fletch_item_details(item_id, start, end)
        sales_data['date'] = pd.to_datetime(sales_data['date'])
        sales_data['sd_eom'] = sales_data['date'] + pd.offsets.MonthEnd(0)
        sales_data['month_id'] = sales_data['sd_eom'].dt.strftime('%Y-%m')
        sales_data['amount'] = sales_data['dt'] - sales_data['cr']
        sales_data['quant'] = sales_data['quant_dt'] - sales_data['quant_cr']
        # Расчитываем цену по продажам но можно будет и по выручки
        sales_data['price'] = np.where(sales_data['quant_dt']>0, sales_data['dt']/sales_data['quant_dt'], 0)        
                
        # Расчитываем статистику по продажам
        def item_stats(period = 'Все данные', dataset = 'amount'):
            stats_df = sales_data[sales_data['period'] == period].copy()
            stats_df = pd.merge(timeline, stats_df, on='month_id', how='left')
            # print(stats_df.columns)
            stats_df['month_num'] = stats_df['eom'].dt.month
            stats_df['year'] = stats_df['eom'].dt.year.astype(str)
            stats_df['Месяц'] = stats_df['eom'].dt.strftime('%b').str.capitalize()
            
            stats_df = stats_df.pivot_table(
                index=['month_num','Месяц'],
                columns='year',
                values=dataset,
                aggfunc='sum',
                
            ).reset_index().sort_values('month_num')
            stats_df = stats_df.drop(columns='month_num')
            stats_df = stats_df.set_index('Месяц')
            stats_df.loc['Итого'] = stats_df.select_dtypes('number').sum()
            # stats_df.columns.names = [None, None]
            # stats_df.index.names = [None]
            ss = stats_df.columns
            html_table = (stats_df.style
             .format('{:,.0f}',subset=ss,na_rep='-',thousands='\u202F',)
             .set_table_attributes('class="forecast-table" ')
             .set_caption("Выручка (тут лучше тепловую карту и переключатель цена / кол-во сделать)")
            #  .hide(axis='index')
            ).to_html()
                        
            
            return html_table
            
            
            
            
            #Тут можно тепловую карту сдлелать
        
        
        
        stats = item_stats()
        
        return dmc.Stack(
            [
                dmc.Center([dmc.Title(f"Карточка товара: {fullname} (Артикль: {article})", order=3,c='blue')]),
                dmc.Space(h=10),
                dmc.Divider(label="Статистика", labelPosition="center", variant="dashed"),
                dmc.Space(h=10),
                dmc.Center([dcc.Markdown(stats, dangerously_allow_html=True)]),
            ]
        )
        
           
        
    
    
    def register_callbacks(self, app):
        pass

