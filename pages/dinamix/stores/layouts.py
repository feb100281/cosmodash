# Файл основной разметки таба по магазинам - то что выдно при заходе на таб и сборки drill down components

import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate


import dash # вот здесь был пробел

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
    callback_context,
)
import dash_ag_grid as dag
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import locale

locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")

from components import ValuesRadioGroups, DATES, NoData, BASE_COLORS, COLORS_BY_COLOR, COLORS_BY_SHADE
from data import (
    load_columns_df,
    save_df_to_redis,
    load_df_from_redis,    
    COLS_DICT,
)

# Суда добавляем зависимые класс из других файлов Не забывает писать . 
from .modal_area_chart import StoreAreaChartModal


class StoresComponents:
    def __init__(self,df_id=None): #df_id передаем в качестве аргутента - это ключь к df в редис с данным по слайдеру. Он автоматом меняется при изменении.
        
        self.df_id = df_id if df_id is not None else None
        # В init описываем только id компонентов в формате {'type':'','index':'1'} очень важно так описать
        
        self.chart_data_store_id = {'type':'st_data_store','index':'1'} # ID store JSON для магазинов 
        self.chart_series_store_id = {'type':'st_series_store','index':'1'} # ID store JSON для серий
        self.filters_data_store_id = {'type':'filter_store','index':'1'} # ID  store JSON для фильтров
        self.chanel_multyselect_id = {'type':'chanel_multyselect','index':'1'} # ID для мультиселекта по каналам
        self.store_multyselect_id = {'type':'store_multyselect','index':'1'}  # ID для мультиселекта по магазинам
        self.stores_area_chart_id = {'type':'stores_area_chart','index':'1'}  # ID для графика динамики магазинов
    
    # Делаем метод для загрузки данных и создания динамически компонентов в первый раз. Что бы не обращаться каждый раз к редис создаем компоненты в подметодах    
    def create_components(self):        
        #Сначала грузим данные для компонентов в тело метода
        df_id = self.df_id
        
        if not df_id:
            return 
        
        df_data: pd.DataFrame = load_df_from_redis(df_id) #Загрузка df из редис по выбранным периодам
        
        if df_data.empty:
           return 
        # ====Собираем здесь все df которые могут понадобиться =====
        
        #данные по месяцам        
        df_eom = df_data.pivot_table(
                index=['eom','store_gr_name','chanel'],
                values=['dt','cr','amount','client_order_number','quant'],
                aggfunc={
                    'dt':'sum',
                    'cr':'sum',
                    'amount':'sum',
                    'quant':'sum',
                    'client_order_number':'nunique'
                }
            ).fillna(0).reset_index().sort_values(by='eom')
        
        #Данные для фильтров 
        df_filters = df_eom[['store_gr_name','chanel']].drop_duplicates()
        
        # и так далее ...
        
        # ==== Теперь делаем сами компоненты !!!!!!
        
        
        
        
        # Store для фильтров
        def filter_store():
            return dcc.Store(id=self.filters_data_store_id,data=df_filters.to_dict("records"),storage_type='memory')
        
        # Группа мультиселектов для графика
        def charts_multyselects():
            return dmc.Group(
                children=[
                    dmc.MultiSelect(
                        id=self.chanel_multyselect_id,
                        label='Канал',
                        placeholder='Выберите канал',
                        data=df_filters['chanel'].unique(),
                        w='100%',
                        mb=10,
                        clearable=True,
                        searchable=True,
                        leftSection=DashIconify(icon="tabler:users")                        
                    ),
                    dmc.MultiSelect(
                        id=self.store_multyselect_id,
                        label='Магазин',
                        placeholder='Выберите магазин',
                        data=df_filters['store_gr_name'].unique(),
                        w='100%',
                        mb=10,
                        clearable=True,
                        searchable=True,
                        leftSection=DashIconify(icon="tabler:users")                        
                    ),
                    
                ]
            )
            
        def big_area_chart():
            df = df_eom.copy()
            df = df.pivot_table(
                index='eom',
                columns='store_gr_name',
                values='amount',
                aggfunc='sum'
            ).fillna(0).reset_index().sort_values(by='eom')
            
            df["eom"] = pd.to_datetime(df["eom"], errors="coerce")
            df["month_name"] = df["eom"].dt.strftime("%b\u202F%y").str.capitalize()
            
            df.rename(columns=COLS_DICT)
            
            data = df.to_dict(orient="records")
            columns = [col for col in df.columns if col not in ["eom","month_name"]]
            
            series = [
            {"name": col, "color": COLORS_BY_SHADE[i % len(COLORS_BY_SHADE)]}
            for i, col in enumerate(columns)
            ]
            return dmc.Stack(
                [
                    dmc.AreaChart(
                id = self.stores_area_chart_id,
                h=600,
                dataKey='month_name',
                data=data,
                series=series,
                tooltipAnimationDuration=500,
                areaProps={
                "isAnimationActive": True,
                "animationDuration": 500,
                "animationEasing": "ease-in-out",
                "animationBegin": 500,
                },
                withPointLabels=False,
                valueFormatter={"function": "formatNumberIntl"},
                withLegend=True,
                legendProps={"verticalAlign": "bottom"},
                connectNulls=True, 
                ),
                dcc.Store(id=self.chart_data_store_id,data=data,storage_type='memory'),
                dcc.Store(id=self.chart_series_store_id,data=series,storage_type='memory')
                
                ]
            )
        
        # краткий отчет
        
        def memo():
            min_date = pd.to_datetime(df_eom['eom'].min())
            min_date = min_date.strftime('%d %b %Y')
            max_date = pd.to_datetime(df_eom['eom'].max())
            max_date = max_date.strftime('%d %b %Y')
            
            df_stores = df_eom.pivot_table(
                index = 'store_gr_name',
                values='dt',
                aggfunc='sum'
            ).fillna(0).reset_index().sort_values(by='dt',ascending=False)
            
            l = ''
            
            for _,rows in df_stores.iterrows():
                l += f"- {rows['store_gr_name']}: {rows['dt']/1_000_000:,.2f} млн рублей \n"
            
            #MD лучше выравнивать по левому краю он чувствителен     
            md_text = f"""
## Краткий отчет о продажах по магазинам за период с {min_date} по {max_date}

За рассматриваемый период чистая выручка по магазинам распределилась следующим образом:
{l}
            
            """
            
            return dmc.Spoiler(
                children=dcc.Markdown(md_text,className='markdown-body'),
                maxHeight=50,
                hideLabel='Скрыть',
                showLabel='Читать далее'
            )
        
        
        # Теперь все наши подметоды котрые возвращают компоненты - возврашаем в тупле основной функции      
        return (
            
            filter_store(),
            charts_multyselects(),
            big_area_chart(),
            memo()            
        )        
    
    # Теперь делаем layout для всего таба
    def tab_layout(self):
        
        # проверяем есть ли df_id
        if not self.df_id:
            return NoData().component #возвращаем заглушку
        
        # распарсиваем компонеты 
        
        filter_store, charts_multyselects, big_area_chart,memo = self.create_components()
        
        # Теперь создаем Layout таба по магазинам
        
        return dmc.Container(
            children=[
            dmc.Title('Динамика продаж по магазинам',order=3, c='blue'),
            dmc.Space(h=5),
            memo,
            dmc.Space(h=5),
            dmc.Title('График продаж по магазинам',order=4),
            charts_multyselects,
            big_area_chart,
            dmc.Space(h=5),
            filter_store,
            StoreAreaChartModal().create_components() # Не забываем загрузить модалку для графика    
            ],
            fluid=True
        )
    
    # ======= Теперь далаем все возможные колбэки 
    
    # Смтотри что бы работать с динамическими колбэками нам придетсы использовать MATCH 
    
    def register_callbacks(self,app):
        
        #Для простоты type прописываем здесь
        
        area_chart = self.stores_area_chart_id['type']
        chanel_filter = self.chanel_multyselect_id['type']
        store_filter = self.store_multyselect_id['type']
        series_store = self.chart_series_store_id['type']
        data_store = self.chart_data_store_id['type']
        filter_data = self.filters_data_store_id['type']
        modal = StoreAreaChartModal().modal_id['type']
        modal_conteiner = StoreAreaChartModal().conteiner_id['type']
        
        # Регим модальные колбэки если есть        
        StoreAreaChartModal().registered_callbacks(app)
        
        #Колбэк для вызова модала при клике на график 
        
        @app.callback(
            Output({"type": modal, "index": MATCH}, "opened"),
            Output({"type": modal_conteiner, "index": MATCH}, "children"),
            Input({"type": area_chart, "index": MATCH}, "clickData"),
            Input({"type": area_chart, "index": MATCH}, "clickSeriesName"),            
            State({"type": modal, "index": MATCH}, "opened"),
            prevent_initial_call=True,
        )
        def show_and_update_modal(clickData, clickSeriesName,  opened):            
            container_data = StoreAreaChartModal(clickData, clickSeriesName).update_modal()
            return not opened, container_data
        
        
        #Колбэки главного графика
        @app.callback(
            Output({'type': store_filter, 'index': MATCH}, 'data'),
            Output({'type': area_chart, 'index': MATCH}, 'series'),
            
            Input({'type': chanel_filter, 'index': MATCH}, 'value'),
            Input({'type': store_filter, 'index': MATCH}, 'value'),
            
            State({'type': filter_data, 'index': MATCH}, 'data'),
            State({'type': series_store, 'index': MATCH}, 'data'),
            State({'type': data_store, 'index': MATCH}, 'data'),
            prevent_initial_call=True,
        )
        def update_chart_series_and_store_filter(chanel_filter_val, store_filter_val, filter_data, series_val, data_val):
            
            ctx = callback_context

            # Определяем, какой Input вызвал callback
            triggered_input_id = ctx.triggered[0]['prop_id']
            trigered_input = triggered_input_id
            
            def ensure_list(x):
                if x is None:
                    return []
                if isinstance(x, str):
                    return [x]
                return list(x)
                        
            def filter_series(chanel_filter_val):                
                df = pd.DataFrame(filter_data)
                if chanel_filter_val:
                    df = df[df['chanel'].isin(ensure_list(chanel_filter_val))]
                store_filter_data = [
                    {"value": s, "label": s} for s in df['store_gr_name'].unique().tolist()
                ]
                store_names = [s["value"] for s in store_filter_data]
                new_series = [item for item in series_val if item["name"] in store_names]
                return store_filter_data, new_series
            
            def filter_series_store(store_filter_val, chanel_filter_val):
                store_names = ensure_list(store_filter_val)
                if store_names:
                    new_series = [item for item in series_val if item["name"] in store_names]
                else:
                    # если фильтр пустой, используем фильтрацию по каналу
                    new_series = filter_series(ensure_list(chanel_filter_val))[1]
                return new_series
            
            

            if chanel_filter in trigered_input:
               return  filter_series(chanel_filter_val)

            elif store_filter in trigered_input:
                return no_update, filter_series_store(store_filter_val,chanel_filter_val)
            
                    
                
        
        
        
        
        
       
        
       
       
        
    
    
    





