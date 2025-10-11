import pandas as pd
import numpy as np
from prophet import Prophet
from data import ENGINE
import dash_mantine_components as dmc
from components import NoData
from dash import dcc
import datetime
import locale
locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")


memos = {
    'yearly_seasonality':'Годовая сезонность',
    'weekly_seasonality':'Недельная сезонность',
    'seasonality_mode':'Метод учета сезонности',
}

comments_seasons = {
    "seasonality_mode": "additive — если выручка стабильна (амплитуда колебаний постоянна); multiplicative — если выручка растет, чтобы сезонные колебания масштабировались вместе с уровнем выручки.",
    "yearly_seasonality": "Годовая сезонность — учитывает повторяющиеся годовые циклы (например, рост продаж в декабре, спад летом).",
    "weekly_seasonality": "Недельная сезонность — отражает различия между днями недели (например, меньше продаж в выходные, пик в будни)."
}

comments_trend = {
    "growth": "linear — стандартный линейный тренд без ограничений; logistic — рост с насыщением, требует столбцов cap и floor.",
    "changepoint_prior_scale": "Чувствительность к изменениям тренда: меньше = плавнее, больше = гибче (типичные значения 0.01–0.5).",
    "changepoint_range": "Процент исторических данных, где можно искать точки изменения тренда (0.8 = первые 80%, 1.0 = вся история).",
    "n_changepoints": "Максимальное количество потенциальных изломов тренда (обычно 25)."
}

SEASONS_OPTIONS = [
    {"value": 'additive', 'label': 'аддитивная'},
    {"value": 'multiplicative', 'label': 'мультипликативная'},
]


def historical_data(start=None, end=None)->pd.DataFrame:
    conditions = ''
    if start and end:
       conditions = f"WHERE date BETWEEN '{start}' AND '{end}'" 
    if end and not start:
       conditions = f"WHERE date <= '{end}' "  
    if not end and start:
       conditions = f"WHERE date >= '{start}' "   
    
    q = f"""
    select 
    date as ds,
    sum(dt-cr) as y
    from sales_salesdata
    {conditions}
    group by date
    
    """
   
    return pd.read_sql(q,ENGINE)

def forecast(       
            horizon,
            current_date = None,
            historical_cut_off = None,
            yearly_seasonality = True,
            weekly_seasonality = True,
            seasonality_mode = 'additive',
            changepoint_prior_scale = 0.05,
            changepoint_range = 1,
            n_changepoints = 25,
        ):
    
    
    current_date = pd.Timestamp.now().normalize() if not current_date else pd.to_datetime(current_date).normalize()
    end = current_date.strftime('%Y-%m-%d')
    
    historical_cut_off = pd.to_datetime(historical_cut_off).normalize() if historical_cut_off else None
    start = historical_cut_off.strftime('%Y-%m-%d') if historical_cut_off else None
    
    data = historical_data(start=start,end=end)
    
    horizon = pd.to_datetime(horizon).normalize()
    
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        seasonality_mode=seasonality_mode,
        growth = 'linear',
        changepoint_prior_scale=changepoint_prior_scale,
        changepoint_range = changepoint_range,
        n_changepoints = n_changepoints
    )
    
    
    model.fit(data)
        
    delta = horizon - current_date
    num_days = delta.days
    
   
    future = model.make_future_dataframe(periods=num_days)
        
    forecast = model.predict(future)
    
    def yearly_seasons():
        if not yearly_seasonality:
           return NoData().component
        components = model.predict_seasonal_components(future)
        dff = pd.DataFrame({
            'ds': future['ds'],
            'yearly': components['yearly'].values
        })
        dff['month_num'] = pd.to_datetime(dff['ds']).dt.month
        dff['month'] = pd.to_datetime(dff['ds']).dt.strftime("%b").str.capitalize()

        # группируем и сортируем по номеру месяца
        monthly_profile = (
            dff.groupby(['month_num', 'month'])['yearly']
            .mean()
            .reset_index()
            .sort_values('month_num')
            .to_dict('records')
        )
        
        return dmc.LineChart(
                h=300,
                dataKey="month",
                data=monthly_profile,
                series = [
                    {"name": "yearly", "color": "indigo.6"},
                    
                ],
                curveType="linear",
                tickLine="xy",
                withYAxis=False,
                withDots=False,
                withTooltip=False
            )        
    
    def mape(dff:pd.DataFrame):
        dff['ds'] = pd.to_datetime(dff['ds']).dt.normalize()
        cur_date = pd.to_datetime(current_date).normalize()
        df = dff[dff['ds']<=cur_date]
        df = df.pivot_table(
            index='ds',
            columns='type',
            values='y',
            aggfunc='sum'
        ).reset_index().sort_values('ds').fillna(0)
        df.columns = df.columns.get_level_values(-1)
        
        df['mape'] = np.where(
           df['План'] == 0,0,
           np.abs(df['План'] - df['Факт']) / df['План']
        )
        total_mape = df['mape'].mean() * 100
        
        df['eom'] = pd.to_datetime(df.ds) + pd.offsets.MonthEnd(0)
        
        monthly_mape = df.pivot_table(
            index='eom',
            values='mape',
            aggfunc=('mean')
        ).reset_index().sort_values('eom')
        monthly_mape['mape'] =  monthly_mape['mape'] * 100
        monthly_mape = monthly_mape.tail(24)
        
        return total_mape, monthly_mape
        
    def html_table(dff:pd.DataFrame):
        df = dff.copy()
        df['eom'] = pd.to_datetime(df['ds']) + pd.offsets.MonthEnd(0)
        df['Год'] = df['eom'].dt.year.astype(str)
        df['moonth_id'] = df['eom'].dt.month
        df['Месяц'] = df['eom'].dt.strftime('%b').str.capitalize()
        df = df.pivot_table(
            index=['moonth_id','Месяц'],
            columns=['Год','type'],
            values='y',
            aggfunc='sum'
        ).reset_index().sort_values('moonth_id')
        df = df.drop(columns='moonth_id')
        df = df.set_index('Месяц')
        df.loc['Итого'] = df.select_dtypes('number').sum()
        df.columns.names = [None, None]
        df.index.names = [None]
        

        ss = df.columns
        
       
        html_table = (df.style
             .format('{:,.0f}',subset=ss,na_rep='-',thousands='\u202F',)
             .set_table_attributes('class="forecast-table" ')
             .set_caption("Результаты планирования")
            #  .hide(axis='index')
        ).to_html()
       
        
        
        return dmc.ScrollArea(
            [
                dcc.Markdown(
                    [
                        html_table
                    ],
                    dangerously_allow_html=True
                )
            ]
        )
        
        
    
    
    actuals = historical_data()
    actuals['type'] = 'Факт'
    
    plan = forecast[['ds','yhat']].copy()
    plan.rename(columns={'yhat': 'y'}, inplace=True)
    plan['type'] = 'План'
    
    df = pd.concat([actuals,plan])
    total_mape, mothly_mape = mape(df)
    
    df['ds'] = pd.to_datetime(df['ds']).dt.normalize()
    cur_date = pd.to_datetime(current_date).normalize()
    ad_plan = plan[plan['ds']>=cur_date]
    dff = pd.concat([actuals,ad_plan])
    
    
    
    return df, yearly_seasons(), total_mape, html_table(dff)
    
    
    
    
