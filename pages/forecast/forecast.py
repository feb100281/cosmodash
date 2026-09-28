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


class ForecastInputError(ValueError):
    """Некорректные входные параметры прогноза — показываются пользователю."""


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
    
    
    today = pd.Timestamp.now().normalize()
    current_date = today if not current_date else pd.to_datetime(current_date).normalize()
    horizon_dt = pd.to_datetime(horizon).normalize()
    if current_date > today:
        raise ForecastInputError(
            f"«Текущая дата» ({current_date:%d.%m.%Y}) позже сегодняшнего дня. "
            f"Дату, до которой нужен план, укажите в поле «Дата горизонта планирования», "
            f"а «Текущую дату» оставьте пустой."
        )
    if horizon_dt <= current_date:
        raise ForecastInputError(
            f"Дата горизонта ({horizon_dt:%d.%m.%Y}) должна быть позже текущей даты "
            f"({current_date:%d.%m.%Y}) — иначе планировать нечего."
        )
    end = current_date.strftime('%Y-%m-%d')
    
    historical_cut_off = pd.to_datetime(historical_cut_off).normalize() if historical_cut_off else None
    start = historical_cut_off.strftime('%Y-%m-%d') if historical_cut_off else None
    
    data = historical_data(start=start,end=end)
    if data.empty:
        raise ForecastInputError("Нет продаж за выбранный исторический период.")
    data["ds"] = pd.to_datetime(data["ds"]).dt.normalize()

    horizon = pd.to_datetime(horizon).normalize()
    last_ds = data["ds"].max()
    if horizon <= last_ds:
        raise ForecastInputError(
            f"Горизонт планирования ({horizon:%d.%m.%Y}) должен быть позже последней даты "
            f"продаж в расчёте ({last_ds:%d.%m.%Y})."
        )
    
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
        
    # периоды считаются от последней даты истории, а не от текущей даты
    num_days = int((horizon - last_ds).days)
    future = model.make_future_dataframe(periods=num_days)
        
    forecast = model.predict(future)
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        forecast[col] = forecast[col].clip(lower=0)
    
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
    
    def mape(dff: pd.DataFrame):
        """
        Ошибка модели по месяцам истории: сумма |факт − план| / сумма факта
        (только полностью прошедшие месяцы). Дневная ошибка на продажах
        с «нулевыми» днями неинформативна.
        """
        d = dff.copy()
        d["ds"] = pd.to_datetime(d["ds"]).dt.normalize()
        d = d[(d["ds"] >= data["ds"].min()) & (d["ds"] <= last_ds)]
        d["eom"] = d["ds"] + pd.offsets.MonthEnd(0)
        m = d.pivot_table(index="eom", columns="type", values="y", aggfunc="sum").fillna(0)
        m = m[m.index <= last_ds]
        if m.empty or "Факт" not in m or "План" not in m or m["Факт"].sum() == 0:
            return float("nan"), pd.DataFrame(columns=["eom", "mape"])
        err = (m["Факт"] - m["План"]).abs()
        total = float(err.sum() / m["Факт"].abs().sum() * 100)
        monthly = (err / m["Факт"].replace(0, np.nan) * 100).rename("mape").reset_index().tail(24)
        return total, monthly

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
    ad_plan = plan[pd.to_datetime(plan['ds']) > last_ds]
    dff = pd.concat([actuals,ad_plan])
    
    
    
    export = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    export["ds"] = pd.to_datetime(export["ds"]).dt.normalize()
    fact = actuals.copy()
    fact["ds"] = pd.to_datetime(fact["ds"]).dt.normalize()
    export = export.merge(fact[["ds", "y"]].rename(columns={"y": "fact"}), on="ds", how="outer").sort_values("ds")
    export = export[export["ds"] <= horizon]
    meta = {
        "current_date": cur_date.strftime("%Y-%m-%d"),
        "last_fact_date": last_ds.strftime("%Y-%m-%d"),
        "horizon": horizon.strftime("%Y-%m-%d"),
        "history_start": data["ds"].min().strftime("%Y-%m-%d"),
        "mape": float(total_mape) if pd.notna(total_mape) else None,
        "params": {
            "Годовая сезонность": "да" if yearly_seasonality else "нет",
            "Недельная сезонность": "да" if weekly_seasonality else "нет",
            "Режим сезонности": seasonality_mode,
            "Чувствительность к изменениям тренда": changepoint_prior_scale,
            "Доля истории для поиска изломов": changepoint_range,
            "Максимум изломов тренда": n_changepoints,
        },
    }

    return df, yearly_seasons(), total_mape, html_table(dff), {"frame": export, "meta": meta}
    
    
    
    
