import pandas as pd
from data import ENGINE
import locale
import plotly.express as px
import plotly.graph_objects as go



locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")

RU_WDAYS = ['Понедельник','Вторник','Среда','Четверг','Пятница','Суббота','Воскресенье']
ru_wday_type = pd.api.types.CategoricalDtype(categories=RU_WDAYS, ordered=True)

def get_days_heatmap(start = '2024-10-01', end='2025-10-31', store=None, is_dark=False, weekdays=False):
    
    start = pd.to_datetime(start) + pd.offsets.MonthBegin(-1)
    start = start.strftime('%Y-%m-%d')
    end_dt = pd.to_datetime(end)
       


    # магазин(ы) в заголовок
    if store:
        store_title = ', '.join(store)
        stores_clause = ','.join(f"'{s}'" for s in store)
        stores = f'and sg.name in ({stores_clause})'
    else:
        store_title = 'Все магазины'
        stores = ''

    # текст агрегации
    agg_title = "Средняя выручка по дням недели" if weekdays else "Выручка по дням месяца"
    mode_hint = "(среднее)" if weekdays else "(сумма)"

    # итоговый title (то, что пойдёт в бейдж)
    title = f"{store_title} • {agg_title} {mode_hint}"

    q = f"""
    SELECT 
        s.date,
        sum(s.dt - s.cr) as amount,
        sg.name as store_gr_name

        FROM sales_salesdata as s
        left join corporate_stores as st on st.id = s.store_id
        left join corporate_storegroups as sg on sg.id = st.gr_id

        where date between '{start}' and '{end}'
        {stores}
        group by date, sg.name
    """
    
    df = pd.read_sql(q,ENGINE)
    df['eom'] = pd.to_datetime(df['date']) + pd.offsets.MonthEnd(0)
    df['eom'] = df['eom'].dt.normalize()    
    df['day'] = pd.to_datetime(df['date']).dt.day
    
    # df['wday_num'] = pd.to_datetime(df['date']).dt.weekday  # Пн=0 ... Вс=6
    # df['wday'] = df['wday_num'].map(dict(enumerate(RU_WDAYS))).astype(ru_wday_type)
    df['wday'] = pd.to_datetime(df['date']).dt.day_name('ru_RU.UTF-8')

    # df['wday'] = pd.to_datetime(df['date']).dt.day_name()
    
    df:pd.DataFrame = (
        df.groupby(['day', 'wday', 'eom'], as_index=False, observed=False)['amount']
        .sum()
    )
    
    if weekdays:       
       df = df.pivot_table(
           index='wday',
           columns = 'eom',
           values='amount',
           aggfunc='mean',
           observed=False            
       ).fillna(0).reindex(RU_WDAYS)
    else:
      df = df.pivot_table(
           index='day',
           columns = 'eom',
           values='amount',
           aggfunc='sum',
        observed=False           
       ).fillna(0)   
    
    z = df.to_numpy()
    x = [d.strftime("%b %y") for d in df.columns]    
    y = df.index.to_list()
    
    # 🎨 Цветовая схема в зависимости от темы
    if is_dark:
        bg_color = '#1e1e1e'
        text_color = '#ffffff'
        grid_color = '#333333'
        colorscale = 'Cividis'
        template = 'plotly_dark'
    else:
        bg_color = '#ffffff'
        text_color = '#000000'
        grid_color = '#e0e0e0'
        colorscale = 'Blues'
        template = 'plotly_white'
        
        
    fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=x,
                y=y,
                coloraxis="coloraxis",
                text = z,
                texttemplate="%{text:,.0f}",
                    textfont={"size":8}))
                
      
    fig.update_xaxes(side="top", gridcolor=grid_color)
    fig.update_yaxes(tickmode='linear', dtick=1, autorange='reversed', gridcolor=grid_color)
    fig.update_layout(
            template=template, 
            height=900,  
            margin=dict(l=60, r=20, t=60, b=60),
            coloraxis=dict(
            colorscale=colorscale,
            colorbar=dict(
                tickcolor=text_color,
        
                tickfont=dict(color=text_color),
            ),
        ),
             plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color),
            
            
            
        )
    
    
    return fig, title

