import pandas as pd
from data import ENGINE
import locale
import plotly.express as px
import plotly.graph_objects as go



locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")

def get_days_heatmap(start = '2024-10-01', end='2025-10-31', store=None):
    
    start = pd.to_datetime(start) + pd.offsets.MonthBegin(-1)
    start = start.strftime('%Y-%m-%d')
       
    title = 'Все магазины'
    stores = ''
    if store:
       st = ','.join(f"'{s}'" for s in store)
       stores = f'and sg.name in ({st})' 
       title = ', '.join(store)
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
    
    df = df.pivot_table(
        index='day',
        columns='eom',
        values='amount',
        aggfunc='sum'
    ).fillna(0)
    z = df.to_numpy()
    x = [d.strftime("%b %y") for d in df.columns]    
    y = df.index.to_list()
    fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=x,
                y=y,
                coloraxis="coloraxis",
                text = z,
                texttemplate="%{text:,.0f}",
                    textfont={"size":8}))
                
      
    fig.update_xaxes(side="top")
    fig.update_yaxes(tickmode='linear', dtick=1, autorange='reversed')
    fig.update_layout(
            height=900,  
            margin=dict(l=60, r=20, t=60, b=60),
            coloraxis=dict(
                colorscale="Blues",
            
            ),
            
        )
    
    
    return fig, title

