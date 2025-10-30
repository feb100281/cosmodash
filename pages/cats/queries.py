import numpy as np
import pandas as pd
from data import ENGINE
import dash_mantine_components as dmc
from components import COLORS_BY_SHADE, COLORS_BY_COLOR

VALS_DICT = {
    'amount':'Выручка',
    'dt':'Продажи',
    'cr':'Возвраты',
    'quant':'Количество'
}

OPTIONS_SWITCHS = {
    'cat':"Показать категории",
    'store_gr_name':'Показать магазины'
}

def fletch_cats():
    q = """
    with parent as (
    select
    id as parent_id,
    name as parent
    from corporate_cattree
    where parent_id is Null
    )

    select 
    p.parent_id,
    p.parent,
    cat.id as cat_id,
    cat.name as cat,
    COALESCE(sc.id,0) as subcat_id,
    COALESCE(sc.name,'Нет подкатегории') as subcat
    from corporate_cattree as cat

    left join parent as p on p.parent_id = cat.parent_id
    left join corporate_subcategory as sc on sc.category_id = cat.id
    where cat.parent_id is not Null       
    """
    return pd.read_sql(q,ENGINE)


def get_df(start, end):
    q = f"""
    select
    date,
    sum(s.dt) as dt,
    sum(s.cr) as cr,
    sum(s.dt - s.cr) as amount,
    parent.id as parent_id,
    parent.name as parent,    
    cat.id as cat_id,
    cat.name as cat,
    coalesce(subcat.id,0) as subcat_id,
    coalesce(subcat.name,'Нет подкатегории') as subcat

    from sales_salesdata as s
    left join corporate_items as i on i.id = s.item_id
    left join corporate_cattree as cat on cat.id = i.cat_id
    left join corporate_cattree as parent on parent.id = cat.parent_id
    left join corporate_subcategory as subcat on subcat.id = i.subcat_id
    where date between '{start}' and '{end}'

    GROUP BY date, parent_id, parent, cat_id, cat, subcat_id, subcat;
    """
    return pd.read_sql(q,ENGINE)


def make_summary_df(start, end):
    q = f"""
    select
    date,
    sum(s.dt) as dt,
    sum(s.cr) as cr,
    sum(s.dt - s.cr) as amount,
    sum(s.quant_dt - s.quant_cr) as quant,
    parent.id as parent_id,
    parent.name as parent,    
    cat.id as cat_id,
    cat.name as cat,
	sg.name as store_gr_name
    

    from sales_salesdata as s
    left join corporate_items as i on i.id = s.item_id
    left join corporate_cattree as cat on cat.id = i.cat_id
    left join corporate_cattree as parent on parent.id = cat.parent_id
    left join corporate_stores as st on st.id = s.store_id
    left join corporate_storegroups as sg on sg.id = st.gr_id 
    where date between '{start}' and '{end}'

    GROUP BY date, parent_id, cat_id, store_gr_name
        
    """
    return pd.read_sql(q,ENGINE)




def cats_report(start, end, option = 'cat', val = 'amount'):
    end_start = end[:-2] + "01"
    df_current = make_summary_df(end_start,end)
    df_current['tp'] = 'current'
    df_current_pars = df_current.pivot_table(index='parent',values=val,aggfunc='sum').reset_index().sort_values(val,ascending=False)
    parent_list = df_current_pars['parent'].unique().tolist()
    end_dt = pd.to_datetime(end)
    start_dt = pd.to_datetime(start)
    start_end = start_dt + pd.offsets.MonthEnd(0)
    start_end = start_end.strftime('%Y-%m-%d')
    df_reff = make_summary_df(start, start_end)
    df_reff['tp'] = 'areff'    
    df = pd.concat([df_current, df_reff])  
    
    tabs = []
    i = 0
    
    for parent in parent_list:
        dff:pd.DataFrame = df[df['parent']==parent]
        i += 1
        dff = dff.pivot_table(
            index = option,
            columns = ['tp'],
            values=val,
            aggfunc='sum',            
        ).reset_index().sort_values('current', ascending=False).fillna(0)
        if 'areff' not in dff.columns:
            dff['areff'] = 0
        if 'current' not in dff.columns:
            dff['current'] = 0
        dff['var'] = dff['current'] - dff['areff']
        
        data = []        
        init_value = dff['areff'].sum()
        end_value = dff['current'].sum()
        data.append({"item": f"{VALS_DICT[val]} {start_dt.strftime('%b %Y')}", f"{VALS_DICT[val]}": init_value, "color": 'grape'})
        
        for _, r in dff.iterrows():
            c = 'red' if r['var'] < 0 else 'green'
            data.append({"item": '𝜟 ' + r[option], f"{VALS_DICT[val]}": r['var'], "color": c})
            
        data.append({"item": f"{VALS_DICT[val]} за {end_dt.strftime('%b %Y')}", f"{VALS_DICT[val]}": end_value, "color": 'grape', "standalone": True})
        
        vf = {"function": "formatNumberIntl"}
        if val == 'quant':
            vf =  {"function": "formatIntl"}
        
        tab = dmc.TabsPanel(
            [
                dmc.Container(
                    [   
                        dmc.Space(h=10),
                        dmc.Switch(
                            id={'type':'val_switch', 'index':i},
                            labelPosition="right",
                            label="Показать значения",
                            size="sm",
                            radius="lg",
                            color="#5c7cfa",
                            disabled=False,
                            withThumbIndicator=True,
                            
                            ),
                        dmc.Space(h=10),
                        dmc.BarChart(
                            h=350,
                            data=data,
                            dataKey="item",
                            type="waterfall",
                            series=[{"name": f"{VALS_DICT[val]}", "color": COLORS_BY_SHADE[0]}],
                            withLegend=True,
                            valueFormatter=vf,
                            withBarValueLabel=False,
                            xAxisProps={
                                "interval": 0,       # ← показывать все подписи
                                "angle": -30,        # ← (опционально) наклонить подписи, чтобы влезали
                                "textAnchor": "end"  # ← выровнять подписи к концу
                            
                            },
                            id={'type':'cat_chart', 'index':i},
                           
                        )
                    ],
                    fluid=True
                )
            ],
            value=f"{parent}"
        )
        tabs.append(tab)

    tab_list = [
    dmc.TabsList(
        [
            dmc.TabsTab(parent, value=f"{parent}") 
            for parent in parent_list
        ]
    )
    ] + tabs

    container = dmc.Container(
        children=[        
            dmc.Tabs(
                tab_list,
                value=parent_list[0],
                orientation="horizontal"
            )
        ],
        fluid=True
    )
        
    return container
    
    
    
    


