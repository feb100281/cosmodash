from reporting.report_generator import (
    ReportGenerator,
    MarkdownBlock,
    DataTable,
    Icon,
    BS
)
import pandas as pd
from data import load_df_from_redis
import locale
locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")



def pdf_data_click(df_id=None):

    df = pd.DataFrame()
    if df_id:
        df = load_df_from_redis(df_id)

    
    date_start = pd.to_datetime(df['date'].min())
    date_start = date_start.strftime('%-d %B %Y')
    
    date_finish = pd.to_datetime(df['date'].max())
    date_finish = date_finish.strftime('%-d %B %Y')
    
    dnl_content = ReportGenerator(
        title='Отчет по динамики продаж',
        bootswatch_theme='yeti',
        fontsize='md'
    )
    
    stores_sales = df.pivot_table(
        index = ['chanel','store_gr_name'],
        values=['amount'],
        aggfunc='sum'
    )
    
    md = f"""
## Отчет по динамики продаж
#### за период с {date_start} по {date_finish}

### Резюме:

За рассматриваемый период были получены следующие показатели:

{Icon.Streamline.business_deal_cash_2.render()} Чистая выручка за период - {df.amount.sum()/1_000_000:,.0f} млн рублей;

{Icon.Streamline.pile_of_money_duo.render()} Продажи - {df.dt.sum()/1_000_000:,.0f} млн рублей;

{Icon.Streamline.backpack.render()} Возвраты - {df.cr.sum()/1_000_000:,.0f} млн рублей;

{Icon.Streamline.dangerous_chemical_lab.render()} Коэффициент возвратов - {(df.dt.sum()/df.cr.sum())*100:,.1f}%.

### Распределение выручки по магащинам
{stores_sales.to_markdown()}



"""
    par1 = MarkdownBlock(md)
    dnl_content.add_component(par1)
    
    return dnl_content
    
