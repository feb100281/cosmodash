from reporting.report_generator import (
    ReportGenerator,
    MarkdownBlock,
    DataTable,
    PlotlyFigure,
)
import pandas as pd
from data import load_df_from_redis


def big_button_click(df_id=None):

    df = pd.DataFrame()
    if df_id:
        df = load_df_from_redis(df_id)

        df = df.pivot_table(
            index=["chanel", "store_gr_name"],
            columns="eom",
            values="amount",
            aggfunc="sum",
        ).fillna(0)

    df1 = round(df / 1_000, 2)

    md = """

# Отчет для пробы

## Подзаголовки

Здесь будем давать комментарии. А, теперь понятно 😅 — проблема в том, что @bottom-center не может выходить за пределы @page, и margin-bottom просто сдвигает контент внутри доступной области, но нижняя граница страницы остаётся нулевой (0cm). То есть нельзя просто "поднять" блок за пределы нижнего края через margin.

### Деалем список

- Нижний отступ страницы  создаёт место, куда @bottom-center может вставиться.
- *Впримере 2cm* - **это расстояние** от нижнего края страницы до номера

"""
    par1 = MarkdownBlock(md,font_size='14px')
    table1 = DataTable(df=df1)
    
    return ReportGenerator().add_component(par1,table1)
    
