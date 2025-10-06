from reporting.report_generator import (
    ReportGenerator,
    MarkdownBlock,
    DataTable,
    Icon,
    BS,
    HtmlRender,
)
import pandas as pd
import numpy as np
from data import load_df_from_redis
import locale

locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")


def pdf_data_click(df_id=None):

    pd.options.display.float_format = "{:,.2f}".format

    df = pd.DataFrame()
    if df_id:
        df = load_df_from_redis(df_id)

    # Колонки в int для расчета в копейках
    int_cols = ["dt", "cr", "amount", "quant_dt", "quant_cr", "quant"]
    for col in int_cols:
        df[col] = df[col] * 100
        df[col] = df[col].astype(int)

    date_start = pd.to_datetime(df["date"].min())
    date_start = date_start.strftime("%-d %B %Y")

    date_finish = pd.to_datetime(df["date"].max())
    last_month = date_finish.month
    last_day = date_finish.day
    last_year = date_finish.year
    date_finish = date_finish.strftime("%-d %B %Y")

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = pd.to_datetime(df.eom).dt.year
    df["month"] = pd.to_datetime(df.eom).dt.strftime("%b").str.capitalize()

    df["month_sn"] = pd.to_datetime(df.eom).dt.month
    df["day_sn"] = pd.to_datetime(df.date).dt.day

    df = df[
        (df["month_sn"] < last_month)
        | ((df["month_sn"] == last_month) & (df["day_sn"] <= last_day))
    ]
    df["month"] = np.where(
        df["month_sn"] == last_month,
        df["month"]
        + "\u00A0"
        + BS.Badge.badge_rounded_warning.text(f" за {last_day} дн", size="xs"),
        df["month"],
    )

    month_sales: pd.DataFrame = df.pivot_table(
        index=["month", "month_sn"],
        columns="year",
        values="amount",
        aggfunc=["sum"],
        fill_value=0,
    )
    month_sales = month_sales / 100_000_000
    month_sales = (
        month_sales.reset_index().sort_values("month_sn").drop(columns="month_sn")
    )
    month_sales.columns = month_sales.columns.droplevel(0)

    cols = month_sales.columns
    html_table = ""

    if len(cols) < 3:
        last_col = cols[1]
        col_name = (
            str(last_col) + "\u00A0" + BS.Badge.badge_primary.text("нак", "top", "xs")
        )
        month_sales[col_name] = month_sales[last_col].cumsum()
        tot = month_sales[last_col].sum()
        month_sales["%"] = round(month_sales[last_col] / tot * 100, 1)
        max_percent = int(month_sales["%"].max())
        month_sales["%"] = month_sales["%"].apply(
            lambda x: BS.ProgressBar(
                "text-bg-info", "", x, value_max=max_percent
            ).render
        )
        html_table = (month_sales.style
                      .format({1:'{:.2f}'})
                      
                      
                      ).to_html()

    elif len(cols) >= 3:
        cum_cols = cols[-2:]
        cum_cols1_name = (
            str(cum_cols[0])
            + "\u00A0"
            + BS.Badge.badge_primary.text("нак", "top", "xs")
        )
        cum_cols2_name = (
            str(cum_cols[1])
            + "\u00A0"
            + BS.Badge.badge_primary.text("нак", "top", "xs")
        )
        month_sales[cum_cols1_name] = month_sales[cum_cols[0]].cumsum()
        month_sales[cum_cols2_name] = month_sales[cum_cols[1]].cumsum()
        month_sales["Откл"] = np.where(
            month_sales[cum_cols1_name] == 0,
            0,
            (month_sales[cum_cols2_name] - month_sales[cum_cols1_name])
            / month_sales[cum_cols1_name],
        )
        month_sales["Откл"] = round(month_sales["Откл"] * 100, 1)
        max_abs = month_sales["Откл"].abs().max()
        min_val = -max_abs
        max_val = max_abs

        month_sales["Откл"] = month_sales["Откл"].apply(
            lambda x: BS.ProgressBarRelative(
                value=x,
                value_min=min_val,
                value_max=max_val,
                color_pos="bg-success",
                color_neg="bg-danger",
            ).render
        )
        html_table = (month_sales.style
                      .format({
                          2024:'{:.2f}',
                          2025:'{:.2f}',
                               })
                      
                     
                      
                      
                      ).hide(axis="index").to_html()

    table = month_sales.to_markdown(index=False)

    for col in month_sales.select_dtypes(include="float"):
        month_sales[col] = month_sales[col].map(lambda x: f"{x:.2f}")

    # Делаем класс ReportGenerator
    dnl_content = ReportGenerator(
        title="Отчет по динамики продаж",  # Титул отчета
        bootswatch_theme="yeti",  # Исходная тема
        fontsize="md",  # Исходные размер шрифта
    )

    md = f"""
## Отчет по динамики продаж
#### за период с {date_start} по {date_finish}

---


## Резюме:

За рассматриваемый период были получены следующие показатели:

- {Icon.Streamline.pile_of_money_duo.render()} Чистая выручка за период - {df.amount.sum()/100_000_000:,.0f} млн рублей;

- {Icon.Streamline.receipt_dollar.render()} Продажи - {df.dt.sum()/100_000_000:,.0f} млн рублей;

- {Icon.Streamline.business_deal_cash_2.render()} Возвраты - {df.cr.sum()/100_000_000:,.0f} млн рублей;

- {Icon.Streamline.dangerous_chemical_lab.render()} Коэффициент возвратов - {(df.cr.sum()/df.dt.sum())*100:,.1f}%.

### Распределение выручки по месяцам *(млн руб)*:
{table}



"""
    par1 = MarkdownBlock(md)
    tabl = HtmlRender(html_table)
    dnl_content.add_component(par1)
    dnl_content.add_component(tabl)

    return dnl_content
