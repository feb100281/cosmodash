# from reporting.report_generator import (
#     ReportGenerator,
#     MarkdownBlock,
#     Icon,
#     BS,
#     HtmlRender,
# )
# import pandas as pd
# import numpy as np
# from data import load_df_from_redis
# import locale

# locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")


# def pdf_data_click(df_id=None):
#     pd.options.display.float_format = "{:,.2f}".format

#     # 1) Загружаем данные
#     if not df_id:
#         # Если df_id нет — лучше вернуть пустой отчет, чем падать
#         dnl_content = ReportGenerator(
#             title="Отчет по динамики продаж",
#             bootswatch_theme="yeti",
#             fontsize="md",
#         )
#         dnl_content.add_component(
#             MarkdownBlock("## Отчет по динамики продаж\n\nДанные не выбраны (df_id отсутствует).")
#         )
#         return dnl_content

#     df = load_df_from_redis(df_id)
#     if df is None or len(df) == 0:
#         dnl_content = ReportGenerator(
#             title="Отчет по динамики продаж",
#             bootswatch_theme="yeti",
#             fontsize="md",
#         )
#         dnl_content.add_component(
#             MarkdownBlock("## Отчет по динамики продаж\n\nДанные пустые (в Redis ничего не найдено).")
#         )
#         return dnl_content

#     df = df.copy()

#     # 2) Проверим нужные колонки
#     required = {"date", "eom", "dt", "cr", "amount", "quant_dt", "quant_cr", "quant"}
#     missing = required - set(df.columns)
#     if missing:
#         raise KeyError(f"В df отсутствуют колонки: {sorted(missing)}")

#     # 3) Приводим даты
#     df["date"] = pd.to_datetime(df["date"], errors="coerce")
#     df["eom"] = pd.to_datetime(df["eom"], errors="coerce")
#     df = df.dropna(subset=["date", "eom"])

#     # 4) Колонки в int для расчета в копейках (делаем безопаснее)
#     int_cols = ["dt", "cr", "amount", "quant_dt", "quant_cr", "quant"]
#     for col in int_cols:
#         df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
#         df[col] = (df[col] * 100).round(0).astype("int64")

#     # 5) Диапазон дат
#     date_start_dt = df["date"].min()
#     date_finish_dt = df["date"].max()

#     date_start = pd.to_datetime(date_start_dt).strftime("%-d %B %Y")
#     date_finish = pd.to_datetime(date_finish_dt).strftime("%-d %B %Y")

#     last_month = int(date_finish_dt.month)
#     last_day = int(date_finish_dt.day)

#     # 6) Месяц/год/номера для фильтра и витрины
#     df["year"] = df["eom"].dt.year
#     df["month"] = df["eom"].dt.strftime("%b").str.capitalize()
#     df["month_sn"] = df["eom"].dt.month
#     df["day_sn"] = df["date"].dt.day

#     # 7) Обрезаем последний месяц до last_day (как у тебя)
#     df = df[
#         (df["month_sn"] < last_month)
#         | ((df["month_sn"] == last_month) & (df["day_sn"] <= last_day))
#     ].copy()

#     # 8) Подпись "за N дн" для последнего месяца
#     df["month"] = np.where(
#         df["month_sn"] == last_month,
#         df["month"] + "\u00A0" + BS.Badge.badge_rounded_warning.text(f" за {last_day} дн", size="xs"),
#         df["month"],
#     )

#     # 9) Pivot по месяцам
#     month_sales: pd.DataFrame = df.pivot_table(
#         index=["month", "month_sn"],
#         columns="year",
#         values="amount",
#         aggfunc="sum",
#         fill_value=0,
#     )

#     # В млн руб
#     month_sales = month_sales / 100_000_000
#     month_sales = month_sales.reset_index().sort_values("month_sn").drop(columns="month_sn")

#     # HTML-таблица (со стилями/прогресс-барами)
#     cols = [c for c in month_sales.columns if c != "month"]
#     html_table = ""

#     # Если есть хотя бы один год
#     if len(cols) >= 1:
#         # Если только 1 год данных (или "меньше 3 колонок" по твоей логике)
#         if len(cols) < 3 and len(cols) >= 2:
#             last_col = cols[1]  # у тебя это было так; предполагается, что cols = ['month', 2024]?? осторожно
#         # Я сделаю более корректно: последний год — самый правый числовой столбец
#         year_cols = [c for c in month_sales.columns if isinstance(c, (int, np.integer))]
#         year_cols = sorted(year_cols)
#         if len(year_cols) == 0:
#             html_table = month_sales.style.hide(axis="index").to_html()
#         elif len(year_cols) == 1:
#             y = year_cols[0]
#             col_name = str(y) + "\u00A0" + BS.Badge.badge_primary.text("нак", "top", "xs")
#             month_sales[col_name] = month_sales[y].cumsum()

#             tot = float(month_sales[y].sum()) if float(month_sales[y].sum()) != 0 else 1.0
#             month_sales["%"] = (month_sales[y] / tot * 100).round(1)

#             max_percent = float(month_sales["%"].max()) if float(month_sales["%"].max()) != 0 else 1.0
#             month_sales["%"] = month_sales["%"].apply(
#                 lambda x: BS.ProgressBar("text-bg-info", "", float(x), value_max=float(max_percent)).render
#             )

#             html_table = (
#                 month_sales.style
#                 .format({y: "{:.2f}", col_name: "{:.2f}"})
#                 .hide(axis="index")
#                 .to_html()
#             )
#         else:
#             # Берём два последних года
#             y1, y2 = year_cols[-2], year_cols[-1]

#             cum1_name = str(y1) + "\u00A0" + BS.Badge.badge_primary.text("нак", "top", "xs")
#             cum2_name = str(y2) + "\u00A0" + BS.Badge.badge_primary.text("нак", "top", "xs")

#             month_sales[cum1_name] = month_sales[y1].cumsum()
#             month_sales[cum2_name] = month_sales[y2].cumsum()

#             month_sales["Откл"] = np.where(
#                 month_sales[cum1_name] == 0,
#                 0,
#                 (month_sales[cum2_name] - month_sales[cum1_name]) / month_sales[cum1_name],
#             )
#             month_sales["Откл"] = (month_sales["Откл"] * 100).round(1)

#             max_abs = float(month_sales["Откл"].abs().max())
#             max_abs = max_abs if max_abs != 0 else 1.0

#             month_sales["Откл"] = month_sales["Откл"].apply(
#                 lambda x: BS.ProgressBarRelative(
#                     value=float(x),
#                     value_min=-max_abs,
#                     value_max=max_abs,
#                     color_pos="bg-success",
#                     color_neg="bg-danger",
#                 ).render
#             )

#             html_table = (
#                 month_sales.style
#                 .format({y1: "{:.2f}", y2: "{:.2f}", cum1_name: "{:.2f}", cum2_name: "{:.2f}"})
#                 .hide(axis="index")
#                 .to_html()
#             )

#     # Markdown-таблица (без HTML-рендеров)
#     table_md = month_sales.copy()
#     # В markdown лучше не тащить HTML из "Откл", поэтому сформируем до подмены/или исключим
#     if "Откл" in table_md.columns:
#         # там уже HTML от ProgressBarRelative; для markdown уберём
#         table_md["Откл"] = table_md["Откл"].astype(str).str.replace(r"<.*?>", "", regex=True)

#     table = table_md.to_markdown(index=False)

#     # 10) Отчет
#     dnl_content = ReportGenerator(
#         title="Отчет по динамики продаж",
#         bootswatch_theme="yeti",
#         fontsize="md",
#     )

#     # Защита от деления на 0
#     dt_sum = df["dt"].sum()
#     cr_sum = df["cr"].sum()
#     amount_sum = df["amount"].sum()
#     return_rate = (cr_sum / dt_sum * 100) if dt_sum else 0

#     md = f"""
# ## Отчет по динамики продаж
# #### за период с {date_start} по {date_finish}

# ---

# ## Резюме:

# За рассматриваемый период были получены следующие показатели:

# - {Icon.Streamline.pile_of_money_duo.render()} Чистая выручка за период - {amount_sum/100_000_000:,.0f} млн рублей;
# - {Icon.Streamline.receipt_dollar.render()} Продажи - {dt_sum/100_000_000:,.0f} млн рублей;
# - {Icon.Streamline.business_deal_cash_2.render()} Возвраты - {cr_sum/100_000_000:,.0f} млн рублей;
# - {Icon.Streamline.dangerous_chemical_lab.render()} Коэффициент возвратов - {return_rate:,.1f}%.

# ### Распределение выручки по месяцам *(млн руб)*:
# {table}
# """

#     dnl_content.add_component(MarkdownBlock(md))

#     # html_table может быть пустым — тогда просто не добавляем
#     if html_table:
#         dnl_content.add_component(HtmlRender(html_table))

#     return dnl_content



from reporting.report_generator import (
    ReportGenerator,
    MarkdownBlock,
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

    # --- загрузка ---
    if not df_id:
        dnl_content = ReportGenerator(
            title="Отчет по динамики продаж",
            bootswatch_theme="yeti",
            fontsize="md",
        )
        dnl_content.add_component(
            MarkdownBlock("## Отчет по динамики продаж\n\nДанные не выбраны (df_id отсутствует).")
        )
        return dnl_content

    df = load_df_from_redis(df_id)
    if df is None or len(df) == 0:
        dnl_content = ReportGenerator(
            title="Отчет по динамики продаж",
            bootswatch_theme="yeti",
            fontsize="md",
        )
        dnl_content.add_component(
            MarkdownBlock("## Отчет по динамики продаж\n\nДанные пустые.")
        )
        return dnl_content

    df = df.copy()

    # --- обязательные колонки ---
    required = {"date", "eom", "dt", "cr", "amount", "quant_dt", "quant_cr", "quant"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"В df отсутствуют колонки: {sorted(missing)}")

    # --- новые колонки (опционально) ---
    dims = ["manager_name", "manu", "store"]
    for c in dims:
        if c not in df.columns:
            df[c] = "—"  # чтобы отчёт не падал и можно было группировать

    # --- даты ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["eom"] = pd.to_datetime(df["eom"], errors="coerce")
    df = df.dropna(subset=["date", "eom"])

    # --- копейки (int) ---
    int_cols = ["dt", "cr", "amount", "quant_dt", "quant_cr", "quant"]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[col] = (df[col] * 100).round(0).astype("int64")

    # --- диапазон ---
    date_start_dt = df["date"].min()
    date_finish_dt = df["date"].max()

    date_start = date_start_dt.strftime("%-d %B %Y")
    date_finish = date_finish_dt.strftime("%-d %B %Y")

    last_month = int(date_finish_dt.month)
    last_day = int(date_finish_dt.day)

    # --- служебные поля ---
    df["year"] = df["eom"].dt.year
    df["month"] = df["eom"].dt.strftime("%b").str.capitalize()
    df["month_sn"] = df["eom"].dt.month
    df["day_sn"] = df["date"].dt.day

    # обрезаем последний месяц до last_day
    df = df[
        (df["month_sn"] < last_month)
        | ((df["month_sn"] == last_month) & (df["day_sn"] <= last_day))
    ].copy()

    # бейдж "за N дн" на последнем месяце
    df["month"] = np.where(
        df["month_sn"] == last_month,
        df["month"] + "\u00A0" + BS.Badge.badge_rounded_warning.text(f" за {last_day} дн", size="xs"),
        df["month"],
    )

    # --- KPI по периоду ---
    dt_sum = int(df["dt"].sum())
    cr_sum = int(df["cr"].sum())
    amount_sum = int(df["amount"].sum())

    dt_mln = dt_sum / 100_000_000
    cr_mln = cr_sum / 100_000_000
    amount_mln = amount_sum / 100_000_000

    # дни в периоде (по календарю)
    days_total = (date_finish_dt.normalize() - date_start_dt.normalize()).days + 1
    days_total = max(days_total, 1)

    return_rate = (cr_sum / dt_sum * 100) if dt_sum else 0
    avg_day_mln = amount_sum / 100_000_000 / days_total

    # --- helper: топ-таблица с долями ---
    def make_top_table(df_in: pd.DataFrame, group_col: str, value_col: str = "amount", top_n: int = 5) -> str:
        tmp = (
            df_in.groupby(group_col, as_index=False)[value_col]
            .sum()
            .sort_values(value_col, ascending=False)
            .head(top_n)
        )
        total = float(df_in[value_col].sum()) if float(df_in[value_col].sum()) != 0 else 1.0
        tmp["Доля, %"] = (tmp[value_col] / total * 100).round(1)

        # в млн руб
        tmp["Выручка, млн"] = (tmp[value_col] / 100_000_000).round(2)

        max_percent = float(tmp["Доля, %"].max()) if float(tmp["Доля, %"].max()) != 0 else 1.0
        tmp["Доля"] = tmp["Доля, %"].apply(
            lambda x: BS.ProgressBar("text-bg-info", "", float(x), value_max=max_percent).render
        )

        tmp = tmp[[group_col, "Выручка, млн", "Доля"]]

        return (
            tmp.style
            .hide(axis="index")
            .to_html()
        )

    top_manager_html = make_top_table(df, "manager_name")
    top_manu_html = make_top_table(df, "manu")
    top_store_html = make_top_table(df, "store")

    # --- матрица store x manager ---
    matrix = df.pivot_table(
        index="store",
        columns="manager_name",
        values="amount",
        aggfunc="sum",
        fill_value=0
    )
    matrix = (matrix / 100_000_000).round(2)

    matrix_html = (
        matrix.style
        .format("{:.2f}")
        .to_html()
    )

    # --- динамика по месяцам (как было) ---
    month_sales = df.pivot_table(
        index=["month", "month_sn"],
        columns="year",
        values="amount",
        aggfunc="sum",
        fill_value=0,
    )

    month_sales = month_sales / 100_000_000
    month_sales = month_sales.reset_index().sort_values("month_sn").drop(columns="month_sn")

    # добавим "Откл" по двум последним годам, если их >=2
    year_cols = sorted([c for c in month_sales.columns if isinstance(c, (int, np.integer))])
    month_sales_html = ""

    if len(year_cols) >= 2:
        y1, y2 = year_cols[-2], year_cols[-1]
        c1 = f"{y1}\u00A0" + BS.Badge.badge_primary.text("нак", "top", "xs")
        c2 = f"{y2}\u00A0" + BS.Badge.badge_primary.text("нак", "top", "xs")
        month_sales[c1] = month_sales[y1].cumsum()
        month_sales[c2] = month_sales[y2].cumsum()

        month_sales["Откл"] = np.where(
            month_sales[c1] == 0,
            0,
            (month_sales[c2] - month_sales[c1]) / month_sales[c1],
        )
        month_sales["Откл"] = (month_sales["Откл"] * 100).round(1)

        max_abs = float(month_sales["Откл"].abs().max())
        max_abs = max_abs if max_abs != 0 else 1.0

        month_sales["Откл"] = month_sales["Откл"].apply(
            lambda x: BS.ProgressBarRelative(
                value=float(x),
                value_min=-max_abs,
                value_max=max_abs,
                color_pos="bg-success",
                color_neg="bg-danger",
            ).render
        )

        month_sales_html = (
            month_sales.style
            .format({y1: "{:.2f}", y2: "{:.2f}", c1: "{:.2f}", c2: "{:.2f}"})
            .hide(axis="index")
            .to_html()
        )
    else:
        # если только 1 год — просто табличка
        month_sales_html = (
            month_sales.style
            .format("{:.2f}")
            .hide(axis="index")
            .to_html()
        )

    # markdown-таблица для текста (без html)
    month_sales_md = month_sales.copy()
    if "Откл" in month_sales_md.columns:
        month_sales_md["Откл"] = month_sales_md["Откл"].astype(str).str.replace(r"<.*?>", "", regex=True)
    month_table_md = month_sales_md.to_markdown(index=False)

    # --- сборка отчета ---
    dnl_content = ReportGenerator(
        title="Отчет по динамики продаж",
        bootswatch_theme="yeti",
        fontsize="md",
    )

    md = f"""
## Отчет по динамики продаж
#### за период с {date_start} по {date_finish}

---

## Резюме

- {Icon.Streamline.pile_of_money_duo.render()} Чистая выручка: **{amount_mln:,.0f} млн ₽**
- {Icon.Streamline.receipt_dollar.render()} Продажи (без возвратов): **{dt_mln:,.0f} млн ₽**
- {Icon.Streamline.business_deal_cash_2.render()} Возвраты: **{cr_mln:,.0f} млн ₽**
- {Icon.Streamline.dangerous_chemical_lab.render()} Коэффициент возвратов: **{return_rate:,.1f}%**
- {Icon.Streamline.calendar_date.render() if hasattr(Icon.Streamline,'calendar_date') else ""} Дней в периоде: **{days_total}**
- {Icon.Streamline.graph_bar.render() if hasattr(Icon.Streamline,'graph_bar') else ""} Среднедневная выручка: **{avg_day_mln:,.1f} млн ₽/день**

---

## Динамика по месяцам *(млн ₽)*
{month_table_md}

---

## ТОП-5 по чистой выручке

### Менеджеры
"""

    dnl_content.add_component(MarkdownBlock(md))
    dnl_content.add_component(HtmlRender(top_manager_html))

    dnl_content.add_component(MarkdownBlock("### Фабрики (manu)"))
    dnl_content.add_component(HtmlRender(top_manu_html))

    dnl_content.add_component(MarkdownBlock("### Магазины (store)"))
    dnl_content.add_component(HtmlRender(top_store_html))

    dnl_content.add_component(MarkdownBlock("---\n## Матрица: магазин × менеджер *(млн ₽)*"))
    dnl_content.add_component(HtmlRender(matrix_html))

    dnl_content.add_component(MarkdownBlock("---\n## Таблица динамики (HTML)"))
    dnl_content.add_component(HtmlRender(month_sales_html))

    return dnl_content


