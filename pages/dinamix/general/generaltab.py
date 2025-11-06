import pandas as pd
import numpy as np
from data import (
    load_columns_df,
    save_df_to_redis,
    load_df_from_redis,
    delete_df_from_redis,
    save_report,
    delete_report,    
    REPORTS,
    ENGINE
)
from dash.exceptions import PreventUpdate
from datetime import datetime, date, timedelta
import io
import dash
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
)
import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import locale

locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")

from components import ValuesRadioGroups, DATES, NoData, month_str_to_date,InDevNotice,ClickOnNotice, DownLoadMenu, LoadingScreen, CsvAGgridDownloader
from data import (
    load_df_from_redis,
    load_columns_dates,
    save_df_to_redis,
    COLS_DICT,
)
from .downloading_content import pdf_data_click

class AreaChartModal:
    def __init__(self, month=None):
        self.month = month if month is not None else None

        self.modal_id = {"type": "gt_area_chart_modal", "index": "1"}
        self.modal_text_id = {"type": "gt_area_chart_modal_text", "index": "1"}

        self.check_distiribution_chart_id = {
            "type": "check_distiribution_chart",
            "index": "1",
        }
        self.check_disttibution_ag_id = {"type": "check_distiribution_ag", "index": "1"}
        self.check_disttibution_ag_id_dnl = {'type':'check_distiribution_ag_dnl','index':1}
        self.ag_gr_id = {'type':'orders-grid','index':1}

    def make_modal(self):

        return dmc.Container(
            children=[
                dmc.Modal(
                    children=[dmc.Text(id=self.modal_text_id)],
                    id=self.modal_id,
                    size="90%",
                )
            ]
        )

    def update_ag(self, d, rrgrid_className):
        def parcer(d):
            if not d or "bins" not in d:
                return None, None

            # Берем все ключи кроме 'bins'
            months = [k.split(" — ")[0] for k in d.keys() if k != "bins"]
            if not months:
                return d["bins"], None

            # Последний месяц
            last_month = pd.to_datetime(months[-1], format="%b %y", errors="coerce")
            return d["bins"], last_month

        _bin, _month = parcer(d)
        

        _month = pd.to_datetime(_month, errors="coerce") + pd.offsets.MonthEnd(0)
        
        
                

        COLS = [
            "date",
            "dt",
            "store",
            "eom",
            "chanel",
            "manager",
            "client_order",
            "quant",
            "client_order_number",
        ]
        
        month = _month.strftime('%Y-%m-%d')
        
        q = f"""
        select 
        LAST_DAY(s.date) as eom,
        s.date,
        s.dt,
        s.cr,
        (s.dt - s.cr) as amount,
        coalesce(sg.name,'Магазин не указан') as store,
        coalesce(st.chanel,'Канал не указан') as chanel,
        coalesce(m.report_name,'Менеджер не указан') as manager,
        coalesce(cat.name,'Нет категории') as cat,
        coalesce(sc.name,'Нет подкатегории') as subcat,
        s.client_order,
        (s.quant_dt - s.quant_cr) as quant,
        s.client_order_number
        -- mn.name as manager,
        -- b.name,
        -- i.fullname,
        -- s.quant_cr
        from sales_salesdata as s

        left join corporate_items as i on i.id = s.item_id
        left join corporate_stores as st on st.id = s.store_id
        left join corporate_storegroups as sg on sg.id = st.gr_id
        left join corporate_managers as m on m.id = s.manager_id
        left join corporate_cattree as cat on cat.id = i.cat_id
        left join corporate_subcategory as sc on sc.id = i.subcat_id
        -- left join corporate_itemmanufacturer as mn on mn.id = i.manufacturer_id
        -- left join corporate_itembrend as b on b.id = i.brend_id

        where LAST_DAY(s.date)  in ('{month}')
                
        """
        df_current = pd.read_sql(q,ENGINE)
               
        

        # df_current = load_columns_dates(COLS, [_month])
        df_current["orders_type"] = np.where(
            df_current["client_order"] == "<Продажи без заказа>",
            "Прочие",
            "Заказы клиента",
        )

        df_orders = (
            df_current.pivot_table(
                index=["client_order_number"],
                values=["dt", "quant", "store", "manager", "date", "orders_type"],
                aggfunc={
                    "dt": "sum",
                    "quant": "sum",
                    "store": "last",
                    "manager": "last",
                    "date": "last",
                    "orders_type": "last",
                },
            )
            .reset_index()
            .sort_values("date")
        )
        df_orders = df_orders[df_orders["dt"] > 1]

        step = 50_000
        last_edge = 350_000

        # формируем границы
        bin_edges = list(np.arange(0, last_edge, step)) + [np.inf]

        # формируем подписи
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            left = bin_edges[i]
            right = bin_edges[i + 1]

            if right == np.inf:
                label = f"от ₽{left/1000:,.0f} тыс и выше"
            elif left == 0:
                label = f"до ₽{right/1000:,.0f} тыс"
            else:
                label = f"от ₽{left/1000:,.0f} до ₽{right/1000:,.0f} тыс"

            bin_labels.append(label)

        # нарезаем на бины
        df_orders["bins"] = pd.cut(
            df_orders["dt"],
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True,
        )

        dff: pd.DataFrame = df_orders[df_orders["bins"] == _bin].copy()
        dff["date"] = pd.to_datetime(dff["date"]).dt.strftime("%Y-%m-%dT%H:%M:%S")

        # columns = [{"headerName": col, "field": col} for col in dff.columns]
        # print(columns)
        date_obj = "d3.timeParse('%Y-%m-%dT%H:%M:%S')(params.data.date)"
        cols = [
            {"headerName": "Номер заказа", 
             "field": "client_order_number", "cellClass": "ag-firstcol-bg",     "pinned": "left",},
            {
            "headerName": "Дата",
            "field": "date",
            "valueFormatter": {"function": "RussianDate(params.value)"}
            },
            {"headerName": "Тип", 
             "field": "orders_type"},
            {"headerName": "Магазин", 
             "field": "store"},
            {
            "headerName": "Сумма",
            "field": "dt",
            "valueFormatter": {"function": "RUB(params.value)"}, "cellClass": "ag-firstcol-bg",
            #"valueFormatter": {"function": "params.value ? '₽'+ d3.format(',.0f')(params.value).replace(/,/g, '\\u202F') : ''" },
            },
            {
                "headerName": "Коли-во товара",
                "field": "quant",
                "valueFormatter": {
                    "function": "FormatWithUnit(params.value,'ед')"
                },
            },
            {"headerName": "Манеджер", "field": "manager",     "pinned": "right",},
        ]

        return dmc.Stack([
            dmc.Space(h=5),
            dmc.Title(f"Детали заказов в ценовом сегменте {_bin}",order=4),
            dmc.Space(h=10),
            CsvAGgridDownloader(self.check_disttibution_ag_id_dnl).dnl_button,
            dag.AgGrid(            
            id=self.ag_gr_id,
            rowData=dff.to_dict("records"),
            columnDefs=cols,
            csvExportParams={
                "fileName": "Заказы_по_цен_сегментам.csv",
            },
            defaultColDef={
                "sortable": True,
                "filter": True,
                "resizable": True,
            },
            dashGridOptions={
                "rowSelection": "single",
                "pagination": True,
                "paginationPageSize": 20,
            },
            style={"height": "600px", "width": "100%"},
            className=rrgrid_className,
            dangerously_allow_code=True            
        ),
            
            
        ]
        )

    def update_modal(self):

        month = pd.to_datetime(self.month, errors="coerce")
        last_month = month + pd.offsets.MonthEnd(-1)
        chart_id = self.check_distiribution_chart_id

       
        
        q = f"""
        select 
        LAST_DAY(s.date) as eom,
        s.date,
        s.dt,
        s.cr,
        (s.dt - s.cr) as amount,
        coalesce(sg.name,'Магазин не указан') as store,
        coalesce(st.chanel,'Канал не указан') as chanel,
        coalesce(m.report_name,'Менеджер не указан') as manager,
        coalesce(cat.name,'Нет категории') as cat,
        coalesce(sc.name,'Нет подкатегории') as subcat,
        s.client_order,
        (s.quant_dt - s.quant_cr) as quant,
        s.client_order_number
        -- mn.name as manager,
        -- b.name,
        -- i.fullname,
        -- s.quant_cr
        from sales_salesdata as s

        left join corporate_items as i on i.id = s.item_id
        left join corporate_stores as st on st.id = s.store_id
        left join corporate_storegroups as sg on sg.id = st.gr_id
        left join corporate_managers as m on m.id = s.manager_id
        left join corporate_cattree as cat on cat.id = i.cat_id
        left join corporate_subcategory as sc on sc.id = i.subcat_id
        -- left join corporate_itemmanufacturer as mn on mn.id = i.manufacturer_id
        -- left join corporate_itembrend as b on b.id = i.brend_id

        where LAST_DAY(s.date)  in ('{month}','{last_month}')
        """
        df_current = pd.read_sql(q,ENGINE)
        
        
        
        # df_current = load_columns_dates(COLS, dates)
        df_current["orders_type"] = np.where(
            df_current["client_order"] == "<Продажи без заказа>",
            "Прочие",
            "Заказы клиента",
        )

        max_date = pd.to_datetime(df_current["date"].max(), errors="coerce")
        min_date = max_date - pd.offsets.MonthBegin(1)

        df_orders = df_current.pivot_table(
            index=["eom", "client_order_number", "orders_type"],
            values=["dt", "quant"],
            aggfunc="sum",
        ).reset_index()

        df_orders = df_orders[df_orders["dt"] > 1]
        df_orders["av_price"] = df_orders["dt"] / df_orders["quant"]

        def check_distibution_chart():
            dff = df_orders[["eom", "orders_type", "dt"]].copy()

            step = 50_000
            last_edge = 350_000

            # формируем границы
            bin_edges = list(np.arange(0, last_edge, step)) + [np.inf]

            # формируем подписи
            bin_labels = []
            for i in range(len(bin_edges) - 1):
                left = bin_edges[i]
                right = bin_edges[i + 1]

                if right == np.inf:
                    label = f"от ₽{left/1000:,.0f} тыс и выше"
                elif left == 0:
                    label = f"до ₽{right/1000:,.0f} тыс"
                else:
                    label = f"от ₽{left/1000:,.0f} до ₽{right/1000:,.0f} тыс"

                bin_labels.append(label)

            # нарезаем на бины
            dff["bins"] = pd.cut(
                dff["dt"],
                bins=bin_edges,
                labels=bin_labels,
                include_lowest=True,
            )

            res = dff.pivot_table(
                index="bins",
                columns=["eom", "orders_type"],
                values="dt",
                aggfunc="count",
                observed=False,
            )

            res.columns = [
                f"{pd.to_datetime(col[0], errors='coerce').strftime('%b %y').capitalize()} — {col[1]}"
                for col in res.columns
            ]

            monthes = list(res.columns)
            res = res.reset_index()

            chart_data = res.to_dict("records")

            chart_series = []
            if len(monthes) > 2:
                chart_series = [
                    {"name": monthes[0], "color": "violet.6", "stackId": "a"},
                    {"name": monthes[1], "color": "violet.3", "stackId": "a"},
                    {"name": monthes[2], "color": "blue.6", "stackId": "b"},
                    {"name": monthes[3], "color": "blue.3", "stackId": "b"},
                ]
            else:
                chart_series = [
                    {"name": monthes[0], "color": "violet.6", "stackId": "a"},
                    {"name": monthes[1], "color": "violet.3", "stackId": "a"},
                ]

            xAxisProps = {
                "angle": -45,
                "tickMargin": 10,  # можно задать отступ между метками и осью
                "textAnchor": "end",  # выравнивание по концу
            }

            return dmc.Stack(
                children=[
                    dmc.Title("График распредления среднего чека по ценовым категориям", order=5),
                    dmc.BarChart(
                        h=300,
                        dataKey="bins",  # должно соответствовать названию столбца с бинами
                        data=chart_data,
                        series=chart_series,
                        tickLine="xy",
                        gridAxis="x",
                        withXAxis=True,
                        withYAxis=True,
                        id=chart_id,
                        # style={"marginBottom": 60},
                        yAxisLabel="Количество",
                        withLegend=True,
                        legendProps={"verticalAlign": "bottom", "height": 50},
                    ),
                ],
            )

        def memo():

            df = df_orders.copy()
            df["month_id"] = pd.factorize(df.eom, sort=True)[0]

            last_vs = pd.to_datetime(last_month).date()
            last_vs: str = last_vs.strftime("%b %y")
            vs = f"vs {last_vs.capitalize()}"

            summary_orders_type = df.pivot_table(
                index="orders_type",
                columns="month_id",
                values=["dt", "client_order_number", "quant"],
                aggfunc={
                    "dt": ["sum", "mean", "median", "max", "min"],
                    "client_order_number": "nunique",
                    "quant": "sum",
                },
            ).fillna(0)
            summary_orders_type.columns = [
                "_".join(map(str, col)).strip()
                for col in summary_orders_type.columns.values
            ]

            [
                "client_order_number_nunique_0",
                "client_order_number_nunique_1",
                "dt_max_0",
                "dt_max_1",
                "dt_mean_0",
                "dt_mean_1",
                "dt_median_0",
                "dt_median_1",
                "dt_min_0",
                "dt_min_1",
                "dt_sum_0",
                "dt_sum_1",
                "quant_sum_0",
                "quant_sum_1",
            ]

            summary_clients_orders = summary_orders_type.head(1)
            summary_other_orders = summary_orders_type.tail(1)

            df["all"] = "all"
            summary_all = df.pivot_table(
                index="all",
                columns="month_id",
                values=["dt", "client_order_number", "quant"],
                aggfunc={
                    "dt": ["sum", "mean", "median", "max", "min"],
                    "client_order_number": "nunique",
                    "quant": "sum",
                },
            ).fillna(0)
            summary_all.columns = [
                f"{val}_{func}_{month}" for val, func, month in summary_all.columns
            ]

            summary_all = summary_all.reset_index(drop=True)

            n_month = df["month_id"].unique().tolist()
            comp = True
            if len(n_month) > 1:
                comp = True
            else:
                comp = False

            l = [
                "client_order_number_nunique_0",
                "client_order_number_nunique_1",
                "dt_max_0",
                "dt_max_1",
                "dt_mean_0",
                "dt_mean_1",
                "dt_median_0",
                "dt_median_1",
                "dt_min_0",
                "dt_min_1",
                "dt_sum_0",
                "dt_sum_1",
                "quant_sum_0",
                "quant_sum_1",
            ]

            tot_orders = (
                summary_all["client_order_number_nunique_1"].sum()
                if comp
                else summary_all["client_order_number_nunique_0"].sum()
            )
            tot_orders_last = (
                summary_all["client_order_number_nunique_0"].sum() if comp else None
            )
            tot_orders_change = (
                (tot_orders - tot_orders_last) / tot_orders_last * 100
                if tot_orders_last
                else None
            )
            if tot_orders_change:
                if tot_orders_change > 0:
                    tot_orders_change = f"*(📈 рост на {tot_orders_change:,.1f}% по сравнению с предыдущем месяцем)*"
                else:
                    tot_orders_change = f"*(📉 падение на {abs(tot_orders_change):,.1f}% по сравнению с предыдущем месяцем)*"
            else:
                tot_orders_change = ""

            clients_orders = (
                summary_clients_orders["client_order_number_nunique_1"].sum()
                if comp
                else summary_clients_orders["client_order_number_nunique_0"].sum()
            )
            clients_orders_last = (
                summary_clients_orders["client_order_number_nunique_0"].sum()
                if comp
                else None
            )

            clients_orders_change = (
                (clients_orders - clients_orders_last) if clients_orders_last else None
            )
            if clients_orders_change:
                if clients_orders_change > 0:
                    clients_orders_change = (
                        f"*(+ {clients_orders_change:,.0f} зак. {vs})*"
                    )
                else:
                    clients_orders_change = (
                        f"*(- {abs(clients_orders_change):,.0f} зак. {vs})*"
                    )
            else:
                clients_orders_change = ""

            other_orders = (
                summary_other_orders["client_order_number_nunique_1"].sum()
                if comp
                else summary_other_orders["client_order_number_nunique_0"].sum()
            )
            other_orders_last = (
                summary_other_orders["client_order_number_nunique_0"].sum()
                if comp
                else None
            )

            other_orders_change = (
                (other_orders - other_orders_last) if other_orders_last else None
            )
            if other_orders_change:
                if other_orders_change > 0:
                    other_orders_change = f"*(+ {other_orders_change:,.0f} зак. {vs})*"
                else:
                    other_orders_change = (
                        f"*(- {abs(other_orders_change):,.0f} зак. {vs})*"
                    )
            else:
                other_orders_change = ""

            av_check_tot = (
                summary_all["dt_mean_1"].sum()
                if comp
                else summary_all["dt_mean_0"].sum()
            )
            av_check_tot_last = summary_all["dt_mean_0"].sum() if comp else None
            av_check_tot_change = (
                (av_check_tot - av_check_tot_last) / av_check_tot_last * 100
                if av_check_tot_last
                else None
            )
            if av_check_tot_change:
                if av_check_tot_change > 0:
                    av_check_tot_change = f"*(📈 рост на {av_check_tot_change:,.1f}% по сравнению с предыдущем месяцем)*"
                else:
                    av_check_tot_change = f"*(📉 падение на {abs(av_check_tot_change):,.1f}% по сравнению с предыдущем месяцем)*"
            else:
                av_check_tot_change = ""

            clients_orders_check = (
                summary_clients_orders["dt_mean_1"].sum()
                if comp
                else summary_clients_orders["dt_mean_0"].sum()
            )
            clients_order_check_last = (
                summary_clients_orders["dt_mean_0"].sum() if comp else None
            )

            clients_orders_check_change = (
                (clients_orders_check - clients_order_check_last)
                if clients_order_check_last
                else None
            )
            if clients_orders_check_change:
                if clients_orders_check_change > 0:
                    clients_orders_check_change = (
                        f"*(+ {clients_orders_check_change:,.0f} руб. {vs})*"
                    )
                else:
                    clients_orders_check_change = (
                        f"*(- {abs(clients_orders_check_change):,.0f} руб. {vs})*"
                    )
            else:
                clients_orders_check_change = ""

            other_orders_check = (
                summary_other_orders["dt_mean_1"].sum()
                if comp
                else summary_other_orders["dt_mean_0"].sum()
            )
            other_orders_check_last = (
                summary_other_orders["dt_mean_0"].sum() if comp else None
            )

            other_orders_check_change = (
                (other_orders_check - other_orders_check_last)
                if other_orders_check_last
                else None
            )
            if other_orders_check_change:
                if other_orders_check_change > 0:
                    other_orders_check_change = (
                        f"*(+ {other_orders_check_change:,.0f} руб. {vs})*"
                    )
                else:
                    other_orders_check_change = (
                        f"*(- {abs(other_orders_check_change):,.0f} руб. {vs})*"
                    )
            else:
                other_orders_check_change = ""

            max_check_tot = (
                summary_all["dt_max_1"].sum() if comp else summary_all["dt_max_0"].sum()
            )
            max_check_tot_last = summary_all["dt_max_0"].sum() if comp else None
            max_check_tot_change = (
                (max_check_tot - max_check_tot_last) if max_check_tot_last else None
            )
            if max_check_tot_change:
                if max_check_tot_change > 0:
                    max_check_tot_change = (
                        f"*(+ {max_check_tot_change:,.0f} руб. {vs})*"
                    )
                else:
                    max_check_tot_change = (
                        f"*(- {abs(max_check_tot_change):,.0f} руб. {vs})*"
                    )
            else:
                max_check_tot_change = ""

            min_check_tot = (
                summary_all["dt_min_1"].sum() if comp else summary_all["dt_min_0"].sum()
            )
            min_check_tot_last = summary_all["dt_min_0"].sum() if comp else None
            min_check_tot_change = (
                (min_check_tot - min_check_tot_last) if min_check_tot_last else None
            )
            if min_check_tot_change:
                if min_check_tot_change > 0:
                    min_check_tot_change = (
                        f"*(+ {min_check_tot_change:,.0f} руб. {vs})*"
                    )
                else:
                    min_check_tot_change = (
                        f"*(- {abs(min_check_tot_change):,.0f} руб. {vs})*"
                    )
            else:
                min_check_tot_change = ""

            madian_check_tot = (
                summary_all["dt_median_1"].sum()
                if comp
                else summary_all["dt_median_0"].sum()
            )
            median_check_tot_last = summary_all["dt_median_0"].sum() if comp else None
            median_check_tot_change = (
                (madian_check_tot - median_check_tot_last) / median_check_tot_last * 100
                if median_check_tot_last
                else None
            )
            if median_check_tot_change:
                if median_check_tot_change > 0:
                    median_check_tot_change = f"*(📈 рост на {median_check_tot_change:,.1f}% по сравнению с предыдущем месяцем)*"
                else:
                    median_check_tot_change = f"*(📉 падение на {abs(median_check_tot_change):,.1f}% по сравнению с предыдущем месяцем)*"
            else:
                median_check_tot_change = ""

            clients_orders_check_median = (
                summary_clients_orders["dt_median_1"].sum()
                if comp
                else summary_clients_orders["dt_median_0"].sum()
            )
            clients_order_check_median_last = (
                summary_clients_orders["dt_median_0"].sum() if comp else None
            )

            clients_orders_check_median_change = (
                (clients_orders_check_median - clients_order_check_median_last)
                if clients_order_check_median_last
                else None
            )
            if clients_orders_check_median_change:
                if clients_orders_check_median_change > 0:
                    clients_orders_check_median_change = (
                        f"*(+ {clients_orders_check_median_change:,.0f} руб. {vs})*"
                    )
                else:
                    clients_orders_check_median_change = f"*(- {abs(clients_orders_check_median_change):,.0f} руб. {vs})*"
            else:
                clients_orders_check_median_change = ""

            other_orders_check_median = (
                summary_other_orders["dt_median_1"].sum()
                if comp
                else summary_other_orders["dt_median_0"].sum()
            )
            other_orders_check_median_last = (
                summary_other_orders["dt_median_0"].sum() if comp else None
            )

            other_orders_check_median_change = (
                (other_orders_check_median - other_orders_check_median_last)
                if other_orders_check_median_last
                else None
            )
            if other_orders_check_median_change:
                if other_orders_check_median_change > 0:
                    other_orders_check_median_change = (
                        f"*(+ {other_orders_check_median_change:,.0f} руб. {vs})*"
                    )
                else:
                    other_orders_check_median_change = (
                        f"*(- {abs(other_orders_check_median_change):,.0f} руб. {vs})*"
                    )
            else:
                other_orders_check_median_change = ""

            text_md = f"""
            ### Статистика средних чеков по заказам
            
            За рассматриваемый период было выполнено **{tot_orders}** заказов {tot_orders_change}, из которых:
            
            - {clients_orders} составили заказы клиентов {clients_orders_change};
            - {other_orders} — прочие заказы {other_orders_change}.
            
            **Средний чек** по всем заказам составил **{av_check_tot:,.0f}** руб. {av_check_tot_change}, при этом:
            
            - средний чек по заказам клиентов составил {clients_orders_check:,.0f} руб. {clients_orders_check_change}; 
            - прочим заказам - {other_orders_check:,.0f} руб. {other_orders_check_change}.
            
            Cумма максимального заказа составила **{max_check_tot:,.0f}** руб {max_check_tot_change}, минимального — **{min_check_tot:,.0f}** руб {min_check_tot_change}.
            
            **Медианный чек** по всем заказам составил **{madian_check_tot:,.0f}** руб. {median_check_tot_change}, при этом:
            
            - медианный чек по заказам клиентов составил {clients_orders_check_median:,.0f} руб. {clients_orders_check_median_change}; 
            - прочим заказам - {other_orders_check_median:,.0f} руб. {other_orders_check_median_change}.
            """

            return dcc.Markdown(text_md, className="markdown-body")

        return dmc.Container(
            children=[
                dmc.Title(
                    f"Отчет по заказам за период с {min_date.strftime('%-d %B %Y')} по {max_date.strftime('%-d %B %Y')}  ",
                    order=3,
                    c="blue",
                ),
                memo(),
                check_distibution_chart(),                
                dmc.Container(
                    children=dmc.Text(
                        "кликните на график что бы увидеть детали", size="sm", c='grape'
                    ),
                    id=self.check_disttibution_ag_id,
                    fluid=True,
                ),
                dmc.Space(h=20),
                dcc.Markdown('### Изменения средних чеков по магазинам',className="markdown-body"),
                dmc.Space(h=5),
                # InDevNotice().in_dev_conteines
                ClickOnNotice(
                notice="Кликните на график, чтобы просмотреть заказы чеков по выбранным ценовым категориям",
                icon="streamline-ultimate:task-finger-show",  
                color="#007BFF",
            ).component,
                
                    ],
                    fluid=True,    
                                    
                ),
                


    def registered_callacks(self, app):
        destrib_chart_type = self.check_distiribution_chart_id["type"]
        destrib_ag_type = self.check_disttibution_ag_id["type"]
        
        ag = self.ag_gr_id['type']
        dnl = self.check_disttibution_ag_id_dnl['type']

        @app.callback(
            Output({"type": destrib_ag_type, "index": MATCH}, "children"),
            Input({"type": destrib_chart_type, "index": MATCH}, "clickData"),
            State("theme_switch", "checked"),
            prevent_initial_call=True,
        )
        def update_ag(clickData, checked):
            rrgrid_className = "ag-theme-alpine-dark" if checked else "ag-theme-alpine"
            a = self.update_ag(clickData, rrgrid_className)

            return a
        
        @app.callback(
            Output({"type":ag,"index":MATCH},'exportDataAsCsv'),
            Input({"type":dnl,"index":MATCH},'n_clicks'),
            prevent_initial_call=True,
        )
        def export_data_as_csv(n_clicks):
            if n_clicks:
                return True
            return False
        


class Components:
    def __init__(self, df_id=None):

        self.df_id = df_id if df_id is not None else None
        self.area_chart_id = {"type": "gt_big_area_chart", "index": "1"}
        self.av_check_chart_id = {"type": "gt_av_check_chart", "index": "1"}
        self.check_selector_id = {"type": "selector", "index": "1"}
        self.memo_id = {"type": "memo", "index": "1"}
        
        self.dnl_buton_id = {'type':'content_download_button_for_gt','index':'1'}
        self.dcc_download_id = {'type':'content_download_dcc_for_gt','index':'1'}
        
        self.pdf_dnl_type = 'gt_pdf_dnl_type_id_unique'

    def data(self):
        df_id = self.df_id

        if not df_id:
            return None

        df_data: pd.DataFrame = load_df_from_redis(df_id)

        if df_data.empty:
            return None

        df_eom = (
            df_data.pivot_table(
                index="eom", values=["dt", "cr", "amount"], aggfunc="sum"
            )
            .fillna(0)
            .reset_index()
            .sort_values("eom")
        )

        def update_area_chart():
            df: pd.DataFrame = df_eom.copy()
            df.rename(columns=COLS_DICT, inplace=True)
            for col in ["Продажи", "Возвраты", "Чистая выручка"]:
                df[f"{col}_from_first"] = ((df[col] / df[col].iloc[0] - 1) * 100).round(
                    2
                )

            # отклонение от предыдущего значения (в %)
            for col in ["Продажи", "Возвраты", "Чистая выручка"]:
                df[f"{col}_from_prev"] = ((df[col] / df[col].shift(1) - 1) * 100).round(
                    2
                )

            df = df.fillna(0)

            df["eom"] = pd.to_datetime(df["eom"], errors="coerce")
            df["eom"] = df["eom"].dt.strftime("%b\u202F%y").str.capitalize()

            data = df.to_dict(orient="records")

            series = [
                {"name": "Чистая выручка", "color": "indigo.3", "type": "bar"},
                {"name": "Продажи", "color": "green.6", "type": "line"},
                {"name": "Возвраты", "color": "red.6", "type": "line"},
            ]

            return dmc.CompositeChart(
                id=self.area_chart_id,
                h=400,
                dataKey="eom",
                data=data,
                tooltipAnimationDuration=500,
                areaProps={
                    "isAnimationActive": True,
                    "animationDuration": 500,
                    "animationEasing": "ease-in-out",
                    "animationBegin": 500,
                },
                withPointLabels=False,
                series=series,
                valueFormatter={"function": "formatNumberIntl"},
                # type="stacked",
                withLegend=True,
                legendProps={"verticalAlign": "bottom"},
                tooltipProps={"content": {"function": "chartTooltip"}},
                # type="default",
            )

        def update_av_check_chart():
            df: pd.DataFrame = df_data[["eom", "dt", "client_order_number"]]
            df = df.dropna()
            if df.empty:
                # return NoData().component
                return LoadingScreen().component
            df_sum = (
                df.groupby(["eom", "client_order_number"], as_index=False)["dt"].sum()
            ).reset_index()
            df_sum = df_sum[df_sum["dt"] != 0]
            df = df_sum.pivot_table(
                index="eom",
                values=["dt", "client_order_number"],
                aggfunc={
                    "dt": ["sum", "median", "max", "min"],
                    "client_order_number": "nunique",
                },
            ).fillna(0)
            df.columns = ["_".join(col).strip() for col in df.columns.values]
            df = df.reset_index().sort_values(by="eom")
            df.rename(
                columns={"dt_median": "Медиана", "dt_max": "Макс", "dt_min": "Мин","client_order_number_nunique":"Количество заказов"},
                inplace=True,
            )

            df["Средний чек"] = df["dt_sum"] / df["Количество заказов"]
            df["eom"] = pd.to_datetime(df["eom"], errors="coerce")
            df["month_name"] = df["eom"].dt.strftime("%b %y").str.capitalize()

            data = df.to_dict(orient="records")

            series = [
                {"name": "Средний чек", "color": "green.6",'type':'line'},
                {"name": "Медиана", "color": "blue.6",'type':'line'},
            ]

            selector_vals = {"1": "Средн / Медиана", "2": "Макс/Мин"}

            return dmc.Stack(
                [
                    ValuesRadioGroups(
                        id_radio=self.check_selector_id,
                        options_dict=selector_vals,
                        val="1",
                    ),
                    dmc.CompositeChart(
                        id=self.av_check_chart_id,
                        h=200,
                        dataKey="month_name",
                        data=data,
                        tooltipAnimationDuration=500,
                        areaProps={
                            "isAnimationActive": True,
                            "animationDuration": 500,
                            "animationEasing": "ease-in-out",
                            "animationBegin": 500,
                        },
                        withPointLabels=False,
                        series=series,
                        valueFormatter={"function": "formatNumberIntl"},
                        # type="stacked",
                        withLegend=True,
                        legendProps={"verticalAlign": "top"},
                        connectNulls=True,
                        composedChartProps={"syncId": "av_check_total"}
                        # tooltipProps={"content":  {"function": "chartTooltip"}},
                        # type="default",
                    ),
             ClickOnNotice(
                notice="Кликните на график, чтобы просмотреть отчет по заказам за выбранный месяц",
                icon="streamline-ultimate:task-finger-show",  
                color="#007BFF",
            ).component,

                    dmc.Text('Количество заказов',size='md'),
                    dmc.CompositeChart(
                        h=100,
                        dataKey="month_name",
                        data=data,
                        tooltipAnimationDuration=500,
                        areaProps={
                            "isAnimationActive": True,
                            "animationDuration": 500,
                            "animationEasing": "ease-in-out",
                            "animationBegin": 500,
                        },
                        withPointLabels=False,
                        series=[{'name':'Количество заказов',"dataKey": "Количество заказов","color":"red",'type':'bar'}],
                        composedChartProps={"syncId": "av_check_total"}                       
                        
                    )
                    
                ]
            )

        def memo():

            from_date = pd.to_datetime(
                df_data["date"].min(), errors="coerce"
            ).normalize()
            to_date = pd.to_datetime(df_data["date"].max(), errors="coerce").normalize()

            md_text = f"""
            ## Краткий отчет за период с {from_date.strftime('%d %B %Y')} по {to_date.strftime('%d %B %Y')} 
            
            За рассматриваемый период:
            
            - **чистая выручка от реализации** составила: {df_data['amount'].sum()/1_000_000:,.2f} млн рублей;
            - общие продажи: {df_data['dt'].sum()/1_000_000:,.2f} млн рублей;
            - возвраты {df_data['cr'].sum()/1_000_000:,.2f} млн рублей.
            """

            return dmc.Spoiler(
                children=[dcc.Markdown(md_text)],
                maxHeight=50,
                hideLabel="Скрыть",
                showLabel="Читать далее",
            )

        def dnl_button():
            return dmc.Flex(DownLoadMenu(
                xls_disable=True,
                html_disable=True,
                pdf_disable=False,
                pdf_id_type=self.pdf_dnl_type                
            ).menu,justify='flex-end')
        
        
        return update_area_chart(), update_av_check_chart(), memo(), dnl_button()


def layout(df_id=None):
    df_id = df_id if df_id is not None else None
    comp = Components(df_id)
    area_chart, av_check_chart, memo, dnl_button  = comp.data()
    try:
        area_chart, av_check_chart, memo, dnl_button  = comp.data()

        return dmc.Container(
            children=[
                dnl_button,
                dmc.Space(h=10),
                memo,
                dmc.Space(h=20),
                dmc.Title("Динамика целевых показателей продаж", order=4, c="blue"),
                dmc.Space(h=10),
                area_chart,                
                dmc.Space(h=20),
                dmc.Title("Динамика средних чеков по заказам", order=4, c="blue"),
                dmc.Space(h=10),
                av_check_chart,
                AreaChartModal().make_modal(),
            ],
            fluid=True
        )
    except:
        # return NoData().component
        return LoadingScreen().component


def registed_callbacks(app):
    comp = Components()
    modal = AreaChartModal()

    av_check_chart_type = comp.av_check_chart_id["type"]
    av_check_selectir_type = comp.check_selector_id["type"]
    av_check_modal_type = modal.modal_id["type"]
    av_check_modal_text_type = modal.modal_text_id["type"]
    pdf_dnl_button = comp.pdf_dnl_type
    dcc_dnl = comp.dcc_download_id['type']

    # График среднего чека
    @app.callback(
        Output({"type": av_check_chart_type, "index": MATCH}, "series"),
        Input({"type": av_check_selectir_type, "index": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def change_series(val):
        # print("Button clicked", n_clicks)

        if val == "1":  # пример переключения
            series = [
                {"name": "Средний чек", "dataKey": "Средний чек", "color": "green.6",'type':'line'},
                {"name": "Медиана", "dataKey": "Медиана", "color": "blue.6",'type':'line'},
            ]
        else:
            series = [
                {"name": "Макс", "dataKey": "Макс", "color": "green.6",'type':'line'},
                {"name": "Мин", "dataKey": "Мин", "color": "blue.6",'type':'line'},
            ]

        return series

    # @app.callback(
    #     Output({"type": av_check_modal_type, "index": MATCH}, "opened"),
    #     Output({"type": av_check_modal_type, "index": MATCH}, "children"),
    #     Input({"type": av_check_chart_type, "index": MATCH}, "clickData"),
    #     State({"type": av_check_modal_type, "index": MATCH}, "opened"),
    #     prevent_initial_call=True,
    # )
    # def av_check_report_modal(clickData, opened):
    #     data = AreaChartModal(clickData["eom"]).update_modal()

    #     return not opened, data
    
    


    @app.callback(
        Output({"type": av_check_modal_type, "index": MATCH}, "opened"),
        Output({"type": av_check_modal_type, "index": MATCH}, "children"),
        Output({"type": av_check_chart_type, "index": MATCH}, "clickData"),  # сбросим при закрытии
        Input({"type": av_check_chart_type, "index": MATCH}, "clickData"),
        Input({"type": av_check_modal_type, "index": MATCH}, "opened"),       # слушаем закрытие
        prevent_initial_call=True,
    )
    def av_check_report_modal(clickData, modal_opened):
        trig = dash.ctx.triggered_id  # dash.callback_context в старых версиях

        # 1) Кликнули по графику → открыть и заполнить
        if isinstance(trig, dict) and trig.get("type") == av_check_chart_type:
            if not clickData:
                return no_update, no_update, no_update

            # Надёжно извлечь eom из разных вариантов структуры clickData
            payload0 = (clickData.get("activePayload") or [{}])[0]
            eom = (
                payload0.get("eom")
                or (payload0.get("payload") or {}).get("eom")
                or clickData.get("eom")
            )

            data = AreaChartModal(eom).update_modal()
            return True, data, no_update   # всегда открываем, clickData пока не трогаем

        # 2) Закрыли модалку крестиком/оверлеем → сбрасываем clickData
        if modal_opened is False:
            return no_update, no_update, None

        # 3) Остальные случаи — без изменений
        return no_update, no_update, no_update


    
    @app.callback(
    Output('pdf_download','data'),
    Output('preview_modal','opened'),
    Output('pdf_content','children'),
    Output("loading-indicator", "children"),
    Output('PreviewModal_theme_selector','value'),
    Output('PreviewModal_size_chose','value'),
    Input({'type':pdf_dnl_button, 'index': ALL}, 'n_clicks'),
    State('df_store','data'),
    State('pdf_download','data'),
    State('preview_modal','opened'),
    prevent_initial_call=True
    )
    def get_report(n_clicks, data, last_id, opened):
        if not n_clicks or all(x is None for x in n_clicks):
            raise dash.exceptions.PreventUpdate  
        
        df_id = data['df_id']

        # генерируем новый объект отчёта
        report = pdf_data_click(df_id)
        
        #дефолтные значения собираем
        current_theme = report.bootswatch_theme
        currnet_fontsize = report.fontsize
        
        content = report.return_iframe()

        # удаляем старый отчёт из памяти
        if last_id:
            delete_report(last_id)

        # сохраняем новый отчёт
        report_id = save_report(report)      

        # возвращаем rid в pdf_download.data
        return report_id, not opened, content, "", current_theme, currnet_fontsize
        
    
    
    
    
    AreaChartModal().registered_callacks(app)
