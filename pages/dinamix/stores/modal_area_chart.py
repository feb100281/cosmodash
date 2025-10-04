# Файл для модалки для AreaChart по магазинам

# import pandas as pd
# import numpy as np
# from dash.exceptions import PreventUpdate
# import dash
# from dash import (
#     Dash,
#     dcc,
#     html,
#     Input,
#     Output,
#     State,
#     _dash_renderer,
#     clientside_callback,
#     MATCH,
#     ALL,
#     ctx,
#     Patch,
#     no_update,
# )
# import dash_ag_grid as dag
# from dash_iconify import DashIconify
# import dash_mantine_components as dmc
# import locale

# locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")

# from components import ValuesRadioGroups, DATES, NoData, BASE_COLORS, COLORS_BY_COLOR, COLORS_BY_SHADE, InDevNotice
# from data import (
#     load_columns_df,
#     save_df_to_redis,
#     load_df_from_redis,  
#     load_columns_dates,  
#     COLS_DICT,
# )

# # Делаем класс с модалкой

# class StoreAreaChartModal:
#     def __init__(self, clickdata=None,clickSeriesName=None): # clickdata - это возвращенный словарь при клике на areachart
        
#         self.clickdata = clickdata
#         self.clickSeriesName = clickSeriesName
#         # Также прописываем id компонентов        
#         self.modal_id = {'type':'store_area_chart_modal','index':'1'} #ID самого модала
#         self.conteiner_id = {'type':'store_area_chart_modal_conteiner','index':'1'} #ID контейнера модала куда будем все складывать
    
#     # Делаем первоначальный модал для layout    
#     def create_components(self):        
#         return  dmc.Modal(
#                     children=[dmc.Container(id=self.conteiner_id)],
#                     id=self.modal_id,
#                     size="90%",
#                 )
    
#     # Делаем метод для апдэйта модала при нажатие на график
    
#     def update_modal(self):
        
#         df = pd.DataFrame()
        
#         #Загрудаем данные из редиски за текущий и предыдущий месяц
        
#         if not self.clickdata:
#            return 
#         month = self.clickdata['eom']
#         month = pd.to_datetime(month, errors="coerce")
#         last_month = month + pd.offsets.MonthEnd(-1)
#         last_year = month - pd.DateOffset(years=1)
#         last_year = last_year + pd.offsets.MonthEnd(0)
        

#         COLS = [
#             "eom",
#             "date",
#             "dt",
#             "cr",
#             "amount",
#             "store_gr_name",
#             "eom",
#             "chanel",
#             "manager",
#             "cat",
#             "subcat",
#             "client_order",
#             "quant",
#             "client_order_number",
#             "store_gr_name_amount_ytd",
            
#         ]
#         dates = [month, last_month, last_year]
#         df_current = load_columns_dates(COLS, dates)
#         df_current["orders_type"] = np.where(
#             df_current["client_order"] == "<Продажи без заказа>",
#             "Продажи без заказ",
#             "Заказы клиента",
#         ) 
        
#         # просто для примера
        
#         df_selected_store:pd.DataFrame =  df_current[df_current['store_gr_name']==self.clickSeriesName].copy()
#         df_selected_store['date']  = pd.to_datetime(df_selected_store['date'],errors='coerce')
#         df_selected_store['day'] = df_selected_store['date'].dt.day 
#         df_selected_store = df_selected_store.pivot_table(
#             index = 'day',
#             columns='eom',
#             values=['amount'],
#             aggfunc='sum'
#         ).reset_index().sort_values(by='day').fillna(0)
        
#         if isinstance(df_selected_store.columns, pd.MultiIndex):
#             df_selected_store.columns = [col[1] if col[0] == 'amount' else col[0] for col in df_selected_store.columns]
            
        
        
        
#         # Делаем отчет (пока такой)
        
#         def memo():
#             md_text = f"""
            
# ## Отчет по магазину {self.clickSeriesName} за {self.clickdata['month_name']}



# ### Это clickdata 

# ```
# {self.clickdata}
# ```

# можно ее использовать что бы загружать данные из редиски      



# ```
# {self.clickSeriesName}
# ```

# можно использовать для филтрации данных по выбраному магазину

# например 

# ```
# df = pd.DataFrame()
        
# #Загрудаем данные из редиски за текущий и предыдущий месяц

# if not self.clickdata:
#     return 
# month = self.clickdata['eom']
# month = pd.to_datetime(month, errors="coerce")
# last_month = month + pd.offsets.MonthEnd(-1)
# last_year = month - pd.DateOffset(years=1)
# last_year = last_year + pd.offsets.MonthEnd(0)


# COLS = [
#     "eom",
#     "date",
#     "dt",
#     "cr",
#     "amount",
#     "store_gr_name",
#     "eom",
#     "chanel",
#     "manager",
#     "cat",
#     "subcat",
#     "client_order",
#     "quant",
#     "client_order_number",
#     "store_gr_name_amount_ytd",
    
# ]
# dates = [month, last_month, last_year]
# df_current = load_columns_dates(COLS, dates)
# df_current["orders_type"] = np.where(
#     df_current["client_order"] == "<Продажи без заказа>",
#     "Прочие",
#     "Заказы клиента",
# )

# # просто для примера

# df_selected_store:pd.DataFrame =  df_current[df_current['store_gr_name']==self.clickSeriesName].copy()
# df_selected_store['date']  = pd.to_datetime(df_selected_store['date'],errors='coerce')
# df_selected_store['day'] = df_selected_store['date'].dt.day 
# df_selected_store = df_selected_store.pivot_table(
#     index = 'day',
#     columns='eom',
#     values=['amount'],
#     aggfunc='sum'
# ).reset_index().sort_values(by='day')

# if isinstance(df_selected_store.columns, pd.MultiIndex):
#     df_selected_store.columns = [col[1] if col[0] == 'amount' else col[0] for col in df_selected_store.columns]
# ```


# Вот df-ка получается какая

# ## Данные по магазину {self.clickSeriesName} за {self.clickdata['month_name']}

# {df_selected_store.to_markdown(index=False)}

            
            
#             """
            
#             return dcc.Markdown(md_text,className='markdown-body')
        
#         # Делаем дфку в md для пробы пера
        
#         def temp_grid():
#             return InDevNotice().in_dev_conteines
        
#         # Основной return

#         return memo(), temp_grid()
    
#     def modal_children(self):
        
#         if not self.clickdata:
#            return NoData().component
        
        
#         memo, temp_grid = self.update_modal()
        
#         return dmc.Stack(
#             [
#                 dmc.Title('Пробный модал по клику на график'),
#                 memo,
#                 temp_grid
#             ]
#         )
    
#     def registered_callbacks(self,app): # Сюда колбэки добавляем которые с модалкой связанны будут 
#         pass


# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from dash import dcc, html, no_update
# import dash_mantine_components as dmc
# import dash_ag_grid as dag
# from dash_iconify import DashIconify

# from components import (
#     ValuesRadioGroups,
#     DATES,
#     NoData,
#     BASE_COLORS,
#     COLORS_BY_COLOR,
#     COLORS_BY_SHADE,
#     InDevNotice,
# )
# from data import (
#     load_columns_df,
#     save_df_to_redis,
#     load_df_from_redis,
#     load_columns_dates,
#     COLS_DICT,
# )


# # ------------------------------------------------------------
# # StoreAreaChartModal — модальное окно «Отчёт по магазину»
# # ------------------------------------------------------------
# # Ключевые фичи:
# # 1) Красивый заголовок с месяцем/магазином, бейджами сравнения (м/м, г/г)
# # 2) KPI-карточки: Выручка, Заказы, Средний чек, Доля «без заказа»
# # 3) Табы: Обзор • Динамика по дням • Разрезы • Таблица операций
# # 4) Графики: Area (по дням), Area кумулятивно, столбцы «без заказа vs заказы»
# # 5) Экспорт: кнопка-меню (PDF/PNG/Excel) — только UI, без колбэков
# # 6) Стойкое поведение при пустых данных
# # ------------------------------------------------------------


# class StoreAreaChartModal:
#     def __init__(self, clickdata=None, clickSeriesName=None):
#         """
#         clickdata: dict с ключами 'eom' (YYYY-MM-DD), 'month_name' (строка), ...
#         clickSeriesName: имя магазина (seriesName из AreaChart)
#         """
#         self.clickdata = clickdata
#         self.clickSeriesName = clickSeriesName

#         # ID компонентов
#         self.modal_id = {"type": "store_area_chart_modal", "index": "1"}
#         self.container_id = {"type": "store_area_chart_modal_container", "index": "1"}

#     # --------------- Публичный API ---------------
#     def create_components(self):
#         return dmc.Modal(
#             id=self.modal_id,
#             size="90%",
#             withCloseButton=True,
#             children=[dmc.Container(id=self.container_id, fluid=True, px="md", py="md")],
#         )

#     def modal_children(self):
#         if not self.clickdata or not self.clickSeriesName:
#             return NoData().component

#         content = self._build_content()
#         return content

#     # ✅ Backward-compat: старое имя метода, чтобы не трогать твой колбэк
#     def update_modal(self):
#         return self.modal_children()

#     def registered_callbacks(self, app):
#         """Заглушка для последующих clientside/server колбэков при необходимости."""
#         pass

#     # --------------- Внутренняя логика ---------------
#     def _build_content(self):
#         # 1) Загрузка данных
#         month, last_month, last_year = self._derive_periods(self.clickdata)
#         df_current = self._load_base_df([month, last_month, last_year])

#         if df_current.empty:
#             return self._empty_state()

#         # 2) Фильтр по магазину
#         store = self.clickSeriesName
#         df_store = df_current.loc[df_current["store_gr_name"] == store].copy()
#         if df_store.empty:
#             return self._empty_state(f"Нет данных по магазину: {store}")

#         # 3) Подготовка дневных рядов
#         daily_stats, daily_cum = self._prepare_daily_series(df_store)

#         # 4) Метрики/KPI
#         kpis = self._compute_kpis(df_store, month, last_month, last_year)

#         # 5) Разрезы
#         slices = self._make_slices(df_store)

#         # 6) Компоненты UI
#         header = self._header(store, self.clickdata.get("month_name"))
#         kpi_grid = self._kpi_grid(kpis)
#         tabs = self._tabs(daily_stats, daily_cum, slices, df_store)

#         return dmc.Stack([header, kpi_grid, tabs], gap="md")

#     # ----------------- Data -----------------
#     @staticmethod
#     def _derive_periods(clickdata):
#         month = pd.to_datetime(clickdata.get("eom"), errors="coerce")
#         last_month = month + pd.offsets.MonthEnd(-1)
#         last_year = (month - pd.DateOffset(years=1)) + pd.offsets.MonthEnd(0)
#         return month, last_month, last_year

#     @staticmethod
#     def _load_base_df(dates):
#         COLS = [
#             "eom",
#             "date",
#             "dt",
#             "cr",
#             "amount",
#             "store_gr_name",
#             "chanel",
#             "manager",
#             "cat",
#             "subcat",
#             "client_order",
#             "quant",
#             "client_order_number",
#             "store_gr_name_amount_ytd",
#         ]
#         df = load_columns_dates(COLS, dates)
#         if df is None or len(df) == 0:
#             return pd.DataFrame(columns=COLS)

#         df["orders_type"] = np.where(
#             df["client_order"].eq("<Продажи без заказа>"),
#             "Продажи без заказа",
#             "Заказы клиента",
#         )
#         # date -> datetime, day
#         df["date"] = pd.to_datetime(df["date"], errors="coerce")
#         df["day"] = df["date"].dt.day
#         return df

#     @staticmethod
#     def _prepare_daily_series(df_store: pd.DataFrame):
#         # Сумма по дням для каждого периода (eom)
#         daily = (
#             df_store.pivot_table(index="day", columns="eom", values="amount", aggfunc="sum")
#             .fillna(0)
#             .sort_index()
#         )
#         # Кумулятив
#         daily_cum = daily.cumsum()
#         # Для удобного отображения в Markdown/таблицах — сбросить индекс
#         daily_stats = daily.reset_index()
#         daily_cum_stats = daily_cum.reset_index()
#         return daily_stats, daily_cum_stats

#     @staticmethod
#     def _compute_kpis(df_store: pd.DataFrame, month, last_month, last_year):
#         def _sum_for_eom(eom):
#             return float(df_store.loc[df_store["eom"].eq(eom), "amount"].sum())

#         def _cnt_orders(eom):
#             return int(df_store.loc[df_store["eom"].eq(eom) & df_store["client_order_number"].notna(), "client_order_number"].nunique())

#         def _qty(eom):
#             return float(df_store.loc[df_store["eom"].eq(eom), "quant"].sum())

#         cur = _sum_for_eom(month)
#         prev_m = _sum_for_eom(last_month)
#         prev_y = _sum_for_eom(last_year)

#         orders_cur = _cnt_orders(month)
#         qty_cur = _qty(month)
#         aov = cur / max(orders_cur, 1)

#         no_order_cur = float(
#             df_store.loc[(df_store["eom"].eq(month)) & (df_store["orders_type"] == "Продажи без заказа"), "amount"].sum()
#         )
#         share_no_order = (no_order_cur / cur) if cur else 0.0

#         def _pct(a, b):
#             if b == 0:
#                 return None
#             return (a - b) / b

#         return {
#             "revenue_cur": cur,
#             "revenue_mom": _pct(cur, prev_m),
#             "revenue_yoy": _pct(cur, prev_y),
#             "orders_cur": orders_cur,
#             "aov": aov,
#             "qty_cur": qty_cur,
#             "share_no_order": share_no_order,
#         }

#     @staticmethod
#     def _make_slices(df_store: pd.DataFrame):
#         def _agg(col):
#             out = (
#                 df_store.groupby(col, dropna=False)["amount"]
#                 .sum()
#                 .sort_values(ascending=False)
#                 .reset_index()
#             )
#             out["amount"] = out["amount"].astype(float)
#             return out

#         return {
#             "orders_type": _agg("orders_type"),
#             "cat": _agg("cat").head(15),
#             "subcat": _agg("subcat").head(15),
#             "manager": _agg("manager").head(15),
#             "chanel": _agg("chanel").head(15),
#         }

#     # ----------------- UI building -----------------
#     def _header(self, store: str, month_name: str | None):
#         subtitle = month_name or "Выбранный месяц"
#         return dmc.Group(
#             [
#                 dmc.Stack([
#                     dmc.Title(f"Отчёт по магазину: {store}", order=2),
#                     dmc.Text(subtitle, size="sm", opacity=0.7),
#                 ], gap=2),
#                 dmc.Menu(
#                     shadow="md",
#                     position="bottom-end",
#                     withArrow=True,
#                     children=[
#                         dmc.MenuTarget(
#                             dmc.Button(
#                                 leftSection=DashIconify(icon="solar:download-linear"),
#                                 variant="light",
#                                 children="Экспорт",
#                             )
#                         ),
#                         dmc.MenuDropdown([
#                             dmc.MenuItem("Скачать PDF", leftSection=DashIconify(icon="solar:document-linear")),
#                             dmc.MenuItem("Скачать PNG", leftSection=DashIconify(icon="solar:image-linear")),
#                             dmc.MenuItem("Скачать Excel", leftSection=DashIconify(icon="mdi:file-excel")),
#                         ]),
#                     ],
#                 ),
#             ],
#             justify="space-between",
#             align="flex-start",
#         )

#     def _kpi_grid(self, kpis: dict):
#         def card(title: str, value: str, badge=None):
#             return dmc.Paper(
#                 withBorder=True,
#                 radius="xl",
#                 p="md",
#                 children=dmc.Stack([
#                     dmc.Text(title, size="sm", opacity=0.6),
#                     dmc.Text(value, fw=700, fz="xl"),
#                     badge if badge is not None else html.Div(),
#                 ], gap=4),
#             )

#         mom_badge = self._pct_badge(kpis.get("revenue_mom"), label="м/м")
#         yoy_badge = self._pct_badge(kpis.get("revenue_yoy"), label="г/г")
#         share_badge = dmc.Badge(f"{self._fmt_pct(kpis.get('share_no_order'))} без заказа", variant="light")

#         return dmc.SimpleGrid(
#             cols=4,
#             spacing="md",

#             children=[
#                 card("Выручка", self._fmt_cur(kpis.get("revenue_cur")), dmc.Group([mom_badge, yoy_badge], gap=6)),
#                 card("Кол-во заказов", f"{kpis.get('orders_cur', 0):,}".replace(",", " ")),
#                 card("Средний чек", self._fmt_cur(kpis.get("aov"))),
#                 card("Товары (qty)", f"{kpis.get('qty_cur', 0):,.0f}".replace(",", " "), share_badge),
#             ],
#         )

#     def _tabs(self, daily_stats: pd.DataFrame, daily_cum: pd.DataFrame, slices: dict, df_store: pd.DataFrame):
#         return dmc.Tabs(
#             value="overview",
#             children=[
#                 dmc.TabsList([
#                     dmc.TabsTab("Обзор", value="overview"),
#                     dmc.TabsTab("Динамика по дням", value="daily"),
#                     dmc.TabsTab("Кумулятивно", value="cum"),
#                     dmc.TabsTab("Разрезы", value="slices"),
#                     dmc.TabsTab("Таблица", value="table"),
#                 ]),
#                 dmc.TabsPanel(self._overview_block(slices), value="overview"),
#                 dmc.TabsPanel(self._daily_block(daily_stats), value="daily"),
#                 dmc.TabsPanel(self._cum_block(daily_cum), value="cum"),
#                 dmc.TabsPanel(self._slices_block(slices), value="slices"),
#                 dmc.TabsPanel(self._table_block(df_store), value="table"),
#             ],
#         )

#     # ----------- Tab panels -----------
#     def _overview_block(self, slices: dict):
#         # Мини-гистограмма orders_type
#         fig = self._bars_from_slice(slices["orders_type"], x="orders_type", title="Структура: заказы vs без заказа")
#         return dmc.Stack([
#             dmc.Text("Короткий обзор структуры", size="sm", opacity=0.7),
#             dcc.Graph(figure=fig, config={"displayModeBar": False}),
#         ])

#     def _daily_block(self, daily_stats: pd.DataFrame):
#         fig = self._daily_area(daily_stats, title="Динамика по дням (выручка)")
#         return dmc.Stack([
#             dmc.Text("Сумма по дням в выбранном месяце vs прошлые периоды", size="sm", opacity=0.7),
#             dcc.Graph(figure=fig, config={"displayModeBar": False}),
#         ])

#     def _cum_block(self, daily_cum: pd.DataFrame):
#         fig = self._daily_area(daily_cum, title="Кумулятивно с начала месяца")
#         return dmc.Stack([
#             dmc.Text("Нарастающим итогом", size="sm", opacity=0.7),
#             dcc.Graph(figure=fig, config={"displayModeBar": False}),
#         ])

#     def _slices_block(self, slices: dict):
#         # Несколько топов
#         blocks = []
#         for key, title in [("cat", "Топ категорий"), ("subcat", "Топ подкатегорий"), ("manager", "Топ менеджеров"), ("chanel", "Каналы продаж")]:
#             fig = self._bars_from_slice(slices[key], x=key, title=title, top_n=10)
#             blocks.append(dmc.Paper(withBorder=True, radius="lg", p="md", children=dcc.Graph(figure=fig, config={"displayModeBar": False})))

#         return dmc.SimpleGrid(cols=2, spacing="md",  children=blocks)

#     def _table_block(self, df_store: pd.DataFrame):
#         # Приведём к понятной таблице транзакций текущего месяца
#         eom = df_store["eom"].max()
#         tx = df_store.loc[df_store["eom"].eq(eom), [
#             "date",
#             "client_order_number",
#             "orders_type",
#             "cat",
#             "subcat",
#             "manager",
#             "chanel",
#             "quant",
#             "amount",
#         ]].copy()
#         tx["date"] = pd.to_datetime(tx["date"], errors="coerce").dt.strftime("%d.%m.%Y")

#         column_defs = [
#             {"headerName": "Дата", "field": "date", "sortable": True},
#             {"headerName": "№ заказа", "field": "client_order_number", "sortable": True},
#             {"headerName": "Тип", "field": "orders_type", "sortable": True},
#             {"headerName": "Категория", "field": "cat", "sortable": True},
#             {"headerName": "Подкатегория", "field": "subcat", "sortable": True},
#             {"headerName": "Менеджер", "field": "manager", "sortable": True},
#             {"headerName": "Канал", "field": "chanel", "sortable": True},
#             {"headerName": "Кол-во", "field": "quant", "type": "rightAligned"},
#             {"headerName": "Сумма", "field": "amount", "type": "rightAligned", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
#         ]

#         return dmc.Paper(
#             withBorder=True,
#             radius="lg",
#             p=0,
#             children=dag.AgGrid(
#                 id={"type": "store_area_chart_table", "index": "1"},
#                 rowData=tx.to_dict("records"),
#                 columnDefs=column_defs,
#                 defaultColDef={"resizable": True, "filter": True},
#                 dashGridOptions={"animateRows": True, "rowSelection": "single"},
#                 style={"height": "480px", "width": "100%"},
#             ),
#         )

#     # ----------------- Figures -----------------
#     @staticmethod
#     def _daily_area(daily_df: pd.DataFrame, title: str = ""):
#         # daily_df: columns -> ["day", eom1, eom2, eom3]
#         fig = go.Figure()
#         cols = [c for c in daily_df.columns if c != "day"]
#         for col in cols:
#             fig.add_trace(
#                 go.Scatter(
#                     x=daily_df["day"],
#                     y=daily_df[col],
#                     mode="lines",
#                     stackgroup=None,  # без накопления, чтобы сравнивать линии между периодами
#                     name=str(pd.to_datetime(col).date()) if "-" in str(col) else str(col),
#                 )
#             )
#         fig.update_layout(
#             title=title,
#             margin=dict(l=8, r=8, t=40, b=8),
#             xaxis_title="День",
#             yaxis_title="Сумма",
#             hovermode="x unified",
#         )
#         return fig

#     @staticmethod
#     def _bars_from_slice(df_slice: pd.DataFrame, x: str, title: str = "", top_n: int | None = None):
#         src = df_slice.copy()
#         if top_n:
#             src = src.head(top_n)
#         fig = go.Figure(go.Bar(x=src[x].astype(str), y=src["amount"]))
#         fig.update_layout(title=title, margin=dict(l=8, r=8, t=40, b=8), xaxis_title="", yaxis_title="Сумма")
#         return fig

#     # ----------------- Helpers -----------------
#     @staticmethod
#     def _fmt_cur(x: float | None):
#         if x is None:
#             return "—"
#         return f"{x:,.0f}".replace(",", " ")

#     @staticmethod
#     def _fmt_pct(x: float | None):
#         if x is None:
#             return "—"
#         return f"{x*100:.1f}%"

#     @staticmethod
#     def _pct_badge(x: float | None, label: str = ""):
#         color = "gray"
#         text = "—"
#         if x is not None:
#             if x > 0:
#                 color = "green"
#                 text = f"▲ {x*100:.1f}% {label}" if label else f"▲ {x*100:.1f}%"
#             elif x < 0:
#                 color = "red"
#                 text = f"▼ {abs(x)*100:.1f}% {label}" if label else f"▼ {abs(x)*100:.1f}%"
#             else:
#                 color = "gray"
#                 text = f"0.0% {label}" if label else "0.0%"
#         return dmc.Badge(text, color=color, variant="light")

#     @staticmethod
#     def _empty_state(text: str = "Данные отсутствуют"):
#         return dmc.Center(
#             dmc.Stack([
#                 DashIconify(icon="mdi:database-off", width=48),
#                 dmc.Text(text, size="sm", opacity=0.7),
#             ], align="center"),
#             mih=240,
#         )



# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from dash import dcc, html, no_update
# import dash_mantine_components as dmc
# import dash_ag_grid as dag
# from dash_iconify import DashIconify

# from components import (
#     ValuesRadioGroups,
#     DATES,
#     NoData,
#     BASE_COLORS,
#     COLORS_BY_COLOR,
#     COLORS_BY_SHADE,
#     InDevNotice,
# )
# from data import (
#     load_columns_df,
#     save_df_to_redis,
#     load_df_from_redis,
#     load_columns_dates,
#     COLS_DICT,
# )


# # ------------------------------------------------------------
# # StoreAreaChartModal — модальное окно «Отчёт по магазину»
# # ------------------------------------------------------------
# # Ключевые фичи:
# # 1) Красивый заголовок с месяцем/магазином, бейджами сравнения (м/м, г/г)
# # 2) KPI-карточки: Выручка, Заказы, Средний чек, Доля «без заказа»
# # 3) Табы: Обзор • Динамика по дням • Разрезы • Таблица операций
# # 4) Графики: Area (по дням), Area кумулятивно, столбцы «без заказа vs заказы»
# # 5) Экспорт: кнопка-меню (PDF/PNG/Excel) — только UI, без колбэков
# # 6) Стойкое поведение при пустых данных
# # ------------------------------------------------------------


# class StoreAreaChartModal:
#     def __init__(self, clickdata=None, clickSeriesName=None):
#         """
#         clickdata: dict с ключами 'eom' (YYYY-MM-DD), 'month_name' (строка), ...
#         clickSeriesName: имя магазина (seriesName из AreaChart)
#         """
#         self.clickdata = clickdata
#         self.clickSeriesName = clickSeriesName

#         # ID компонентов
#         self.modal_id = {"type": "store_area_chart_modal", "index": "1"}
#         self.container_id = {"type": "store_area_chart_modal_container", "index": "1"}

#     # --------------- Публичный API ---------------
#     def create_components(self):
#         return dmc.Modal(
#             id=self.modal_id,
#             size="90%",
#             withCloseButton=True,
#             children=[dmc.Container(id=self.container_id, fluid=True, px="md", py="md")],
#         )

#     def modal_children(self):
#         if not self.clickdata or not self.clickSeriesName:
#             return NoData().component

#         content = self._build_content()
#         return content

#     # ✅ Backward-compat: старое имя метода, чтобы не трогать твой колбэк
#     def update_modal(self):
#         return self.modal_children()

#     def registered_callbacks(self, app):
#         """Заглушка для последующих clientside/server колбэков при необходимости."""
#         pass

#     # --------------- Внутренняя логика ---------------
#     def _build_content(self):
#         # 1) Загрузка данных
#         month, last_month, last_year = self._derive_periods(self.clickdata)
#         df_current = self._load_base_df([month, last_month, last_year])

#         if df_current.empty:
#             return self._empty_state()

#         # 2) Фильтр по магазину
#         store = self.clickSeriesName
#         df_store = df_current.loc[df_current["store_gr_name"] == store].copy()
#         if df_store.empty:
#             return self._empty_state(f"Нет данных по магазину: {store}")

#         # 3) Подготовка дневных рядов
#         daily_stats, daily_cum = self._prepare_daily_series(df_store)

#         # 4) Метрики/KPI
#         kpis = self._compute_kpis(df_store, month, last_month, last_year)

#         # 5) Разрезы
#         slices = self._make_slices(df_store)

#         # 6) Компоненты UI
#         header = self._header(store, self.clickdata.get("month_name"))
#         kpi_grid = self._kpi_grid(kpis)
#         tabs = self._tabs(daily_stats, daily_cum, slices, df_store)

#         return dmc.Stack([header, kpi_grid, tabs], gap="md")

#     # ----------------- Data -----------------
#     @staticmethod
#     def _derive_periods(clickdata):
#         month = pd.to_datetime(clickdata.get("eom"), errors="coerce")
#         last_month = month + pd.offsets.MonthEnd(-1)
#         last_year = (month - pd.DateOffset(years=1)) + pd.offsets.MonthEnd(0)
#         return month, last_month, last_year

#     @staticmethod
#     def _load_base_df(dates):
#         COLS = [
#             "eom",
#             "date",
#             "dt",
#             "cr",
#             "amount",
#             "store_gr_name",
#             "chanel",
#             "manager",
#             "cat",
#             "subcat",
#             "client_order",
#             "quant",
#             "client_order_number",
#             "store_gr_name_amount_ytd",
#         ]
#         df = load_columns_dates(COLS, dates)
#         if df is None or len(df) == 0:
#             return pd.DataFrame(columns=COLS)

#         df["orders_type"] = np.where(
#             df["client_order"].eq("<Продажи без заказа>"),
#             "Продажи без заказа",
#             "Заказы клиента",
#         )
#         # date -> datetime, day
#         # приведение eom к Timestamp, чтобы сравнения работали корректно
#         df["eom"] = pd.to_datetime(df["eom"], errors="coerce")
#         df["eom"] = df["eom"] + pd.offsets.MonthEnd(0)
#         df["date"] = pd.to_datetime(df["date"], errors="coerce")
#         df["day"] = df["date"].dt.day
#         return df

#     @staticmethod
#     def _prepare_daily_series(df_store: pd.DataFrame):
#         # Сумма по дням для каждого периода (eom)
#         daily = (
#             df_store.pivot_table(index="day", columns="eom", values="amount", aggfunc="sum")
#             .fillna(0)
#             .sort_index()
#         )
#         # Кумулятив
#         daily_cum = daily.cumsum()
#         # Для удобного отображения в Markdown/таблицах — сбросить индекс
#         daily_stats = daily.reset_index()
#         daily_cum_stats = daily_cum.reset_index()
#         return daily_stats, daily_cum_stats

#     @staticmethod
#     def _compute_kpis(df_store: pd.DataFrame, month, last_month, last_year):
#         def _sum_for_eom(eom):
#             return float(df_store.loc[df_store["eom"].eq(eom), "amount"].sum())

#         def _cnt_orders(eom):
#             return int(df_store.loc[df_store["eom"].eq(eom) & df_store["client_order_number"].notna(), "client_order_number"].nunique())

#         def _qty(eom):
#             return float(df_store.loc[df_store["eom"].eq(eom), "quant"].sum())

#         cur = _sum_for_eom(month)
#         prev_m = _sum_for_eom(last_month)
#         prev_y = _sum_for_eom(last_year)

#         orders_cur = _cnt_orders(month)
#         qty_cur = _qty(month)
#         aov = cur / max(orders_cur, 1)

#         no_order_cur = float(
#             df_store.loc[(df_store["eom"].eq(month)) & (df_store["orders_type"] == "Продажи без заказа"), "amount"].sum()
#         )
#         share_no_order = (no_order_cur / cur) if cur else 0.0

#         def _pct(a, b):
#             if b == 0:
#                 return None
#             return (a - b) / b

#         return {
#             "revenue_cur": cur,
#             "revenue_mom": _pct(cur, prev_m),
#             "revenue_yoy": _pct(cur, prev_y),
#             "orders_cur": orders_cur,
#             "aov": aov,
#             "qty_cur": qty_cur,
#             "share_no_order": share_no_order,
#         }

#     @staticmethod
#     def _make_slices(df_store: pd.DataFrame):
#         def _agg(col):
#             out = (
#                 df_store.groupby(col, dropna=False)["amount"]
#                 .sum()
#                 .sort_values(ascending=False)
#                 .reset_index()
#             )
#             out["amount"] = out["amount"].astype(float)
#             return out

#         return {
#             "orders_type": _agg("orders_type"),
#             "cat": _agg("cat").head(15),
#             "subcat": _agg("subcat").head(15),
#             "manager": _agg("manager").head(15),
#             "chanel": _agg("chanel").head(15),
#         }

#     # ----------------- UI building -----------------
#     def _header(self, store: str, month_name: str | None):
#         subtitle = month_name or "Выбранный месяц"
#         # Убрали экспорт; только заголовок и подзаголовок
#         return dmc.Group(
#             [
#                 dmc.Stack([
#                     dmc.Title(f"Отчёт по магазину: {store}", order=2),
#                     dmc.Text(subtitle, size="sm", opacity=0.7),
#                 ], gap=2),
#             ],
#             justify="space-between",
#             align="flex-start",
#         )
#         subtitle = month_name or "Выбранный месяц"
#         return dmc.Group(
#             [
#                 dmc.Stack([
#                     dmc.Title(f"Отчёт по магазину: {store}", order=2),
#                     dmc.Text(subtitle, size="sm", opacity=0.7),
#                 ], gap=2),
#                 dmc.Menu(
#                     shadow="md",
#                     position="bottom-end",
#                     withArrow=True,
#                     children=[
#                         dmc.MenuTarget(
#                             dmc.Button(
#                                 leftSection=DashIconify(icon="solar:download-linear"),
#                                 variant="light",
#                                 children="Экспорт",
#                             )
#                         ),
#                         dmc.MenuDropdown([
#                             dmc.MenuItem("Скачать PDF", leftSection=DashIconify(icon="solar:document-linear")),
#                             dmc.MenuItem("Скачать PNG", leftSection=DashIconify(icon="solar:image-linear")),
#                             dmc.MenuItem("Скачать Excel", leftSection=DashIconify(icon="mdi:file-excel")),
#                         ]),
#                     ],
#                 ),
#             ],
#             justify="space-between",
#             align="flex-start",
#         )

#     def _kpi_grid(self, kpis: dict):
#         def card(title: str, value: str, badge=None):
#             return dmc.Paper(
#                 withBorder=True,
#                 radius=0,  # прямые углы
#                 p="md",
#                 children=dmc.Stack([
#                     dmc.Text(title, size="sm", opacity=0.6),
#                     dmc.Text(value, fw=700, fz="xl"),
#                     badge if badge is not None else html.Div(),
#                 ], gap=4),
#             )

#         mom_badge = self._pct_badge(kpis.get("revenue_mom"), label="м/м")
#         yoy_badge = self._pct_badge(kpis.get("revenue_yoy"), label="г/г")
#         share_badge = dmc.Badge(f"{self._fmt_pct(kpis.get('share_no_order'))} без заказа", variant="light")

#         return dmc.SimpleGrid(
#             cols=4,
#             spacing="md",

#             children=[
#                 card("Выручка", self._fmt_cur(kpis.get("revenue_cur")), dmc.Group([mom_badge, yoy_badge], gap=6)),
#                 card("Кол-во заказов", f"{kpis.get('orders_cur', 0):,}".replace(",", " ")),
#                 card("Средний чек", self._fmt_cur(kpis.get("aov"))),
#                 card("Товары (qty)", f"{kpis.get('qty_cur', 0):,.0f}".replace(",", " "), share_badge),
#             ],
#         )

#     def _tabs(self, daily_stats: pd.DataFrame, daily_cum: pd.DataFrame, slices: dict, df_store: pd.DataFrame):
#         return dmc.Tabs(
#             value="overview",
#             children=[
#                 dmc.TabsList([
#                     dmc.TabsTab("Обзор", value="overview"),
#                     dmc.TabsTab("Динамика по дням", value="daily"),
#                     dmc.TabsTab("Кумулятивно", value="cum"),
#                     dmc.TabsTab("Разрезы", value="slices"),
#                     dmc.TabsTab("Таблица", value="table"),
#                 ]),
#                 dmc.TabsPanel(self._overview_block(slices), value="overview"),
#                 dmc.TabsPanel(self._daily_block(daily_stats), value="daily"),
#                 dmc.TabsPanel(self._cum_block(daily_cum), value="cum"),
#                 dmc.TabsPanel(self._slices_block(slices), value="slices"),
#                 dmc.TabsPanel(self._table_block(df_store), value="table"),
#             ],
#         )

#     # ----------- Tab panels -----------
#     def _overview_block(self, slices: dict):
#         # Мини-гистограмма orders_type
#         fig = self._bars_from_slice(slices["orders_type"], x="orders_type", title="Структура: заказы vs без заказа")
#         return dmc.Stack([
#             dmc.Text("Короткий обзор структуры", size="sm", opacity=0.7),
#             dcc.Graph(figure=fig, config={"displayModeBar": False}),
#         ])

#     def _daily_block(self, daily_stats: pd.DataFrame):
#         fig = self._daily_area(daily_stats, title="Динамика по дням (выручка)")
#         return dmc.Stack([
#             dmc.Text("Сумма по дням в выбранном месяце vs прошлые периоды", size="sm", opacity=0.7),
#             dcc.Graph(figure=fig, config={"displayModeBar": False}),
#         ])

#     def _cum_block(self, daily_cum: pd.DataFrame):
#         fig = self._daily_area(daily_cum, title="Кумулятивно с начала месяца")
#         return dmc.Stack([
#             dmc.Text("Нарастающим итогом", size="sm", opacity=0.7),
#             dcc.Graph(figure=fig, config={"displayModeBar": False}),
#         ])

#     def _slices_block(self, slices: dict):
#         # Несколько топов
#         blocks = []
#         for key, title in [("cat", "Топ категорий"), ("subcat", "Топ подкатегорий"), ("manager", "Топ менеджеров"), ("chanel", "Каналы продаж")]:
#             fig = self._bars_from_slice(slices[key], x=key, title=title, top_n=10)
#             blocks.append(dmc.Paper(withBorder=True, radius="lg", p="md", children=dcc.Graph(figure=fig, config={"displayModeBar": False})))

#         return dmc.SimpleGrid(cols=2, spacing="md",  children=blocks)

#     def _table_block(self, df_store: pd.DataFrame):
#         # Приведём к понятной таблице транзакций текущего месяца
#         eom = df_store["eom"].max()
#         tx = df_store.loc[df_store["eom"].eq(eom), [
#             "date",
#             "client_order_number",
#             "orders_type",
#             "cat",
#             "subcat",
#             "manager",
#             "chanel",
#             "quant",
#             "amount",
#         ]].copy()
#         tx["date"] = pd.to_datetime(tx["date"], errors="coerce").dt.strftime("%d.%m.%Y")

#         column_defs = [
#             {"headerName": "Дата", "field": "date", "sortable": True},
#             {"headerName": "№ заказа", "field": "client_order_number", "sortable": True},
#             {"headerName": "Тип", "field": "orders_type", "sortable": True},
#             {"headerName": "Категория", "field": "cat", "sortable": True},
#             {"headerName": "Подкатегория", "field": "subcat", "sortable": True},
#             {"headerName": "Менеджер", "field": "manager", "sortable": True},
#             {"headerName": "Канал", "field": "chanel", "sortable": True},
#             {"headerName": "Кол-во", "field": "quant", "type": "rightAligned"},
#             {"headerName": "Сумма", "field": "amount", "type": "rightAligned", "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
#         ]

#         return dmc.Paper(
#             withBorder=True,
#             radius="lg",
#             p=0,
#             children=dag.AgGrid(
#                 id={"type": "store_area_chart_table", "index": "1"},
#                 rowData=tx.to_dict("records"),
#                 columnDefs=column_defs,
#                 defaultColDef={"resizable": True, "filter": True},
#                 dashGridOptions={"animateRows": True, "rowSelection": "single"},
#                 style={"height": "480px", "width": "100%"},
#             ),
#         )

#     # ----------------- Figures -----------------
#     @staticmethod
#     def _daily_area(daily_df: pd.DataFrame, title: str = ""):
#         # daily_df: columns -> ["day", eom1, eom2, eom3]
#         fig = go.Figure()
#         cols = [c for c in daily_df.columns if c != "day"]
#         for col in cols:
#             name = str(pd.to_datetime(col).strftime('%b %y')) if pd.notna(col) else str(col)
#             fig.add_trace(
#                 go.Scatter(
#                     x=daily_df["day"],
#                     y=daily_df[col],
#                     mode="lines",
#                     stackgroup=None,
#                     name=name,
#                 )
#             )
#         fig.update_layout(
#             title=title,
#             margin=dict(l=8, r=8, t=40, b=8),
#             xaxis_title="День",
#             yaxis_title="Сумма",
#             hovermode="x unified",
#             legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
#         )
#         return fig

#     @staticmethod
#     def _bars_from_slice(df_slice: pd.DataFrame, x: str, title: str = "", top_n: int | None = None):
#         src = df_slice.copy()
#         if top_n:
#             src = src.head(top_n)
#         fig = go.Figure(go.Bar(x=src[x].astype(str), y=src["amount"]))
#         fig.update_layout(title=title, margin=dict(l=8, r=8, t=40, b=8), xaxis_title="", yaxis_title="Сумма")
#         return fig

#     # ----------------- Helpers -----------------
#     @staticmethod
#     def _fmt_cur(x: float | None):
#         if x is None:
#             return "—"
#         return f"{x:,.0f}".replace(",", " ")

#     @staticmethod
#     def _fmt_pct(x: float | None):
#         if x is None:
#             return "—"
#         return f"{x*100:.1f}%"

#     @staticmethod
#     def _pct_badge(x: float | None, label: str = ""):
#         color = "gray"
#         text = "—"
#         if x is not None:
#             if x > 0:
#                 color = "green"
#                 text = f"▲ {x*100:.1f}% {label}" if label else f"▲ {x*100:.1f}%"
#             elif x < 0:
#                 color = "red"
#                 text = f"▼ {abs(x)*100:.1f}% {label}" if label else f"▼ {abs(x)*100:.1f}%"
#             else:
#                 color = "gray"
#                 text = f"0.0% {label}" if label else "0.0%"
#         return dmc.Badge(text, color=color, variant="light")

#     @staticmethod
#     def _empty_state(text: str = "Данные отсутствуют"):
#         return dmc.Center(
#             dmc.Stack([
#                 DashIconify(icon="mdi:database-off", width=48),
#                 dmc.Text(text, size="sm", opacity=0.7),
#             ], align="center"),
#             mih=240,
#         )




import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, no_update
import dash_mantine_components as dmc
import dash_ag_grid as dag
from dash_iconify import DashIconify

from components import (
    ValuesRadioGroups,
    DATES,
    NoData,
    BASE_COLORS,
    COLORS_BY_COLOR,
    COLORS_BY_SHADE,
    InDevNotice,
)
from data import (
    load_columns_df,
    save_df_to_redis,
    load_df_from_redis,
    load_columns_dates,
    COLS_DICT,
)


class StoreAreaChartModal:
    def __init__(self, clickdata=None, clickSeriesName=None):

        self.clickdata = clickdata
        self.clickSeriesName = clickSeriesName

        # ID компонентов (внешний контейнер и ВНУТРЕННИЙ контейнер)
        self.modal_id = {"type": "store_area_chart_modal", "index": "1"}
        self.container_id = {"type": "store_area_chart_modal_container", "index": "1"}  # внешний (его обновляет твой колбэк)
        self.inner_container_id = {"type": "store_area_chart_modal_inner", "index": "1"}  # внутренний (его обновляет класс)
        self.metric_id = {"type": "store_area_chart_metric", "index": "1"}

    # --------------- Публичный API ---------------
    def create_components(self):
        return dmc.Modal(
            id=self.modal_id,
            size="90%",
            withCloseButton=True,
            children=[dmc.Container(id=self.container_id, fluid=True, px="md", py="md")],
        )

    def modal_children(self):
        if not self.clickdata or not self.clickSeriesName:
            return dmc.Box(id=self.inner_container_id, children=NoData().component)
        content = self._build_content()
        return dmc.Box(id=self.inner_container_id, children=content)

    def update_modal(self):
        return self.modal_children()

    def registered_callbacks(self, app):
        """
        Колбэк для переключения метрики (amount/quant/aov).
        ВАЖНО: обновляем ТОЛЬКО inner_container, чтобы не пересекаться с внешним колбэком.
        """
        from dash import Input, Output, State, no_update



    # --------------- Внутренняя логика ---------------
    def _build_content(self, metric: str = "amount"):
        # 1) Загрузка данных
        month, last_month, last_year = self._derive_periods(self.clickdata)
        df_current = self._load_base_df([month, last_month, last_year])

        if df_current.empty:
            return self._empty_state()

        # 2) Фильтр по магазину
        store = self.clickSeriesName
        df_store = df_current.loc[df_current["store_gr_name"] == store].copy()
        if df_store.empty:
            return self._empty_state(f"Нет данных по магазину: {store}")

        daily_stats, daily_cum = self._prepare_daily_series(df_store, metric=metric)

        # 5) Разрезы (всегда по amount — для структуры выручки)
        slices = self._make_slices(df_store)

        header = self._header(store, self.clickdata.get("month_name"), metric)

        tabs = self._tabs(daily_stats, daily_cum, slices, df_store, month, metric)


        return dmc.Stack([header,  tabs], gap="md")

    # ----------------- Data -----------------
    @staticmethod
    def _derive_periods(clickdata):
        month = pd.to_datetime(clickdata.get("eom"), errors="coerce")
        last_month = month + pd.offsets.MonthEnd(-1)
        last_year = (month - pd.DateOffset(years=1)) + pd.offsets.MonthEnd(0)
        return month, last_month, last_year

    @staticmethod
    def _load_base_df(dates):
        COLS = [
            "eom",
            "date",
            "dt",
            "cr",
            "amount",
            "store_gr_name",
            "chanel",
            "manager",
            "cat",
            "subcat",
            "client_order",
            "quant",
            "client_order_number",
            "store_gr_name_amount_ytd",
            "manu",
            "brend"
            
        ]
        df = load_columns_dates(COLS, dates)
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=COLS)

        df["orders_type"] = np.where(
            df["client_order"].eq("<Продажи без заказа>"),
            "Продажи без заказа",
            "Заказы клиента",
        )
        df["eom"] = pd.to_datetime(df["eom"], errors="coerce") + pd.offsets.MonthEnd(0)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["day"] = df["date"].dt.day
        return df

    @staticmethod
    def _prepare_daily_series(df_store: pd.DataFrame, metric: str = "amount"):
        col = "amount" if metric in ("amount", "aov") else "quant"
        daily = (
            df_store.pivot_table(index="day", columns="eom", values=col, aggfunc="sum")
            .fillna(0)
            .sort_index()
        )
        daily_cum = daily.cumsum()
        daily_stats = daily.reset_index()
        daily_cum_stats = daily_cum.reset_index()
        return daily_stats, daily_cum_stats

    @staticmethod
    def _compute_kpis(df_store: pd.DataFrame, month, last_month, last_year, metric: str = "amount"):
        def _sum_for_eom(eom, col="amount"):
            return float(df_store.loc[df_store["eom"].eq(eom), col].sum())

        def _cnt_orders(eom):
            return int(
                df_store.loc[
                    df_store["eom"].eq(eom) & df_store["client_order_number"].notna(),
                    "client_order_number",
                ].nunique()
            )

        def _qty(eom):
            return float(df_store.loc[df_store["eom"].eq(eom), "quant"].sum())

        cur_amount = _sum_for_eom(month, "amount")
        prev_m_amount = _sum_for_eom(last_month, "amount")
        prev_y_amount = _sum_for_eom(last_year, "amount")

        orders_cur = _cnt_orders(month)
        qty_cur = _qty(month)

        if metric == "quant":
            cur = _sum_for_eom(month, "quant")
            prev_m = _sum_for_eom(last_month, "quant")
            prev_y = _sum_for_eom(last_year, "quant")
        elif metric == "aov":
            cur = cur_amount / max(orders_cur, 1)
            prev_m = (prev_m_amount / max(_cnt_orders(last_month), 1)) if prev_m_amount else 0
            prev_y = (prev_y_amount / max(_cnt_orders(last_year), 1)) if prev_y_amount else 0
        else:
            cur = cur_amount
            prev_m = prev_m_amount
            prev_y = prev_y_amount

        no_order_cur = float(
            df_store.loc[
                (df_store["eom"].eq(month)) & (df_store["orders_type"] == "Продажи без заказа"),
                "amount",
            ].sum()
        )
        share_no_order = (no_order_cur / cur_amount) if cur_amount else 0.0

        def _pct(a, b):
            if b == 0:
                return None
            return (a - b) / b

        return {
            "metric_label": {"amount": "Выручка", "quant": "Кол-во", "aov": "Средний чек"}[metric],
            "metric_value": cur,
            "metric_mom": _pct(cur, prev_m),
            "metric_yoy": _pct(cur, prev_y),
            "orders_cur": orders_cur,
            "aov": cur_amount / max(orders_cur, 1),
            "qty_cur": qty_cur,
            "share_no_order": share_no_order,
        }

    @staticmethod
    def _make_slices(df_store: pd.DataFrame):
        def _agg(col):
            out = (
                df_store.groupby(col, dropna=False)["amount"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            out["amount"] = out["amount"].astype(float)
            return out

        return {
            "orders_type": _agg("orders_type"),
            "cat": _agg("cat").head(15),
            "subcat": _agg("subcat").head(15),
            "manager": _agg("manager").head(15),
            "chanel": _agg("chanel").head(15),
        }

    # ----------------- UI building -----------------
    def _header(self, store: str, month_name: str | None, metric: str):
        subtitle = month_name or "Выбранный месяц"
        return dmc.Group(
            [
                dmc.Stack(
                    [
                        dmc.Title(f"Отчёт по магазину: {store}", order=2),
                        dmc.Badge(subtitle,size="lg",
	                                    radius="xs", variant="light",),
                    ],
                    gap=2,
                ),
                
            ],
            justify="space-between",
            align="center",
        )



    def _tabs(self, daily_stats: pd.DataFrame, _daily_cum_unused: pd.DataFrame, slices: dict, df_store: pd.DataFrame, month, metric: str):

        return dmc.Tabs(
            value="overview",
            children=[
                dmc.TabsList(
                    [
                        dmc.TabsTab("Обзор", value="overview"),
                        dmc.TabsTab("Кумулятивно", value="cum"),
                        dmc.TabsTab("Разрезы", value="slices"),
                        dmc.TabsTab("Тёпловая карта", value="heat"),
                        dmc.TabsTab("Таблица", value="table"),
                    ]
                ),

                dmc.TabsPanel(self._overview_block(df_store, month), value="overview"),
                dmc.TabsPanel(self._cum_block(daily_stats, month, metric), value="cum"),
                dmc.TabsPanel(self._slices_block(slices), value="slices"),
                dmc.TabsPanel(self._heatmap_block(df_store, month, metric), value="heat"),
                dmc.TabsPanel(self._table_block(df_store), value="table"),
            ],
        )

    # ----------- Tab panels -----------
#     def _overview_block(self, df_store: pd.DataFrame, month):
#         # ===== Подготовка периода и данных =====
#         period_label = self._period_label(month)
#         cur = df_store.loc[df_store["eom"].eq(month)].copy()
#         if cur.empty:
#             return dmc.Stack(
#                 [dmc.Text(f"Обзор за {period_label}", size="sm", opacity=0.7), self._empty_state("Нет данных за выбранный период")],
#                 gap="sm",
#             )

#         # прошлые периоды для MoM/YoY
#         last_month = month + pd.offsets.MonthEnd(-1)
#         last_year  = (month - pd.DateOffset(years=1)) + pd.offsets.MonthEnd(0)

#         has_dt = "dt" in cur.columns
#         has_cr = "cr" in cur.columns
#         has_amount = "amount" in cur.columns

#         def _sum_for(eom, col):
#             src = df_store.loc[df_store["eom"].eq(eom)]
#             return float(src[col].sum()) if col in src.columns else 0.0

#         # продажи/возвраты/чистая
#         sales_cur   = _sum_for(month, "dt") if has_dt else float(cur.loc[cur["amount"] > 0, "amount"].sum())
#         returns_cur = _sum_for(month, "cr") if has_cr else float(abs(cur.loc[cur["amount"] < 0, "amount"].sum()))
#         net_cur     = _sum_for(month, "amount") if has_amount else sales_cur - returns_cur

#         sales_pm   = _sum_for(last_month, "dt") if has_dt else float(df_store.loc[(df_store["eom"].eq(last_month)) & (df_store["amount"] > 0), "amount"].sum())
#         returns_pm = _sum_for(last_month, "cr") if has_cr else float(abs(df_store.loc[(df_store["eom"].eq(last_month)) & (df_store["amount"] < 0), "amount"].sum()))
#         net_pm     = _sum_for(last_month, "amount") if has_amount else sales_pm - returns_pm

#         sales_py   = _sum_for(last_year, "dt") if has_dt else float(df_store.loc[(df_store["eom"].eq(last_year)) & (df_store["amount"] > 0), "amount"].sum())
#         returns_py = _sum_for(last_year, "cr") if has_cr else float(abs(df_store.loc[(df_store["eom"].eq(last_year)) & (df_store["amount"] < 0), "amount"].sum()))
#         net_py     = _sum_for(last_year, "amount") if has_amount else sales_py - returns_py

#         def _pct(a, b):
#             if b == 0:
#                 return None
#             return (a - b) / b

#         # заказы и чеки
#         orders = cur.loc[cur["client_order_number"].notna()].copy()
#         if orders.empty:
#             aov = median_check = max_check = 0.0
#             orders_cnt = 0
#         else:
#             order_checks = orders.groupby("client_order_number", dropna=True)["amount"].sum().astype(float)
#             order_checks = order_checks[order_checks > 0]
#             orders_cnt = int(order_checks.shape[0])
#             aov = float(order_checks.mean()) if orders_cnt else 0.0
#             median_check = float(order_checks.median()) if orders_cnt else 0.0
#             max_check = float(order_checks.max()) if orders_cnt else 0.0

#         # AOV динамика
#         def _aov_for(eom):
#             src = df_store.loc[df_store["eom"].eq(eom)]
#             if src.empty:
#                 return 0.0
#             oc = src.loc[src["client_order_number"].notna()].groupby("client_order_number")["amount"].sum()
#             oc = oc[oc > 0]
#             return float(oc.mean()) if not oc.empty else 0.0

#         aov_pm = _aov_for(last_month)
#         aov_py = _aov_for(last_year)

#         # доли
#         returns_share = (returns_cur / sales_cur) if sales_cur else 0.0
#         no_order_share = 0.0
#         if "orders_type" in cur.columns and has_amount:
#             no_order_amt = float(cur.loc[cur["orders_type"].eq("Продажи без заказа"), "amount"].clip(lower=0).sum())
#             no_order_share = (no_order_amt / sales_cur) if sales_cur else 0.0

#                 # === подготовка датафрейма для возвратов (оставляем как у тебя) ===
#         return_df = cur.copy()
#         if has_cr:
#             return_df["return_value"] = return_df["cr"].astype(float)
#         else:
#             return_df["return_value"] = return_df["amount"].where(return_df["amount"] < 0, 0).abs().astype(float)

#         def _arrow(val):
#             if val is None:
#                 return "—"
#             if val > 0:
#                 return f'<span style="color:#2ecc71;">⬆️ {val*100:.1f}%</span>'
#             if val < 0:
#                 return f'<span style="color:#e74c3c;">⬇️ {abs(val)*100:.1f}%</span>'
#             return '<span style="color:gray;">0.0%</span>'

#         # === helper: таблица возвратов по одному разрезу ===
#         def _make_returns_table(col: str, title: str, top_n: int = 15):
#             if col not in return_df.columns:
#                 return dmc.Alert(f"Нет данных по: {title}", color="gray", variant="light")

#             grp = (
#                 return_df.groupby(col, dropna=False)["return_value"]
#                 .sum()
#                 .sort_values(ascending=False)
#             )
#             grp = grp[grp > 0].head(top_n)

#             if grp.empty:
#                 return dmc.Alert(f"— по «{title}» нет возвратов", color="gray", variant="light")

#             total_returns = float(returns_cur) if returns_cur else 0.0

#             thead = dmc.TableThead(
#                 dmc.TableTr([
#                     dmc.TableTh("№"),
#                     dmc.TableTh(title),
#                     dmc.TableTh("Возвраты, ₽"),
#                     dmc.TableTh("Доля"),
#                 ])
#             )

#             rows = []
#             for i, (k, v) in enumerate(grp.items(), start=1):
#                 name = str(k) if pd.notna(k) else "—"
#                 share = (v / total_returns) if total_returns else 0.0
#                 rows.append(
#                     dmc.TableTr([
#                         dmc.TableTd(f"{i}"),
#                         dmc.TableTd(name),
#                         dmc.TableTd(self._fmt_cur(float(v))),
#                         dmc.TableTd(self._fmt_pct(float(share))),
#                     ])
#                 )

#             table = dmc.Table(
#                 children=[thead, dmc.TableTbody(rows)],
#                 striped=True,
#                 highlightOnHover=True,
#                 horizontalSpacing="sm",
#                 verticalSpacing="xs",
#                 withTableBorder=True,     # если в твоей версии нет — замени на withBorder=True
#                 withColumnBorders=False,  # при необходимости убери
                
#             )
#             return table

#         # === вкладки с таблицами по возвратам ===
#         returns_tabs = dmc.Tabs(
#             [
#                 dmc.TabsList([
#                     dmc.TabsTab("Категории", value="cat"),
#                     dmc.TabsTab("Производители", value="manu"),
#                     dmc.TabsTab("Бренды", value="brand"),
#                 ]),
#                 dmc.TabsPanel(_make_returns_table("cat", "Категория"), value="cat"),
#                 dmc.TabsPanel(_make_returns_table("manu", "Производитель"), value="manu"),
#                 dmc.TabsPanel(
#                     _make_returns_table("brand", "Бренд") if "brand" in return_df.columns
#                     else _make_returns_table("brend", "Бренд"),
#                     value="brand",
#                 ),
#             ],
#             value="cat",
#             color="pink",      # поставь корпоративный цвет при желании
#             variant="pills",   # можно "outline"
#             keepMounted=False, # чтобы не считать невидимые вкладки
#         )

#         # === markdown без списков возвратов ===
#         md = f"""
# __Обзор за период {period_label} по магазину {self.clickSeriesName}__

# - **Продажи:** {self._fmt_cur(sales_cur)} руб.
# - **Возвраты:** {self._fmt_cur(returns_cur)} руб.
# - **Чистая выручка:** **{self._fmt_cur(net_cur)} руб.**  
# MoM: {_arrow(_pct(net_cur, net_pm))} · YoY: {_arrow(_pct(net_cur, net_py))}
# - **Кол-во заказов:** {orders_cnt:,} шт.
# - **Средний чек:** {self._fmt_cur(aov)} · MoM: {_arrow(_pct(aov, aov_pm))} · YoY: {_arrow(_pct(aov, aov_py))}
# - **Медианный / Макс чек:** {self._fmt_cur(median_check)} руб. / {self._fmt_cur(max_check)} руб.
# - **Доля продаж без заказа:** {self._fmt_pct(no_order_share)}
# - **Доля возвратов:** {self._fmt_pct(returns_share)}
# """

#         # === финальный return ===
#         return dmc.Stack(
#             [
#                 dcc.Markdown(md, dangerously_allow_html=True),
#                 dmc.Divider(variant="dashed"),
#                 dmc.Title("Возвраты", order=4),
#                 returns_tabs,
#             ],
#             gap="sm",
#         )


    def _overview_block(self, df_store: pd.DataFrame, month):
        # ===== Подготовка периода и данных =====
        period_label = self._period_label(month)
        cur = df_store.loc[df_store["eom"].eq(month)].copy()
        if cur.empty:
            return dmc.Stack(
                [dmc.Text(f"Обзор за {period_label}", size="sm", opacity=0.7), self._empty_state("Нет данных за выбранный период")],
                gap="sm",
            )

        # прошлые периоды для MoM/YoY
        last_month = month + pd.offsets.MonthEnd(-1)
        last_year  = (month - pd.DateOffset(years=1)) + pd.offsets.MonthEnd(0)

        has_dt = "dt" in cur.columns
        has_cr = "cr" in cur.columns
        has_amount = "amount" in cur.columns

        def _sum_for(eom, col):
            src = df_store.loc[df_store["eom"].eq(eom)]
            return float(src[col].sum()) if col in src.columns else 0.0

        # продажи/возвраты/чистая
        sales_cur   = _sum_for(month, "dt") if has_dt else float(cur.loc[cur["amount"] > 0, "amount"].sum())
        returns_cur = _sum_for(month, "cr") if has_cr else float(abs(cur.loc[cur["amount"] < 0, "amount"].sum()))
        net_cur     = _sum_for(month, "amount") if has_amount else sales_cur - returns_cur

        sales_pm   = _sum_for(last_month, "dt") if has_dt else float(df_store.loc[(df_store["eom"].eq(last_month)) & (df_store["amount"] > 0), "amount"].sum())
        returns_pm = _sum_for(last_month, "cr") if has_cr else float(abs(df_store.loc[(df_store["eom"].eq(last_month)) & (df_store["amount"] < 0), "amount"].sum()))
        net_pm     = _sum_for(last_month, "amount") if has_amount else sales_pm - returns_pm

        sales_py   = _sum_for(last_year, "dt") if has_dt else float(df_store.loc[(df_store["eom"].eq(last_year)) & (df_store["amount"] > 0), "amount"].sum())
        returns_py = _sum_for(last_year, "cr") if has_cr else float(abs(df_store.loc[(df_store["eom"].eq(last_year)) & (df_store["amount"] < 0), "amount"].sum()))
        net_py     = _sum_for(last_year, "amount") if has_amount else sales_py - returns_py

        def _pct(a, b):
            if b == 0:
                return None
            return (a - b) / b

        # заказы и чеки
        orders = cur.loc[cur["client_order_number"].notna()].copy()
        if orders.empty:
            aov = median_check = max_check = 0.0
            orders_cnt = 0
        else:
            order_checks = orders.groupby("client_order_number", dropna=True)["dt"].sum().astype(float)
            order_checks = order_checks[order_checks > 0]
            orders_cnt = int(order_checks.shape[0])
            aov = float(order_checks.mean()) if orders_cnt else 0.0
            median_check = float(order_checks.median()) if orders_cnt else 0.0
            max_check = float(order_checks.max()) if orders_cnt else 0.0

        # AOV динамика
        def _aov_for(eom):
            src = df_store.loc[df_store["eom"].eq(eom)]
            if src.empty:
                return 0.0
            oc = src.loc[src["client_order_number"].notna()].groupby("client_order_number")["amount"].sum()
            oc = oc[oc > 0]
            return float(oc.mean()) if not oc.empty else 0.0

        aov_pm = _aov_for(last_month)
        aov_py = _aov_for(last_year)

        # доли
        returns_share = (returns_cur / sales_cur) if sales_cur else 0.0
        no_order_share = 0.0
        if "orders_type" in cur.columns and has_amount:
            no_order_amt = float(cur.loc[cur["orders_type"].eq("Продажи без заказа"), "amount"].clip(lower=0).sum())
            no_order_share = (no_order_amt / sales_cur) if sales_cur else 0.0

        # === подготовка датафрейма для возвратов ===
        return_df = cur.copy()
        if has_cr:
            return_df["return_value"] = return_df["cr"].astype(float)
        else:
            return_df["return_value"] = return_df["amount"].where(return_df["amount"] < 0, 0).abs().astype(float)

        # ===== UI helpers =====
        def _delta_badge(val):
            if val is None:
                return dmc.Badge("—", color="gray", variant="light",  radius="xs", )
            txt = f"{val*100:+.1f}%"
            if val > 0:
                return dmc.Badge(txt, color="green", variant="filled", radius="xs", leftSection=DashIconify(icon="mdi:trending-up", width=16,   ))
            if val < 0:
                return dmc.Badge(txt, color="red", variant="filled",radius="xs", leftSection=DashIconify(icon="mdi:trending-down", width=16, ))
            return dmc.Badge("0.0%", color="gray", variant="light", radius="xs",)

        def _kpi_card(title: str, value: str, subtitle=None, right=None):
            return dmc.Paper(
                withBorder=True, radius="md", p="md",
                children=dmc.Stack(
                    [
                        dmc.Group([dmc.Text(title, size="sm", c="dimmed")], justify="space-between"),
                        dmc.Group(
                            [dmc.Text(value, fw=700, size="lg"), right] if right else [dmc.Text(value, fw=700, size="lg")],
                            justify="space-between", align="center"
                        ),
                        subtitle if subtitle else dmc.Space(h=0),
                    ],
                    gap=4,
                ),
            )

        # ===== KPI header =====
        kpi_grid = dmc.SimpleGrid(
            cols={ "base": 1, "sm": 2, "lg": 4 },
            spacing="md",
            children=[
                _kpi_card(
                    "Продажи", self._fmt_cur(sales_cur),
                    
                ),
                _kpi_card("Возвраты", self._fmt_cur(returns_cur),
                        subtitle=dmc.Text(f"Доля возвратов: {self._fmt_pct(returns_share)}", size="xs", c="dimmed")),
                _kpi_card("Чистая выручка", self._fmt_cur(net_cur),
                         right=dmc.Stack([dmc.Text("MoM/YoY по чистой", size="xs", c="dimmed"),
                                    dmc.Group([_delta_badge(_pct(net_cur, net_pm)), _delta_badge(_pct(net_cur, net_py))], gap=6)], gap=2)),
                _kpi_card("Заказы", f"{orders_cnt:,}".replace(",", " "),
                        subtitle=dmc.Text(f"Средний чек: {self._fmt_cur(aov)}", size="xs", c="dimmed")),
                _kpi_card("Медианный чек", self._fmt_cur(median_check)),
                _kpi_card("Макс. чек", self._fmt_cur(max_check)),
                _kpi_card("Продажи без заказа", self._fmt_pct(no_order_share)),
                _kpi_card("Период", period_label),
            ],
        )

        # === helper: таблица возвратов (с ИТОГО) ===
        def _make_returns_table(col: str, title: str, top_n: int = 15):
            if col not in return_df.columns:
                return dmc.Alert(f"Нет данных по: {title}", color="gray", variant="light")

            grp = (
                return_df.groupby(col, dropna=False)["return_value"]
                .sum()
                .sort_values(ascending=False)
            )
            grp = grp[grp > 0]

            if grp.empty:
                return dmc.Alert(f"— по «{title}» нет возвратов", color="gray", variant="light")

            grp_top = grp.head(top_n)
            total_returns = float(returns_cur) if returns_cur else 0.0

            thead = dmc.TableThead(
                dmc.TableTr([
                    dmc.TableTh("№"),
                    dmc.TableTh(title),
                    dmc.TableTh("Возвраты, ₽", ta="right"),
                    dmc.TableTh("Доля", ta="right"),
                ])
            )

            rows = []
            for i, (k, v) in enumerate(grp_top.items(), start=1):
                name = str(k) if pd.notna(k) else "—"
                share = (v / total_returns) if total_returns else 0.0
                rows.append(
                    dmc.TableTr([
                        dmc.TableTd(f"{i}"),
                        dmc.TableTd(name),
                        dmc.TableTd(self._fmt_cur(float(v)), ta="right"),
                        dmc.TableTd(self._fmt_pct(float(share)), ta="right"),
                    ])
                )

            # === ИТОГО по видимым строкам ===
            total_in_table = float(grp_top.sum())
            share_in_table = (total_in_table / total_returns) if total_returns else 0.0
            tfoot = dmc.TableTfoot(
                dmc.TableTr([
                    dmc.TableTh("ИТОГО"),
                    dmc.TableTh("-"),
                    dmc.TableTh(self._fmt_cur(total_in_table), ta="right"),
                    dmc.TableTh(self._fmt_pct(share_in_table), ta="right"),
                ])
            )



            return dmc.Table(
                children=[thead, dmc.TableTbody(rows), tfoot],
                striped=True,
                highlightOnHover=True,
                horizontalSpacing="sm",
                verticalSpacing="xs",
                withTableBorder=True,
                withColumnBorders=False,

            )

        # === вкладки с таблицами по возвратам ===
        returns_tabs = dmc.Tabs(
            [
                dmc.TabsList([
                    dmc.TabsTab("Категории", value="cat"),
                    dmc.TabsTab("Производители", value="manu"),
                    dmc.TabsTab("Бренды", value="brand"),
                ]),
                dmc.TabsPanel(_make_returns_table("cat", "Категория"), value="cat"),
                dmc.TabsPanel(_make_returns_table("manu", "Производитель"), value="manu"),
                dmc.TabsPanel(
                    _make_returns_table("brand", "Бренд") if "brand" in return_df.columns
                    else _make_returns_table("brend", "Бренд"),
                    value="brand",
                ),
            ],
            value="cat",
            color="pink",
            variant="pills",
            keepMounted=False,
        )

        # === финальный return ===
        return dmc.Stack(
            [
                dmc.Title(f"Обзор за {period_label} · {self.clickSeriesName}", order=3),
                kpi_grid,
                dmc.Divider(variant="dashed"),
                dmc.Group([dmc.Title("Возвраты", order=4)], justify="space-between"),
                returns_tabs,
            ],
            gap="md",
        )



    def _cum_block(self, daily_stats: pd.DataFrame, month, metric: str):
        ds = self._ensure_aov_series(daily_stats, month, metric)
        cum = ds.copy()
        for c in [c for c in cum.columns if c != "day"]:
            cum[c] = cum[c].cumsum()
        fig = self._daily_area(cum, title="Кумулятивно с начала месяца")
        return dmc.Stack(
            [
                dmc.Text("Нарастающим итогом", size="sm", opacity=0.7),
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
            ]
        )

    def _slices_block(self, slices: dict):
        blocks = []
        for key, title in [
            ("cat", "Топ категорий"),
            ("subcat", "Топ подкатегорий"),
            ("manager", "Топ менеджеров"),
            ("chanel", "Каналы продаж"),
        ]:
            fig = self._bars_from_slice(slices[key], x=key, title=title, top_n=10)
            blocks.append(
                dmc.Paper(withBorder=True, radius=0, p="md", children=dcc.Graph(figure=fig, config={"displayModeBar": False}))
            )
        return dmc.SimpleGrid(cols=2, spacing="md", children=blocks)

    def _table_block(self, df_store: pd.DataFrame):
        eom = df_store["eom"].max()
        tx = df_store.loc[
            df_store["eom"].eq(eom),
            [
                "date",
                "client_order_number",
                "orders_type",
                "cat",
                "subcat",
                "manager",
                "chanel",
                "quant",
                "amount",
            ],
        ].copy()
        tx["date"] = pd.to_datetime(tx["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        column_defs = [
            {"headerName": "Дата", "field": "date", "sortable": True},
            {"headerName": "№ заказа", "field": "client_order_number", "sortable": True},
            {"headerName": "Тип", "field": "orders_type", "sortable": True},
            {"headerName": "Категория", "field": "cat", "sortable": True},
            {"headerName": "Подкатегория", "field": "subcat", "sortable": True},
            {"headerName": "Менеджер", "field": "manager", "sortable": True},
            {"headerName": "Канал", "field": "chanel", "sortable": True},
            {"headerName": "Кол-во", "field": "quant", "type": "rightAligned"},
            {"headerName": "Сумма", "field": "amount", "type": "rightAligned", "valueFormatter": {"function": "d3.format(',.2f')(value)"}},
        ]

        return dmc.Paper(
            withBorder=True,
            radius="lg",
            p=0,
            children=dag.AgGrid(
                id={"type": "store_area_chart_table", "index": "1"},
                rowData=tx.to_dict("records"),
                columnDefs=column_defs,
                defaultColDef={"resizable": True, "filter": True},
                dashGridOptions={"animateRows": True, "rowSelection": "single"},
                style={"height": "480px", "width": "100%"},
            ),
        )

    # ----------------- Figures -----------------
    @staticmethod
    def _daily_area(daily_df: pd.DataFrame, title: str = ""):
        fig = go.Figure()
        cols = [c for c in daily_df.columns if c != "day"]
        for col in cols:
            name = str(pd.to_datetime(col).strftime("%b %y")) if pd.notna(col) else str(col)
            fig.add_trace(go.Scatter(x=daily_df["day"], y=daily_df[col], mode="lines", name=name))
        fig.update_layout(
            title=title,
            margin=dict(l=8, r=8, t=40, b=8),
            xaxis_title="День",
            yaxis_title="Сумма",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        return fig

    @staticmethod
    def _bars_from_slice(df_slice: pd.DataFrame, x: str, title: str = "", top_n: int | None = None):
        src = df_slice.copy()
        if top_n:
            src = src.head(top_n)
        fig = go.Figure(go.Bar(x=src[x].astype(str), y=src["amount"]))
        fig.update_layout(title=title, margin=dict(l=8, r=8, t=40, b=8), xaxis_title="", yaxis_title="Сумма")
        return fig

    # ----------------- Helpers -----------------
    @staticmethod
    def _fmt_cur(x: float | None):
        if x is None:
            return "—"
        return f"{x:,.0f}".replace(",", " ")

    @staticmethod
    def _fmt_pct(x: float | None):
        if x is None:
            return "—"
        return f"{x*100:.1f}%"

    @staticmethod
    def _pct_badge(x: float | None, label: str = ""):
        color = "gray"
        text = "—"
        if x is not None:
            if x > 0:
                color = "green"
                text = f"▲ {x*100:.1f}% {label}" if label else f"▲ {x*100:.1f}%"
            elif x < 0:
                color = "red"
                text = f"▼ {abs(x)*100:.1f}% {label}" if label else f"▼ {abs(x)*100:.1f}%"
            else:
                color = "gray"
                text = f"0.0% {label}" if label else "0.0%"
        return dmc.Badge(text, color=color, variant="light", radius="xs",)

    @staticmethod
    def _empty_state(text: str = "Данные отсутствуют"):
        return dmc.Center(
            dmc.Stack(
                [
                    DashIconify(icon="mdi:database-off", width=48),
                    dmc.Text(text, size="sm", opacity=0.7),
                ],
                align="center",
            ),
            mih=240,
        )

    
    @staticmethod
    def _ensure_aov_series(daily_stats: pd.DataFrame, month, metric: str):
        if metric != "aov":
            return daily_stats
        # здесь оставляем amount как прокси
        return daily_stats
    
    @staticmethod
    def _period_label(month):
        # "01–30 сентября 2025"
        if pd.isna(month):
            return "выбранный месяц"
        start = (month - pd.offsets.MonthBegin(1)).date()
        end = month.date()
        # русские месяцы для красоты
        months = ["января","февраля","марта","апреля","мая","июня","июля","августа","сентября","октября","ноября","декабря"]
        def _fmt(d):
            return f"{d.day} {months[d.month-1]} {d.year}"
        # компактный диапазон: "1–30 сентября 2025"
        if start.month == end.month and start.year == end.year:
            return f"{start.day}–{end.day} {months[end.month-1]} {end.year}"
        return f"{_fmt(start)} — {_fmt(end)}"

    @staticmethod
    def _stat_card(title: str, value: str):
        return dmc.Paper(
            withBorder=True, radius=0, p="md",
            children=dmc.Stack(
                [dmc.Text(title, size="sm", opacity=0.6), dmc.Text(value, fw=700, fz="xl")],
                gap=4
            )
        )

    


    # @staticmethod
    # def _heatmap_block(df_store: pd.DataFrame, month, metric: str):
    #     cur = df_store.loc[df_store["eom"].eq(month)].copy()
    #     if cur.empty:
    #         return dmc.Stack([dmc.Text("Нет данных для тепловой карты", size="sm", opacity=0.7)])

    #     cur["week_of_month"] = ((cur["date"].dt.day - 1) // 7) + 1
    #     cur["weekday"] = cur["date"].dt.weekday  # 0=Пн
    #     val_col = "amount" if metric in ("amount", "aov") else "quant"

    #     pivot = (
    #         cur.pivot_table(index="week_of_month", columns="weekday", values=val_col, aggfunc="sum")
    #         .fillna(0)
    #         .sort_index()
    #     )

    #     # Формат чисел (в тыс. ₽)
    #     fmt = lambda x: f"{x/1000:.1f} тыс" if x >= 1000 else f"{x:.0f}"

    #     weekday_names = {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Вс"}
    #     cols_sorted = sorted(pivot.columns)
    #     x_labels = [weekday_names[i] for i in cols_sorted]

    #     # Подписи с форматированной выручкой
    #     text_values = np.vectorize(fmt)(pivot.values)

    #     fig = go.Figure(
    #         data=go.Heatmap(
    #             z=pivot.values,
    #             x=x_labels,
    #             y=[f"Неделя {i}" for i in pivot.index],
    #             coloraxis="coloraxis",
    #             text=text_values,
    #             texttemplate="%{text}",
    #             textfont={"size": 12, "color": "white"},
    #             hovertemplate="<b>%{x}</b><br>%{y}<br>Выручка: %{z:,.0f} ₽<extra></extra>",
    #         )
    #     )

    #     fig.update_layout(
    #         title={
    #             "text": "Тепловая карта продаж по дням недели",
    #             "x": 0.5,
    #             "xanchor": "center",
    #             "yanchor": "top",
    #             "font": {"size": 18, "family": "Manrope"},
    #         },
    #         margin=dict(l=30, r=30, t=60, b=30),
    #         coloraxis=dict(
    #             colorscale="Blues",
    #             colorbar=dict(title="Выручка, ₽", tickformat=",.0f"),
    #         ),
    #         plot_bgcolor="rgba(0,0,0,0)",
    #         paper_bgcolor="rgba(0,0,0,0)",
    #     )

    #     return dmc.Paper(
    #         withBorder=True,
    #         radius="md",
    #         p="md",
    #         shadow="sm",
    #         children=[
    #             dmc.Group(
    #                 [
    #                     dmc.Text("Календарь продаж по неделям", fw=600, size="md"),
    #                     dmc.Badge("₽ в ячейках — дневная выручка", color="gray", variant="light"),
    #                 ],
    #                 justify="space-between",
    #                 align="center",
    #                 mb="sm",
    #             ),
    #             dcc.Graph(figure=fig, config={"displayModeBar": False}),
    #         ],
    #     )



    @staticmethod
    def _heatmap_block(df_store: pd.DataFrame, month, metric: str):
        cur = df_store.loc[df_store["eom"].eq(month)].copy()
        if cur.empty:
            return dmc.Stack([dmc.Text("Нет данных для тепловой карты", size="sm", opacity=0.7)])

        # подготовка признаков
        cur["week_of_month"] = ((cur["date"].dt.day - 1) // 7) + 1
        cur["weekday"] = cur["date"].dt.weekday  # 0=Пн
        val_col = "amount" if metric in ("amount", "aov") else "quant"

        pivot = (
            cur.pivot_table(index="week_of_month", columns="weekday", values=val_col, aggfunc="sum")
            .fillna(0)
            .sort_index()
        )

        # --- авто-единицы (руб/тыс/млн) для компактных подписей
        z = pivot.values
        z_max = float(z.max()) if z.size else 0.0
        if z_max >= 1_000_000:
            div, suffix = 1_000_000.0, " млн"
        elif z_max >= 100_000:
            div, suffix = 1_000.0, " тыс"
        else:
            div, suffix = 1.0, " ₽"

        def fmt(v: float) -> str:
            v = float(v)
            if div == 1_000_000.0:
                return f"{v/div:.1f}{suffix}"
            if div == 1_000.0:
                return f"{v/div:.1f}{suffix}"
            return f"{v:,.0f} ₽".replace(",", " ")

        weekday_names = {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Вс"}
        cols_sorted = sorted(pivot.columns)            # 0..6
        x_labels = [weekday_names[i] for i in cols_sorted]
        y_labels = [f"Неделя {i}" for i in pivot.index]

        # базовая тепловая карта (без текста внутри)
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=x_labels,
                y=y_labels,
                coloraxis="coloraxis",
                hovertemplate="<b>%{x}</b><br>%{y}<br>Значение: %{z:,.0f} ₽<extra></extra>",
            )
        )

        # динамический цвет подписей через аннотации
        z_min, z_max = float(z.min()) if z.size else 0.0, float(z.max()) if z.size else 1.0
        z_mid = (z_min + z_max) / 2.0

        for r_idx, y_lab in enumerate(y_labels):
            for c_idx, x_lab in enumerate(x_labels):
                val = float(z[r_idx, c_idx]) if z.size else 0.0
                txt = fmt(val)
                # простое правило контраста: правее середины шкалы — тёмный фон => белый текст
                color = "white" if val > z_mid else "black"
                fig.add_annotation(
                    x=x_lab, y=y_lab, text=txt,
                    showarrow=False,
                    font=dict(size=12, color=color),
                )

        fig.update_layout(
            title={
                "text": "Тепловая карта продаж по дням недели",
                "x": 0.5, "xanchor": "center",
                "font": {"size": 18, "family": "Manrope"},
            },
            margin=dict(l=30, r=30, t=60, b=30),
            coloraxis=dict(
                colorscale="Blues",
                colorbar=dict(title="Выручка, ₽", tickformat=",.0f"),
            ),
            xaxis=dict(type="category"),
            yaxis=dict(type="category"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        return dmc.Paper(
            withBorder=True, radius="md", p="md", shadow="sm",
            children=[
                dmc.Group(
                    [
                        dmc.Text("Календарь продаж по неделям", fw=600, size="md"),
                        dmc.Badge(f"Подписи: {('млн' if div==1_000_000 else 'тыс' if div==1_000 else '₽')}", color="gray", variant="light"),
                    ],
                    justify="space-between", align="center", mb="sm",
                ),
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
            ],
        )
