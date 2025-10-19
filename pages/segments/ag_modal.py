# import pandas as pd
# import numpy as np
# from dash_iconify import DashIconify
# import dash_mantine_components as dmc
# from .db_queries import fletch_item_details
# from dash import dcc, html, Input, Output, State, no_update, MATCH
# import plotly.graph_objects as go
# import calendar

# import locale
# locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')


# class AGModal:
#     def __init__(self):
#         self.modal_id = {"type": "segments_ag_modal", "index": "1"}
#         self.modal_conteiner_id = {"type": "segments_ag_modal_container", "index": "1"}
#         # controls
#         self.metric_id = {"type": "ag_metric", "index": "1"}
#         # containers
#         self.stats_container_id = {"type": "ag_stats_container", "index": "1"}
#         self.heatmap_container_id = {"type": "ag_heatmap_container", "index": "1"}
#         self.month_selector_id = {"type": "ag_month_selector", "index": "1"}
#         self.kpi_container_id = {"type": "ag_kpi_container", "index": "1"}
#         # stores
#         self.store_sales_id = {"type": "ag_store_sales", "index": "1"}
#         self.store_timeline_id = {"type": "ag_store_timeline", "index": "1"}

#     def layout(self):
#         return dmc.Modal(
#             id=self.modal_id,
#             size="90%",
#             opened=False,
#             children=[
#                 dmc.Container(id=self.modal_conteiner_id, fluid=True),
#             ],
#         )

#     # ========================= core UI builder ========================= #
#     def update_modal(self, d, start, end):
#         """Строим layout модалки и заполняем Stores исходными данными."""
#         item_id = int(d["item_id"])
#         fullname = d["fullname"]
#         init_date = d["init_date"]
#         article = d["article"]
#         manu = d["manu"]
#         brend = d["brend"]

#         first_date = pd.to_datetime(init_date)
#         last_date = pd.to_datetime(end)

#         # Таймлайн по месяцам (конец месяца)
#         timeline = pd.date_range(
#             start=first_date + pd.offsets.MonthEnd(0),
#             end=last_date + pd.offsets.MonthEnd(0),
#             freq="ME",
#         )
#         timeline = pd.DataFrame({"eom": timeline})
#         timeline["month_id"] = timeline["eom"].dt.strftime("%Y-%m")

#         # Данные по продажам товара
#         sales_data = fletch_item_details(item_id, start, end)
#         sales_data["date"] = pd.to_datetime(sales_data["date"])
#         sales_data["sd_eom"] = sales_data["date"] + pd.offsets.MonthEnd(0)
#         sales_data["month_id"] = sales_data["sd_eom"].dt.strftime("%Y-%m")

#         # Суммы и кол-ва с учётом возвратов
#         sales_data["amount"] = sales_data["dt"] - sales_data["cr"]
#         sales_data["quant"] = sales_data["quant_dt"] - sales_data["quant_cr"]
#         # price по строкам не используем — посчитаем на агрегации

#         # Таблица по умолчанию — выручка, теплокарта overall
#         stats_html = self._make_stats_html(
#             timeline=timeline,
#             sales_data=sales_data,
#             dataset="amount",
#             heatmap=True,
#             heatmap_axis="overall",
#             period="Все данные",
#         )

#         # ====== UI ======
#         header = dmc.Center(
#             [dmc.Title(f"Карточка товара: {fullname} (Артикль: {article})", order=3, c="blue")]
#         )

#         controls = dmc.Group(
#             align="center",
#             grow=True,
#             children=[
#                 dmc.SegmentedControl(
#                     id=self.metric_id,
#                     value="amount",
#                     data=[
#                         {"label": "Выручка", "value": "amount"},
#                         {"label": "Кол-во", "value": "quant"},
#                         {"label": "Средняя цена", "value": "price"},
#                     ],
#                     size="sm",
#                     color="blue",
#                 ),
#             ],
#         )

#         table_block = dmc.Stack(
#             [
#                 dmc.Divider(label="Статистика", labelPosition="center", variant="dashed"),
#                 dmc.Center(
#                     html.Div(
#                         id=self.stats_container_id,
#                         children=dcc.Markdown(stats_html, dangerously_allow_html=True),
#                         # важное изменение: контейнер «по размеру контента»
#                         style={"display": "inline-block", "maxWidth": "100%"},
#                     )
#                 ),

#             ]
#         )

#         # ====== Heatmap block (по дням, как у тебя) ======
#         month_options = [
#             {"label": pd.to_datetime(m).strftime("%b %Y"), "value": str(m)}
#             for m in timeline["eom"].sort_values().unique()
#         ]
#         default_month = str(timeline["eom"].max()) if not timeline.empty else None

#         heatmap_controls = dmc.Group(
#             [
#                 dmc.Select(
#                     id=self.month_selector_id,
#                     data=month_options,
#                     value=default_month,
#                     placeholder="Месяц",
#                     searchable=True,
#                     nothingFoundMessage="Нет данных",
#                     w=260,
#                 ),
#             ],
#             align="center",
#         )

#         heatmap_block = dmc.Stack(
#             [
#                 dmc.Divider(label="Тепловая карта по дням", labelPosition="center", variant="dashed"),
#                 heatmap_controls,
#                 html.Div(id=self.heatmap_container_id),
#             ]
#         )

#         # ====== Summary KPI ======
#         kpi_block = dmc.Stack(
#             [
#                 dmc.Divider(label="Итоги", labelPosition="center", variant="dashed"),
#                 html.Div(id=self.kpi_container_id),
#             ]
#         )

#         # Stores
#         stores = [
#             dcc.Store(id=self.store_sales_id, data=sales_data.to_dict("records")),
#             dcc.Store(
#                 id=self.store_timeline_id,
#                 data=timeline.assign(eom=timeline["eom"].astype(str)).to_dict("records"),
#             ),
#         ]

#         return dmc.Stack([
#             header,
#             dmc.Space(h=10),
#             kpi_block,
#             dmc.Space(h=10),
#             controls,
#             dmc.Space(h=10),
#             table_block,
#             dmc.Space(h=10),
#             heatmap_block,
#             *stores
#         ])

#     # ========================= helpers ========================= #
#     @staticmethod
#     def _weighted_price_table(pvt: pd.DataFrame) -> pd.DataFrame:
#         """Из pivot с мультиколонками (amount/quant × year) делаем таблицу цен по годам."""
#         years = [c for c in pvt.columns.get_level_values(1).unique()]
#         res = pd.DataFrame(index=pvt.index)
#         for y in years:
#             a = pvt[("amount", y)] if ("amount", y) in pvt.columns else 0
#             q = pvt[("quant", y)] if ("quant", y) in pvt.columns else 0
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 val = np.where(q != 0, a / q, np.nan)
#             res[y] = val
#         return res

#     @staticmethod
#     def _apply_simple_gradient(styler, axis_mode: str):
#         """Фолбэк-градиент без matplotlib: белый → #90caf9 (синий)."""
#         df = styler.data.copy()

#         def hex_interp(v):
#             start = np.array([255, 255, 255], dtype=float)
#             end = np.array([144, 202, 249], dtype=float)  # #90caf9
#             rgb = (start + (end - start) * v).astype(int)
#             return [f"background-color: rgb({r},{g},{b})" for r, g, b in rgb]

#         if axis_mode == "rows":
#             styled = df.apply(lambda s: pd.Series(hex_interp((s - s.min())/(s.max()-s.min() or 1)), index=s.index), axis=1)
#         elif axis_mode == "columns":
#             styled = df.apply(lambda s: pd.Series(hex_interp((s - s.min())/(s.max()-s.min() or 1)), index=s.index))
#         else:
#             vmin = np.nanmin(df.values)
#             vmax = np.nanmax(df.values)
#             rng = (vmax - vmin) if np.isfinite(vmax - vmin) and (vmax - vmin) != 0 else 1
#             norm = (df - vmin) / rng
#             styled = norm.apply(lambda s: pd.Series(hex_interp(s), index=s.index))
#         return styler.apply(lambda _: styled, axis=None)

#     @staticmethod
#     def _make_stats_html(
#         timeline: pd.DataFrame,
#         sales_data: pd.DataFrame,
#         dataset: str = "amount",
#         heatmap: bool = True,
#         heatmap_axis: str = "overall",
#         period: str = "Все данные",
#     ) -> str:
#         """Сводная по месяцам × годам (+ теплокарта)."""
#         sdf = sales_data.copy()
#         if "period" in sdf.columns:
#             sdf = sdf[sdf["period"].fillna("Все данные") == period]

#         stats_df = pd.merge(
#             timeline[["eom", "month_id"]], sdf[["month_id", "amount", "quant"]],
#             on="month_id", how="left"
#         )
#         stats_df["month_num"] = pd.to_datetime(stats_df["eom"]).dt.month
#         stats_df["year"] = pd.to_datetime(stats_df["eom"]).dt.year.astype(str)
#         stats_df["Месяц"] = pd.to_datetime(stats_df["eom"]).dt.strftime("%b").str.capitalize()

#         pvt = stats_df.pivot_table(
#             index=["month_num", "Месяц"],
#             columns="year",
#             values=["amount", "quant"],
#             aggfunc="sum",
#             observed=True,
#         )

#         # 1) Сортируем мультиколонки (lexsorted)
#         pvt = pvt.sort_index(axis=1)

#         # 2) Сортируем строки по номеру месяца и аккуратно «срезаем» level индекса
#         pvt = pvt.sort_index(level=0)              # сортируем по month_num
#         pvt = pvt.reset_index(level=0, drop=True)  # убираем month_num из индекса
#         pvt.index.name = "Месяц"


#         if dataset == "amount":
#             show = pvt["amount"].copy()
#         elif dataset == "quant":
#             show = pvt["quant"].copy()
#         else:
#             show = AGModal._weighted_price_table(pvt)

#         # Итоги
#         total_row = {}
#         for col in show.columns:
#             if dataset == "price":
#                 a_sum = pvt[("amount", col)].sum(min_count=1)
#                 q_sum = pvt[("quant", col)].sum(min_count=1)
#                 total_row[col] = (a_sum / q_sum) if q_sum not in (0, np.nan) else np.nan
#             else:
#                 base = pvt[(dataset, col)]
#                 total_row[col] = base.sum(min_count=1)
#         show.loc["Итого"] = pd.Series(total_row)

#         # sty = (
#         #     show.style
#         #     .format("{:,.0f}", na_rep="-", thousands=" ")
#         #     .set_table_attributes('class="forecast-table"')
#         #     .set_caption("Сводная таблица · переключатель метрики сверху (выручка/кол-во/цена)")
#         # )
        
#         sty = (
#             show.style
#             .format("{:,.0f}", na_rep="-", thousands=" ")
#             .set_table_attributes('class="forecast-table"')
#             .set_table_styles([
#                 {'selector': 'table', 'props': [('margin-left', 'auto'), ('margin-right', 'auto')]},
#                 {'selector': 'caption', 'props': [('caption-side', 'top'), ('text-align', 'center')]}
#             ])
#             .set_caption("Сводная таблица · переключатель метрики сверху (выручка/кол-во/цена)")
#         )


#         if heatmap:
#             try:
#                 sty = sty.background_gradient(cmap="YlGnBu", axis=None)  # overall
#             except Exception:
#                 sty = AGModal._apply_simple_gradient(sty, "overall")

#         return sty.to_html()

#     # -------------------- KPI summary -------------------- #
#     @staticmethod
#     def _build_kpis(df_scope: pd.DataFrame):
#         if df_scope.empty:
#             return dmc.Text("Нет данных", c="dimmed")
#         df = df_scope.copy()

#         # гарантируем нужные колонки
#         if "amount" not in df.columns:
#             df["amount"] = df.get("dt", 0) - df.get("cr", 0)
#         if "quant" not in df.columns:
#             df["quant"] = df.get("quant_dt", 0) - df.get("quant_cr", 0)

#         # агрегаты
#         gross_amount = float(pd.to_numeric(df.get("dt", 0)).sum())
#         returns_amount = float(pd.to_numeric(df.get("cr", 0)).sum())
#         net_amount = float(pd.to_numeric(df["amount"]).sum())

#         gross_quant = float(pd.to_numeric(df.get("quant_dt", 0)).sum())
#         returns_quant = float(pd.to_numeric(df.get("quant_cr", 0)).sum())
#         net_quant = float(pd.to_numeric(df["quant"]).sum())

#         avg_price = (net_amount / net_quant) if net_quant else np.nan
#         return_rate_amount = (returns_amount / gross_amount * 100) if gross_amount else np.nan

#         def card(title, value, subtitle=None, icon=None):
#             left = DashIconify(icon=icon or "mdi:chart-box-outline", width=18)
#             return dmc.Paper(
#                 withBorder=True, radius="md", p="md", shadow="sm",
#                 children=[
#                     dmc.Group(
#                         [
#                             dmc.Avatar(left, size="sm", radius="xl", color="blue", variant="light"),
#                             dmc.Text(title, size="sm", c="dimmed"),
#                         ],
#                         gap="sm",
#                     ),
#                     dmc.Text(value, fw=700, size="xl"),
#                     dmc.Text(subtitle or "", size="xs", c="dimmed"),
#                 ],
#             )

#         grid = dmc.SimpleGrid(
#             cols=5, spacing="md",
#             children=[
#                 card("Итого выручка", f"{net_amount:,.0f} ₽".replace(",", " \u202F"), icon="mdi:cash"),
#                 card("Итого продано", f"{net_quant:,.0f} шт".replace(",", " \u202F"), icon="mdi:package-variant"),
#                 card("Средняя цена", (f"{avg_price:,.0f} ₽/шт".replace(",", " \u202F") if np.isfinite(avg_price) else "—"), icon="mdi:tag-outline"),
#                 card("Итого возвратов", f"{returns_amount:,.0f} ₽".replace(",", " \u202F"),
#                      subtitle=f"Кол-во: {returns_quant:,.0f} шт".replace(",", " \u202F"),
#                      icon="mdi:undo-variant"),
#                 card("Коэфф. возвратов", (f"{return_rate_amount:.1f}%" if np.isfinite(return_rate_amount) else "—"),
#                      subtitle="по сумме (CR/DT)", icon="mdi:percent-outline"),
#             ],
#         )
#         return grid

#     # ========================= callbacks ========================= #
#     def register_callbacks(self, app):
#         # Таблица: без переключателей теплокарты — всегда overall
#         @app.callback(
#             Output(self.stats_container_id, "children"),
#             Input(self.metric_id, "value"),
#             State(self.store_sales_id, "data"),
#             State(self.store_timeline_id, "data"),
#             prevent_initial_call=True,
#         )
#         def _update_table(metric_value, sales_records, timeline_records):
#             if not sales_records or not timeline_records:
#                 return no_update

#             sales_df = pd.DataFrame(sales_records)
#             timeline_df = pd.DataFrame(timeline_records)

#             for col in ("sd_eom", "date"):
#                 if col in sales_df.columns:
#                     with pd.option_context('mode.chained_assignment', None):
#                         try:
#                             sales_df[col] = pd.to_datetime(sales_df[col])
#                         except Exception:
#                             pass
#             if "eom" in timeline_df.columns:
#                 try:
#                     timeline_df["eom"] = pd.to_datetime(timeline_df["eom"])
#                 except Exception:
#                     pass

#             html_table = AGModal._make_stats_html(
#                 timeline=timeline_df,
#                 sales_data=sales_df,
#                 dataset=(metric_value or "amount"),
#                 heatmap=True,
#                 heatmap_axis="overall",
#                 period="Все данные",
#             )
#             return dcc.Markdown(html_table, dangerously_allow_html=True)

#         # Дневная теплокарта
#         @app.callback(
#             Output(self.heatmap_container_id, "children"),
#             Input(self.month_selector_id, "value"),
#             Input(self.metric_id, "value"),
#             State(self.store_sales_id, "data"),
#             prevent_initial_call=False,
#         )
#         def _update_heatmap(month_value, metric_value, sales_records):
#             if not sales_records:
#                 return dmc.Text("Нет данных", c="dimmed")
#             sales_df = pd.DataFrame(sales_records)
#             if "date" in sales_df.columns:
#                 try:
#                     sales_df["date"] = pd.to_datetime(sales_df["date"]).dt.normalize()
#                 except Exception:
#                     pass
#             month_ts = pd.to_datetime(month_value) if month_value else (
#                 sales_df["date"].max() if "date" in sales_df.columns else None
#             )
#             return AGModal._heatmap_block(sales_df, month_ts, metric_value or "amount")

#         # KPI
#         @app.callback(
#             Output(self.kpi_container_id, "children"),
#             Input(self.store_sales_id, "data"),
#             prevent_initial_call=False,
#         )
#         def _update_kpis(sales_records):
#             if not sales_records:
#                 return dmc.Text("Нет данных", c="dimmed")
#             df = pd.DataFrame(sales_records)
#             return AGModal._build_kpis(df)

#     # -------------------- Heatmap builder (daily) -------------------- #
#     @staticmethod
#     def _heatmap_block(df_scope: pd.DataFrame, month, metric: str):
#         if month is None or df_scope.empty:
#             return dmc.Stack([dmc.Text("Нет данных для тепловой карты", size="sm", c="dimmed")])

#         start = (pd.to_datetime(month) - pd.offsets.MonthBegin(1)).normalize()
#         end = pd.to_datetime(month).normalize()
#         cur = df_scope.loc[(df_scope["date"] >= start) & (df_scope["date"] <= end)].copy()
#         if cur.empty:
#             return dmc.Stack([dmc.Text("Нет данных для тепловой карты", size="sm", c="dimmed")])

#         # агрегат по дате: amount/quant; price считаем как amount_sum/quant_sum на день
#         daily = cur.groupby(cur["date"].dt.date).agg({"amount": "sum", "quant": "sum"}).astype(float)
#         if metric == "price":
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 daily_vals = daily["amount"] / daily["quant"]
#             val_series = daily_vals
#         elif metric == "quant":
#             val_series = daily["quant"]
#         else:
#             val_series = daily["amount"]

#         # календарная сетка Пн..Вс
#         y = int(pd.to_datetime(month).year)
#         m = int(pd.to_datetime(month).month)
#         calendar.setfirstweekday(calendar.MONDAY)
#         weeks = calendar.monthcalendar(y, m)  # 0 = пустая клетка

#         # матрицы: значения, подписи-дат, маска реальных дат
#         z, custom, mask = [], [], []
#         for wk in weeks:
#             row_vals, row_custom, row_mask = [], [], []
#             for d in wk:
#                 if d == 0:
#                     row_vals.append(np.nan)
#                     row_custom.append("")
#                     row_mask.append(False)
#                 else:
#                     the_date = pd.Timestamp(y, m, d).date()
#                     v = float(val_series.get(the_date, np.nan))
#                     row_vals.append(v)
#                     row_custom.append(pd.Timestamp(the_date).strftime("%d.%m"))
#                     row_mask.append(True)
#             z.append(row_vals); custom.append(row_custom); mask.append(row_mask)

#         z_arr = np.array(z, dtype=float)
#         finite = z_arr[np.isfinite(z_arr)]
#         z_max = float(np.nanmax(z_arr)) if finite.size else 0.0

#         suffix = {"amount": "₽", "quant": "шт", "price": "₽/шт"}[metric if metric in ("amount","quant","price") else "amount"]
#         def fmt(v: float) -> str:
#             if not np.isfinite(v):
#                 return "—"
#             return f"{v:,.0f} {suffix}".replace(",", " ")

       

#         weekday_names = ["Пн","Вт","Ср","Чт","Пт","Сб","Вс"]
#         x_labels = weekday_names

#         # === подписи строк (недели) ===
#         # готовим мапу по дням: amount, quant, price
#         daily_tbl = cur.groupby(cur["date"].dt.date).agg(amount=("amount","sum"),
#                                                         quant=("quant","sum"))
#         with np.errstate(divide='ignore', invalid='ignore'):
#             daily_tbl["price"] = daily_tbl["amount"] / daily_tbl["quant"]

#         row_values = []  # значение для подписи каждой недели
#         for r_idx, week in enumerate(weeks):
#             # собираем даты этой строки
#             dates_in_week = [pd.Timestamp(y, m, d).date() for d in week if d != 0]
#             if not dates_in_week:
#                 row_values.append(np.nan)
#                 continue
#             w = daily_tbl.reindex(dates_in_week)
#             if metric == "price":
#                 # ВЗВЕШЕННАЯ средняя за неделю
#                 a_sum = np.nansum(w["amount"].values)
#                 q_sum = np.nansum(w["quant"].values)
#                 row_values.append(a_sum / q_sum if (q_sum and np.isfinite(q_sum)) else np.nan)
#             elif metric == "quant":
#                 row_values.append(np.nansum(w["quant"].values))
#             else:  # amount
#                 row_values.append(np.nansum(w["amount"].values))

#         # подписи строк: для цены — «ср. цена», иначе — «сумма»
#         unit = {"amount": "₽", "quant": "шт", "price": "₽/шт"}[metric]
#         def fmt_row(v):
#             if not np.isfinite(v): return "—"
#             return f"{v:,.0f} {unit}".replace(",", " ")

#         y_labels = [
#             (f"Неделя {i+1} — ср. цена {fmt_row(v)}" if metric == "price"
#             else f"Неделя {i+1} — {fmt_row(v)}")
#             for i, v in enumerate(row_values)
#         ]

#         # перерисовываем фигуру с новыми подписями
#         fig = go.Figure(
#             data=go.Heatmap(
#                 z=z_arr, x=x_labels, y=y_labels, coloraxis="coloraxis",
#                 hovertemplate="<b>%{customdata}</b><br>%{y} / %{x}<br>Значение: %{z:,.0f}<extra></extra>",
#                 customdata=custom,
#                 zmin=np.nanmin(z_arr[np.isfinite(z_arr)]) if np.isfinite(z_arr).any() else None,
#                 zmax=np.nanmax(z_arr[np.isfinite(z_arr)]) if np.isfinite(z_arr).any() else None,
#             )
#         )

#         # аннотации в ячейках без изменений
#         z_min = float(np.nanmin(z_arr[np.isfinite(z_arr)])) if np.isfinite(z_arr).any() else 0.0
#         z_max = float(np.nanmax(z_arr[np.isfinite(z_arr)])) if np.isfinite(z_arr).any() else 0.0
#         z_mid = (z_min + (z_max if z_max > 0 else 1.0)) / 2.0
#         for r_idx, _ in enumerate(y_labels):
#             for c_idx, _ in enumerate(x_labels):
#                 if not mask[r_idx][c_idx]:
#                     continue
#                 val = float(z_arr[r_idx, c_idx]) if np.isfinite(z_arr[r_idx, c_idx]) else np.nan
#                 date_txt = custom[r_idx][c_idx]
#                 txt = f"{date_txt}<br>{fmt(val)}"
#                 color = "white" if (np.isfinite(val) and val > z_mid) else "black"
#                 fig.add_annotation(x=x_labels[c_idx], y=y_labels[r_idx], text=txt,
#                                 showarrow=False, font=dict(size=11, color=color))

#         colorbar_title = {"amount": "Выручка", "quant": "Кол-во", "price": "Цена"}[metric if metric in ("amount","quant","price") else "amount"]
#         fig.update_layout(
#             margin=dict(l=30, r=30, t=30, b=20),
#             coloraxis=dict(colorscale="Blues", colorbar=dict(title=colorbar_title)),
#             xaxis=dict(type="category"),
#             yaxis=dict(type="category"),
#             plot_bgcolor="rgba(0,0,0,0)",
#             paper_bgcolor="rgba(0,0,0,0)",
#         )

#         flat = [(float(v), custom[r][c]) for r in range(len(weeks)) for c, v in enumerate(z_arr[r]) if np.isfinite(v)]
#         top3 = sorted(flat, key=lambda x: x[0], reverse=True)[:3]
#         top3_block = dmc.Group(
#             [
#                 dmc.Badge(f"{d} — {fmt(v)}", color="blue", variant="light", radius="xs")
#                 for v, d in top3
#             ] if top3 else [dmc.Badge("Нет данных", color="gray", variant="light")],
#             gap="xs",
#         )

#         return dmc.Paper(
#             withBorder=True, radius="md", p="md", shadow="sm",
#             children=[
#                 dcc.Graph(figure=fig, config={"displayModeBar": False}),
#                 dmc.Group([dmc.Text("Top-3 дня:", fw=600), top3_block], gap="sm", mt="sm"),
#             ],
#         )



import pandas as pd
import numpy as np
from dash_iconify import DashIconify
import dash_mantine_components as dmc
from .db_queries import fletch_item_details
from dash import dcc, html, Input, Output, State, no_update, MATCH
import plotly.graph_objects as go
import calendar
import locale

# RU локаль (оставляю как у тебя; если на окружении нет ru_RU.UTF-8, можно обернуть в try)
locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')


class AGModal:
    def __init__(self):
        self.modal_id = {"type": "segments_ag_modal", "index": "1"}
        self.modal_conteiner_id = {"type": "segments_ag_modal_container", "index": "1"}
        # controls
        self.metric_id = {"type": "ag_metric", "index": "1"}
        # containers
        self.stats_container_id = {"type": "ag_stats_container", "index": "1"}
        self.heatmap_container_id = {"type": "ag_heatmap_container", "index": "1"}
        self.month_selector_id = {"type": "ag_month_selector", "index": "1"}
        self.kpi_container_id = {"type": "ag_kpi_container", "index": "1"}
        # stores
        self.store_sales_id = {"type": "ag_store_sales", "index": "1"}
        self.store_timeline_id = {"type": "ag_store_timeline", "index": "1"}

    def layout(self):
        return dmc.Modal(
            id=self.modal_id,
            size="90%",
            opened=False,
            children=[
                dmc.Container(id=self.modal_conteiner_id, fluid=True),
            ],
        )

    # ========================= core UI builder ========================= #
    def update_modal(self, d, start, end):
        """Строим layout модалки и заполняем Stores исходными данными."""
        item_id = int(d["item_id"])
        fullname = d["fullname"]
        init_date = d["init_date"]
        article = d["article"]
        manu = d["manu"]
        brend = d["brend"]

        first_date = pd.to_datetime(init_date)
        last_date = pd.to_datetime(end)

        # Таймлайн по месяцам (конец месяца)
        timeline = pd.date_range(
            start=first_date + pd.offsets.MonthEnd(0),
            end=last_date + pd.offsets.MonthEnd(0),
            freq="ME",
        )
        timeline = pd.DataFrame({"eom": timeline})
        timeline["month_id"] = timeline["eom"].dt.strftime("%Y-%m")

        # Данные по продажам товара
        sales_data = fletch_item_details(item_id, start, end)
        sales_data["date"] = pd.to_datetime(sales_data["date"]).dt.normalize()
        sales_data["sd_eom"] = sales_data["date"] + pd.offsets.MonthEnd(0)
        sales_data["month_id"] = sales_data["sd_eom"].dt.strftime("%Y-%m")

        # Суммы и кол-ва с учётом возвратов
        sales_data["amount"] = sales_data["dt"] - sales_data["cr"]
        sales_data["quant"] = sales_data["quant_dt"] - sales_data["quant_cr"]
        # price по строкам не используем — считаем на агрегации

        # Таблица по умолчанию — выручка, теплокарта overall
        stats_html = self._make_stats_html(
            timeline=timeline,
            sales_data=sales_data,
            dataset="amount",
            heatmap=True,
            heatmap_axis="overall",
            period="Все данные",
        )

        # ====== UI ======
        header = dmc.Center(
            [dmc.Title(f"Карточка товара: {fullname} (Артикль: {article})", order=3, c="blue")]
        )

        controls = dmc.Group(
            align="center",
            grow=True,
            children=[
                dmc.SegmentedControl(
                    id=self.metric_id,
                    value="amount",
                    data=[
                        {"label": "Выручка", "value": "amount"},
                        {"label": "Кол-во", "value": "quant"},
                        {"label": "Средняя цена", "value": "price"},
                    ],
                    size="sm",
                    color="blue",
                ),
            ],
        )

        table_block = dmc.Stack(
            [
                dmc.Divider(label="Статистика", labelPosition="center", variant="dashed"),
                dmc.Center(
                    html.Div(
                        id=self.stats_container_id,
                        children=dcc.Markdown(stats_html, dangerously_allow_html=True),
                        # контейнер «по размеру контента»
                        style={"display": "inline-block", "maxWidth": "100%"},
                    )
                ),
            ]
        )

        # ====== Heatmap block (по дням) ======
        month_options = [
            {"label": pd.to_datetime(m).strftime("%b %Y"), "value": str(m)}
            for m in timeline["eom"].sort_values().unique()
        ]
        default_month = str(timeline["eom"].max()) if not timeline.empty else None

        heatmap_controls = dmc.Group(
            [
                dmc.Select(
                    id=self.month_selector_id,
                    data=month_options,
                    value=default_month,
                    placeholder="Месяц",
                    searchable=True,
                    nothingFoundMessage="Нет данных",
                    w=260,
                ),
            ],
            align="center",
        )

        heatmap_block = dmc.Stack(
            [
                dmc.Divider(label="Тепловая карта по дням", labelPosition="center", variant="dashed"),
                heatmap_controls,
                html.Div(id=self.heatmap_container_id),
            ]
        )

        # ====== Summary KPI ======
        kpi_block = dmc.Stack(
            [
                dmc.Divider(label="Итоги", labelPosition="center", variant="dashed"),
                html.Div(id=self.kpi_container_id),
            ]
        )

        # Stores
        stores = [
            dcc.Store(id=self.store_sales_id, data=sales_data.to_dict("records")),
            dcc.Store(
                id=self.store_timeline_id,
                data=timeline.assign(eom=timeline["eom"].astype(str)).to_dict("records"),
            ),
        ]

        return dmc.Stack([
            header,
            dmc.Space(h=10),
            kpi_block,
            dmc.Space(h=10),
            controls,
            dmc.Space(h=10),
            table_block,
            dmc.Space(h=10),
            heatmap_block,
            *stores
        ])

    # ========================= helpers ========================= #
    @staticmethod
    def _weighted_price_table(pvt: pd.DataFrame) -> pd.DataFrame:
        """Из pivot с мультиколонками (amount/quant × year) делаем таблицу цен по годам."""
        years = list(pvt.columns.get_level_values(1).unique())
        res = pd.DataFrame(index=pvt.index)
        for y in years:
            a = pvt.get(("amount", y), pd.Series(0.0, index=pvt.index))
            q = pvt.get(("quant", y),  pd.Series(0.0, index=pvt.index))
            with np.errstate(divide='ignore', invalid='ignore'):
                res[y] = np.where(q != 0, a / q, np.nan)
        return res

    @staticmethod
    def _apply_simple_gradient(styler, axis_mode: str):
        """Фолбэк-градиент без matplotlib: белый → #90caf9 (синий)."""
        df = styler.data.copy()

        def hex_interp(v):
            start = np.array([255, 255, 255], dtype=float)
            end = np.array([144, 202, 249], dtype=float)  # #90caf9
            rgb = (start + (end - start) * v).astype(int)
            return [f"background-color: rgb({r},{g},{b})" for r, g, b in rgb]

        if axis_mode == "rows":
            styled = df.apply(lambda s: pd.Series(hex_interp((s - s.min())/(s.max()-s.min() or 1)), index=s.index), axis=1)
        elif axis_mode == "columns":
            styled = df.apply(lambda s: pd.Series(hex_interp((s - s.min())/(s.max()-s.min() or 1)), index=s.index))
        else:
            vmin = np.nanmin(df.values)
            vmax = np.nanmax(df.values)
            rng = (vmax - vmin) if np.isfinite(vmax - vmin) and (vmax - vmin) != 0 else 1
            norm = (df - vmin) / rng
            styled = norm.apply(lambda s: pd.Series(hex_interp(s), index=s.index))
        return styler.apply(lambda _: styled, axis=None)

    @staticmethod
    def _make_stats_html(
        timeline: pd.DataFrame,
        sales_data: pd.DataFrame,
        dataset: str = "amount",
        heatmap: bool = True,
        heatmap_axis: str = "overall",
        period: str = "Все данные",
    ) -> str:
        """Сводная по месяцам × годам (+ теплокарта)."""
        sdf = sales_data.copy()
        if "period" in sdf.columns:
            sdf = sdf[sdf["period"].fillna("Все данные") == period]

        stats_df = pd.merge(
            timeline[["eom", "month_id"]],
            sdf[["month_id", "amount", "quant"]],
            on="month_id", how="left"
        )
        stats_df["month_num"] = pd.to_datetime(stats_df["eom"]).dt.month
        stats_df["year"] = pd.to_datetime(stats_df["eom"]).dt.year.astype(str)
        stats_df["Месяц"] = pd.to_datetime(stats_df["eom"]).dt.strftime("%b").str.capitalize()

        # --- pivot без опасных reset/drop по колонкам ---
        pvt = stats_df.pivot_table(
            index=["month_num", "Месяц"],
            columns="year",
            values=["amount", "quant"],
            aggfunc="sum",
            observed=True,
        )

        # колонки MultiIndex приводим к lexsorted
        pvt = pvt.sort_index(axis=1)

        # строки: сортируем по номеру месяца и аккуратно «срезаем» level индекса
        pvt = pvt.sort_index(level=0).reset_index(level=0, drop=True)
        pvt.index.name = "Месяц"

        # выбор метрики
        if dataset == "amount":
            show = pvt["amount"].copy()
        elif dataset == "quant":
            show = pvt["quant"].copy()
        else:
            show = AGModal._weighted_price_table(pvt)

        # Итоги (по колонкам-годам)
        total_row = {}
        for col in show.columns:
            if dataset == "price":
                a_sum = pvt[("amount", col)].sum(min_count=1)
                q_sum = pvt[("quant", col)].sum(min_count=1)
                total_row[col] = (a_sum / q_sum) if q_sum not in (0, np.nan) else np.nan
            else:
                total_row[col] = show[col].sum(min_count=1)
        show.loc["Итого"] = pd.Series(total_row)

        sty = (
            show.style
            .format("{:,.0f}", na_rep="-", thousands=" ")
            .set_table_attributes('class="forecast-table"')
            .set_table_styles([
                {'selector': 'table', 'props': [('margin-left', 'auto'), ('margin-right', 'auto')]},
                {'selector': 'caption', 'props': [('caption-side', 'top'), ('text-align', 'center')]}
            ])
            .set_caption("Сводная таблица · переключатель метрики сверху (выручка/кол-во/цена)")
        )

        if heatmap:
            try:
                sty = sty.background_gradient(cmap="YlGnBu", axis=None)  # overall
            except Exception:
                sty = AGModal._apply_simple_gradient(sty, "overall")

        return sty.to_html()

    # -------------------- KPI summary -------------------- #
    @staticmethod
    def _build_kpis(df_scope: pd.DataFrame):
        if df_scope.empty:
            return dmc.Text("Нет данных", c="dimmed")
        df = df_scope.copy()

        # гарантируем нужные колонки
        if "amount" not in df.columns:
            df["amount"] = df.get("dt", 0) - df.get("cr", 0)
        if "quant" not in df.columns:
            df["quant"] = df.get("quant_dt", 0) - df.get("quant_cr", 0)

        # агрегаты
        gross_amount = float(pd.to_numeric(df.get("dt", 0)).sum())
        returns_amount = float(pd.to_numeric(df.get("cr", 0)).sum())
        net_amount = float(pd.to_numeric(df["amount"]).sum())

        gross_quant = float(pd.to_numeric(df.get("quant_dt", 0)).sum())
        returns_quant = float(pd.to_numeric(df.get("quant_cr", 0)).sum())
        net_quant = float(pd.to_numeric(df["quant"]).sum())

        avg_price = (net_amount / net_quant) if net_quant else np.nan
        return_rate_amount = (returns_amount / gross_amount * 100) if gross_amount else np.nan

        def card(title, value, subtitle=None, icon=None):
            left = DashIconify(icon=icon or "mdi:chart-box-outline", width=18)
            return dmc.Paper(
                withBorder=True, radius="md", p="md", shadow="sm",
                children=[
                    dmc.Group(
                        [
                            dmc.Avatar(left, size="sm", radius="xl", color="blue", variant="light"),
                            dmc.Text(title, size="sm", c="dimmed"),
                        ],
                        gap="sm",
                    ),
                    dmc.Text(value, fw=700, size="xl"),
                    dmc.Text(subtitle or "", size="xs", c="dimmed"),
                ],
            )

        grid = dmc.SimpleGrid(
            cols=5, spacing="md",
            children=[
                card("Итого выручка", f"{net_amount:,.0f} ₽".replace(",", " \u202F"), icon="mdi:cash"),
                card("Итого продано", f"{net_quant:,.0f} шт".replace(",", " \u202F"), icon="mdi:package-variant"),
                card("Средняя цена", (f"{avg_price:,.0f} ₽/шт".replace(",", " \u202F") if np.isfinite(avg_price) else "—"), icon="mdi:tag-outline"),
                card("Итого возвратов", f"{returns_amount:,.0f} ₽".replace(",", " \u202F"),
                     subtitle=f"Кол-во: {returns_quant:,.0f} шт".replace(",", " \u202F"),
                     icon="mdi:undo-variant"),
                card("Коэфф. возвратов", (f"{return_rate_amount:.1f}%" if np.isfinite(return_rate_amount) else "—"),
                     subtitle="по сумме (CR/DT)", icon="mdi:percent-outline"),
            ],
        )
        return grid

    # ========================= callbacks ========================= #
    def register_callbacks(self, app):
        # Таблица: без переключателей теплокарты — всегда overall
        @app.callback(
            Output(self.stats_container_id, "children"),
            Input(self.metric_id, "value"),
            State(self.store_sales_id, "data"),
            State(self.store_timeline_id, "data"),
            prevent_initial_call=True,
        )
        def _update_table(metric_value, sales_records, timeline_records):
            if not sales_records or not timeline_records:
                return no_update

            sales_df = pd.DataFrame(sales_records)
            timeline_df = pd.DataFrame(timeline_records)

            for col in ("sd_eom", "date"):
                if col in sales_df.columns:
                    with pd.option_context('mode.chained_assignment', None):
                        try:
                            sales_df[col] = pd.to_datetime(sales_df[col])
                        except Exception:
                            pass
            if "eom" in timeline_df.columns:
                try:
                    timeline_df["eom"] = pd.to_datetime(timeline_df["eom"])
                except Exception:
                    pass

            html_table = AGModal._make_stats_html(
                timeline=timeline_df,
                sales_data=sales_df,
                dataset=(metric_value or "amount"),
                heatmap=True,
                heatmap_axis="overall",
                period="Все данные",
            )
            return dcc.Markdown(html_table, dangerously_allow_html=True)

        # Дневная теплокарта
        @app.callback(
            Output(self.heatmap_container_id, "children"),
            Input(self.month_selector_id, "value"),
            Input(self.metric_id, "value"),
            State(self.store_sales_id, "data"),
            prevent_initial_call=False,
        )
        def _update_heatmap(month_value, metric_value, sales_records):
            if not sales_records:
                return dmc.Text("Нет данных", c="dimmed")

            sales_df = pd.DataFrame(sales_records)

            # ❗️СИНХРОНИЗИРУЕМ period с таблицей
            if "period" in sales_df.columns:
                sales_df = sales_df[sales_df["period"].fillna("Все данные") == "Все данные"]

            if "date" in sales_df.columns:
                try:
                    sales_df["date"] = pd.to_datetime(sales_df["date"]).dt.normalize()
                except Exception:
                    pass

            month_ts = pd.to_datetime(month_value) if month_value else (
                sales_df["date"].max() if "date" in sales_df.columns else None
            )
            return AGModal._heatmap_block(sales_df, month_ts, metric_value or "amount")


        # KPI
        @app.callback(
            Output(self.kpi_container_id, "children"),
            Input(self.store_sales_id, "data"),
            prevent_initial_call=False,
        )
        def _update_kpis(sales_records):
            if not sales_records:
                return dmc.Text("Нет данных", c="dimmed")
            df = pd.DataFrame(sales_records)
            return AGModal._build_kpis(df)

    # -------------------- Heatmap builder (daily) -------------------- #
    @staticmethod
    def _heatmap_block(df_scope: pd.DataFrame, month, metric: str):
        if month is None or df_scope.empty:
            return dmc.Stack([dmc.Text("Нет данных для тепловой карты", size="sm", c="dimmed")])

        month = pd.to_datetime(month)
        # синхронизированные границы месяца
        start = (month - pd.offsets.MonthBegin(1)).normalize()   # 1-е число
        end = (month + pd.offsets.MonthEnd(0)).normalize()       # строго EOM 00:00

        cur = df_scope.loc[(df_scope["date"] >= start) & (df_scope["date"] <= end)].copy()
        if cur.empty:
            return dmc.Stack([dmc.Text("Нет данных для тепловой карты", size="sm", c="dimmed")])

        # агрегат по дате: amount/quant; price считаем как amount_sum/quant_sum на день
        daily = (cur.groupby(cur["date"].dt.date, observed=True)
                 .agg(amount=("amount", "sum"), quant=("quant", "sum"))
                 .astype(float))
        with np.errstate(divide='ignore', invalid='ignore'):
            daily["price"] = daily["amount"] / daily["quant"]

        field = metric if metric in ("amount", "quant", "price") else "amount"
        vals = daily[field].to_dict()

        # календарная сетка Пн..Вс
        y = int(month.year)
        m = int(month.month)
        calendar.setfirstweekday(calendar.MONDAY)
        weeks = calendar.monthcalendar(y, m)  # 0 = пустая клетка

        # матрицы: значения, подписи-дат, маска реальных дат
        z, custom, mask = [], [], []
        for wk in weeks:
            row_vals, row_custom, row_mask = [], [], []
            for d in wk:
                if d == 0:
                    row_vals.append(np.nan)
                    row_custom.append("")
                    row_mask.append(False)
                else:
                    the_date = pd.Timestamp(y, m, d).date()
                    v = float(vals.get(the_date, np.nan))
                    row_vals.append(v)
                    row_custom.append(pd.Timestamp(the_date).strftime("%d.%m"))
                    row_mask.append(True)
            z.append(row_vals)
            custom.append(row_custom)
            mask.append(row_mask)

        z_arr = np.array(z, dtype=float)

        suffix = {"amount": "₽", "quant": "шт", "price": "₽/шт"}[field]

        def fmt(v: float) -> str:
            if not np.isfinite(v):
                return "—"
            return f"{v:,.0f} {suffix}".replace(",", " ")

        weekday_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
        x_labels = weekday_names

        # значения для подписи каждой недели (сумма или взвешенная средняя для price)
        row_values = []
        for week in weeks:
            dates_in_week = [pd.Timestamp(y, m, d).date() for d in week if d != 0]
            if not dates_in_week:
                row_values.append(np.nan)
                continue
            w = daily.reindex(dates_in_week)
            if field == "price":
                a_sum = np.nansum(w["amount"].values)
                q_sum = np.nansum(w["quant"].values)
                row_values.append(a_sum / q_sum if (q_sum and np.isfinite(q_sum)) else np.nan)
            elif field == "quant":
                row_values.append(np.nansum(w["quant"].values))
            else:  # amount
                row_values.append(np.nansum(w["amount"].values))

        unit = {"amount": "₽", "quant": "шт", "price": "₽/шт"}[field]

        def fmt_row(v):
            if not np.isfinite(v):
                return "—"
            return f"{v:,.0f} {unit}".replace(",", " ")

        y_labels = [
            (f"Неделя {i+1} — ср. цена {fmt_row(v)}" if field == "price"
             else f"Неделя {i+1} — {fmt_row(v)}")
            for i, v in enumerate(row_values)
        ]

        # перерисовываем фигуру
        finite_vals = z_arr[np.isfinite(z_arr)]
        zmin = float(np.nanmin(finite_vals)) if finite_vals.size else None
        zmax = float(np.nanmax(finite_vals)) if finite_vals.size else None

        fig = go.Figure(
            data=go.Heatmap(
                z=z_arr,
                x=x_labels,
                y=y_labels,
                coloraxis="coloraxis",
                hovertemplate="<b>%{customdata}</b><br>%{y} / %{x}<br>Значение: %{z:,.0f}<extra></extra>",
                customdata=custom,
                zmin=zmin,
                zmax=zmax,
            )
        )

        # аннотации в ячейках
        if finite_vals.size:
            z_min = float(np.nanmin(finite_vals))
            z_max = float(np.nanmax(finite_vals))
            z_mid = (z_min + (z_max if z_max > 0 else 1.0)) / 2.0
        else:
            z_mid = 0.0

        for r_idx, _ in enumerate(y_labels):
            for c_idx, _ in enumerate(x_labels):
                if not mask[r_idx][c_idx]:
                    continue
                val = float(z_arr[r_idx, c_idx]) if np.isfinite(z_arr[r_idx, c_idx]) else np.nan
                date_txt = custom[r_idx][c_idx]
                txt = f"{date_txt}<br>{fmt(val)}"
                color = "white" if (np.isfinite(val) and val > z_mid) else "black"
                fig.add_annotation(
                    x=x_labels[c_idx],
                    y=y_labels[r_idx],
                    text=txt,
                    showarrow=False,
                    font=dict(size=11, color=color),
                )

        colorbar_title = {"amount": "Выручка", "quant": "Кол-во", "price": "Цена"}[field]
        fig.update_layout(
            margin=dict(l=30, r=30, t=30, b=20),
            coloraxis=dict(colorscale="Blues", colorbar=dict(title=colorbar_title)),
            xaxis=dict(type="category"),
            yaxis=dict(type="category"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        # Top-3 дней
        flat = [
            (float(v), custom[r][c])
            for r in range(len(weeks))
            for c, v in enumerate(z_arr[r])
            if np.isfinite(v)
        ]
        top3 = sorted(flat, key=lambda x: x[0], reverse=True)[:3]
        top3_block = dmc.Group(
            [
                dmc.Badge(f"{d} — {fmt(v)}", color="blue", variant="light", radius="xs")
                for v, d in top3
            ] if top3 else [dmc.Badge("Нет данных", color="gray", variant="light")],
            gap="xs",
        )

        return dmc.Paper(
            withBorder=True, radius="md", p="md", shadow="sm",
            children=[
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
                dmc.Group([dmc.Text("Top-3 дня:", fw=600), top3_block], gap="sm", mt="sm"),
            ],
        )
