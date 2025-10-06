# Файл основной разметки таба по магазинам

import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate
from decimal import Decimal, ROUND_HALF_UP
import dash_ag_grid as dag


import dash
from dash import (
    dcc, html, Input, Output, State,
    MATCH, no_update, callback_context as ctx
)
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import locale

locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")

from components import NoData, BASE_COLORS, COLORS_BY_SHADE
from data import load_df_from_redis
from .modal_area_chart import StoreAreaChartModal




def seg_item(icon: str, text: str, value: str, color: str = None):
    return {
        "value": value,
        "label": dmc.Group(
            gap=6,
            wrap=False,
            align="center",
            children=[
                DashIconify(icon=icon, width=18, color=color),
                dmc.Text(text, size="sm"),
            ],
        ),
    }
    
def layout(df_id: str, period_label: str | None = None):
    return StoresComponents(df_id=df_id, period_label=period_label).tab_layout()


def fmt_abs(val, kind):
                # kind: 'mln' (деньги в млн ₽) | 'int' (целое) | 'rub' (в ₽ с разделителями)
                if kind == 'mln':
                    v = Decimal(val) / Decimal('1000000')
                    v = v.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                    return f"{v:,.2f} млн ₽"
                if kind == 'rub':
                    v = Decimal(val).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                    s = f"{v:,.2f}"
                    return s.replace(",", "X").replace(".", ",").replace("X", "\u202F") + " ₽"
                # int
                try:
                    return f"{int(round(val)):,}".replace(",", "\u202F")
                except:
                    return "0"

def delta_node(curr, prev, good_when_up=True, as_pct=True, w=110, ta="left", kind_abs='mln'):
    import math, pandas as pd
    if prev in (None, 0) or pd.isna(prev) or math.isclose(prev, 0.0):
        return dmc.Text("—", c="gray", ta=ta, w=w, ff="tabular-nums")

    diff = curr - prev
    is_up_good = (diff > 0) if good_when_up else (diff < 0)
    arrow_char = "▲" if diff > 0 else ("▼" if diff < 0 else "■")
    color = "green" if is_up_good else ("red" if diff != 0 else "gray")

    if as_pct:
        val = (diff / prev) * 100.0
        # округлим «по-человечески», без хвостов
        txt = f"{abs(Decimal(val).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))}%"
    else:
        # формат в зависимости от метрики
        txt = fmt_abs(abs(diff), kind_abs)

    return dmc.Text(f"{arrow_char} {txt}", c=color, ta=ta, w=w, ff="tabular-nums")



class StoresComponents:
    def __init__(self, df_id=None, period_label: str | None = None):
        self.df_id = df_id if df_id is not None else None
        self.period_label = period_label

        # IDs (pattern ids)
        self.chart_data_store_id   = {'type':'st_data_store','index':'1'}
        self.chart_series_store_id = {'type':'st_series_store','index':'1'}
        self.filters_data_store_id = {'type':'filter_store','index':'1'}
        self.raw_eom_store_id      = {'type':'st_raw_eom','index':'1'}

        self.chanel_multyselect_id = {'type':'chanel_multyselect','index':'1'}
        self.region_multyselect_id = {'type':'region_multyselect','index':'1'}
        self.store_multyselect_id  = {'type':'store_multyselect','index':'1'}
        self.metric_segment_id     = {'type':'st_metric','index':'1'}

        self.stores_area_chart_id  = {'type':'stores_area_chart','index':'1'}

        self.report_drawer_id      = {'type':'st_report_drawer','index':'1'}
        self.report_button_id      = {'type':'st_report_open','index':'1'}
        
        self.heatmap_wrap_id   = {'type':'st_heatmap_wrap','index':'1'}
        self.heatmap_metric_id = {'type':'st_heatmap_metric','index':'1'}
        self.heatmap_range_id  = {'type':'st_heatmap_range','index':'1'}
        self.daily_store_id    = {'type':'st_daily','index':'1'}
        self.heatmap_scope_id = {'type':'st_heatmap_scope','index':'1'}
 

    # ======================= COMPONENTS =======================





    def _heatmap_period_block(self, df_scope: pd.DataFrame, start, end, metric: str, *, is_dark: bool=False):

        import plotly.graph_objects as go


        # --- валидация
        if df_scope is None or df_scope.empty or pd.isna(start) or pd.isna(end):
            return dmc.Alert("Нет данных для теплокарты", color="gray", variant="light", radius="md")

        # --- приведение диапазона
        df = df_scope.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        start = pd.to_datetime(start).normalize()
        end   = pd.to_datetime(end).normalize()
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        if df.empty:
            return dmc.Alert("За выбранный период данных нет", color="yellow", variant="light", radius="md")

        # --- выбор метрики (amount | cr)
        val_col = "amount" if (metric or "amount") in ("amount", "aov", "amount") else "cr"
        day_sum = df.groupby(df["date"].dt.date)[val_col].sum().astype(float)

        # --- сетка недель (Пн–Вс)
        first_monday = (start - pd.Timedelta(days=start.weekday())).normalize()
        last_sunday  = (end + pd.Timedelta(days=(6 - end.weekday()))).normalize()
        weeks, cur = [], first_monday
        while cur <= last_sunday:
            weeks.append(cur)
            cur += pd.Timedelta(days=7)

        # --- матрицы
        z, custom_date, valid_mask, row_sums = [], [], [], []
        for ws in weeks:
            row_vals, row_dates, row_mask = [], [], []
            for d in range(7):
                the_day = (ws + pd.Timedelta(days=d)).date()
                if the_day < start.date() or the_day > end.date():
                    row_vals.append(np.nan); row_dates.append(""); row_mask.append(False)
                else:
                    v = float(day_sum.get(the_day, 0.0))
                    row_vals.append(v)
                    row_dates.append(pd.Timestamp(the_day).strftime("%d.%m"))
                    row_mask.append(True)
            z.append(row_vals); custom_date.append(row_dates); valid_mask.append(row_mask)
            row_sums.append(np.nansum(row_vals))

        z_arr = np.array(z, dtype=float)
        finite_vals = z_arr[np.isfinite(z_arr)]
        z_max = float(np.nanmax(finite_vals)) if finite_vals.size else 0.0

        # --- формат чисел
        if   z_max >= 1_000_000: div, suffix = 1_000_000.0, " млн"
        elif z_max >=   100_000: div, suffix = 1_000.0,     " тыс"
        else:                    div, suffix = 1.0,         " ₽"

        def fmt_sum(v: float) -> str:
            if not np.isfinite(v): return "—"
            if div == 1_000_000.0: return f"{v/div:.1f}{suffix}"
            if div == 1_000.0:     return f"{v/div:.1f}{suffix}"
            return f"{v:,.0f} ₽".replace(",", " ")

        weekday_names = ["Пн","Вт","Ср","Чт","Пт","Сб","Вс"]
        x_labels = weekday_names

        # подписи строк: «dd.mm–dd.mm — сумма»
        y_labels = []
        for i, ws in enumerate(weeks):
            we = ws + pd.Timedelta(days=6)
            y_labels.append(f"{ws.strftime('%d.%m')}–{we.strftime('%d.%m')} — {fmt_sum(row_sums[i])}")

        # --- цвета под тему
        text_primary = "#11181C" if not is_dark else "#E6E8EB"
        graph_bg = "rgba(0,0,0,0)"  # прозрачный — впишется в тему
        badge_bg = "rgba(255,255,255,0.80)" if not is_dark else "rgba(0,0,0,0.55)"
        badge_border = "rgba(0,0,0,0.18)" if not is_dark else "rgba(255,255,255,0.18)"
        badge_text_default = "#1F2A44" if not is_dark else "#E6E8EB"

        # --- теплокарта
        fig = go.Figure(
            data=go.Heatmap(
                z=z_arr,
                x=x_labels,
                y=y_labels,
                coloraxis="coloraxis",
                customdata=custom_date,
                hovertemplate="<b>%{customdata}</b><br>%{y} / %{x}<br>Значение: %{z:,.0f}<extra></extra>",
                zmin=None,           # позволяем <0
                zsmooth=False
            )
        )

        # порог контраста (для цвета ДАТЫ над плиткой)
        z_min = float(np.nanmin(finite_vals)) if finite_vals.size else 0.0
        z_mid = (z_min + (z_max if z_max > 0 else 1.0)) / 2.0

        # --- динамические размеры
        num_rows = len(weeks)
        row_h = 36 if num_rows <= 20 else (30 if num_rows <= 32 else 26)
        top_m, bot_m = 64, 44
        height_px = int(top_m + bot_m + num_rows * row_h)

        y_tick_size  = 12 if num_rows <= 20 else (11 if num_rows <= 32 else 10)
        date_font_sz = 12 if num_rows <= 20 else (11 if num_rows <= 32 else 10)
        sum_font_sz  = 11 if num_rows <= 20 else (10 if num_rows <= 32 else 9)

        # вертикальные сдвиги внутри ячейки
        yshift_date = int(row_h * 0.28)
        yshift_sum  = -int(row_h * 0.28)

        # --- аннотации: дата (верх) + сумма в «бейдже» (низ)
        for r_idx, y_lab in enumerate(y_labels):
            for c_idx, x_lab in enumerate(x_labels):
                if not valid_mask[r_idx][c_idx]:
                    continue
                raw_val = z_arr[r_idx, c_idx]
                val = float(raw_val) if np.isfinite(raw_val) else 0.0
                date_txt = custom_date[r_idx][c_idx]

                # 1) дата
                date_color = "#FFFFFF" if (val > z_mid) else text_primary
                fig.add_annotation(
                    x=x_lab, y=y_lab, text=date_txt,
                    showarrow=False, yshift=yshift_date,
                    font=dict(size=date_font_sz, color=date_color,
                            family="Inter, system-ui, -apple-system, Segoe UI")
                )

                # 2) сумма (красным, если < 0)
                value_color = "red" if val < 0 else badge_text_default
                fig.add_annotation(
                    x=x_lab, y=y_lab, text=fmt_sum(val),
                    showarrow=False, yshift=yshift_sum,
                    font=dict(size=sum_font_sz, color=value_color,
                            family="Inter, system-ui, -apple-system, Segoe UI"),
                    bgcolor=badge_bg,
                    bordercolor=badge_border,
                    borderwidth=1,
                    borderpad=2
                )

        # --- оформление
        colorbar_title = "Выручка, ₽" if val_col == "amount" else "Возвраты, ₽"
        fig.update_layout(
            height=height_px,
            margin=dict(l=30, r=30, t=56, b=36),
            coloraxis=dict(
                colorscale="Blues",
                colorbar=dict(
                    title=dict(text=colorbar_title, font=dict(color=text_primary)),  # <-- правильный способ
                    tickformat=",.0f",
                    tickfont=dict(color=text_primary)
                )
            ),
            xaxis=dict(type="category", tickfont=dict(size=12, color=text_primary)),
            yaxis=dict(type="category", tickfont=dict(size=y_tick_size, color=text_primary)),
            plot_bgcolor=graph_bg,
            paper_bgcolor=graph_bg,
            title=dict(
                text=f"Календарь {start.strftime('%d.%m.%Y')} — {end.strftime('%d.%m.%Y')}",
                x=0.5, xanchor="center",
                font=dict(size=16, color=text_primary,
                        family="Inter, system-ui, -apple-system, Segoe UI")
            )
        )

        return dcc.Graph(
            figure=fig,
            config={"displayModeBar": False},
            style={"height": f"{height_px}px"}
        )



    def create_components(self):
        if not self.df_id:
            return
        
        df_data: pd.DataFrame = load_df_from_redis(self.df_id)
        
        def store_full_df_for_returns():
            cols = [
                "date", "cr", "store_gr_name", "subcat", 'cat', 'fullname',
                "client_order", "quant_cr", "client_order_number", "manager",
                "chanel", "store_region", "brend", "manu",
            ]
            for c in cols:
                if c not in df_data.columns:
                    df_data[c] = None
            return dcc.Store(id="df_returns_store", data=df_data[cols].to_dict("records"), storage_type="memory")


        
        
        def heatmap_period_block():
            import pandas as pd

            data_min = pd.to_datetime(df_data['date'].min()).normalize()
            data_max = pd.to_datetime(df_data['date'].max()).normalize()

            today = pd.Timestamp.today().normalize()
            default_end = min(today, data_max)
            curr_month_start = default_end.replace(day=1)
            prev_month_start = (curr_month_start - pd.offsets.MonthBegin(1))
            default_start = max(prev_month_start, data_min)

            def _d(x): return x.to_pydatetime().date()

            controls = dmc.Group(
                justify="space-between", align="center", wrap="wrap", gap="sm",
                children=[
                    dmc.Group(gap="xs", align="center", children=[
                        DashIconify(icon="tabler:calendar-due", width=18),
                        dmc.Text("Календарь за период", fw=700),
                    ]),
                    dmc.Group(gap="sm", align="center", children=[
                        dmc.SegmentedControl(
                            id=self.heatmap_metric_id,
                            data=[{"label":"Выручка","value":"amount"},
                                {"label":"Возвраты","value":"cr"}],
                            value="amount", size="sm", radius="sm", color="blue"
                        ),
                        dmc.DatePickerInput(
                            id=self.heatmap_range_id,
                            type="range",
                            size="sm",
                            radius="sm",
                            allowSingleDateInRange=True,
                            value=[_d(default_start), _d(default_end)],
                            minDate=_d(data_min),
                            maxDate=_d(data_max),
                        ),
                        # ← вот этот бейдж будет меняться колбэком
                        dmc.Badge(
                            id=self.heatmap_scope_id,
                            children="Все магазины",
                            variant="outline", radius="xs", size="md",
                        ),
                    ]),
                ],
            )

            return dmc.Paper(
                withBorder=True, radius="md", p="md", shadow="sm",
                children=[
                    controls,
                    dmc.Space(h=6),
                    dcc.Loading(
                        id={'type': 'st_heatmap_loading', 'index': '1'},
                        type="circle",
                        fullscreen=False,
                        children=html.Div(
                            id=self.heatmap_wrap_id,
                            style={"minHeight": 360}
                        ),
                    ),
                ],
            )




        def store_daily_scope():
            # дневные данные для теплокарты (уважит фильтры в колбэке)
            need = ['date','amount','quant','dt','cr','store_gr_name','chanel','store_region']
            tmp = df_data.copy()
            for c in need:
                if c not in tmp.columns:
                    tmp[c] = 0 if c in ('amount','cr','dt','quant') else None
            return dcc.Store(id=self.daily_store_id, data=tmp[need].to_dict('records'), storage_type='memory')

        if df_data is None or df_data.empty:
            return
        df_data['store_region'] = df_data['store_region'].fillna('Регион не указан')

        # гарантируем наличие store_region
        if 'store_region' not in df_data.columns:
            df_data['store_region'] = 'Все регионы'

        # ======= ЗАКАЗЫ: считаем строго как в av_check =========
        # берём только нужные поля и убираем NaN в ключевых колонках
        df_store_checks = df_data[['eom','store_gr_name','dt','client_order_number']].dropna(
            subset=['eom','store_gr_name','dt','client_order_number']
        )
        # 1) сумма каждого чека по магазину и месяцу
        df_stores_checks_agg = (
            df_store_checks
            .groupby(['eom','store_gr_name','client_order_number'], as_index=False)['dt']
            .sum()
        )
        # 2) убираем нулевые чеки
        df_stores_checks_agg = df_stores_checks_agg[df_stores_checks_agg['dt'] != 0]

        # 3) агрегаты по магазинам: число заказов и сумма "живых" чеков (может пригодиться)
        df_stores_checks_analytics = (
            df_stores_checks_agg
            .groupby(['eom','store_gr_name'], as_index=False)
            .agg(orders=('client_order_number','nunique'),
                orders_amount=('dt','sum'))
        )
        # при желании можно посчитать средний чек по магазину:
        df_stores_checks_analytics['av_check'] = np.where(
            df_stores_checks_analytics['orders'] > 0,
            df_stores_checks_analytics['orders_amount'] / df_stores_checks_analytics['orders'],
            0.0
        )

        # ======= БАЗОВЫЕ СУММЫ ПО МЕТРИКАМ =======
        def present(cols): 
            return [c for c in cols if c in df_data.columns]

        value_cols = present(['dt','cr','amount','quant','quant_dt','quant_cr'])
        df_eom = (
            df_data.pivot_table(
                index=['eom','store_gr_name','chanel','store_region'],
                values=value_cols,
                aggfunc='sum'
            )
            .fillna(0).reset_index().sort_values('eom')
        )

        # мержим корректные заказы и средний чек
        df_eom = df_eom.merge(
            df_stores_checks_analytics[['eom','store_gr_name','orders','av_check']],
            on=['eom','store_gr_name'], how='left'
        )
        df_eom['orders']   = df_eom['orders'].fillna(0).astype(int)
        df_eom['av_check'] = df_eom['av_check'].fillna(0.0)

        # фильтры
        df_filters = df_eom[['store_gr_name','chanel','store_region']].drop_duplicates()

        # ---- stores
        def store_filters():
            return dcc.Store(
                id=self.filters_data_store_id,
                data=df_filters.to_dict('records'),
                storage_type='memory'
            )

        def store_raw():
            wanted = ['eom','store_gr_name','chanel','store_region',
                    'amount','dt','cr','quant','quant_dt','quant_cr',
                    'orders','av_check', ]
            tmp = df_eom.copy()
            for c in wanted:
                if c not in tmp.columns:
                    tmp[c] = 0
            return dcc.Store(
                id=self.raw_eom_store_id,
                data=tmp[wanted].to_dict('records'),
                storage_type='memory'
            )

        # ---- right drawer (отчёт)
        def report_drawer():
            return dmc.Drawer(
                id=self.report_drawer_id,
                title="Отчёт",
                position="right",
                size=520,
                padding="md",
                opened=False,
                children=html.Div()  # наполняем колбэком
            )

        # ---- sidebar / controls (как у тебя было)
        def controls():
            return dmc.Card(
                withBorder=True, shadow="xs", radius=0, p="md",
                children=[
                    dmc.Group(
                        justify="space-between", align="center", mb=8,
                        children=[
                            dmc.Group(
                                gap="sm", align="center",
                                children=[
                                    dmc.ThemeIcon(radius=0, size="lg", variant="transparent",
                                                children=DashIconify(icon="tabler:adjustments-alt")),
                                    dmc.Text("Панель настроек", fw=700),
                                ],
                            ),
                            dmc.Group(
                                gap="xs",
                                children=[
                                    dmc.Tooltip(label="Сбросить фильтры", withArrow=True),
                                    dmc.ActionIcon(
                                        id={'type': 'st_controls_reset', 'index': '1'},
                                        radius=0, variant="light",
                                        children=DashIconify(icon="tabler:refresh"),
                                    ),
                                    # dmc.Tooltip(label="Отчёт по текущему отбору", withArrow=True),
                                    # dmc.ActionIcon(
                                    #     id=self.report_button_id,
                                    #     radius=0, variant="light",
                                    #     children=DashIconify(icon="tabler:report-analytics"),
                                    # ),
                                ],
                            ),
                        ],
                    ),
                    dmc.Divider(variant="dashed", my=10),
                    dmc.Stack(gap="sm", children=[
                        dmc.MultiSelect(
                            id=self.chanel_multyselect_id, label="Канал", placeholder="Выберите канал",
                            data=sorted(df_filters["chanel"].dropna().unique().tolist()),
                            w="100%", radius=0, clearable=True, searchable=True,
                            leftSection=DashIconify(icon="tabler:arrows-shuffle"),
                        ),
                        dmc.MultiSelect(
                            id=self.region_multyselect_id, label="Регион", placeholder="Выберите регион",
                            data=sorted(df_filters["store_region"].dropna().unique().tolist()),
                            w="100%", radius=0, clearable=True, searchable=True,
                            leftSection=DashIconify(icon="tabler:map-pin"),
                        ),
                        dmc.MultiSelect(
                            id=self.store_multyselect_id, label="Магазин", placeholder="Выберите магазин",
                            data=sorted(df_filters["store_gr_name"].dropna().unique().tolist()),
                            w="100%", radius=0, clearable=True, searchable=True,
                            leftSection=DashIconify(icon="tabler:building-store"),
                        ),
                        dmc.SegmentedControl(
                            id=self.metric_segment_id,
                            data=[
                                seg_item("tabler:chart-line",       "Выручка",            "amount"),   
                                seg_item("tabler:cash-banknote",    "Возвраты ₽",         "ret_amt_abs"),   
                                seg_item("tabler:box-seam",         "Возвраты шт",        "ret_qty_abs"),  
                                seg_item("tabler:arrows-left-right","Коэф. возвратов, %", "ret_coef"),   
                                seg_item("tabler:shopping-cart",    "Кол-во заказов",     "count_order"), 
                                seg_item("tabler:credit-card",      "Средний чек",        "avr_recept"),  
                            ],
                            value="amount",
                            radius="sm",
                            fullWidth=True,
                            orientation="vertical",
                            color="blue",
                        ),
                        dmc.Divider(variant="dashed", my=10),
                        dmc.Switch(
                            id={"type": "st_switch_legend", "index": "1"},
                            label="Показывать легенду", checked=True, radius="xs",
                        ),
                        dmc.Switch(
                            id={"type": "st_switch_points", "index": "1"},
                            label="Маркеры точек", checked=False, radius="xs",
                        ),
                    ]),
                ],
            )

        # ---- main chart 
        def chart():
            df = df_eom.pivot_table(index='eom', columns='store_gr_name', values='amount', aggfunc='sum') \
                    .fillna(0).reset_index().sort_values('eom')
            df["eom"] = pd.to_datetime(df["eom"], errors="coerce")
            df["month_name"] = df["eom"].dt.strftime("%b\u202F%y").str.capitalize()
            store_cols = [c for c in df.columns if c not in ["eom","month_name"]]
            df["Все магазины"] = df[store_cols].sum(axis=1)
            data = df.to_dict("records")

            all_cols = ["Все магазины"] + store_cols
            series_full = []
            for i, col in enumerate(all_cols):
                color = BASE_COLORS[0] if col == "Все магазины" else COLORS_BY_SHADE[i % len(COLORS_BY_SHADE)]
                series_full.append({"name": col, "color": color})
            default_series = [s for s in series_full if s["name"] == "Все магазины"]

            return dmc.Stack([
                dmc.AreaChart(
                    id=self.stores_area_chart_id, h=600, dataKey='month_name',
                    data=data, series=default_series,
                    tooltipAnimationDuration=500,
                    areaProps={
                        "isAnimationActive": True, "animationDuration": 500,
                        "animationEasing": "ease-in-out", "animationBegin": 500,
                    },
                    withPointLabels=False,
                    valueFormatter={"function": "formatNumberIntl"},
                    withLegend=True, 
                    connectNulls=True,
                    yAxisProps={"domain": ["dataMin", "auto"]},
                ),
                dcc.Store(id=self.chart_data_store_id, data=data, storage_type='memory'),
                dcc.Store(id=self.chart_series_store_id, data=series_full, storage_type='memory'),

              
            ])

       

        def memo():
            # ===== период вида "01 МАР 24 - 29 СЕН 25" =====
            MONTHS_RU_3 = ["ЯНВ","ФЕВ","МАР","АПР","МАЙ","ИЮН","ИЮЛ","АВГ","СЕН","ОКТ","НОЯ","ДЕК"]
            def fmt_dd_MMM_yy(d):
                if d is None or pd.isna(d): return ""
                d = pd.to_datetime(d)
                return f"{d.day:02d} {MONTHS_RU_3[d.month-1]} {d.strftime('%y')}"

            d_min = pd.to_datetime(df_data['date'].min(), errors='coerce')
            d_max = pd.to_datetime(df_data['date'].max(), errors='coerce')
            period_text = f"{fmt_dd_MMM_yy(d_min)} - {fmt_dd_MMM_yy(d_max)}"

            # --- агрегаты на стартовый рендер (base=period, delta=%) ---
            total_sales   = float(df_data['dt'].sum())
            total_net     = float(df_data['amount'].sum())
            total_returns = float(df_data['cr'].sum())
            total_orders  = int(df_eom['orders'].sum()) if 'orders' in df_eom.columns else 0

            eom_series = pd.to_datetime(df_eom['eom'], errors='coerce')
            first_eom, last_eom = eom_series.min(), eom_series.max()

            def _fmt_mln(x): 
                try: return f"{x/1_000_000:,.2f} млн ₽"
                except: return "0,00 млн ₽"
            def _fmt_int(x):
                try: return f"{int(x):,}".replace(",", "\u202F")
                except: return "0"

            def sum_at(col, eom):
                if col not in df_eom.columns or pd.isna(eom): return 0.0
                m = pd.to_datetime(df_eom['eom']) == eom
                return float(df_eom.loc[m, col].sum())

            first_amount, last_amount = sum_at('amount', first_eom), sum_at('amount', last_eom)
            first_dt,     last_dt     = sum_at('dt', first_eom),     sum_at('dt', last_eom)
            first_cr,     last_cr     = sum_at('cr', first_eom),     sum_at('cr', last_eom)
            first_ord,    last_ord    = sum_at('orders', first_eom), sum_at('orders', last_eom)

            
            def metric_row(label, value_text, delta):
                return dmc.Group(
                    gap="sm", align="center",
                    children=[
                        dmc.Text(f"{label}:", w=180, ta="left"),
                        dmc.Text(value_text, fw=700, w=140, ta="right", ff="tabular-nums"),
                        delta
                    ]
                )

            # стартовые строки (base=первый месяц, delta=%)
            init_list = html.Ul(
                style={"listStyleType":"disc","margin":0,"paddingLeft":"1.2rem"},
                children=[
                    html.Li(metric_row("Чистая выручка", _fmt_mln(total_net),
                                    delta_node(last_amount, first_amount, True, True))),
      
                    html.Li(metric_row("Общие продажи",  _fmt_mln(total_sales),
                                    delta_node(last_dt, first_dt, True, True))),
                    
                    html.Li(metric_row("Кол-во заказов", _fmt_int(total_orders),
                                    delta_node(last_ord, first_ord, True, True))),
                    html.Li(metric_row("Возвраты",       _fmt_mln(total_returns),
                                    delta_node(last_cr, first_cr, good_when_up=False, as_pct=True))),
                ]
            )

            # ===== UI =====
            header = dmc.Group(
                gap="xs",
                children=[
                    dmc.Text(f"Краткий отчёт за период: {period_text}", fw=700),
            
                ],
            )


            
            
            controls = dmc.Paper(
                withBorder=True, shadow="xs", radius="sm", p="sm",
                style={"background": "rgba(0,102,255,0.04)"},
                children=dmc.Group(
                    justify="space-between", align="center", gap="sm", wrap="wrap",
                    children=[
                        dmc.Group(
                            gap="md", wrap="wrap",
                            children=[
                                # База сравнения (period / last_month / custom)
                                dmc.Stack(gap=2, w=360, children=[
                                    dmc.Group(gap=6, align="center", children=[
                                        DashIconify(icon="tabler:calendar-stats", width=14),
                                        dmc.Text("База сравнения", size="xs", c="dimmed"),
                                    ]),
                                    dmc.SegmentedControl(
                                        id={'type':'sum_base_mode','index':'1'},
                                        data=[
                                            {"label":"за период",   "value":"period"},
                                            {"label":"посл. месяц", "value":"last_month"},
                                            {"label":"выбрать…",    "value":"custom"},
                                        ],
                                        value="period", size="sm", radius="sm", color="blue", fullWidth=True
                                    ),
                                   dmc.Box(
                                        id={'type':'sum_base_custom_box','index':'1'},
                                        children=dmc.MonthPickerInput(
                                            id={'type':'sum_base_custom','index':'1'},
                                            placeholder="Выберите месяц",
                                            size="sm",
                                            value=None,
                                            clearable=True,
                                            w="100%"
                                        ),
                                        style={"display":"none"}  # показываем только при base_mode=custom
                                    )
                                ]),

                                # Формат дельты
                                dmc.Stack(gap=2, w=170, children=[
                                    dmc.Group(gap=6, align="center", children=[
                                        DashIconify(icon="tabler:arrows-diff", width=14),
                                        dmc.Text("Формат дельты", size="xs", c="dimmed"),
                                    ]),
                                    dmc.SegmentedControl(
                                        id={'type':'sum_delta_mode','index':'1'},
                                        data=[{"label":"Абс.","value":"abs"},{"label":"%","value":"pct"}],
                                        value="pct", size="sm", radius="sm", color="blue", fullWidth=True
                                    ),
                                ]),

                                # Формат суммы
                                dmc.Stack(gap=2, w=190, children=[
                                    dmc.Group(gap=6, align="center", children=[
                                        DashIconify(icon="tabler:currency-ruble", width=14),
                                        dmc.Text("Формат суммы", size="xs", c="dimmed"),
                                    ]),
                                    dmc.SegmentedControl(
                                        id={'type':'sum_number_format','index':'1'},
                                        data=[{"label":"млн ₽","value":"mln"},{"label":"полные ₽","value":"full"}],
                                        value="mln", size="sm", radius="sm", color="blue", fullWidth=True
                                    ),
                                ]),
                            ],
                        ),

                        # подпись сравнения справа
                        dmc.Badge(
                            id={'type':'sum_caption','index':'1'},
                            size="md", radius="sm", variant="outline",
                            style={"whiteSpace":"nowrap"}
                        ),
                    ]
                )
            )
            
            
            
            
        


            
            df_store_total = (
                df_eom.groupby('store_gr_name', as_index=False)['dt'].sum()
                .rename(columns={'dt': 'sum_dt'})
            )
            def store_at(eom):
                m = pd.to_datetime(df_eom['eom']) == eom
                return (df_eom.loc[m]
                        .groupby('store_gr_name', as_index=False)['dt'].sum()
                        .rename(columns={'dt': 'val'}))
            st_first = store_at(first_eom).rename(columns={'val':'first_dt'})
            st_last  = store_at(last_eom).rename(columns={'val':'last_dt'})
            st = (df_store_total.merge(st_first, on='store_gr_name', how='left')
                                .merge(st_last,  on='store_gr_name', how='left')
                                .fillna(0.0)
                                .sort_values('sum_dt', ascending=False))
            def store_row(rank, name, sum_dt, first_dt_s, last_dt_s):
                share_pct = float(sum_dt / total_sales * 100) if total_sales else 0.0
                return dmc.Group(
                    gap="sm", align="center",
                    children=[
                        dmc.Badge(str(rank), variant="filled", color="teal", w=40, ta="center", radius="xs",),
                        dmc.Text(str(name), w=220, ta="left"),
                        dmc.Text(_fmt_mln(sum_dt), fw=600, w=140, ta="right"),
                        
                        delta_node(last_dt_s, first_dt_s, True, True, w=90),
                        dmc.Group(align="center", gap=8, children=[
                            dmc.Progress(value=share_pct, w=180, size="lg", radius="xs"),
                            dmc.Text(f"{share_pct:.1f}%", w=44, ta="right", c="dimmed", ff="tabular-nums"),
                        ]),
                    ]
                )
            store_block = dmc.Stack(
                gap="xs",
                children=[store_row(i+1, r['store_gr_name'], r['sum_dt'], r['first_dt'], r['last_dt'])
                        for i, (_, r) in enumerate(st.iterrows())]
            )

            return dmc.Alert(
                title=None, color="blue", radius="md", variant="light",
                children=[
                    header,
                    dmc.Divider(variant="dashed", my=8),
                    controls,
                    dmc.Divider(variant="dashed", my=8),

                
                dcc.Loading(
                    id={'type':'sum_loading','index':'1'},          
                    children=dmc.SimpleGrid(
                        id={'type':'sum_rows','index':'1'},
                        cols=2, spacing="md",
                        children=[]
                    ),

                ),
                    dmc.Divider(variant="dashed", my=8),
                    
                    
                    dmc.Spoiler(
                        showLabel=dmc.Badge(
                            "Продажи по магазинам",
                            variant="light", color="blue", radius="xs", size="md",
                        ),
                        hideLabel=dmc.Badge(
                            "Скрыть",
                            variant="light", color="gray", radius="xs", size="md",
                            leftSection=DashIconify(icon="tabler:chevron-up")
                        ),
                        maxHeight=0, transitionDuration=200,
                        children=[
                            # dmc.Paper(
                            #     withBorder=True, radius="md", p="sm", mt="xs",
                            #     children=dcc.Loading(
                            #         id={'type':'store_loading','index':'1'},    
                            #         children=dmc.Stack(
                            #             id={'type':'store_block','index':'1'},
                            #             gap="xs"
                            #         ),
                                    
                            #     )
                            # )
                            dmc.Paper(
    withBorder=True, radius="md", p="sm", mt="xs",
    children=[
        dmc.Group(
            gap="xs", align="center",
            children=[
                dmc.SegmentedControl(
                    id={'type':'store_metric_mode','index':'1'},
                    value="amount",
                    data=[
                        {"label": "Выручка", "value": "amount"},
                        {"label": "Возвраты ₽", "value": "cr"},
                        {"label": "Коэф. возвратов %", "value": "cr_ratio"},
                        {"label": "Средний чек", "value": "avg_check"},
                    ],
                    size="sm", radius="sm", color="blue",
                ),
                dmc.Text("Сортировка и доля — по выручке текущего месяца", c="dimmed", size="xs")
            ]
        ),
        dmc.Divider(variant="dashed", my=8),

        dcc.Loading(
            id={'type':'store_loading','index':'1'},
            type="circle",
            children=dmc.Stack(id={'type':'store_block','index':'1'}, gap="xs"),
        ),
    ]
)

                        ]
                    )

                ]
            )
            
          
        
        returns_modal = dmc.Modal(
            id="returns_modal",
            title=dmc.Group([
                DashIconify(icon="tabler:eye", width=18),
                dmc.Text("Детализация возвратов", fw=700),
                dmc.SegmentedControl(
                    id="returns_range",
                    data=[
                        {"label": "Последний месяц", "value": "last"},
                        {"label": "Весь период",     "value": "all"},
                    ],
                    value="last", size="sm", radius="xs", color="blue",
                ),
            ]),
            size="90%",
            centered=True,
            overlayProps={"opacity": 0.55, "blur": 2},
            children=dmc.Box([
                
                dmc.LoadingOverlay(
                id="returns_loading",
                visible=False,
                zIndex=1000,
                overlayProps={"opacity": 0.6, "blur": 2},
            ),
                
                # ——— ПОНЧИК — распределение возвратов  ———

                dmc.SimpleGrid(
                    cols=2,
                    spacing="md",
                    children=[
                        # ===== ПОНЧИК №1: по категориям =====
                        dmc.Paper(
                            withBorder=True, radius="md", p="sm",
                            children=[
                                dmc.Group(
                                    justify="space-between", align="center",
                                    children=[
                                        dmc.Group(gap="xs", children=[
                                            DashIconify(icon="tabler:chart-donut", width=18),
                                            dmc.Text("Распределение возвратов по категориям", fw=700),
                                            dmc.Space(h=10),
                                        ]),
                                    ],
                                ),
                                dmc.Group(
                                    align="start", gap="lg", wrap=False,
                                    children=[
                                        dcc.Loading(

              
                                            children=dmc.DonutChart(
                                                id="returns_cat_donut",
                                                data=[],
                                                withTooltip=True,
                                                withLabels=False,
                                                strokeWidth=2,
                                                paddingAngle=2,
                                                chartLabel="",
                                              
                                     
                        
                                                size=260,
                                                thickness=35,
                                            ),
                                        ),
                                        dmc.Stack(
                                            id="returns_cat_legend",
                                            gap=6,
                                            style={"minWidth": 260, "flex": 1},
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        # ===== ПОНЧИК №2: по производителям =====
                        dmc.Paper(
                            withBorder=True, radius="md", p="sm",
                            children=[
                                dmc.Group(
                                    justify="space-between", align="center",
                                    children=[
                                        dmc.Group(gap="xs", children=[
                                            DashIconify(icon="fluent:building-factory-16-regular", width=18),
                                            dmc.Text("Распределение возвратов по производителям", fw=700),
                                            dmc.Space(h=10),
                                        ]),
                                    ],
                                ),
                                dmc.Group(
                                    align="start", gap="lg", wrap=False,
                                    children=[
                                        dcc.Loading(
                 
                                            children=dmc.DonutChart(
                                                id="returns_manu_donut",
                                                data=[],
                                                withTooltip=True,
                                                withLabels=False,
                                                strokeWidth=2,
                                                paddingAngle=2,
                                                chartLabel="",
                                                
                        
                                                size=260,
                                                thickness=35,
                                            ),
                                        ),
                                        dmc.Stack(
                                            id="returns_manu_legend",
                                            gap=6,
                                            style={"minWidth": 260, "flex": 1},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                
                dmc.Space(h=15),





                # ——— ТАБЛИЦА — возвраты ———
                dcc.Loading(

                    children=dag.AgGrid(
                        id="returns_grid",
                        columnDefs=[
                            {"headerName": "Дата", "field": "date", "minWidth": 120,
                            "cellClass": "ag-firstcol-bg", "pinned": "left", "headerClass": "ag-center-header"},

                            {"headerName": "№ заказа", "field": "client_order_number", "minWidth": 130,
                            "headerClass": "ag-center-header"},

                            # ГРУППА «Номенклатура»
                            {
                                "headerName": "Номенклатура",
                                "groupId": "product",
                                "marryChildren": True,
                                "headerClass": "ag-center-header",
                                "children": [
                                    {"headerName": "Номенклатура", "field": "fullname",
                                    "minWidth": 220, "type": "leftAligned", "cellClass": "ag-firstcol-bg",
                                    "headerClass": "ag-center-header"},

                                    {"headerName": "Бренд", "field": "brend",
                                    "minWidth": 180, "type": "leftAligned", "columnGroupShow": "open",
                                    "headerClass": "ag-center-header",
                                    # опционально: подставляем «Бренд не указан» прямо в ячейке
                                    # "valueGetter": {"function": "(p)=> (p.data && p.data.brend && String(p.data.brend).trim()) ? p.data.brend : 'Бренд не указан'"}
                                    },

                                    {"headerName": "Производитель", "field": "manu",
                                    "minWidth": 180, "type": "leftAligned", "columnGroupShow": "open",
                                    "headerClass": "ag-center-header",
                                    # "valueGetter": {"function": "(p)=> (p.data && p.data.manu && String(p.data.manu).trim()) ? p.data.manu : 'Производитель не указан'"}
                                    },

                                    {"headerName": "Категория", "field": "cat",
                                    "minWidth": 160, "type": "leftAligned", "columnGroupShow": "open",
                                    "headerClass": "ag-center-header"},

                                    {"headerName": "Подкатегория", "field": "subcat",
                                    "minWidth": 180, "type": "leftAligned", "columnGroupShow": "open",
                                    "headerClass": "ag-center-header"},
                                ],
                            },

                            {"headerName": "Кол-во", "field": "quant_cr", "minWidth": 100, "type": "centerAligned",
                            "valueFormatter": {"function": "params.value != null ? d3.format(',.0f')(params.value).replace(/,/g, '\\u202F') : ''"},
                            "headerClass": "ag-center-header"},

                            {"headerName": "Возвраты ₽", "field": "cr", "minWidth": 130, "type": "rightAligned",
                            "valueFormatter": {"function": "params.value ? '₽'+ d3.format(',.2f')(params.value).replace(/,/g, '\\u202F') : ''"},
                            "cellClass": "ag-firstcol-bg", "headerClass": "ag-center-header"},

                            {"headerName": "Магазин", "field": "store_gr_name", "minWidth": 160, "type": "leftAligned",
                            "headerClass": "ag-center-header"},

                            {"headerName": "Менеджер", "field": "manager", "minWidth": 150, "type": "leftAligned",
                            "pinned": "right", "headerClass": "ag-center-header"},
                        ],
                        defaultColDef={
                            "sortable": True,
                            "filter": True,
                            "resizable": True,
                            "floatingFilter": True,
                            "headerClass": "ag-center-header",  # центр для всех заголовков по умолчанию
                        },
                        rowData=[],  # заполняется колбэком
                        dashGridOptions={
                            "rowHeight": 32,
                            "animateRows": True,
                            "domLayout": "normal",
                        },
                        style={"width": "100%", "height": "70vh"},
                        className="ag-theme-alpine",
                        dangerously_allow_code=True
                    ),
                ),
            ]),
        )







            





        # ---- right drawer
        def report_drawer():
            return dmc.Drawer(
                id=self.report_drawer_id,
                title="Отчёт",
                position="right",
                size=520,
                padding="md",
                opened=False,
                children=html.Div()
            )

        # вернём все 6 элементов (как ожидает tab_layout)
        return (
            store_filters(),
            store_raw(),
            controls(),
            chart(),
            report_drawer(),
            memo(),
            returns_modal,
            store_full_df_for_returns(),
            heatmap_period_block(),   
            store_daily_scope(),    

        )
    
    



    # ======================= LAYOUT =======================

    def tab_layout(self):
        if not self.df_id:
            return NoData().component
        filter_store, raw_store, controls, chart, report_drawer, memo, returns_modal, df_returns_store, heatmap_block, daily_store = self.create_components()
        return dmc.Container(
            fluid=True,
            children=[
                dmc.Title('Динамика продаж по магазинам', order=3, c='blue'),
                dmc.Space(h=6),
                memo,
                dmc.Space(h=10),
                heatmap_block,
                dmc.Grid(
                    gutter="lg", align="stretch",
                    children=[
                        dmc.GridCol(
                            span={"base": 12, "lg": 3},
                            children=dmc.Card(
                                withBorder=True, radius=0, shadow="sm", p="md",
                                style={"position": "sticky", "top": 70, "alignSelf": "start"},
                                children=controls,
                            ),
                        ),
                        dmc.GridCol(
                            span={"base": 12, "lg": 9},
                            children=[
                                dmc.Title('График продаж по магазинам', order=4),
                                dmc.Space(h=10),
                                chart,
                            ],
                        ),
                    ],
                ),
                dmc.Space(h=6),
                filter_store,
                raw_store,
                report_drawer,
                returns_modal, 
                df_returns_store, 
                daily_store,
                StoreAreaChartModal().create_components(),
   

            ],
        )
    
    

        



    # ======================= CALLBACKS =======================

    def register_callbacks(self, app):
        area_chart   = self.stores_area_chart_id['type']
        ch_filter    = self.chanel_multyselect_id['type']
        rg_filter    = self.region_multyselect_id['type']
        st_filter    = self.store_multyselect_id['type']
        metric_id    = self.metric_segment_id['type']
        series_store = self.chart_series_store_id['type']
        data_store   = self.chart_data_store_id['type']
        filter_data  = self.filters_data_store_id['type']
        modal        = StoreAreaChartModal().modal_id['type']
        modal_cont   = StoreAreaChartModal().container_id['type']

        # модальные колбэки из твоего класса
        StoreAreaChartModal().registered_callbacks(app)

        @app.callback(
            Output({"type": modal, "index": MATCH}, "opened"),
            Output({"type": modal_cont, "index": MATCH}, "children"),
            Output({"type": area_chart, "index": MATCH}, "clickData"),   
            Input({"type": area_chart, "index": MATCH}, "clickData"),
            Input({"type": area_chart, "index": MATCH}, "clickSeriesName"),
            Input({"type": modal, "index": MATCH}, "opened"),
            prevent_initial_call=True,
        )
        def handle_modal_and_click_reset(clickData, clickSeriesName, opened):
            trig = dash.ctx.triggered_id  
            if isinstance(trig, dict) and trig.get("type") == area_chart and trig.get("index") is not None:
                if clickData:
                    cont = StoreAreaChartModal(clickData, clickSeriesName).update_modal()
                    return True, cont, no_update  

            if opened is False:
                return no_update, no_update, None   
            return no_update, no_update, no_update
        









        # === главный апдейт: фильтры + выбор метрики ===
        @app.callback(
            Output({'type': st_filter,  'index': MATCH}, 'data'),        # options "Магазин"
            Output({'type': area_chart, 'index': MATCH}, 'series'),      # серии
            Output({'type': area_chart, 'index': MATCH}, 'data'),        # данные графика
            Output({'type': data_store, 'index': MATCH}, 'data'),        # кэш данных
            Output({'type': area_chart, 'index': MATCH}, 'valueFormatter'),  # форматтер значений
            Input({'type': ch_filter,   'index': MATCH}, 'value'),
            Input({'type': rg_filter,   'index': MATCH}, 'value'),
            Input({'type': st_filter,   'index': MATCH}, 'value'),
            Input({'type': metric_id,   'index': MATCH}, 'value'),       # amount | ret_amt_abs | ret_qty_abs | ret_coef
            State({'type': filter_data, 'index': MATCH}, 'data'),
            State({'type': series_store,'index': MATCH}, 'data'),
            State({'type': 'st_raw_eom','index': MATCH}, 'data'),
            prevent_initial_call=True,
        )
        def update_chart(ch_val, rg_val, st_val, metric, filter_df, series_val, raw_eom):
            def L(x):
                if x is None: return []
                return x if isinstance(x, list) else [x]

            # --- фильтрация сырых данных
            rdf = pd.DataFrame(raw_eom)
            if L(ch_val): rdf = rdf[rdf['chanel'].isin(L(ch_val))]
            if L(rg_val): rdf = rdf[rdf['store_region'].isin(L(rg_val))]
            
            

            # --- ветки по метрике
            if metric == 'amount':
                formatter = {"function": "formatNumberIntl"}
                pivot = (
                    rdf.pivot_table(index='eom', columns='store_gr_name', values='amount', aggfunc='sum')
                    .fillna(0).sort_index().reset_index()
                )

            elif metric == 'ret_amt_abs':
                formatter = {"function": "formatNumberIntl"}
                pivot = (
                    rdf.pivot_table(index='eom', columns='store_gr_name', values='cr', aggfunc='sum')
                    .fillna(0).sort_index().reset_index()
                )

            elif metric == 'ret_qty_abs':
                formatter = {"function": "formatIntl"}   # целые значения
                pivot = (
                    rdf.pivot_table(index='eom', columns='store_gr_name', values='quant_cr', aggfunc='sum')
                    .fillna(0).sort_index().reset_index()
                )
            
            elif metric == 'count_order':
                formatter = {"function": "formatIntl"}  # целые
                pivot = (
                    rdf.pivot_table(
                        index='eom',
                        columns='store_gr_name',
                        values='orders',
                        aggfunc='first'   # <--- не суммируем повторно
                    )
                    .fillna(0).sort_index().reset_index()
                )
                # «Все магазины» = сумма по магазинам
                store_cols = [c for c in pivot.columns if c not in ['eom']]
                pivot['Все магазины'] = pivot[store_cols].sum(axis=1)


            elif metric == 'avr_recept':
                # Средний чек = сумма amount / сумма orders
                formatter = {"function": "formatNumberIntl"}
                by = (
                    rdf.groupby(['eom','store_gr_name'])
                    .agg(amount_sum=('dt','sum'), orders=('orders','sum'))
                    .reset_index()
                )
                by['value'] = np.where(by['orders'] > 0, by['amount_sum'] / by['orders'], 0.0)
                pivot = (
                    by.pivot_table(index='eom', columns='store_gr_name', values='value', aggfunc='mean')
                    .fillna(0).sort_index().reset_index()
                )
                
                tot = (
                    rdf.groupby('eom')
                    .agg(amount_sum=('dt','sum'), orders=('orders','sum'))
                    .reset_index()
                )
                tot['Все магазины'] = np.where(tot['orders'] > 0, tot['amount_sum'] / tot['orders'], 0.0)
                pivot = pivot.merge(tot[['eom','Все магазины']], on='eom', how='left')

            
            

            else:  # 'ret_coef' — коэффициент возвратов (₽), %
                formatter = {"function": "formatPercentIntl"}
                g = rdf.groupby(['eom','store_gr_name'])[['cr','dt']].sum().reset_index()
                g['value'] = np.where(g['dt'] > 0, 100.0 * g['cr'] / g['dt'], 0.0)
                pivot = (
                    g.pivot_table(index='eom', columns='store_gr_name', values='value', aggfunc='mean')
                    .fillna(0).sort_index().reset_index()
                )
                # pooled ratio для «Все магазины»
                tot = rdf.groupby('eom')[['cr','dt']].sum().reset_index()
                tot['Все магазины'] = np.where(tot['dt'] > 0, 100.0 * tot['cr'] / tot['dt'], 0.0)
                pivot = pivot.merge(tot[['eom','Все магазины']], on='eom', how='left')

            # --- сумма «Все магазины» для абсолютных метрик
            if metric in ('amount', 'ret_amt_abs', 'ret_qty_abs'):
                store_cols = [c for c in pivot.columns if c not in ['eom']]
                pivot['Все магазины'] = pivot[store_cols].sum(axis=1)

            # --- оформление дат
            pivot['eom'] = pd.to_datetime(pivot['eom'], errors='coerce')
            pivot = pivot.sort_values('eom').reset_index(drop=True)
            pivot['month_name'] = pivot['eom'].dt.strftime("%b\u202F%y").str.capitalize()

            # переносим month_name в конец
            cols_no_mn = [c for c in pivot.columns if c != 'month_name']
            pivot = pivot[cols_no_mn + ['month_name']]
            chart_data = pivot.to_dict('records')

            # --- options для "Магазин" после фильтров
            fdf = pd.DataFrame(filter_df)
            if L(ch_val): fdf = fdf[fdf['chanel'].isin(L(ch_val))]
            if L(rg_val): fdf = fdf[fdf['store_region'].isin(L(rg_val))]
            store_options = [{"value": s, "label": s} for s in sorted(fdf['store_gr_name'].unique().tolist())]

            available = {o["value"] for o in store_options}
            selected  = [s for s in L(st_val) if s in available]

            def series_by(names): return [s for s in series_val if s["name"] in names]
            def total_series():   return [s for s in series_val if s["name"] == "Все магазины"]

            out_series = series_by(selected) if selected else total_series()

            return store_options, out_series, chart_data, chart_data, formatter



        # переключатели легенды/поинтов
        @app.callback(
            Output({'type': area_chart, 'index': MATCH}, 'withLegend'),
            Output({'type': area_chart, 'index': MATCH}, 'withPointLabels'),
            Input({'type': 'st_switch_legend', 'index': MATCH}, 'checked'),
            Input({'type': 'st_switch_points', 'index': MATCH}, 'checked'),
            prevent_initial_call=True,
        )
        def update_chart_view(legend_on, points_on):
            return bool(legend_on), bool(points_on)

        # === отчёт в правом Drawer: открытие + генерация контента ===
        @app.callback(
            Output({'type': 'st_report_drawer', 'index': MATCH}, 'opened'),
            Output({'type': 'st_report_drawer', 'index': MATCH}, 'title'),
            Output({'type': 'st_report_drawer', 'index': MATCH}, 'children'),
            Input({'type': 'st_report_open', 'index': MATCH}, 'n_clicks'),
            State({'type': ch_filter, 'index': MATCH}, 'value'),
            State({'type': rg_filter, 'index': MATCH}, 'value'),
            State({'type': st_filter, 'index': MATCH}, 'value'),
            State({'type': metric_id,  'index': MATCH}, 'value'),
            State({'type': 'st_raw_eom', 'index': MATCH}, 'data'),
            prevent_initial_call=True,
        )
        def open_report(n, ch_val, rg_val, st_val, metric, raw_eom):
            def L(x): return [] if x is None else (x if isinstance(x, list) else [x])

            rdf = pd.DataFrame(raw_eom)
            if L(ch_val): rdf = rdf[rdf['chanel'].isin(L(ch_val))]
            if L(rg_val): rdf = rdf[rdf['store_region'].isin(L(rg_val))]
            scope = "Все магазины" if not L(st_val) else ", ".join(L(st_val))
            if L(st_val): rdf = rdf[rdf['store_gr_name'].isin(L(st_val))]

            # агрегаты для текста
            rdf['eom'] = pd.to_datetime(rdf['eom'])
            last_eom = rdf['eom'].max()
            uniq_sorted = sorted(rdf['eom'].dropna().unique())
            prev_eom = uniq_sorted[-2] if len(uniq_sorted) >= 2 else None

            def sum_col(c): return float(rdf[c].sum()) if c in rdf.columns else 0.0

            # ---- ветки по метрикам (значение "metric" совпадает с SegmentedControl) ----
            if metric == 'amount':
                metric_name = "Чистая выручка"
                y_series_name = "Выручка"
                total = sum_col('amount')
                last = float(rdf.loc[rdf['eom'] == last_eom, 'amount'].sum())
                prev = float(rdf.loc[rdf['eom'] == prev_eom, 'amount'].sum()) if prev_eom is not None else 0.0
                trend = 0 if prev == 0 else round(100 * (last - prev) / prev, 1)
                # мини-график: сумма amount по месяцам
                g = rdf.groupby('eom')['amount'].sum().reset_index().sort_values('eom')
                g['month_name'] = pd.to_datetime(g['eom']).dt.strftime("%b\u202F%y").str.capitalize()
                mini_data = g[['month_name', 'amount']].rename(columns={'amount': 'value'}).to_dict('records')
                value_suffix = " ₽"
                trend_suffix = " %"

            elif metric == 'ret_amt_abs':
                metric_name = "Возвраты (₽)"
                y_series_name = "CR (₽)"
                total = sum_col('cr')
                last = float(rdf.loc[rdf['eom'] == last_eom, 'cr'].sum())
                prev = float(rdf.loc[rdf['eom'] == prev_eom, 'cr'].sum()) if prev_eom is not None else 0.0
                trend = 0 if prev == 0 else round(100 * (last - prev) / prev, 1)
                # мини-график: сумма cr по месяцам
                g = rdf.groupby('eom')['cr'].sum().reset_index().sort_values('eom')
                g['month_name'] = pd.to_datetime(g['eom']).dt.strftime("%b\u202F%y").str.capitalize()
                mini_data = g[['month_name', 'cr']].rename(columns={'cr': 'value'}).to_dict('records')
                value_suffix = " ₽"
                trend_suffix = " %"

            elif metric == 'ret_qty_abs':
                metric_name = "Возвраты (шт.)"
                y_series_name = "QCR (шт.)"
                total = sum_col('quant_cr')
                last = float(rdf.loc[rdf['eom'] == last_eom, 'quant_cr'].sum())
                prev = float(rdf.loc[rdf['eom'] == prev_eom, 'quant_cr'].sum()) if prev_eom is not None else 0.0
                trend = 0 if prev == 0 else round(100 * (last - prev) / prev, 1)
                # мини-график: сумма quant_cr по месяцам
                g = rdf.groupby('eom')['quant_cr'].sum().reset_index().sort_values('eom')
                g['month_name'] = pd.to_datetime(g['eom']).dt.strftime("%b\u202F%y").str.capitalize()
                mini_data = g[['month_name', 'quant_cr']].rename(columns={'quant_cr': 'value'}).to_dict('records')
                value_suffix = ""
                trend_suffix = " %"

            else:  # 'ret_coef' — коэффициент возвратов (₽), %
                metric_name = "Коэф. возвратов (₽), %"
                y_series_name = "CR/DT, %"
                # текущие и прошлые значения как процент
                def ratio_at(eom):
                    part = rdf.loc[rdf['eom'] == eom]
                    dt_sum = float(part['dt'].sum())
                    cr_sum = float(part['cr'].sum())
                    return 0.0 if dt_sum == 0 else (100.0 * cr_sum / dt_sum)

                last = ratio_at(last_eom)
                prev = ratio_at(prev_eom) if prev_eom is not None else 0.0
                trend = round(last - prev, 1)
                total = sum_col('cr')  # просто чтобы было что показать в summary (не используется в тексте)
                # мини-график: процент по месяцам
                g = rdf.groupby('eom')[['cr', 'dt']].sum().reset_index().sort_values('eom')
                g['value'] = np.where(g['dt'] > 0, 100.0 * g['cr'] / g['dt'], 0.0)
                g['month_name'] = pd.to_datetime(g['eom']).dt.strftime("%b\u202F%y").str.capitalize()
                mini_data = g[['month_name', 'value']].to_dict('records')
                value_suffix = " %"
                trend_suffix = " п.п."

            title = f"Отчёт • {scope}"
            trend_txt = ("+" if trend > 0 else "") + f"{trend}{trend_suffix}"

            rows = [
                dmc.Group(justify="space-between", children=[
                    dmc.Text(metric_name, fw=700),
                    dmc.Badge(scope, 	radius="xs",),
                ]),
                dmc.Text(f"Последний период: {last_eom.strftime('%b %Y').capitalize()}"),
                dmc.Text(f"Тек. значение: {last:,.2f}{value_suffix}"),
                dmc.Text(f"Изменение к пред.: {trend_txt}"),
                dmc.Divider(variant="dashed", my=8),
                dmc.AreaChart(
                    h=220, dataKey='month_name',
                    data=mini_data,
                    series=[{"name": y_series_name, "color": BASE_COLORS[0], "valueKey": "value"}],
                    withLegend=False, withPointLabels=False, connectNulls=True,
                ),
            ]
            return True, title, dmc.Stack(children=rows, gap="xs")

        
        
        # КНОПКА СБРОСИТЬ
        @app.callback(
            Output({'type': ch_filter, 'index': MATCH}, 'value'),
            Output({'type': rg_filter, 'index': MATCH}, 'value'),
            Output({'type': st_filter, 'index': MATCH}, 'value'),
            Output({'type': 'st_metric', 'index': MATCH}, 'value'),
            Input({'type': 'st_controls_reset', 'index': MATCH}, 'n_clicks'),
            prevent_initial_call=True,
        )
        def reset_filters(n_clicks):
            return None, None, None, 'amount'
        
        
        
        
        # #### ОБНОВЛЕНИЕ САММАРИ
        from pandas.tseries.offsets import MonthEnd

        @app.callback(
            Output({'type':'sum_rows','index':MATCH}, 'children'),
            Output({'type':'sum_caption','index':MATCH}, 'children'),
            Input({'type':'sum_base_mode','index':MATCH}, 'value'),      # period | last_month | custom
            Input({'type':'sum_delta_mode','index':MATCH}, 'value'),     # abs | pct
            Input({'type':'sum_number_format','index':MATCH}, 'value'),  # mln | full
            Input({'type':'sum_base_custom','index':MATCH}, 'value'),    # dmc.MonthPickerInput
            State({'type':'st_raw_eom','index':MATCH}, 'data'),
            prevent_initial_call=False,
        )
        def update_summary_rows(base_mode, delta_mode, num_format, base_custom_val, raw_eom):
            import pandas as pd, numpy as np, math
            from decimal import Decimal, ROUND_HALF_UP

            rdf = pd.DataFrame(raw_eom or [])
            if rdf.empty:
                return [dmc.Alert("Нет данных для расчёта KPI", color="gray", variant="light", radius="md")], ""

            # --- даты
            rdf['eom'] = pd.to_datetime(rdf['eom'], errors='coerce')
            rdf = rdf.dropna(subset=['eom'])
            eoms = np.sort(rdf['eom'].unique())
            if len(eoms) == 0:
                return [dmc.Alert("Нет валидных дат EOM", color="gray", variant="light", radius="md")], ""

            last_eom, first_eom = eoms[-1], eoms[0]
            prev_eom = eoms[-2] if len(eoms) >= 2 else None

            # кастом → конец месяца
            custom_eom = None
            if base_custom_val:
                try:
                    custom_eom = pd.to_datetime(base_custom_val) + MonthEnd(0)
                except Exception:
                    custom_eom = None

            # --- форматтеры
            NBSP = "\u202F"
            def fmt_money_full(x: float) -> str:
                try:
                    return f"{int(round(x)):,}".replace(",", NBSP) + " ₽"
                except:
                    return "0 ₽"

            def fmt_money_mln(x: float) -> str:
                try:
                    return f"{x/1_000_000:,.2f}".replace(",", " ").replace(".", ",") + " млн ₽"
                except:
                    return "0,00 млн ₽"

            def fmt_money(x: float) -> str:
                return fmt_money_mln(x) if num_format == "mln" else fmt_money_full(x)

            def fmt_int(x: float) -> str:
                try:
                    return f"{int(x):,}".replace(",", NBSP)
                except:
                    return "0"

            MONTHS_RU_3 = ["ЯНВ","ФЕВ","МАР","АПР","МАЙ","ИЮН","ИЮЛ","АВГ","СЕН","ОКТ","НОЯ","ДЕК"]
            def mon_yy(d):
                if d is None or pd.isna(d): return "—"
                d = pd.to_datetime(d)
                return f"{MONTHS_RU_3[d.month-1]} {d.strftime('%y')}"

            # агрегаторы
            def at(col, eom):
                if col not in rdf.columns or eom is None: return 0.0
                return float(rdf.loc[rdf['eom']==eom, col].sum())

            # пара для сравнения (current vs base)
            if base_mode == 'last_month':
                current_eom, base_eom = last_eom, prev_eom
            elif base_mode == 'custom' and (custom_eom is not None):
                current_eom, base_eom = last_eom, custom_eom
            else:  # period
                current_eom, base_eom = last_eom, first_eom

            # ряды для спарклайнов
            by_month = (rdf.groupby('eom')[['amount','dt','orders','cr','quant_cr']]
                        .sum().sort_index())

            def series(col):
                return by_month[col].to_list() if col in by_month.columns else []

            def ratio_series():
                if not {'cr','dt'}.issubset(by_month.columns): return []
                g = by_month[['cr','dt']].copy()
                return (100.0 * g['cr'] / g['dt']).replace([np.inf, -np.inf], 0.0).fillna(0.0).to_list()

            def sval(vals, eom_):
                if eom_ is None or len(vals) == 0: return None
                try:
                    pos = by_month.index.get_loc(eom_)
                    if isinstance(pos, slice): pos = pos.start or 0
                    if isinstance(pos, (list, np.ndarray)): pos = int(pos[0]) if len(pos) else 0
                    return vals[max(0, min(int(pos), len(vals)-1))]
                except Exception:
                    return None

            # визуалка
            # def spark_color(curr_val, base_val, good_when_up: bool):
            #     if curr_val is None or base_val is None: return "gray"
            #     if math.isclose(curr_val, base_val, rel_tol=0.005, abs_tol=1e-9): return "gray"
            #     up = curr_val > base_val
            #     return "teal" if ((up and good_when_up) or ((not up) and (not good_when_up))) else "red"

            def delta_node(curr, prev, *, good_up=True, pct_mode=True, is_money=False, is_pct_metric=False):
                if prev in (None, 0) or (isinstance(prev, float) and math.isclose(prev, 0.0)):
                    return dmc.Text("—", c="gray", ff="tabular-nums", fw=700, style={"whiteSpace":"nowrap"})
                diff = (curr or 0) - (prev or 0)
                color = "teal" if ((diff > 0 and good_up) or (diff < 0 and not good_up)) else ("red" if diff != 0 else "gray")
                arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "■")
                if pct_mode:
                    # относительное изменение, всегда в %
                    val = Decimal(diff/prev*100).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
                    txt = f"{arrow} {abs(val)}%"
                else:
                    # абсолютное изменение
                    if is_pct_metric:
                        # для процентных метрик — в п.п.
                        txt = f"{arrow} {abs(diff):.1f} п.п."
                    else:
                        txt = f"{arrow} " + (fmt_money(abs(diff)) if is_money else fmt_int(abs(diff)))
                return dmc.Text(txt, c=color, ff="tabular-nums", fw=700, style={"whiteSpace":"nowrap"})


            # --- фиксированная мини-сетка для колонок curr/base: [МЕСЯЦ | ЗНАЧЕНИЕ]
            MONTH_W = 54   # ширина под «СЕН 25», чтобы месяцы стояли строго столбцом
            GAP = 6

            def pill(text):
                return dmc.Kbd(text, size="xs", style={"opacity":0.9, "width": f"{MONTH_W}px", "textAlign": "center"})

            def cell_box(children):
                return dmc.Box(
                    children=children,
                    style={
                        "display": "grid",
                        "gridTemplateColumns": f"{MONTH_W}px 1fr",
                        "alignItems": "center",
                        "columnGap": f"{GAP}px",
                        "justifyItems": "start",
                        "whiteSpace": "nowrap",
                    }
                )

            # ячейки «текущий/база»
            def cell_money(col, eom_):
                return cell_box([
                    pill(mon_yy(eom_)),
                    dmc.Text(fmt_money(at(col, eom_)), fw=700, ff="tabular-nums"),
                ])

            def cell_int(col, eom_):
                return cell_box([
                    pill(mon_yy(eom_)),
                    dmc.Text(fmt_int(at(col, eom_)), fw=700, ff="tabular-nums"),
                ])

            def cell_pct(val, eom_):
                return cell_box([
                    pill(mon_yy(eom_)),
                    dmc.Text(f"{val:.1f}%", fw=700, ff="tabular-nums"),
                ])

            # грид-строка: label(3) | curr(3) | base(3) | delta(2) | spark(1)
            def metric_row_grid(label, *, curr_cell, base_cell, delta_cell):
                return html.Li(
                    dmc.Grid(gutter="xs", align="center", columns=12, children=[
                        dmc.GridCol(dmc.Text(f"{label}:", c="inherit"), span=3),
                        dmc.GridCol(curr_cell,  span=3, style={"textAlign": "left"}),
                        dmc.GridCol(base_cell,  span=3, style={"textAlign": "left"}),
                        dmc.GridCol(delta_cell, span=3, style={"textAlign": "left"}),
                        # dmc.GridCol(dmc.Box(spark, style={"width": "100%"}), span=1),
                    ]),
          
                )

            # спарклайны
            # def spark_for(col, *, good_up=True):
            #     vals = series(col)
            #     color = spark_color(sval(vals, current_eom), sval(vals, base_eom), good_when_up=good_up)
            #     return dmc.Sparkline(data=vals, w="100%", h=24, color=color, fillOpacity=0.5, curveType="Linear", strokeWidth=2)

            # коэффициент возвратов
            def ratio_eom(eom_):
                num, den = at('cr', eom_), at('dt', eom_)
                return (num/den*100.0) if den > 0 else 0.0

            # ПРОДАЖИ
            sales_rows = [
                metric_row_grid(
                    "Чистая выручка",
                    curr_cell = cell_money('amount', current_eom),
                    base_cell = cell_money('amount', base_eom),
                    delta_cell= delta_node(at('amount', current_eom), at('amount', base_eom),
                                        good_up=True, pct_mode=(delta_mode=='pct'), is_money=True),
                    # spark     = spark_for('amount', good_up=True)
                ),
                metric_row_grid(
                    "Общие продажи",
                    curr_cell = cell_money('dt', current_eom),
                    base_cell = cell_money('dt', base_eom),
                    delta_cell= delta_node(at('dt', current_eom), at('dt', base_eom),
                                        good_up=True, pct_mode=(delta_mode=='pct'), is_money=True),
                    # spark     = spark_for('dt', good_up=True)
                ),
                metric_row_grid(
                    "Кол-во заказов",
                    curr_cell = cell_int('orders', current_eom),
                    base_cell = cell_int('orders', base_eom),
                    delta_cell= delta_node(at('orders', current_eom), at('orders', base_eom),
                                        good_up=True, pct_mode=(delta_mode=='pct'), is_money=False),
                    # spark     = spark_for('orders', good_up=True)
                ),
            ]
            left_block = dmc.Stack(gap=6, children=[
                dmc.Text("Продажи", fw=700, c="dimmed"),
                html.Ul(style={"margin": 0, "paddingLeft": "1rem"}, children=sales_rows)
            ])

            # ВОЗВРАТЫ
            returns_rows = [
                metric_row_grid(
                    "Возвраты ₽",
                    curr_cell = cell_money('cr', current_eom),
                    base_cell = cell_money('cr', base_eom),
                    delta_cell= delta_node(at('cr', current_eom), at('cr', base_eom),
                                        good_up=False, pct_mode=(delta_mode=='pct'), is_money=True),
                    # spark     = spark_for('cr', good_up=False)
                ),
                metric_row_grid(
                    "Возвраты шт",
                    curr_cell = cell_int('quant_cr', current_eom),
                    base_cell = cell_int('quant_cr', base_eom),
                    delta_cell= delta_node(at('quant_cr', current_eom), at('quant_cr', base_eom),
                                        good_up=False, pct_mode=(delta_mode=='pct'), is_money=False),
                    # spark     = spark_for('quant_cr', good_up=False)
                ),
                metric_row_grid(
                    "Коэф. возвратов",
                    curr_cell = cell_pct(ratio_eom(current_eom), current_eom),
                    base_cell = cell_pct(ratio_eom(base_eom),    base_eom),
                    delta_cell= delta_node(
                        ratio_eom(current_eom),
                        ratio_eom(base_eom),
                        good_up=False,
                        pct_mode=(delta_mode == 'pct'),   # ← теперь уважаем режим
                        is_money=False,
                        is_pct_metric=True                # ← ключевая строчка: п.п. в ABS
                    ),
                    # spark     = dmc.Sparkline(
                    #     data=ratio_series(), w="100%", h=24,
                    #     color=spark_color(sval(ratio_series(), current_eom), sval(ratio_series(), base_eom), good_when_up=False),
                    #     fillOpacity=0.5, curveType="Linear", strokeWidth=2
                    # )
                ),

            ]
            # right_block = dmc.Stack(gap=6, children=[
            #     dmc.Text("Возвраты", fw=700, c="dimmed"),
            #     html.Ul(style={"margin": 0, "paddingLeft": "1rem"}, children=returns_rows)
            # ])
            
            
            right_block = dmc.Stack(gap=6, children=[
                dmc.Group(
                    align="center",
                    gap="xs",
                    children=[
                        dmc.Text("Возвраты", fw=700, c="dimmed"),
                        dmc.Tooltip(
                            label="Детализация возвратов",
                            position="right",
                            withArrow=True,
                            children=dmc.ActionIcon(
                                id="open_returns_modal",               
                                variant="light",
                                size="lg",
                                children=DashIconify(icon="tabler:eye", width=20, height=20)
                            )
                        ),
                    ]
                ),
                # html.Ul(style={"margin": 0, "paddingLeft": "1rem"}, children=returns_rows)
                dcc.Loading(
                        type="circle",
                        children=html.Ul(
                            style={"margin": 0, "paddingLeft": "1rem"},
                            children=returns_rows,
                        )),
            ])
            # подпись справа (легенда режима)
            def cap():
                dmode = "Δ %" if delta_mode == "pct" else "Δ абс."
                nfmt  = "млн ₽" if num_format == "mln" else "полные ₽"

                # кто база?
                if base_mode == "last_month":
                    b = mon_yy(prev_eom)
                    b_hint = "пред. месяц"
                elif base_mode == "custom" and (base_eom is not None):
                    b = mon_yy(base_eom)
                    b_hint = "кастом"
                else:
                    b = mon_yy(first_eom)
                    b_hint = "1-й мес. периода"

                a = mon_yy(last_eom)  # всегда последний месяц

                # если базы нет (например, единственный месяц данных)
                if b in ("—", "", None):
                    return f"{a} • {dmode} • {nfmt}"

                return f"{a} vs {b} ({b_hint}) • {dmode} • {nfmt}"

            cap_text = cap()

 
            return [left_block, right_block], cap_text


        
        
        
        
        
        # ### Показ/скрытие MonthPickerInput
        @app.callback(
            Output({'type':'sum_base_custom_box','index':MATCH}, 'style'),
            Input({'type':'sum_base_mode','index':MATCH}, 'value'),
            prevent_initial_call=False,
        )
        def toggle_custom_box(base_mode):
            return {"display":"block"} if base_mode == "custom" else {"display":"none"}
        

        
        


        @app.callback(
            Output({'type':'store_block','index':MATCH}, 'children'),
            Input({'type':'sum_base_mode','index':MATCH}, 'value'),         # period | last_month | custom
            Input({'type':'sum_delta_mode','index':MATCH}, 'value'),        # abs | pct
            Input({'type':'sum_number_format','index':MATCH}, 'value'),     # mln | full
            Input({'type':'sum_base_custom','index':MATCH}, 'value'),       # dmc.MonthPickerInput
            Input({'type':'store_metric_mode','index':MATCH}, 'value'),     # amount | cr | cr_ratio | avg_check
            State({'type':'st_raw_eom','index':MATCH}, 'data'),
            prevent_initial_call=False,
        )
        def update_store_block(base_mode, delta_mode, num_format, base_custom_val, store_metric, raw_eom):
            import pandas as pd, numpy as np, math
            from decimal import Decimal, ROUND_HALF_UP
            from pandas.tseries.offsets import MonthEnd

            rdf = pd.DataFrame(raw_eom or [])
            if rdf.empty:
                raise PreventUpdate

            # --- даты
            rdf['eom'] = pd.to_datetime(rdf['eom'], errors='coerce')
            rdf = rdf.dropna(subset=['eom'])
            eoms = np.sort(rdf['eom'].unique())
            if len(eoms) == 0:
                raise PreventUpdate

            last_eom  = eoms[-1]
            prev_eom  = eoms[-2] if len(eoms) >= 2 else None
            first_eom = eoms[0]

            # кастом → конец месяца
            custom_eom = None
            if base_custom_val:
                try:
                    custom_eom = pd.to_datetime(base_custom_val) + MonthEnd(0)
                except Exception:
                    custom_eom = None

            # --- форматтеры
            NBSP = "\u202F"
            def fmt_money_full(x: float) -> str:
                try:
                    return f"{int(round(x)):,}".replace(",", NBSP) + " ₽"
                except:
                    return "0 ₽"

            def fmt_money_mln(x: float) -> str:
                try:
                    return f"{x/1_000_000:,.2f}".replace(",", " ").replace(".", ",") + " млн ₽"
                except:
                    return "0,00 млн ₽"

            def fmt_money(x: float) -> str:
                return fmt_money_mln(x) if num_format == "mln" else fmt_money_full(x)

            def fmt_int(x: float) -> str:
                try:
                    return f"{int(x):,}".replace(",", NBSP)
                except:
                    return "0"

            MONTHS_RU_3 = ["ЯНВ","ФЕВ","МАР","АПР","МАЙ","ИЮН","ИЮЛ","АВГ","СЕН","ОКТ","НОЯ","ДЕК"]
            def mon_yy(d):
                if d is None or pd.isna(d): return "—"
                d = pd.to_datetime(d); return f"{MONTHS_RU_3[d.month-1]} {d.strftime('%y')}"

            # --- СВОДКИ ПО ВЫРУЧКЕ ДЛЯ СОРТИРОВКИ/ДОЛИ (всегда по amount)
            METRIC_COL = 'amount'

            def sum_at(eom, col=METRIC_COL, alias='val'):
                if eom is None:
                    out = rdf[['store_gr_name']].drop_duplicates().copy()
                    out[alias] = 0.0
                    return out
                part = rdf.loc[rdf['eom']==eom]
                return (part.groupby('store_gr_name', as_index=False)[col]
                            .sum().rename(columns={col: alias}))

            last_sales   = sum_at(last_eom,   alias='last_val')
            prev_sales   = sum_at(prev_eom,   alias='prev_val')
            first_sales  = sum_at(first_eom,  alias='first_val')
            custom_sales = sum_at(custom_eom, alias='custom_val') if custom_eom is not None else None

            st = last_sales.copy()
            if prev_sales is not None:   st = st.merge(prev_sales,  on='store_gr_name', how='left')
            if first_sales is not None:  st = st.merge(first_sales, on='store_gr_name', how='left')
            if custom_sales is not None: st = st.merge(custom_sales,on='store_gr_name', how='left')
            st = st.fillna(0.0)

            # выбор базы для сравнения
            if base_mode == 'last_month':
                current_eom, base_eom = last_eom, prev_eom
            elif base_mode == 'custom' and (custom_eom is not None):
                current_eom, base_eom = last_eom, custom_eom
            else:  # period
                current_eom, base_eom = last_eom, first_eom

            # доля и сортировка ВСЕГДА по чистой выручке текущего месяца
            denom = float(st['last_val'].sum()) or 0.0
            st['share'] = np.where(denom>0, st['last_val']/denom*100.0, 0.0)
            st['value'] = st['last_val']
            st = st.sort_values('value', ascending=False).reset_index(drop=True)

            # --- ПОСТРОЕНИЕ ВЫБИРАЕМОЙ МЕТРИКИ
            # агрегаты по магазинам/месяцам для базовых колонок
            by_store_month_sum = (rdf
                .groupby(['store_gr_name','eom'])[['amount','dt','orders','cr']]
                .sum()
                .sort_index()
            )

            # производные таблицы (stores x months)
            period_eoms = (rdf[['eom']].drop_duplicates().sort_values('eom')['eom']).tolist()
            cr_ratio_tbl = (100.0 * by_store_month_sum['cr'] / by_store_month_sum['dt']).replace([np.inf,-np.inf], 0.0).fillna(0.0)
            cr_ratio_tbl = cr_ratio_tbl.unstack(fill_value=0.0).reindex(columns=period_eoms, fill_value=0.0)

            avg_check_tbl = (by_store_month_sum['dt'] / by_store_month_sum['orders']).replace([np.inf,-np.inf], 0.0).fillna(0.0)
            avg_check_tbl = avg_check_tbl.unstack(fill_value=0.0).reindex(columns=period_eoms, fill_value=0.0)

            # конфиг метрик
            metric_cfg = {
                "amount":    {"label": "Выручка",           "is_money": True,  "is_pct": False, "good_up": True},
                "cr":        {"label": "Возвраты ₽",        "is_money": True,  "is_pct": False, "good_up": False},
                "cr_ratio":  {"label": "Коэф. возвратов",   "is_money": False, "is_pct": True,  "good_up": False},
                "avg_check": {"label": "Средний чек",       "is_money": True,  "is_pct": False, "good_up": True},
            }
            cfg = metric_cfg.get(store_metric or "amount", metric_cfg["amount"])
            LABEL = cfg["label"]

            # значение метрики по магазину/месяцу
            def metric_value(store, eom):
                if eom is None:
                    return 0.0
                if store_metric == "amount":
                    try:    return float(by_store_month_sum.loc[(store, eom)]['amount'])
                    except: return 0.0
                if store_metric == "cr":
                    try:    return float(by_store_month_sum.loc[(store, eom)]['cr'])
                    except: return 0.0
                if store_metric == "cr_ratio":
                    try:    return float(cr_ratio_tbl.loc[store].get(eom, 0.0))
                    except: return 0.0
                if store_metric == "avg_check":
                    try:    return float(avg_check_tbl.loc[store].get(eom, 0.0))
                    except: return 0.0
                return 0.0

            # ряды для спарклайна по выбранной метрике
            def metric_series_for_store(store):
                return [metric_value(store, e) for e in period_eoms]

            def spark_color_for_store(vals, store_name):
                if not vals or base_eom is None: return "gray"
                last_val = vals[-1] if len(vals) else None
                base_val = metric_value(store_name, base_eom)
                if last_val is None or base_val is None:
                    return "gray"
                if math.isclose(last_val, base_val, rel_tol=0.005, abs_tol=1e-9):
                    return "gray"
                return "teal" if ((last_val > base_val) == cfg["good_up"]) else "red"

            # мини-сетка для колонок curr/base
            MONTH_W = 64
            GAP = 6
            def pill(text):
                return dmc.Kbd(text, size="xs", style={"opacity":0.9, "width": f"{MONTH_W}px", "textAlign":"center"})
            def cell_box(children):
                return dmc.Box(children=children, style={
                    "display":"grid", "gridTemplateColumns": f"{MONTH_W}px 1fr",
                    "alignItems":"center", "columnGap": f"{GAP}px", "justifyItems":"start",
                    "whiteSpace":"nowrap",
                })

            def fmt_val(x):
                if cfg["is_pct"]:
                    return f"{x:.1f}%"
                return fmt_money(x) if cfg["is_money"] else fmt_int(x)

            def curr_cell(store):
                v = metric_value(store, current_eom)
                return cell_box([pill(mon_yy(current_eom)), dmc.Text(fmt_val(v), fw=700, ff="tabular-nums")])

            def base_cell(store):
                v = metric_value(store, base_eom)
                return cell_box([pill(mon_yy(base_eom)), dmc.Text(fmt_val(v), fw=700, ff="tabular-nums")])

            # дельта
            def delta_node_for_store(store):
                curr = metric_value(store, current_eom)
                prev = metric_value(store, base_eom)
                good_up = cfg["good_up"]
                if prev in (None, 0) or (isinstance(prev, float) and math.isclose(prev, 0.0)):
                    return dmc.Text("—", c="gray", ff="tabular-nums", fw=700, style={"whiteSpace":"nowrap"})
                diff = (curr or 0.0) - (prev or 0.0)
                color = "teal" if ((diff>0 and good_up) or (diff<0 and not good_up)) else ("red" if diff!=0 else "gray")
                arrow = "▲" if diff>0 else ("▼" if diff<0 else "■")
                if delta_mode == 'pct':  # относительная
                    val = Decimal(diff/prev*100).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
                    txt = f"{arrow} {abs(val)}%"
                else:                    # абсолют
                    if cfg["is_pct"]:
                        txt = f"{arrow} {abs(diff):.1f} п.п."
                    else:
                        txt = f"{arrow} " + (fmt_money(abs(diff)) if cfg["is_money"] else fmt_int(abs(diff)))
                return dmc.Text(txt, c=color, ff="tabular-nums", fw=700, style={"whiteSpace":"nowrap"})

            # --- заголовок
            header = dmc.Grid(gutter="xs", align="center", columns=12, children=[
                dmc.GridCol(dmc.Text("Магазин", c="dimmed", fw=600), span=4),
                dmc.GridCol(dmc.Text(f"Текущий ({mon_yy(last_eom)}) — {LABEL}", c="dimmed", fw=600), span=2),
                dmc.GridCol(dmc.Text(f"База ({mon_yy(base_eom)}) — {LABEL}",   c="dimmed", fw=600), span=2),
                dmc.GridCol(dmc.Text("Дельта", c="dimmed", fw=600), span=2, style={"textAlign":"left"}),
                dmc.GridCol(dmc.Text("Тренд",  c="dimmed", fw=600), span=1, style={"textAlign":"left"}),
                dmc.GridCol(dmc.Text("Доля",   c="dimmed", fw=600), span=1, style={"textAlign":"right"}),
            ])

            # --- строки
            def store_row(rank, row):
                store = row['store_gr_name']
                vals  = metric_series_for_store(store)
                spark = dmc.Sparkline(
                    data=vals, w="100%", h=22, color=spark_color_for_store(vals, store),
                    fillOpacity=0.5, curveType="Linear", strokeWidth=2
                )
                share_badge = dmc.Badge(f"{row['share']:.1f}%", variant="outline", radius="xs",
                                        style={"whiteSpace":"nowrap", "justifySelf":"end"})
                return html.Li(
                    dmc.Grid(gutter="xs", align="center", columns=12, children=[
                        dmc.GridCol(
                            dmc.Group(gap=8, align="center", wrap="nowrap", children=[
                                dmc.Badge(str(rank), variant="filled", color="teal", radius="xs", w=48, ta="center"),
                                dmc.Text(store, style={"whiteSpace":"nowrap","overflow":"hidden","textOverflow":"ellipsis"})
                            ]),
                            span=4
                        ),
                        dmc.GridCol(curr_cell(store), span=2),
                        dmc.GridCol(base_cell(store), span=2),
                        dmc.GridCol(delta_node_for_store(store), span=2, style={"textAlign":"left"}),
                        dmc.GridCol(dmc.Box(spark, style={"width":"100%"}), span=1),
                        dmc.GridCol(share_badge, span=1, style={"textAlign":"right"}),
                    ]),
                    style={"listStyleType":"none", "margin":0, "padding":"2px 0"}
                )

            rows = [store_row(i+1, r) for i, r in st.iterrows()]
            return [dmc.Stack(gap=6, children=[header, html.Ul(style={"margin":0, "paddingLeft":"0"}, children=rows)])]




   
        ### ПОКАЗЫВАТЬ ОПРЕДЕЛЕННЫЕ МЕСЯЦА В КАЛЕНДАРЕ МАКСИМЛАЬНЫЕ И МИНИМАЛЬНЫЕ
        @app.callback(
            Output({'type':'sum_base_custom','index':MATCH}, 'minDate'),
            Output({'type':'sum_base_custom','index':MATCH}, 'maxDate'),
            Output({'type':'sum_base_custom','index':MATCH}, 'disabled'),
            Output({'type':'sum_base_custom','index':MATCH}, 'value'),
            Input({'type':'st_raw_eom','index':MATCH}, 'data'),
            State({'type':'sum_base_custom','index':MATCH}, 'value'),
            prevent_initial_call=False,
        )
        def setup_custom_month_bounds(raw_eom, current_value):
            import pandas as pd, numpy as np
            # по умолчанию всё отключено
            min_date = max_date = None
            disabled = True
            out_value = current_value

            rdf = pd.DataFrame(raw_eom or [])
            if rdf.empty or 'eom' not in rdf.columns:
                return min_date, max_date, disabled, None

            # нормализуем даты
            rdf['eom'] = pd.to_datetime(rdf['eom'], errors='coerce')
            rdf = rdf.dropna(subset=['eom'])
            eoms = np.sort(rdf['eom'].unique())
            if len(eoms) == 0:
                return min_date, max_date, disabled, None

            first_eom = pd.to_datetime(eoms[0])
            last_eom  = pd.to_datetime(eoms[-1])

            # max = предыдущий месяц от последнего
            if len(eoms) >= 2:
                prev_eom = pd.to_datetime(eoms[-2])
                min_date = first_eom.to_pydatetime()
                max_date = prev_eom.to_pydatetime()
                disabled = False
            else:
                # всего один месяц — выбирать не из чего
                min_date = first_eom.to_pydatetime()
                max_date = first_eom.to_pydatetime()
                disabled = True

            # если текущее значение вне границ — очистим
            if current_value:
                try:
                    v = pd.to_datetime(current_value) + MonthEnd(0)
                    if (v < pd.to_datetime(min_date)) or (v > pd.to_datetime(max_date)):
                        out_value = None
                except Exception:
                    out_value = None

            return min_date, max_date, disabled, out_value
        

        
        
        # === ОТКРЫТИЕ МОДАЛКИ: сразу показываем последний месяц ===
        
        # ---  колбэк открытия: мгновенно показать модалку и анимацию
        @app.callback(
            Output("returns_modal", "opened"),
            Output("returns_loading", "visible"),
            Input("open_returns_modal", "n_clicks"),
            prevent_initial_call=True,
        )
        def open_returns_modal(n):
            if not n:                          # None или 0 -> игнор
                raise PreventUpdate
            return True, True                  # открыть и показать анимацию




        # --- 2)  колбэк: загрузить данные и выключить оверлей

        @app.callback(
            Output("returns_grid", "rowData"),
            # donut by category
            Output("returns_cat_donut", "data"),
            Output("returns_cat_donut", "chartLabel"),
            Output("returns_cat_legend", "children"),
            # donut by manufacturer
            Output("returns_manu_donut", "data"),
            Output("returns_manu_donut", "chartLabel"),
            Output("returns_manu_legend", "children"),
            # выключаем оверлей в конце — помечаем как дубликат
            Output("returns_loading", "visible", allow_duplicate=True),
            Input("returns_modal", "opened"),
            Input("returns_range", "value"),
            State("df_returns_store", "data"),
            prevent_initial_call=True,
        )
        def open_or_update_returns(is_opened, mode, df_data):
            import pandas as pd, numpy as np
            if not is_opened:
                raise PreventUpdate

            df = pd.DataFrame(df_data or [])
            if df.empty:
                return [], [], "", [], [], "", [], False

            needed = ["date","client_order_number","manager","cr","quant_cr",
                    "store_gr_name","subcat","cat","fullname","brend","manu"]
            for c in needed:
                if c not in df.columns:
                    df[c] = None

            def _label_or_default(s, default_txt):
                return (s.astype("string").fillna("").str.strip()
                        .replace({"None": "", "nan": ""})
                        .map(lambda x: default_txt if x == "" else x))
            df["brend"] = _label_or_default(df["brend"], "Бренд не указан")
            df["manu"]  = _label_or_default(df["manu"],  "Произв. не указан")

            df["date"]     = pd.to_datetime(df["date"], errors="coerce")
            df["cr"]       = pd.to_numeric(df["cr"], errors="coerce").fillna(0)
            df["quant_cr"] = pd.to_numeric(df["quant_cr"], errors="coerce").fillna(0)
            df = df[df["cr"] != 0]
            if df.empty:
                return [], [], "", [], [], "", [], False

            mode = mode or "last"
            if mode == "last":
                last_date  = df["date"].max()
                start      = last_date.replace(day=1)
                next_start = start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)
                df = df[(df["date"] >= start) & (df["date"] < next_start)]

            df["date"] = df["date"].dt.strftime("%Y-%m-%d")

            NBSP = "\u202F"
            def fmt_rub0(x): return f"₽{int(round(float(x))):,}".replace(",", NBSP)
            palette = ["blue.6","teal.6","grape.6","orange.6","cyan.6","red.6","lime.6","violet.6","indigo.6","pink.6"]
            TOP_N = 10

            def build_donut(df_, key_col, other_label="Прочее"):
                pie = (df_.groupby(key_col, as_index=False)["cr"].sum().sort_values("cr", ascending=False))
                if len(pie) > TOP_N:
                    other = float(pie["cr"].iloc[TOP_N:].sum())
                    pie   = pd.concat([pie.head(TOP_N), pd.DataFrame([{key_col: other_label, "cr": other}])], ignore_index=True)
                total = float(pie["cr"].sum()) or 1.0

                data, legend = [], []
                for i, r in pie.reset_index(drop=True).iterrows():
                    name  = str(r[key_col]); value = float(r["cr"])
                    pct   = value/total*100.0; color = palette[i % len(palette)]
                    data.append({"name": name, "value": value, "color": color})
                    legend.append(
                        dmc.Group(gap="xs", wrap=False, align="center", children=[
                            dmc.ThemeIcon(radius="xl", size=10, variant="filled", color=color),
                            dmc.Text(f"{name} — {fmt_rub0(value)} ({pct:.1f}%)",
                                    size="sm",
                                    style={"whiteSpace":"nowrap","overflow":"hidden","textOverflow":"ellipsis"}),
                        ])
                    )
                return data, fmt_rub0(total), legend

            df_cat = df.assign(cat=df["cat"].astype("string").fillna("").str.strip()
                            .replace({"": "Категория не указана"}))
            cat_data, cat_center, cat_legend = build_donut(df_cat, "cat")
            manu_data, manu_center, manu_legend = build_donut(df, "manu")

            row_out = df[needed].to_dict("records")

            return row_out, cat_data, cat_center, cat_legend, manu_data, manu_center, manu_legend, False

        
        ### ПЕРЕКЛЮЧЕНИЕ ТЕМЫ У ТАБЛИЧКИ
        @app.callback(
            Output("returns_grid", "className"),
            Input("theme_switch", "checked"),
        )
        def toggle_theme(checked):
            theme = "ag-theme-alpine-dark" if checked else "ag-theme-alpine"
            return theme
        
        
        #### КАЛЕНДАРЬ ПО ВЫРУЧКЕ НА ГЛАВНОЙ СТРАНИЦЕ
        # @dash.callback(
        #     Output({'type':'st_heatmap_wrap','index':MATCH}, 'children'),
        #     Output({'type':'st_heatmap_scope','index':MATCH}, 'children'),   # ← добавили
        #     Input({'type':'st_heatmap_metric','index':MATCH}, 'value'),
        #     Input({'type':'st_heatmap_range','index':MATCH}, 'value'),
        #     Input({'type':'chanel_multyselect','index':MATCH}, 'value'),
        #     Input({'type':'region_multyselect','index':MATCH}, 'value'),
        #     Input({'type':'store_multyselect','index':MATCH}, 'value'),
        #     State({'type':'st_daily','index':MATCH}, 'data'),
        #     prevent_initial_call=False,
        # )
        # def render_period_heatmap(metric, date_range, ch_val, rg_val, st_val, daily_data):
        #     import pandas as pd

        #     def L(x): return [] if x is None else (x if isinstance(x, list) else [x])

        #     # --- вычислим подпись для бейджа
        #     stores = L(st_val)
        #     if not stores:
        #         scope_text = "Все магазины"
        #     elif len(stores) == 1:
        #         scope_text = stores[0]
        #     elif len(stores) == 2:
        #         scope_text = f"{stores[0]}, {stores[1]}"
        #     else:
        #         scope_text = f"{len(stores)} магазинов"

        #     df = pd.DataFrame(daily_data or [])
        #     if df.empty:
        #         return dmc.Alert("Нет данных для теплокарты", color="gray", variant="light", radius="md"), scope_text

        #     if L(ch_val): df = df[df['chanel'].isin(L(ch_val))]
        #     if L(rg_val): df = df[df['store_region'].isin(L(rg_val))]
        #     if L(st_val): df = df[df['store_gr_name'].isin(L(st_val))]

        #     # период
        #     if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        #         start = pd.to_datetime(date_range[0])
        #         end   = pd.to_datetime(date_range[1])
        #     else:
        #         start = pd.to_datetime(df['date'].min())
        #         end   = pd.to_datetime(df['date'].max())

        #     df['date'] = pd.to_datetime(df['date'], errors='coerce')
        #     df = df[(df['date'] >= start) & (df['date'] <= end)]
        #     if df.empty:
        #         return dmc.Alert("За выбранный период данных нет", color="yellow", variant="light", radius="md"), scope_text

        #     # построение графика
        #     graph = self._heatmap_period_block(df, start, end, metric)

        #     return graph, scope_text
        
        
        @dash.callback(
            Output({'type':'st_heatmap_wrap','index':MATCH}, 'children'),
            Input({'type':'st_heatmap_metric','index':MATCH}, 'value'),
            Input({'type':'st_heatmap_range','index':MATCH}, 'value'),
            Input({'type':'chanel_multyselect','index':MATCH}, 'value'),
            Input({'type':'region_multyselect','index':MATCH}, 'value'),
            Input({'type':'store_multyselect','index':MATCH}, 'value'),
            Input("theme_switch", "checked"),                    # ← ТЕМА
            State({'type':'st_daily','index':MATCH}, 'data'),
            prevent_initial_call=False,
        )
        def render_period_heatmap(metric, date_range, ch_val, rg_val, st_val, theme_checked, daily_data):
            import pandas as pd
            def L(x): return [] if x is None else (x if isinstance(x, list) else [x])

            df = pd.DataFrame(daily_data or [])
            if df.empty:
                return dmc.Alert("Нет данных для теплокарты", color="gray", variant="light", radius="md")

            if L(ch_val): df = df[df['chanel'].isin(L(ch_val))]
            if L(rg_val): df = df[df['store_region'].isin(L(rg_val))]
            if L(st_val): df = df[df['store_gr_name'].isin(L(st_val))]

            if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
                start = pd.to_datetime(date_range[0])
                end   = pd.to_datetime(date_range[1])
            else:
                start = pd.to_datetime(df['date'].min())
                end   = pd.to_datetime(df['date'].max())

            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df[(df['date'] >= start) & (df['date'] <= end)]
            if df.empty:
                return dmc.Alert("За выбранный период данных нет", color="yellow", variant="light", radius="md")

            # строим с учётом темы
            return self._heatmap_period_block(df, start, end, metric or "amount", is_dark=bool(theme_checked))




