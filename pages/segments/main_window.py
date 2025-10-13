import pandas as pd
import numpy as np
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import dash_ag_grid as dag

from dash import dcc
import plotly.express as px
import plotly.graph_objects as go

from typing import Optional



from data import (
    load_columns_df,
    load_df_from_redis,
    delete_df_from_redis,
    save_df_to_redis,
)
from components import MonthSlider, DATES, COLORS_BY_SHADE, COLORS_BY_COLOR
from dash import dcc, Input, Output, State, no_update, MATCH
from .db_queries import get_items, fletch_dataset, fletch_agents, fletch_stores
from .drawler import CATS_MANAGEMENT
from .ag_modal import AGModal
AG_MODAL = AGModal()

COLS = [
    "date",
    "eom",
    "init_date",
    "parent_cat",
    "parent_cat_id",
    "cat",
    "cat_id",
    "subcat",
    "subcat_id",
    "item_id",
    "fullname",
    "brend",
    "manu",
    "amount",
    "quant",
]

def title_badge(text: str):
    return dmc.Badge(
        text,
        color="blue",
        size="lg",
        radius="sm",
        # variant="gradient",
        # gradient={"from": "blue", "to": "grape", "deg": 45},
        style={"fontWeight": 600, "letterSpacing": "0.2px"}
    )
    


def kpi_card(label: str, value: str, icon: str = "mdi:chart-line"):
    return dmc.Paper(
        withBorder=True,
        radius="md",
        shadow="sm",
        p="md",
        style={
            "borderColor": "rgba(99,102,241,0.35)",  # оттенок grape
        },
        children=dmc.Group(
            align="center",
            gap="sm",
            children=[
                dmc.ThemeIcon(
                    size="lg", radius="md", variant="light", color="blue",
                    children=DashIconify(icon=icon, width=20)
                ),
                dmc.Stack(gap=2, children=[
                    dmc.Text(value, size="xl", fw=700, style={"lineHeight": 1}),
                    dmc.Text(label, size="sm", opacity=0.7),
                ])
            ]
        )
    )




def id_to_months(start, end):
    return DATES[start].strftime("%Y-%m-%d"), DATES[end].strftime("%Y-%m-%d")

def fmt_int(x): 
    try: return f"{int(round(float(x))):,}".replace(",", " ")
    except: return "0"

def fmt_rub(x):
    try:
        v = float(x)
        s = f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"
        return s.replace(",", " ") + " ₽"
    except: return "0 ₽"

def fmt_pct(x):
    try: return f"{float(x):.2f}%".replace(".", ",")
    except: return "0,00%"

def empty_placeholder():
    return dmc.Paper(
        withBorder=True, radius="md", p="xl",
        style={
            "borderStyle": "dashed",
            "background": "linear-gradient(180deg, rgba(0,0,0,0.02), rgba(0,0,0,0.04))",
        },
        children=dmc.Center(
            dmc.Stack(
                gap="sm", align="center",
                children=[
                    dmc.ThemeIcon(
                        DashIconify(icon="solar:cursor-square-linear", width=88),
                        size=96, radius="xl", variant="light", color="blue"
                    ),
                    dmc.Title("Выберите позиции для анализа", order=3),
                    dmc.Text(
                        "Отметьте группы/бренды/производителей слева, и здесь появится таблица.",
                        c="dimmed", size="sm", ta="center", maw=520
                    ),
                    dmc.Group(gap="xs", justify="center", mt="xs",
                              children=[
                                  dmc.Badge("Шаг 1: выберите период", variant="outline", radius="xs",),
                                  dmc.Badge("Шаг 2: отметьте категории", variant="outline", radius="xs",),
                                  dmc.Badge("Шаг 3: смотрите детали", variant="outline", radius="xs",),
                              ]),
                ],
            )
        ),
    )
    
NBSP_THIN = "\u202F"  # тонкий неразрывный пробел
def fmt_grouped(v: float, money=False):
    try: v = float(v)
    except: v = 0.0
    s = f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"
    s = s.replace(",", " ").replace(" ", NBSP_THIN)
    return f"{s}{NBSP_THIN}₽" if money else s

_UNITS = ["", "тыс", "млн", "млрд", "трлн"]

def fmt_compact(v: float, money=False, digits=2):
    try: v = float(v)
    except: v = 0.0
    n = abs(v)
    k = 0
    while n >= 1000 and k < len(_UNITS)-1:
        n /= 1000.0
        v /= 1000.0
        k += 1
    num = f"{v:.{digits}f}".rstrip("0").rstrip(".")
    num = num.replace(".", ",")
    out = f"{num}{NBSP_THIN}{_UNITS[k]}".strip()
    return f"{out}{NBSP_THIN}₽" if money else out



def pareto_block(df: pd.DataFrame):
    # агрегируем по номенклатуре
    g = (df.groupby("fullname", as_index=False)
           .agg(amount=("dt","sum")))
    g = g.sort_values("amount", ascending=False).reset_index(drop=True)
    g["cum_share"] = (g["amount"].cumsum() / g["amount"].sum() * 100).fillna(0)
    # ограничим топ-30 для читаемости
    top = g.head(30)

    data = top.to_dict("records")
    return dmc.Card(withBorder=True, radius=0, p="md", children=[
        dmc.Group(justify="space-between", children=[
            dmc.Title("Парето по выручке (топ-30)", order=5),
            dmc.Badge("бар = выручка; линия = накопл. доля, %", variant="light"),
        ]),
        dmc.Space(h=6),
        dmc.BarChart(
            h=320, data=data, dataKey="fullname",
            series=[{"name": "amount", "label": "Выручка"}],
            valueFormatter = {"function": "formatNumberIntl"},
            withLegend=False,
            gridAxis="xy",


        ),
        dmc.Space(h=6),
        dmc.Progress(value=float(top["cum_share"].iloc[-1] if not top.empty else 0), size="lg", striped=True, radius="xs"),
    ])

# Блок по магазинам/менеджерам
def stores_block(df_stores: pd.DataFrame):
    dfs = df_stores.copy()

        
    dfs['rank'] = dfs['amount'].rank(method='min', ascending=False)
    dfs['store_sales'] = np.where(dfs['rank'] <= 5, dfs['store_gr_name'], 'Другие магазины')
    dfs = dfs.pivot_table(index='store_sales', values='amount', aggfunc='sum').reset_index().sort_values(by='amount', ascending=False)

    store_data = []
    for i, row in enumerate(dfs.itertuples(index=False)):
        color = COLORS_BY_SHADE[i % len(COLORS_BY_SHADE)]  # чтобы не выйти за пределы
        store_data.append({
            "name": row.store_sales,
            "value": float(row.amount),  # на всякий случай преобразуем в число
            "color": color
        })
    
    stores_list = dmc.List(
        [
            dmc.ListItem(
                f"{name}: {value/1_000_000:,.2f} млн ₽".replace(",", " "),
                icon=dmc.ThemeIcon(
                    DashIconify(icon="tdesign:shop-filled", width=16),
                    size=24,
                    radius="xl",
                    color=color_shop,
                    variant="light",
                ),
            )
            for name, value, color_shop in zip(dfs['store_sales'], dfs['amount'], COLORS_BY_SHADE)
        ]
    )
    
    
    return dmc.Box(
        [
            dmc.Space(h=6),
            dmc.Divider(label="Магазины", my="md"),
            dmc.Title("Распределение продаж по магазинам", order=5),
            dmc.Space(h=6),
            
            dmc.Group(
                [  
                 dmc.Stack(
                     [
                        
                        stores_list
                     ]
                    ),
                
                    
                    dmc.Stack(
                     [
                        
                        dmc.PieChart(
                        h=200,
                        data=store_data,
                        labelsType="percent",
                        withTooltip=True,
                        tooltipDataSource="segment",
                        mx="auto",
                        strokeWidth=2,
                        withLabels=True
                     )
                     ]
                    ),
                ],
                gap='xl',
                grow=True,
            )
        ],
       
    )






def insights_block(
    df: pd.DataFrame,
    tot_revenue: float,
    agent_share: float,
    selected_category: Optional[str] = None,
    yaxis_max: Optional[float] = None,
):
    d = df.copy()

    # ============== Утилиты ==============
    def clamp(x: float) -> float:
        try:
            x = float(x)
        except Exception:
            x = 0.0
        return max(0.0, min(100.0, x))

    def fmt_pct(x, digits=1):
        try:
            s = f"{float(x):.{digits}f}%"
        except Exception:
            s = f"{0:.{digits}f}%"
        return s.replace(".", ",")

    def fmt_compact(x, money=False, digits=1):
        try:
            x = float(x)
        except Exception:
            x = 0.0
        absx = abs(x)
        suffix = ""
        val = x
        if absx >= 1_000_000_000:
            val = x / 1_000_000_000
            suffix = " млрд"
        elif absx >= 1_000_000:
            val = x / 1_000_000
            suffix = " млн"
        elif absx >= 1_000:
            val = x / 1_000
            suffix = " тыс"
        s = f"{val:.{digits}f}".replace(".", ",") + suffix
        if money:
            s += " ₽"
        return s

    def section_card(title: str, children):
        return dmc.Paper(withBorder=True, radius="md", p="md", children=[
            dmc.Text(title, fw=600, mb=8),
            children,
        ])

    def kpi_stat(label: str, value, hint: str = ""):
        return dmc.Paper(withBorder=True, radius="md", p="sm", children=[
            dmc.Text(label, size="xs", c="dimmed"),
            dmc.Text(value, fw=700, fz="lg"),
            (dmc.Text(hint, size="xs", c="dimmed") if hint else dmc.Box()),
        ])

    def find_col(cands):
        lc = {c.lower(): c for c in d.columns}
        for name in cands:
            if name.lower() in lc:
                return lc[name.lower()]
        return None

    # красивый потолок оси Y (ступени 1–2–5)
    def nice_ceil(v: float, pad_ratio: float = 0.15) -> float:
        v = float(v or 0) * (1 + pad_ratio)
        if v <= 0:
            return 1.0
        mag = 10 ** int(np.floor(np.log10(v)))
        base = np.ceil(v / mag)
        step = 1 if base <= 1 else (2 if base <= 2 else (5 if base <= 5 else 10))
        return float(step * mag)

    # ============== Приведение чисел ==============
    for c in ["dt", "cr", "quant_dt", "quant_cr"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
        else:
            d[c] = 0.0

    d["amount"] = d["dt"] - d["cr"]
    d["quant"]  = d["quant_dt"] - d["quant_cr"]
    d["ret_pct"] = (d["cr"] / d["dt"].replace(0, pd.NA)).fillna(0) * 100

    # ============== Имена колонок (расширенно) ==============
    date_col  = find_col(["date", "eom", "doc_date", "operation_date", "sale_date", "init_date", "дата", "Дата"])
    agent_col = find_col(["agent_name"])
    store_col = find_col(["store_gr_name"])
    cat_col   = find_col(["category", "category_name", "Категория", "Группа", "group_name"])
    brand_col = find_col(["brend", "brand", "бренд"])
    manu_col  = find_col(["manu", "manufacturer", "производитель"])
    full_col  = find_col(["fullname", "sku", "наименование"])

    # Фильтр по категории
    if selected_category and (cat_col is not None) and (cat_col in d.columns):
        d = d.loc[d[cat_col].astype(str) == str(selected_category)].copy()

    # ============== Сводные числа ==============
    n_items = int(d[full_col].nunique()) if full_col in d.columns else 0
    n_brand = int(d[brand_col].nunique()) if brand_col in d.columns else 0
    n_manu  = int(d[manu_col].nunique())  if manu_col  in d.columns else 0

    total_dt = float(d["dt"].sum())
    total_cr = float(d["cr"].sum())
    amount   = float(d["amount"].sum())
    q_net    = float(d["quant"].sum())

    avg_price_net = (amount / q_net) if q_net > 0 else 0.0

    revenue_pct = clamp((amount / tot_revenue * 100) if tot_revenue else 0.0)
    ret_coef    = clamp((total_cr / total_dt * 100) if total_dt else 0.0)
    agent_share = clamp(agent_share or 0.0)

    # Доля «дизайнера»
    designer_share = None
    if agent_col is not None and "dt" in d.columns:
        s = d[agent_col].astype(str)
        is_des = (s.str.contains("дизайн", case=False, na=False) |
                  s.str.fullmatch(r"\s*дизайнер\s*", case=False, na=False))
        dt_des = float(d.loc[is_des, "dt"].sum())
        designer_share = clamp((dt_des / total_dt * 100) if total_dt > 0 else 0.0)

    # ============== Парето ==============
    by_fullname_amt = (d.groupby(full_col, as_index=False)["amount"].sum()) if full_col in d.columns else pd.DataFrame(columns=[full_col,"amount"])
    pareto_text = dmc.Text("нет данных", size="sm", c="dimmed")
    pareto_widget = dmc.Box()
    if amount > 0 and n_items > 0 and not by_fullname_amt.empty:
        g = by_fullname_amt.sort_values("amount", ascending=False).reset_index(drop=True)
        total_amt = max(1e-9, g["amount"].sum())
        g["cum_share"] = g["amount"].cumsum() / total_amt * 100
        k80 = int((g["cum_share"] >= 80).idxmax()) + 1 if not g.empty else 0
        k80 = min(k80, n_items)
        share_of_cat = (k80 / n_items * 100) if n_items else 0
        current_cum = float(g["cum_share"].iloc[k80-1]) if k80 > 0 else 0.0

        pareto_text = dmc.Text(
            f"Топ-{k80} SKU (~{share_of_cat:.1f}%) формируют 80% выручки".replace(".", ","),
            size="sm"
        )
        pareto_widget = dmc.Stack(gap=6, children=[
            dmc.Progress(value=clamp(current_cum), size="md", radius="sm"),
            dmc.Group(justify="space-between", children=[
                dmc.Text(f"Достигнуто: {fmt_pct(current_cum)}", size="xs", c="dimmed"),
                dmc.Badge("80%", variant="outline", radius="xs")
            ])
        ])

    # ============== Топы ==============
    def top_money(col, k=3):
        if (col is None) or (col not in d.columns) or d.empty:
            return ["нет данных"]
        g = (d.groupby(col, as_index=False)["amount"].sum()
               .sort_values("amount", ascending=False).head(k))
        if g.empty:
            return ["нет данных"]
        return [f"{r[col]} — {fmt_compact(r['amount'], money=True)}" for _, r in g.iterrows()]

    brands_list = top_money(brand_col)
    manus_list  = top_money(manu_col)

    # Лучшие SKU
    best_amt_txt = ["нет данных"]
    if not by_fullname_amt.empty:
        best_amt = by_fullname_amt.sort_values("amount", ascending=False).head(5)
        best_amt_txt = [f"{r[full_col]} — {fmt_compact(r['amount'], money=True)}" for _, r in best_amt.iterrows()] or ["нет данных"]

    # ============== Возвраты/качество ==============
    def risk_color(p):
        return "red" if p >= 30 else ("yellow" if p >= 20 else "green")

    risky_list = [dmc.ListItem("нет данных")]
    risky_count = 0
    very_high_count = 0
    max_ret = 0.0

    if total_dt > 0:
        min_dt = 0.01 * total_dt
        risky_all = d.loc[
            ((d["quant_dt"] >= 5) | (d["dt"] >= min_dt)) & (d["ret_pct"] >= 20)
        ].copy()

        if not risky_all.empty:
            risky_count = int((risky_all["ret_pct"] >= 20).sum())
            very_high_count = int((risky_all["ret_pct"] >= 30).sum())
            max_ret = float(risky_all["ret_pct"].max())

            risky_top = (risky_all.groupby(full_col, as_index=False)
                        .agg(dt=("dt","sum"), cr=("cr","sum"))
                        .assign(ret_pct=lambda x: (x["cr"] / x["dt"].replace(0, pd.NA)).fillna(0) * 100)
                        .sort_values("ret_pct", ascending=False)
                        .head(5))
            risky_list = [
                dmc.ListItem(
                    dmc.Badge(
                        f"{r[full_col]} — {fmt_pct(r['ret_pct'], 1)}",
                        color=risk_color(float(r["ret_pct"])),
                        variant="filled", radius="xs"
                    )
                ) for _, r in risky_top.iterrows()
            ]

    def quality_state(max_ret_val, count_20, count_30):
        if count_20 == 0:
            return ("green", "Возвраты в норме — критичных позиций нет.")
        if count_30 > 0:
            return ("red", f"Высокие возвраты: {count_30} SKU с ≥30%")
        return ("yellow", f"Есть возвраты выше нормы: {count_20} SKU с ≥20%")

    quality_color, quality_text = quality_state(max_ret, risky_count, very_high_count)
    quality_alert = dmc.Alert(quality_text, color=quality_color, variant="light", withCloseButton=True)

    # ============== Гистограмма цен (опционально) ==============
    price_hist_card = dmc.Box()
    if "quant_dt" in d.columns and (d["quant_dt"] > 0).any():
        d_price = d.loc[(d["quant_dt"] > 0) & (d["dt"] > 0)].copy()
        d_price["price_gross_unit"] = d_price["dt"] / d_price["quant_dt"]
        d_price = d_price.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["price_gross_unit"])
        if not d_price.empty:
            q_lo, q_hi = d_price["price_gross_unit"].quantile([0.01, 0.99])
            mask = d_price["price_gross_unit"].between(q_lo, q_hi, inclusive="both")
            clipped = int((~mask).sum())
            d_clip = d_price.loc[mask].copy()

            fig_hist = px.histogram(
                d_clip,
                x="price_gross_unit",
                nbins=24,
                labels={"price_gross_unit": "Цена за единицу (Gross)"},
            )
            fig_hist.update_xaxes(range=[float(q_lo), float(q_hi)])
            fig_hist.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
            caption = dmc.Text(
                f"Показаны 1–99 перцентили; исключено выбросов: {clipped}",
                size="xs", c="dimmed"
            )
            price_hist_card = section_card(
                "Распределение цен (Gross, за ед.)",
                dmc.Stack(gap=6, children=[dcc.Graph(figure=fig_hist, config={"displayModeBar": False}), caption])
            )

   

    # ============== KPI ==============
    kpis_grid = dmc.SimpleGrid(
        cols={"base": 1, "sm": 2},
        spacing="sm",
        children=[
            kpi_card("Выручка (нетто)", fmt_compact(amount, money=True), icon="mdi:cash-multiple"),
            kpi_card("Средняя цена (Net)", fmt_compact(avg_price_net, money=True), icon="mdi:currency-rub"),
            kpi_card("Сумма возвратов", fmt_compact(total_cr, money=True), icon="mdi:cash-refund"),
            kpi_card("Коэфф. возвратов", fmt_pct(ret_coef, 1), icon="mdi:percent"),
        ]
    )


    # ============== Кольца ==============
    rings_row = dmc.Group(gap="lg", wrap="wrap", children=[
        dmc.Stack(align="center", gap=0, children=[
            dmc.RingProgress(sections=[{"value": float(revenue_pct), "color": "blue"}], size=82, thickness=10),
            dmc.Text("Доля в выручке", size="xs", c="dimmed"),
            dmc.Text(fmt_pct(revenue_pct, 2), fw=600),
        ]),
        dmc.Stack(align="center", gap=0, children=[
            dmc.RingProgress(sections=[{"value": float(ret_coef), "color": "orange"}], size=82, thickness=10),
            dmc.Text("Коэф. возвратов", size="xs", c="dimmed"),
            dmc.Text(fmt_pct(ret_coef, 2), fw=600),
        ]),
        dmc.Stack(align="center", gap=0, children=[
            dmc.RingProgress(sections=[{"value": float(agent_share), "color": "cyan"}], size=82, thickness=10),
            dmc.Text("Доля дизайнеров", size="xs", c="dimmed"),
            dmc.Text(fmt_pct(agent_share, 2), fw=600),
        ]),
        (dmc.Stack(align="center", gap=0, children=[
            dmc.RingProgress(sections=[{"value": float(designer_share or 0), "color": "grape"}], size=82, thickness=10),
            dmc.Text("«Дизайнер»", size="xs", c="dimmed"),
            dmc.Text(fmt_pct((designer_share or 0), 1), fw=600),
        ]) if designer_share is not None else dmc.Box()),
    ])

   

    # ============== SKU-аналитика ==============
    # Лучшие SKU по количеству (как было)
    best_qty_txt = ["нет данных"]
    by_fullname_qty = (
        d.groupby(full_col, as_index=False).agg(q=("quant", "sum"), a=("amount", "sum"))
    ) if full_col in d.columns else pd.DataFrame(columns=[full_col, "q", "a"])
    if not by_fullname_qty.empty:
        bq = by_fullname_qty.sort_values("q", ascending=False).head(5)
        best_qty_txt = []
        for _, r in bq.iterrows():
            price = (r["a"] / r["q"]) if r["q"] > 0 else 0.0
            best_qty_txt.append(f"{r[full_col]} — {int(round(r['q']))} шт, ~{fmt_compact(price, money=True)}/ед")

    # Топ SKU по выручке — фикс
    best_amt_list = ["нет данных"]
    if 'by_fullname_amt' in locals() and not by_fullname_amt.empty:
        top_amt = (
            by_fullname_amt
            .copy()
            .assign(amount=lambda x: pd.to_numeric(x["amount"], errors="coerce").fillna(0.0))
            .query("amount > 0")
            .sort_values("amount", ascending=False)   # ← вот ключевая правка
            .head(5)
        )
        if not top_amt.empty:
            best_amt_list = [f"{r[full_col]} — {fmt_compact(r['amount'], money=True)}" for _, r in top_amt.iterrows()]

    sku_row = dmc.SimpleGrid(
        cols={"base": 1, "sm": 2}, spacing="md",
        children=[
            section_card("Лучшие SKU по выручке", dmc.List([dmc.ListItem(x) for x in best_amt_list], withPadding=True)),

               
           section_card(
            "Лучшие SKU по выручке",
                dmc.List(
                    [dmc.ListItem(x) for x in best_qty_txt],
                    withPadding=True,
                ),
            )

        ],
    )


    # ============== Доп. графики ==============
    def has_children(node) -> bool:
        return getattr(node, "children", None) not in (None, [], ())

    extras_charts = (
        dmc.SimpleGrid(cols={"base":1, "md":2}, spacing="md",
                       children=[price_hist_card, ])
        if (has_children(price_hist_card) )
        else dmc.Box()
    )

    # ============== Итоговый layout ==============
    left_col = dmc.Stack(gap="md", children=[
        section_card(title_badge("KPI"), kpis_grid),
        section_card(
            title_badge("Ассортимент"),
            dmc.Text(
                f"{n_items} номенклатур · {n_brand} брендов · {n_manu} производителей",
                size="sm",
                opacity=0.8
            ),
        ),
        section_card(title_badge("Индикаторы"), rings_row),
    ])

    right_col = dmc.Stack(gap="md", children=[
        section_card("Парето 80/20", dmc.Stack(gap=6, children=[pareto_text, pareto_widget])),
        section_card("Рисковые SKU", dmc.Stack(gap=6, children=[quality_alert, dmc.List(risky_list, withPadding=True)])),
        dmc.SimpleGrid(cols={"base":1, "sm":2}, spacing="md", children=[
            section_card("Топ-бренды по выручке", dmc.List([dmc.ListItem(x) for x in brands_list], withPadding=True)),
            section_card("Топ-производители по выручке", dmc.List([dmc.ListItem(x) for x in manus_list], withPadding=True)),
        ]),
    ])

    return dmc.Stack(gap="md", children=[
        dmc.Group(justify="space-between", align="center", children=[
            dmc.Title("Быстрые выводы", order=4),
            dmc.Badge("Аналитика ассортимента", variant="outline", radius="xs"),
        ]),
        dmc.SimpleGrid(cols={"base": 1, "md": 2}, spacing="lg", children=[left_col, right_col]),
        sku_row,
        extras_charts,

    ])




class SegmentMainWindow:
    def __init__(self):

        self.title = dmc.Title("Сегментный анализ", order=1, c="blue")
        self.memo = dmc.Text(
            "Данный раздел предоставляет аналитику по номенклатурам продукции",
            size="xs",
        )

        self.search_input_id      = "segments_search_fullname"


        self.mslider_id = "segment_analisys_monthslider"
        self.tree_conteiner_id = "segment_analisys_tree_container_very_unique_id"
        self.details_conteiner_id = "segment_analisys_details_container_very_unique_id"
        self.tree_id = "segments_tree_id"
        self.df_store_id = "df_segment_store"
        self.last_update_lb_id = "last_update_segments_lb"
        self.group_box_id = "segment_group_box_id_unique"
        self.assing_cat = "segments_assign_cat_action_button"
        self.assing_manu = "segments_assign_manu_action_button"
        self.assing_brend = "segments_assign_brend_action_button"
        self.ag_grid_id = {'type': 'segments_ag_grid', 'index': '1'}

        self.mslider = MonthSlider(id=self.mslider_id)
        self.tree = dmc.Tree(
            id=self.tree_id,
            data=[],
            expandedIcon=DashIconify(icon="line-md:chevron-right-circle", width=20),
            collapsedIcon=DashIconify(icon="line-md:arrow-up-circle", width=20),
            checkboxes=True,
        )

        group_box_options = [
            {"value": "parent_cat", "label": "Группа"},
            {"value": "brend", "label": "Бренд"},
            {"value": "manu", "label": "Производитель"},
        ]

        self.group_box = dmc.SegmentedControl(
            id=self.group_box_id,
            data=group_box_options,
            value="parent_cat",
        )

        self.df_store = dcc.Store(id=self.df_store_id, storage_type="session")

        self.last_update_lb = dcc.Loading(
            dmc.Badge(
                size="md",
                variant="light",
                radius="xs",
                color="red",
                id=self.last_update_lb_id,
            )
        )
        
        # Делаем иконки для действий с базой данных
        self.actions = dmc.ActionIconGroup(
            [
                dmc.ActionIcon(
                    variant="default",
                    size="lg",
                    children=DashIconify(icon="mdi:file-tree-outline", width=20),
                    id = self.assing_cat,   
                    disabled=True,                 
                ),
                dmc.ActionIcon(
                    variant="default",
                    size="lg",
                    children=DashIconify(icon="mdi:teddy-bear", width=20),
                    id = self.assing_brend,
                    disabled=True, 
                ),
                dmc.ActionIcon(
                    variant="default",
                    size="lg",
                    children=DashIconify(icon="mdi:manufacturing", width=20),
                    id = self.assing_manu,
                    disabled=True, 
                ),
            ],
            orientation="horizontal",
            
            
        )

    def update_ag(self, df, rrgrid_className):
        df = df.copy()
        # безопасные расчёты
        df["dt"]       = pd.to_numeric(df.get("dt"), errors="coerce").fillna(0)
        df["cr"]       = pd.to_numeric(df.get("cr"), errors="coerce").fillna(0)
        df["quant_dt"] = pd.to_numeric(df.get("quant_dt"), errors="coerce").fillna(0)
        df["quant_cr"] = pd.to_numeric(df.get("quant_cr"), errors="coerce").fillna(0)

        df["amount"]    = df["dt"] - df["cr"]
        df["quant"]     = df["quant_dt"] - df["quant_cr"]
        df["ret_ratio"] = (df["cr"] / df["dt"].replace(0, pd.NA)).fillna(0) * 100  # ← ПО ДЕНЬГАМ!

        cols = [
            # ГРУППА «Номенклатура»
            {
                "headerName": "Номенклатура",
                "groupId": "product",
                "marryChildren": True,
                "headerClass": "ag-center-header",
                "children": [
                    {
                        "headerName": "Номенклатура",
                        "field": "fullname",
                        "minWidth": 220,
                        "type": "leftAligned",
                        "cellClass": "ag-firstcol-bg",
                        "headerClass": "ag-center-header",
                         "pinned": "left",
                    },
                    {
                        "headerName": "Бренд",
                        "field": "brend",
                        "minWidth": 160,
                        "type": "leftAligned",
                        "columnGroupShow": "open",
                        "headerClass": "ag-center-header",
                        # опционально:
                        # "valueGetter": {"function": "(p)=> (p.data?.brend && String(p.data.brend).trim()) || 'Бренд не указан'"}
                    },
                    {
                        "headerName": "Производитель",
                        "field": "manu",
                        "minWidth": 160,
                        "type": "leftAligned",
                        "columnGroupShow": "open",
                        "headerClass": "ag-center-header",
                        # "valueGetter": {"function": "(p)=> (p.data?.manu && String(p.data.manu).trim()) || 'Производитель не указан'"}
                    },
                    {
                        "headerName": "Артикль",
                        "field": "article",  # <-- подставь твоё фактическое поле (сейчас у тебя пустая строка)
                        "minWidth": 140,
                        "type": "leftAligned",
                        "columnGroupShow": "open",
                        "headerClass": "ag-center-header",
                    },
                ],
            },

            
            {
                "headerName": "Дата инициализации",
                "field": "init_date",
                "valueFormatter": {"function": "RussianDate(params.value)"},
            },
            {
                "headerName": "Последняя продажа",
                "field": "last_sales_date",
                "valueFormatter": {"function": "RussianDate(params.value)"},
            },
            {
                "headerName": "Выручка",
                "field": "amount",
                "valueFormatter": {"function": "RUB(params.value)"},
                "cellClass": "ag-firstcol-bg",
            },
            {
                "headerName": "Всего продано",
                "field": "quant",
                "valueFormatter": {"function": "FormatWithUnit(params.value,'ед')"},
            },
            {
                "headerName": "Процент возвратов",
                "field": "ret_ratio",
                "valueFormatter": {"function": "FormatWithUnit(params.value,'%')"},
            },
            {"headerName": "Дизайнеры", "field": "agent"},
            {"headerName": "id", "field": "item_id", "hide": True},
]


        return dmc.Stack(
            [
                dmc.Space(h=4),
                dmc.Title("Выбранные позиции", order=4),
                dmc.Space(h=6),
                dag.AgGrid(
                    id=self.ag_grid_id,
                    rowData=df.to_dict("records"),
                    columnDefs=cols,
                    defaultColDef={"sortable": True, "filter": True, "resizable": True},
                    dashGridOptions={
                    "rowSelection": "single", 
                    "pagination": True, 
                    "paginationPageSize": 20,
                    "suppressRowClickSelection": False,
                    #"enableCellTextSelection": True,
                    "ensureDomOrder": True,
                    #"onRowDoubleClicked": {"function": "function(params) { window.dashAgGridFunctions.onRowDoubleClick(params); }"}
                },
                    
                # getRowId="function(params) { return params.data.fullname + '_' + params.data.init_date; }",
                    style={"height": "600px", "width": "100%"},
                    className=rrgrid_className,
                    dangerously_allow_code=True,
                ),
            ]
        )


    def data(self, start_eom, end_eom):
        df = load_columns_df(COLS, start_eom, end_eom)

        return df

    def maketree(self, df_id, group):

        df = load_df_from_redis(df_id)

        if group != "parent_cat":
            df["parent_cat"] = df[group]
            df["parent_cat_id"] = df[f"{group}_id"]
            df["parent_cat"] = df["parent_cat"].fillna("Нет данных")
            df["cat"] = df["cat"].fillna("Нет категории")
            df["subcat"] = df["subcat"].fillna("Нет подкатегории")
            df["parent_cat_id"] = df["parent_cat_id"].fillna(10_000_000)
            df["cat_id"] = df["cat_id"].fillna(10_000_000)
            df["subcat_id"] = df["subcat_id"].fillna(10_000_000)
            df["item_id"] = df["item_id"].fillna(10_000_001)
            df["cat_id"] = df["parent_cat_id"].astype(str) + df["cat_id"].astype(str)
            df["subcat_id"] = df["cat_id"].astype(str) + df["subcat_id"].astype(str)

        else:
            df["parent_cat"] = df["parent_cat"].fillna("Нет группы")
            df["cat"] = df["cat"].fillna("Нет категории")
            df["subcat"] = df["subcat"].fillna("Нет подкатегории")
            df["parent_cat_id"] = df["parent_cat_id"].fillna(10_000_000)
            df["cat_id"] = df["cat_id"].fillna(10_000_000)
            df["subcat_id"] = df["subcat_id"].fillna(10_000_000)
            df["item_id"] = df["item_id"].fillna(10_000_001)

        df["amount"] = df.dt - df.cr
        df["quant"] = df.quant_dt - df.quant_cr

        df = df.pivot_table(
            index=[
                "parent_cat_id",
                "parent_cat",
                "cat_id",
                "cat",
                "subcat_id",
                "subcat",
                "fullname",
                "item_id",
            ],
            # index='fullname',
            values=["amount", "quant"],
            aggfunc={
                "amount": "sum",
                "quant": "sum",
            },
        ).reset_index()
        df["fullname"] = df["fullname"].apply(
            lambda x: x if len(x) <= 50 else x[:50] + "..."
        )

        tree = []

        def find_or_create(lst, value, label):
            """Находит или создаёт узел"""
            for node in lst:
                if node["value"] == str(value):
                    return node
            node = {
                "value": str(value),
                "label": str(label),
                "children": [],
                "_count": 0,  # внутренний счётчик
            }
            lst.append(node)
            return node

        for _, row in df.iterrows():
            pid, pname = row["parent_cat_id"], row["parent_cat"]
            cid, cname = row["cat_id"], row["cat"]
            sid, sname = row["subcat_id"], row["subcat"]
            fullname = (row["item_id"], row["fullname"])

            # Преобразуем 10_000_000 обратно в None
            cid = None if cid == 10_000_000 else cid
            sid = None if sid == 10_000_000 else sid

            # 1 уровень — parent
            parent_node = find_or_create(tree, pid, pname)

            # 2 уровень — cat
            if cid is not None:
                cat_node = find_or_create(parent_node["children"], cid, cname)
            else:
                cat_node = parent_node

            # 3 уровень — subcat
            if sid is not None:
                subcat_node = find_or_create(cat_node["children"], sid, sname)
            else:
                subcat_node = cat_node

            # 4 уровень — fullname
            subcat_node["children"].append(
                {"value": str(fullname[0]), "label": str(fullname[1])}
            )

            # Увеличиваем счётчики на всех уровнях
            parent_node["_count"] += 1
            if cat_node is not parent_node:
                cat_node["_count"] += 1
            if subcat_node not in (parent_node, cat_node):
                subcat_node["_count"] += 1

        # Финальный проход для добавления (N) в label
        def finalize_labels(lst):
            for node in lst:
                count = node.get("_count", 0)
                if count > 0:
                    node["label"] = f"{node['label']} ({count})"
                # только если есть дети
                if "children" in node and node["children"]:
                    finalize_labels(node["children"])
                # удаляем внутренний ключ
                node.pop("_count", None)

        finalize_labels(tree)

        return tree

    
    
    def layout(self):
        # --- ЛЕВАЯ КОЛОНКА  ---
        sidebar = dmc.Card(
            withBorder=True, radius="sm", shadow="sm", p="md",
            style={"position": "sticky", "top": 72, "alignSelf": "flex-start"},
            children=[
                dmc.Stack(
                    gap="sm",
                    children=[
                        dmc.Divider(label="Группировка", labelPosition="left"),
                        self.group_box,
                        
                        dmc.Divider(label="Поиск по номенклатуре", labelPosition="left"),
                            dmc.TextInput(
                                id=self.search_input_id,
                                placeholder="Начните вводить название...",
                                leftSection=DashIconify(icon="mdi:magnify", width=18),
                                debounce=350,
                            ),



                        

                        dmc.Divider(label="Выбор позиций", labelPosition="left"),
                        dcc.Loading(children=dmc.ScrollArea(
                            type="scroll",
                            style={"height": 420},
                            children=self.tree,
                        ), type="cube",        ),

                        dmc.Divider(label="Действия", labelPosition="left"),
                        dmc.Flex(self.actions, justify="flex-start"),
                       
                    ],
                )
            ],
        )

        # --- ПРАВАЯ КОЛОНКА (таблица) ---
        right_panel = dmc.Card(
            withBorder=True, radius="sm", shadow="sm", p="md",
            children=[
                dmc.Group(justify="space-between", children=[dmc.Title("Детали / выборка", order=4)]),
                dmc.Space(h=8),
                dcc.Loading(
                    id="segment-details-loading",
                    # type="default",
                    type="cube", 
                    children=dmc.Container(
                        id=self.details_conteiner_id,
                        fluid=True,
                         children=empty_placeholder(),
                        # children=dmc.Paper("Выберите элементы слева", p="md", withBorder=True, radius="sm"),
                    ),
                ),
            ],
        )

        return dmc.Container(
            fluid=True,
            children=[
                # Заголовок + подзаголовок
                dmc.Group(justify="space-between", align="center",
                        children=[self.title, dmc.Badge("Сегментный анализ", variant="outline", color="blue")]),
                dmc.Text("Данный раздел предоставляет аналитику по номенклатурам продукции", size="xs", c="dimmed"),
                dmc.Space(h=10),

                # --- Ряд "Период" (бейдж справа) ---
                dmc.Group(justify="space-between", align="center",
                        children=[dmc.Text("Период", size="sm", c="dimmed"), self.last_update_lb]),

                # --- СЛАЙДЕР НА ВСЮ ШИРИНУ ---
                dmc.Paper(
                    withBorder=True, radius="sm", p="md",
                    children=self.mslider,  # ← теперь не в колонке, занимает 100%
                ),
                dmc.Space(h=12),
                
                
                # --- ДВЕ КОЛОНКИ ---
                dmc.Grid(
                    gutter="lg", align="stretch",
                    children=[
                        dmc.GridCol(sidebar, span={"base": 12, "md": 5, "lg": 4, "xl": 3}),
                        dmc.GridCol(right_panel, span={"base": 12, "md": 7, "lg": 8, "xl": 9}),
                    ],
                ),

                # служебные блоки
                dcc.Store(id="dummy_imputs_for_segment_slider"),
                dcc.Store(id="dummy_imputs_for_segment_render"),
                self.df_store,
                CATS_MANAGEMENT.make_drawler(),
                AG_MODAL.layout()
            ],
        )



    def register_callbacks(self, app):
        
        ag_grid = self.ag_grid_id['type']
        ag_modal = AG_MODAL.modal_id['type']
        model_container = AG_MODAL.modal_conteiner_id['type']
        
        @app.callback(
            Output(self.df_store_id, "data"),
            Output(self.last_update_lb_id, "children"),
            Input(self.mslider_id, "value"),
            Input("dummy_imputs_for_segment_render", "data"),
            State(self.df_store_id, "data"),
            prevent_initial_call=False,
        )
        def update_df(slider_value, dummy, store_data):
            start, end = id_to_months(slider_value[0], slider_value[1])
            start: pd.Timestamp = pd.to_datetime(start) + pd.offsets.MonthBegin(-1)
            end: pd.Timestamp = pd.to_datetime(end) + pd.offsets.MonthEnd(0)

            start = start.strftime("%Y-%m-%d")
            end = end.strftime("%Y-%m-%d")

            if store_data and "df_id" in store_data:
                if store_data["start"] == start and store_data["end"] == end:
                    df = load_df_from_redis(store_data["df_id"])
                    if df is not None:  # ключ ещё живой в Redis
                        min_date = pd.to_datetime(start)
                        max_date = pd.to_datetime(end)
                        notification = f"{min_date.strftime('%b %y')} - {max_date.strftime('%b %y')}"
                        return no_update, notification

                delete_df_from_redis(store_data["df_id"])

            df = fletch_dataset(start, end)
            tot_revenue = df.dt.sum() - df.cr.sum()

            df_id = save_df_to_redis(df, expire_seconds=1200)

            store_dict = {
                "df_id": df_id,
                "start": start,
                "end": end,
                "tot_revenue": tot_revenue,
                "slider_val": slider_value,
            }

            nnoms = df.fullname.nunique()

            min_date = pd.to_datetime(start)
            max_date = pd.to_datetime(end)

            notificattion = f"{min_date.strftime('%b %y')} - {max_date.strftime('%b %y')} ВСЕГО: {nnoms:.0f} НОМЕНКЛАТУР"

            return store_dict, notificattion

        @app.callback(
            Output(self.tree_id, "data"),
            Input(self.df_store_id, "data"),
            Input(self.group_box_id, "value"),
            Input(self.search_input_id, "value"),
        )
        def update_tabs(store_data, group_val, q):
            id_df = store_data["df_id"]
            df = load_df_from_redis(id_df)
            if q:
                mask = df["fullname"].astype(str).str.contains(str(q), case=False, na=False)
                df_f = df.loc[mask].copy()
                tmp_id = save_df_to_redis(df_f, expire_seconds=600)
                out = self.maketree(tmp_id, group_val)
                delete_df_from_redis(tmp_id)
                return out
            return self.maketree(id_df, group_val)

        @app.callback(
            # Output(self.kpi_container_id, "children"),
            Output(self.details_conteiner_id, "children"),
            Output(self.assing_cat, 'disabled'),
            Output(self.assing_brend, 'disabled'),
            Output(self.assing_manu, 'disabled'),
            Input(self.tree_id, "checked"),
            # Input(self.kpi_compact_switch_id, "checked"),
            State("theme_switch", "checked"),
            State(self.df_store_id, "data"),
            prevent_initial_call=True,
        )
        def get_data(checked,  theme, store_data):
            if not checked:
                return  empty_placeholder(), True, True, True

            rrgrid_className = "ag-theme-alpine-dark" if theme else "ag-theme-alpine"
            
            md = get_items(checked,store_data['start'],store_data['end'])

            agent_summary = fletch_agents(checked, store_data['start'], store_data['end'])
            store_summary = fletch_stores(checked, store_data['start'], store_data['end'])
            dfa = agent_summary.copy()
            dfa['agent_sales'] = np.where(dfa['agent_name'] != 'Без дизайнера', "Через дизайнера", dfa['agent_name'])
            dfa = dfa.pivot_table(index='agent_sales', values='amount', aggfunc='sum').reset_index()
            agent_share = dfa[dfa['agent_sales'] == 'Через дизайнера']['amount'].sum() / dfa['amount'].sum() * 100 if dfa['amount'].sum() > 0 else 0.0

    
            # kpi = build_kpi(md, compact=bool(compact_on))

            details = dmc.Stack(
                [
                    insights_block(md, store_data['tot_revenue'],agent_share),
                    pareto_block(md),    
                    stores_block(store_summary),
                    self.update_ag(md, rrgrid_className),
                   
                    
                ],
                gap="md",
            )

            return  details, False, False, False

        #Вызываем управление категорями
        @app.callback(
            Output(CATS_MANAGEMENT.drawer_id,'opened'),
            Output(CATS_MANAGEMENT.drawer_conteiner_id,'children'),
            Input(self.assing_cat,'n_clicks'),
            State(self.tree_id,'checked'),
            prevent_initial_call=True,            
        )
        def update_cats(nclicks,ids):
            return True, CATS_MANAGEMENT.update_drawer(ids)
                
        @app.callback(
            Output({'type':ag_modal,'index':MATCH},'opened'),
            Output({'type':model_container,'index':MATCH},'children'),
            Input({'type':ag_grid,'index':MATCH},'selectedRows'),
            State(self.df_store_id, "data"),
            prevent_initial_call=True,            
        )
        def open_modal(double_click_data, store_data):
            start = store_data['start']
            end = store_data['end']
            
            if double_click_data:
                
                d = double_click_data[0]
                return True, AG_MODAL.update_modal(d, start, end)
            
            return no_update, no_update
            
            
            
        
        
        
        
        
        CATS_MANAGEMENT.register_callbacks(app)
        AG_MODAL.register_callbacks(app)
        
        