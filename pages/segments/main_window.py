import pandas as pd
import numpy as np
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import dash_ag_grid as dag



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

def _nb_fmt(v: float, money=False):
    try: v = float(v)
    except: v = 0.0
    s = f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"
    s = s.replace(",", " ").replace(" ", NBSP_THIN)
    return f"{s}{NBSP_THIN}₽" if money else s

def kpi_card(label: str, value: str, icon: str, color="blue"):
    return dmc.Paper(
        withBorder=True, radius=0, p="md",
        style={
            "minWidth": 220,              # чтобы карточки не сжимались
            "minHeight": 112,
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "space-between",
        },
        children=[
            dmc.Group(align="center", gap="xs", children=[
                dmc.ThemeIcon(DashIconify(icon=icon, width=18),
                              variant="light", color=color, size=32, radius="xl"),
                dmc.Text(label, c="dimmed", size="sm"),
            ]),
            dmc.Text(
                value, fw=800,
                style={
                    "fontSize": "clamp(20px, 3.2vw, 28px)",   # адаптивно
                    "whiteSpace": "nowrap",
                    "fontVariantNumeric": "tabular-nums",
                },
            ),
        ],
    )

def build_kpi(df: pd.DataFrame, compact: bool = True):
    dt  = pd.to_numeric(df.get("dt"),       errors="coerce").fillna(0).sum()
    cr  = pd.to_numeric(df.get("cr"),       errors="coerce").fillna(0).sum()
    qdt = pd.to_numeric(df.get("quant_dt"), errors="coerce").fillna(0).sum()
    qcr = pd.to_numeric(df.get("quant_cr"), errors="coerce").fillna(0).sum()
    n_items   = df.get("fullname", pd.Series(dtype="object")).nunique()
    amount    = dt - cr
    quant_net = qdt - qcr
    ret_coef  = (cr / dt * 100) if dt > 0 else 0.0
    avg_price = (amount / quant_net) if quant_net > 0 else 0.0

    F = (lambda x, money=False: fmt_compact(x, money)) if compact else (lambda x, money=False: fmt_grouped(x, money))

    row = dmc.Group(
        gap="md", wrap="nowrap",
        style={"overflowX": "auto", "paddingBottom": 4},
        children=[
            kpi_card("Номенклатур",        f"{int(n_items)}",                   "mdi:format-list-bulleted"),
            kpi_card("Продажи",       F(dt, money=True),                   "mdi:cash-multiple",     "green"),
            kpi_card("Возвраты",      F(cr, money=True),                   "mdi:cash-refund",       "red"),
            kpi_card("Чистая выручка",     F(amount, money=True),               "mdi:chart-line-variant","grape"),
            kpi_card("Коэф. возвратов",    f"{ret_coef:.2f}%".replace(".", ","),"mdi:percent-outline",   "orange"),
            kpi_card("Средняя цена / ед",  F(avg_price, money=True),            "mdi:cube-outline",      "indigo"),
        ],
    )
    return row

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
            # valueFormatter={"function":"RUB"},
            withLegend=False,
            gridAxis="xy",
            # добавим линию доли как вторую ось (если в твоей версии MantineCharts есть mixed chart)
            # если нет поддержки линии — можно вывести вторым AreaChart отдельно
        ),
        dmc.Space(h=6),
        dmc.Progress(value=float(top["cum_share"].iloc[-1] if not top.empty else 0), size="sm", striped=True, radius="xs"),
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

def insights_block(df: pd.DataFrame, tot_revenue: float, agent_share: float):
    d = df.copy()

    # числовые столбцы -> float
    for c in ["dt","cr","quant_dt","quant_cr"]:
        d[c] = pd.to_numeric(d.get(c), errors="coerce").fillna(0)

    d["amount"]  = d["dt"] - d["cr"]
    d['tot_revenue'] = tot_revenue
    d['revenue_pct'] = (d['amount'] / d['tot_revenue'] * 100).fillna(0)
    d["quant"]   = d["quant_dt"] - d["quant_cr"]
    d["ret_pct"] = (d["cr"] / d["dt"].replace(0, pd.NA)).fillna(0) * 100

    # размеры витрины
    n_items = d["fullname"].nunique()
    n_brand = d["brend"].nunique() if "brend" in d.columns else 0
    n_manu  = d["manu"].nunique()  if "manu"  in d.columns else 0

    total_dt = float(d["dt"].sum())
    total_cr = float(d["cr"].sum())
    amount   = float(d["amount"].sum())
    revenue_pct = round(float(d["revenue_pct"].sum()), 2)
    q_net    = float(d["quant"].sum())
    avg_price = amount / q_net if q_net > 0 else 0.0
    ret_coef  = (total_cr / total_dt * 100) if total_dt > 0 else 0.0

    # помощник для поиска колонок с альтернативными названиями
    def find_col(cands):
        lc = {c.lower(): c for c in d.columns}
        for name in cands:
            if name in lc: return lc[name]
        return None

    agent_col = find_col(["agent_name"])
    store_col = find_col(["store","store_name","shop","store_gr_name","magazin"])

    # доля «дизайнера», если есть
    designer_share = None
    if agent_col is not None:
        s = d[agent_col].astype(str)
        is_des = (s.str.contains("дизайн", case=False, na=False) |
                  s.str.fullmatch(r"\s*дизайнер\s*", case=False, na=False))
        dt_des = float(d.loc[is_des, "dt"].sum())
        designer_share = (dt_des / total_dt * 100) if total_dt > 0 else 0.0

    # ——— Тексты разделов ———

    # 1) сводка + два колечка
    summary = dmc.Stack(gap=4, children=[
        dmc.Text(f"Ассортимент: {n_items} номенклатур, {n_brand} брендов, {n_manu} производителей."),
        dmc.Text(f"Средняя цена: {fmt_compact(avg_price, money=True)}; выручка: {fmt_compact(amount, money=True)}."),
        dmc.Group(gap="lg", wrap="wrap", children=[
            dmc.Stack(align="center", gap=0, children=[
                dmc.RingProgress(sections=[{"value": float(revenue_pct), "color": "blue"}], size=80, thickness=10),
                dmc.Text("Доля в выручке", size="xs", c="dimmed"),
                dmc.Text(f"{revenue_pct:.2f}%".replace(".", ","), fw=600),
            ]),
            dmc.Stack(align="center", gap=0, children=[
                dmc.RingProgress(sections=[{"value": float(ret_coef), "color": "orange"}], size=80, thickness=10),
                dmc.Text("Коэф. возвратов", size="xs", c="dimmed"),
                dmc.Text(f"{ret_coef:.2f}%".replace(".", ","), fw=600),
            ]),
            dmc.Stack(align="center", gap=0, children=[
                dmc.RingProgress(sections=[{"value": float(agent_share), "color": "cyan"}], size=80, thickness=10),
                dmc.Text("Доля дизайнеров", size="xs", c="dimmed"),
                dmc.Text(f"{agent_share:.2f}%".replace(".", ","), fw=600),
            ]),
            
            (dmc.Stack(align="center", gap=0, children=[
                dmc.RingProgress(sections=[{"value": float(designer_share or 0), "color": "grape"}], size=80, thickness=10),
                dmc.Text("Доля «дизайнера»", size="xs", c="dimmed"),
                dmc.Text(f"{(designer_share or 0):.1f}%".replace(".", ","), fw=600),
            ]) if designer_share is not None else dmc.Box()),
        ]),
    ])

    # 2) Парето-бейдж
    pareto_badge = dmc.Badge("—", variant="light")
    if amount > 0 and n_items > 0:
        g = (d.groupby("fullname", as_index=False)["amount"].sum()
               .sort_values("amount", ascending=False).reset_index(drop=True))
        g["cum_share"] = g["amount"].cumsum() / g["amount"].sum() * 100
        k80 = int((g["cum_share"] >= 80).idxmax()) + 1 if not g.empty else 0
        share_of_cat = 100 * k80 / n_items if n_items > 0 else 0
        pareto_badge = dmc.Badge(
            f"Топ-{k80} SKU (~{share_of_cat:.1f}%) дают 80% выручки".replace(".", ","),
            variant="light", 	radius="xs",
        )

    # 3) Топ-бренды и производители по выручке
    def top_money(col, k=3):
        if col not in d.columns: return ["—"]
        g = (d.groupby(col, as_index=False)["amount"].sum()
               .sort_values("amount", ascending=False).head(k))
        return [f"{r[col]} — {fmt_compact(r['amount'], money=True)}" for _, r in g.iterrows()]

    brands_list = top_money("brend")
    manus_list  = top_money("manu")

    # 4) Лучшие SKU по выручке
    best_amt = (d.groupby("fullname", as_index=False)["amount"].sum()
                  .sort_values("amount", ascending=False).head(5))
    best_amt_txt = [f"{r['fullname']} — {fmt_compact(r['amount'], money=True)}"
                    for _, r in best_amt.iterrows()]

    # 5) Лучшие SKU по количеству + цена за 1 ед
    best_qty = (d.groupby("fullname", as_index=False)
                  .agg(q=("quant","sum"), a=("amount","sum"))
                  .sort_values("q", ascending=False).head(5))
    best_qty_txt = []
    for _, r in best_qty.iterrows():
        price = (r["a"]/r["q"]) if r["q"] > 0 else 0.0
        best_qty_txt.append(f"{r['fullname']} — {int(round(r['q']))} шт, ~{fmt_compact(price, money=True)}/ед")

    # 6) Новые модели за 30 дней (qty, revenue, цена/ед)
    new_models_txt, n_new = [], 0
    if "init_date" in d.columns:
        d["init_date"] = pd.to_datetime(d["init_date"], errors="coerce")
        anchor = pd.to_datetime(d["date"], errors="coerce").max() if "date" in d.columns else d["init_date"].max()
        if pd.notna(anchor):
            cutoff = anchor - pd.Timedelta(days=30)
            new_df = d.loc[d["init_date"] >= cutoff]
            n_new = new_df["fullname"].nunique()
            if n_new > 0:
                agg = (new_df.groupby("fullname", as_index=False)
                             .agg(q=("quant","sum"), a=("amount","sum"))
                             .sort_values("a", ascending=False).head(10))
                for _, r in agg.iterrows():
                    p = (r["a"]/r["q"]) if r["q"] > 0 else 0.0
                    new_models_txt.append(
                        f"{r['fullname']} — {int(round(r['q']))} шт; {fmt_compact(r['a'], money=True)}; ~{fmt_compact(p, money=True)}/ед"
                    )

    # 7) Рисковые SKU (высокий % возвратов)
    min_q = 5
    min_dt = 0.01 * total_dt
    risky = d.loc[((d["quant_dt"] >= min_q) | (d["dt"] >= min_dt)) & (d["ret_pct"] >= 20)]
    risky = (risky.groupby("fullname", as_index=False)
                  .agg(dt=("dt","sum"), cr=("cr","sum"))
                  .assign(ret_pct=lambda x: (x["cr"] / x["dt"].replace(0, pd.NA)).fillna(0) * 100)
                  .sort_values("ret_pct", ascending=False)
                  .head(5))
    risky_txt = [f"{r['fullname']} — {r['ret_pct']:.1f}%".replace(".", ",") for _, r in risky.iterrows()]
    if not risky_txt: risky_txt = ["—"]

    # 8) Топ менеджеров/магазинов (если столбцы есть)
    agents_txt = ["—"]
    if agent_col is not None:
        ag = (d.groupby(agent_col, as_index=False)["amount"].sum()
                .sort_values("amount", ascending=False).head(3))
        if not ag.empty:
            agents_txt = [f"{r[agent_col]} — {fmt_compact(r['amount'], money=True)}" for _, r in ag.iterrows()]

    stores_txt = ["—"]
    if store_col is not None:
        st = (d.groupby(store_col, as_index=False)["amount"].sum()
                .sort_values("amount", ascending=False).head(3))
        if not st.empty:
            stores_txt = [f"{r[store_col]} — {fmt_compact(r['amount'], money=True)}" for _, r in st.iterrows()]

    # ——— Верстка: одна колонка с разделителями ———
    section = lambda title, items: dmc.Stack(gap=4, children=[
        dmc.Text(title, fw=700),
        dmc.List([dmc.ListItem(x) for x in items], withPadding=True),
    ])

    return dmc.Card(withBorder=True, radius=0, p="md", children=[
        dmc.Group(justify="space-between", align="center",
                  children=[dmc.Title("Быстрые выводы", order=5),
                            dmc.Badge("аналитика ассортимента", variant="outline", radius="xs",)]),
        dmc.Space(h=6),

        summary,
        dmc.Space(h=6),
        pareto_badge,

        dmc.Divider(label="Топ-бренды и производители", my="md"),
        section("Топ-бренды по выручке", brands_list),
        section("Топ-производители по выручке", manus_list),

        dmc.Divider(label="SKU-аналитика", my="md"),
        section("Лучшие SKU по выручке", best_amt_txt if best_amt_txt else ["—"]),
        section("Лучшие SKU по количеству", best_qty_txt if best_qty_txt else ["—"]),

        dmc.Divider(label=f"Новые модели за 30 дней: {n_new}", my="md"),
        dmc.List([dmc.ListItem(x) for x in (new_models_txt or ["—"])], withPadding=True),

        dmc.Divider(label="Рисковые SKU (высокий % возвратов)", my="md"),
        dmc.List([dmc.ListItem(x) for x in risky_txt], withPadding=True),

        ((dmc.Divider(label="Менеджеры и магазины", my="md"),
          section("Топ-менеджеры", agents_txt),
          section("Топ-магазины",  stores_txt))
         if (agent_col is not None or store_col is not None) else dmc.Box()),
    ])





class SegmentMainWindow:
    def __init__(self):

        self.title = dmc.Title("Сегментный анализ", order=1, c="blue")
        self.memo = dmc.Text(
            "Данный раздел предоставляет аналитику по номенклатурам продукции",
            size="xs",
        )
        # self.kpi_container_id = "segment_kpi_container"
        # self.kpi_compact_switch_id = "segment_kpi_compact"
        self.search_input_id      = "segments_search_fullname"
        # self.only_returns_id      = "segments_only_returns_switch"
        # self.designer_share_id    = "segments_designer_share_number"   # %
        # self.designer_enable_id   = "segments_designer_enable_switch"
        # self.ret_threshold_id     = "segments_ret_threshold_number"    # %
        # self.ret_enable_id        = "segments_ret_enable_switch"
        # self.pareto_enable_id     = "segments_pareto_enable_switch"


        # self.mslider_id = {"type":"segment_analisys_monthslider", "index":'1'}
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
            {"headerName": "Номенклатура", "field": "fullname"},
            {"headerName": "Дата инициализации", "field": "init_date",
            "valueFormatter": {"function": "RussianDate(params.value)"}},
            {"headerName": "Последняя продажа", "field": "last_sales_date",
            "valueFormatter": {"function": "RussianDate(params.value)"}},

            {"headerName": "Выручка", "field": "amount",
            "valueFormatter": {"function": "RUB(params.value)"},
            "cellClass": "ag-firstcol-bg"},

            {"headerName": "Всего продано", "field": "quant",
            "valueFormatter": {"function": "FormatWithUnit(params.value,'ед')"}},

            {"headerName": "Процент возвратов", "field": "ret_ratio",
            "valueFormatter": {"function": "FormatWithUnit(params.value,'%')"}},

            {"headerName": "Бренд", "field": "brend"},
            {"headerName": "Производитель", "field": "manu"},
            {"headerName": "Дизайнеры", "field": "agent"},
            {'headerName': "Артикль", "field": ""},
            {'headerName':'id', 'field':'item_id', 'hide':True},
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

                            # dmc.Divider(label="Фильтры анализа", labelPosition="left"),
                            # dmc.Stack(gap="xs", children=[
                            #     dmc.Group(gap="sm", grow=True, align="center", children=[
                            #         dmc.Switch(id=self.designer_enable_id, label="Доля продаж через дизайнера ≥", size="sm"),
                            #         dmc.NumberInput(id=self.designer_share_id, value=80, min=0, max=100, step=5, rightSection=dmc.Text("%")),
                            #     ]),
                            #     dmc.Group(gap="sm", grow=True, align="center", children=[
                            #         dmc.Switch(id=self.ret_enable_id, label="Процент возвратов ≥", size="sm"),
                            #         dmc.NumberInput(id=self.ret_threshold_id, value=20, min=0, max=100, step=5, rightSection=dmc.Text("%")),
                            #     ]),
                            #     dmc.Switch(id=self.pareto_enable_id, label="Показать Парето по выручке", size="sm"),
                            # ]),

                        

                        dmc.Divider(label="Выбор позиций", labelPosition="left"),
                        dmc.ScrollArea(
                            type="scroll",
                            style={"height": 420},
                            children=self.tree,
                        ),

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
                    type="default",
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
                
                # # --- Полоса управления KPI ---
                # dmc.Group(
                #     justify="space-between",
                #     children=[
                #         dmc.Text("Показатели (KPI)", fw=600),
                #         dmc.Switch(
                #             id=self.kpi_compact_switch_id,
                #             label="Короткие числа (тыс/млн)",
                #             checked=True,                # ← по умолчанию включено
                #             size="sm",
                #         ),
                #     ],
                # ),
                # dmc.Space(h=6),
                
                # # --- KPI-полоса на всю ширину ---
                # dmc.Paper(
                #     withBorder=True, radius="sm", p="md",
                #     children=dmc.Container(id=self.kpi_container_id, fluid=True),
                # ),
                # dmc.Space(h=12),

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
        
        