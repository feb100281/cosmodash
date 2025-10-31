import numpy as np
import pandas as pd
from data import ENGINE
import dash_mantine_components as dmc
from components import COLORS_BY_SHADE, COLORS_BY_COLOR





VALS_DICT = {
    'amount':'Выручка',
    'dt':'Продажи',
    'cr':'Возвраты',
    'quant':'Количество'
}

OPTIONS_SWITCHS = {
    'cat':"Показать категории",
    'store_gr_name':'Показать магазины'
}

def fletch_cats():
    q = """
    with parent as (
    select
    id as parent_id,
    name as parent
    from corporate_cattree
    where parent_id is Null
    )

    select 
    p.parent_id,
    p.parent,
    cat.id as cat_id,
    cat.name as cat,
    COALESCE(sc.id,0) as subcat_id,
    COALESCE(sc.name,'Нет подкатегории') as subcat
    from corporate_cattree as cat

    left join parent as p on p.parent_id = cat.parent_id
    left join corporate_subcategory as sc on sc.category_id = cat.id
    where cat.parent_id is not Null       
    """
    return pd.read_sql(q,ENGINE)


def get_df(start, end):
    q = f"""
    select
    date,
    sum(s.dt) as dt,
    sum(s.cr) as cr,
    sum(s.dt - s.cr) as amount,
    parent.id as parent_id,
    parent.name as parent,    
    cat.id as cat_id,
    cat.name as cat,
    coalesce(subcat.id,0) as subcat_id,
    coalesce(subcat.name,'Нет подкатегории') as subcat

    from sales_salesdata as s
    left join corporate_items as i on i.id = s.item_id
    left join corporate_cattree as cat on cat.id = i.cat_id
    left join corporate_cattree as parent on parent.id = cat.parent_id
    left join corporate_subcategory as subcat on subcat.id = i.subcat_id
    where date between '{start}' and '{end}'

    GROUP BY date, parent_id, parent, cat_id, cat, subcat_id, subcat;
    """
    return pd.read_sql(q,ENGINE)





def make_summary_df(start, end):
    q = f"""
    select
    date,
    sum(s.dt) as dt,
    sum(s.cr) as cr,
    sum(s.dt - s.cr) as amount,
    sum(s.quant_dt - s.quant_cr) as quant,
    parent.id as parent_id,
    parent.name as parent,    
    cat.id as cat_id,
    cat.name as cat,
	sg.name as store_gr_name
    

    from sales_salesdata as s
    left join corporate_items as i on i.id = s.item_id
    left join corporate_cattree as cat on cat.id = i.cat_id
    left join corporate_cattree as parent on parent.id = cat.parent_id
    left join corporate_stores as st on st.id = s.store_id
    left join corporate_storegroups as sg on sg.id = st.gr_id 
    where date between '{start}' and '{end}'

    GROUP BY date, parent_id, cat_id, store_gr_name
        
    """
    return pd.read_sql(q,ENGINE)


   





# ---- ВСПОМОГАТЕЛЬНОЕ ----
def _fmt_int(n: float) -> str:
    """Человеческое форматирование целых без запятой, с пробелами."""
    try:
        return f"{float(n):,.0f}".replace(",", " ")
    except Exception:
        return str(n)

def _fmt_delta_pct(cur: float, ref: float) -> float:
    if not ref:
        return 0.0
    return (cur - ref) / ref * 100

def _kpi_cards(df_current: pd.DataFrame, df_reff: pd.DataFrame, val: str, start, end):
    cur = float(df_current[val].sum())
    ref = float(df_reff[val].sum())
    delta = cur - ref
    delta_pct = _fmt_delta_pct(cur, ref)
    color = "green" if delta >= 0 else "red"

    # Форматируем названия месяцев
    cur_label = pd.to_datetime(end).strftime("%B %Y")       # например "Октябрь 2025"
    ref_label = pd.to_datetime(start).strftime("%B %Y")     # например "Июль 2025"

    return dmc.Group(
        [
            dmc.Card(withBorder=True, radius="md", p="md", children=[
                dmc.Text(cur_label, size="xs", c="dimmed"),
                dmc.Title(_fmt_int(cur), order=3),
            ]),
            dmc.Card(withBorder=True, radius="md", p="md", children=[
                dmc.Text(ref_label, size="xs", c="dimmed"),
                dmc.Title(_fmt_int(ref), order=3),
            ]),
            dmc.Card(withBorder=True, radius="md", p="md", children=[
                dmc.Text("Изменение", size="xs", c="dimmed"),
                dmc.Group(
                    [
                        dmc.Title(_fmt_int(delta), order=3),
                        dmc.Badge(f"{delta_pct:+.1f}%", color=color, variant="light", radius='xs'),
                    ],
                    gap=8,
                ),
            ]),
        ],
        justify="space-between",
        grow=True,
        wrap="wrap",
        gap="md",
    )

def _top_table(
    dff: pd.DataFrame,
    option: str,
    val: str,
    label_current: str,
    label_reff: str,
    n: int = 100,
):
    """
    dff — pivot с колонками ['areff','current'] + option.
    Показываем оба месяца и дельты.
    """
    base = dff[[option, "areff", "current"]].copy()
    base = base.sort_values("current", ascending=False).head(n)

    base["var"] = base["current"] - base["areff"]
    base["var_pct"] = np.where(base["areff"] != 0, base["var"] / base["areff"] * 100, np.nan)

    rows = []
    for _, r in base.iterrows():
        color = "green" if r["var"] >= 0 else "red"
        rows.append(
            dmc.TableTr(
                [
                    dmc.TableTd(dmc.Text(str(r[option]))),
                    dmc.TableTd(_fmt_int(r["current"])),
                    dmc.TableTd(_fmt_int(r["areff"])),
                    dmc.TableTd(dmc.Badge(_fmt_int(r["var"]), color=color, variant="light", radius='xs')),
                    dmc.TableTd(
                        dmc.Badge(
                            "--" if pd.isna(r["var_pct"]) else f"{r['var_pct']:+.1f}%",
                            color=color,
                            variant="outline",
                            radius='xs'
                        )
                    ),
                ]
            )
        )

    return dmc.Stack(
        [
            dmc.Text(f"Top-10: {label_current} vs {label_reff}", size="sm", c="dimmed"),
            dmc.Table(
                [
                    dmc.TableThead(
                        dmc.TableTr(
                            [
                                dmc.TableTh("Элемент"),
                                dmc.TableTh(label_current),   # текущий месяц словами
                                dmc.TableTh(label_reff),      # первый выбранный месяц словами
                                dmc.TableTh("Δ"),
                                dmc.TableTh("Δ%"),
                            ]
                        )
                    ),
                    dmc.TableTbody(rows),
                ],
                striped=True,
                highlightOnHover=True,
                horizontalSpacing="md",
                verticalSpacing="sm",
            ),
        ],
        gap="xs",
    )

# ---- ОСНОВНАЯ ФУНКЦИЯ ----
def cats_report(start, end, option='cat', val='amount'):
    """
    Возвращает контейнер: KPI-карточки + Tabs по parent
    (водопад + Top-10 с двумя месяцами).
    """
    end_start = end[:-2] + "01"

    # текущий период (с 1-го числа до end)
    df_current = make_summary_df(end_start, end)
    df_current["tp"] = "current"

    # порядок табов — по текущему периоду
    df_current_pars = (
        df_current.pivot_table(index="parent", values=val, aggfunc="sum")
        .reset_index()
        .sort_values(val, ascending=False)
    )
    parent_list = df_current_pars["parent"].unique().tolist()

    # референс: от start до конца стартового месяца
    end_dt = pd.to_datetime(end)
    start_dt = pd.to_datetime(start)
    start_end = (start_dt + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    df_reff = make_summary_df(start, start_end)
    df_reff["tp"] = "areff"

    # подписи месяцев (локаль у тебя уже ставится в main.py)
    label_current = end_dt.strftime("%B %Y").capitalize()
    label_reff = start_dt.strftime("%B %Y").capitalize()

    # общий df
    df = pd.concat([df_current, df_reff], ignore_index=True)

    # агрегаты для бейджей табов
    tabs_totals = (
        df.groupby(["parent", "tp"])[val]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ("areff", "current"):
        if col not in tabs_totals.columns:
            tabs_totals[col] = 0
    tabs_totals["var"] = tabs_totals["current"] - tabs_totals["areff"]

    totals_map = {
        row["parent"]: {
            "areff": float(row["areff"]),
            "current": float(row["current"]),
            "var": float(row["var"]),
        }
        for _, row in tabs_totals.iterrows()
    }

    def _format_badge_value(v: float, metric: str) -> str:
        sign = "+" if v >= 0 else "−"
        num = abs(v)
        if metric != "quant":
            if num >= 1_000_000:
                body = f"{num/1_000_000:.1f} млн"
            elif num >= 1_000:
                body = f"{num/1_000:.0f} тыс"
            else:
                body = f"{num:.0f}"
        else:
            body = f"{num/1_000:.0f}k" if num >= 1_000 else f"{num:.0f}"
        return f"{sign}{body}"

    # KPI сверху
    kpis = _kpi_cards(df_current, df_reff, val, start, end)


    # Tabs
    tabs = []
    i = 0
    for parent in parent_list:
        i += 1
        dff_parent = df[df["parent"] == parent].copy()

        # pivot по выбранной оси (option)
        dff = (
            dff_parent.pivot_table(
                index=option,
                columns=["tp"],
                values=val,
                aggfunc="sum",
            )
            .reset_index()
            .sort_values("current", ascending=False)
            .fillna(0)
        )
        for col in ("areff", "current"):
            if col not in dff.columns:
                dff[col] = 0.0
        dff["var"] = dff["current"] - dff["areff"]

        # данные для waterfall
        data = []
        init_value = float(dff["areff"].sum())
        end_value = float(dff["current"].sum())

        data.append(
            {
                "item": f"{VALS_DICT[val]} {start_dt.strftime('%b %Y')}",
                f"{VALS_DICT[val]}": init_value,
                "color": "grape",
            }
        )
        for _, r in dff.iterrows():
            c = "red" if r["var"] < 0 else "green"
            data.append(
                {
                    "item": "𝜟 " + str(r[option]),
                    f"{VALS_DICT[val]}": float(r["var"]),
                    "color": c,
                }
            )
        data.append(
            {
                "item": f"{VALS_DICT[val]} за {end_dt.strftime('%b %Y')}",
                f"{VALS_DICT[val]}": end_value,
                "color": "grape",
                "standalone": True,
            }
        )

        vf = {"function": "formatNumberIntl"} if val != "quant" else {"function": "formatIntl"}

        tab = dmc.TabsPanel(
            [
                dmc.Container(
                    [
                        dmc.Space(h=10),
                        dmc.Switch(
                            id={"type": "val_switch", "index": i},
                            labelPosition="right",
                            label="Показать значения",
                            size="sm",
                            radius="lg",
                            color="blue",
                            disabled=False,
                            withThumbIndicator=True,
                        ),
                        dmc.Space(h=10),
                        dmc.BarChart(
                            h=350,
                            data=data,
                            dataKey="item",
                            type="waterfall",
                            series=[{"name": f"{VALS_DICT[val]}", "color": COLORS_BY_SHADE[0]}],
                            withLegend=True,
                            valueFormatter=vf,
                            withBarValueLabel=False,

                            xAxisProps={
                                "interval": 0,
                                "angle": -30,
                                "textAnchor": "end",
                                "height": 64,
                                "tickMargin": 8,
                            },
                            id={"type": "cat_chart", "index": i},
                        ),
                        dmc.Space(h=24),
                        _top_table(
                            dff,
                            option,
                            val,
                            label_current=label_current,
                            label_reff=label_reff,
                            n=10,
                        ),
                    ],
                    fluid=True,
                )
            ],
            value=f"{parent}",
        )
        tabs.append(tab)

    # TabsList с бейджами
    tabs_list_items = []
    for parent in parent_list:
        diff_val = totals_map.get(parent, {}).get("var", 0.0)
        badge_color = "green" if diff_val > 0 else ("red" if diff_val < 0 else "gray")
        badge_text = _format_badge_value(diff_val, val)

        tabs_list_items.append(
            dmc.TabsTab(
                dmc.Group(
                    [
                        dmc.Text(parent, size="sm"),
                        dmc.Badge(badge_text, color=badge_color, variant="light", size="sm", radius="xs"),
                    ],
                    gap=6,
                    align="center",
                ),
                value=f"{parent}",
            )
        )

    tab_list = [dmc.TabsList(tabs_list_items)] + tabs

    body = dmc.Tabs(
        tab_list,
        value=parent_list[0] if parent_list else None,
        orientation="horizontal",
    )

    # Итоговый контейнер: KPI + табы
    container = dmc.Container(
        children=[
            kpis,
            dmc.Space(h=12),
            body,
        ],
        fluid=True,
    )
    return container