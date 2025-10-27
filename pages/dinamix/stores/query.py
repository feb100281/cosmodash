import pandas as pd
from data import ENGINE
import locale
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify
from math import ceil
from numpy.polynomial.polynomial import polyfit
from dash import dcc






locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")

RU_WDAYS = ['Понедельник','Вторник','Среда','Четверг','Пятница','Суббота','Воскресенье']
WD_SHORT = dict(zip(RU_WDAYS, ["Пн","Вт","Ср","Чт","Пт","Сб","Вс"]))
ru_wday_type = pd.api.types.CategoricalDtype(categories=RU_WDAYS, ordered=True)

def _fmt_rub(v: float) -> str:
    return "₽" + f"{v:,.0f}".replace(",", " ")

def _fmt_dt(d) -> str:
    return pd.to_datetime(d).strftime("%d.%m.%Y")




# ==================== ХЕЛПЕРЫ ВЫВОДА ====================

def _pct(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.1f}%"

def _color_by_delta(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "dimmed"
    return ("green" if x > 0 else "red") if x != 0 else "dimmed"

def _arrow(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return "↑" if x > 0 else ("↓" if x < 0 else "→")

def _safe_mean(s):
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.mean()) if len(s) else float("nan")

def _safe_cv(values):
    s = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    m = s.mean()
    return float(s.std(ddof=0) / m) if (len(s) and m) else float("nan")

def _tooltip(child, text: str, w=280):
    if not text:
        return child
    return dmc.Tooltip(
        label=text, withArrow=True, multiline=True, maw=w,
        style={"whiteSpace": "normal"}, children=child
    )

def _empty_block(title="Нет данных", note="В выбранном периоде нет записей."):
    return dmc.Alert(
        title=title, color="gray", variant="light",
        children=dmc.Text(note, size="sm")
    )

# (Опционально; больше не используется, но оставляю на случай совместимости)
def _acc_item(icon: str, title: str, panel_children):
    """Иконка из lucide + заголовок для аккуратного AccordionItem."""
    return dmc.AccordionItem(
        value=title,
        children=[
            dmc.AccordionControl(
                dmc.Group(
                    gap="xs",
                    children=[
                        dmc.ThemeIcon(
                            DashIconify(icon=icon, width=16),
                            variant="light", color="grape", radius="xl"
                        ),
                        dmc.Text(title, fw=600)
                    ]
                )
            ),
            dmc.AccordionPanel(children=panel_children)
        ]
    )
    

# ===== ХЕЛПЕРЫ ДЛЯ СЕЗОННОСТИ (стрелки + тултипы) =====
def _index_arrow_cell(pct: float, hint: str = None):
    """Ячейка индекса с зелёной/красной стрелкой + форматированным % и tooltip с расчётом."""
    if np.isnan(pct):
        return dmc.Text("—", size="sm", c="dimmed")

    color = "green" if pct > 0 else ("red" if pct < 0 else "gray")
    icon  = "lucide:arrow-up-right" if pct > 0 else ("lucide:arrow-down-right" if pct < 0 else "lucide:arrow-right")
    cell  = dmc.Group(
        gap=4, align="center",
        children=[
            dmc.ThemeIcon(DashIconify(icon=icon, width=14), color=color, variant="light", radius="xl", size="sm"),
            dmc.Text(_pct(pct), size="sm", c=color, fw=600)
        ]
    )
    return _tooltip(cell, hint) if hint else cell

def _fmt_month_day_idx_hint(day_val: float, dom_mean: float, idx_pct: float):
    """Текст-расчёт для tooltip: как получился индекс в процентах."""
    # idx_pct = (day_val / dom_mean - 1) * 100
    return (
        "Индекс = (среднее по дню / среднее по всем дням − 1) × 100%\n"
        f"= ({_fmt_rub(day_val)} / {_fmt_rub(dom_mean)} − 1) × 100% = {_pct(idx_pct)}"
    )


def _metric_badge(text, color="gray"):
    return dmc.Badge(text, color=color, variant="light", radius="sm", size="sm")

def _format_date_range(dates):
    """Форматирует серию подряд идущих дат в читаемый диапазон."""
    if not dates:
        return ""
    dates = sorted(pd.to_datetime(dates))
    start, end = dates[0], dates[-1]
    same_month = start.month == end.month
    same_year = start.year == end.year

    if same_month and same_year:
        # Пример: 11–15 ноября 2024
        return f"{start.day}–{end.day} {start.strftime('%B %Y')}"
    elif same_year:
        # Пример: 28 февраля – 3 марта 2024
        return f"{start.day} {start.strftime('%B')} – {end.day} {end.strftime('%B %Y')}"
    else:
        # Пример: 29 декабря 2024 – 2 января 2025
        return f"{start.strftime('%d %B %Y')} – {end.strftime('%d %B %Y')}"

def _acc_section(*, value: str, icon: str, title: str, subtitle: str = "",
                 right=None, color="grape", panel_children=None):
    """
    Универсальный конструктор секции аккордеона с иконкой, заголовком, подзаголовком и правыми бейджами.
    """
    return dmc.AccordionItem(
        value=value,
        children=[
            dmc.AccordionControl(
                dmc.Group(
                    justify="space-between",
                    align="center",
                    wrap="nowrap",
                    gap="sm",
                    children=[
                        dmc.Group(
                            gap="xs",
                            align="center",
                            wrap="nowrap",
                            children=[
                                dmc.ThemeIcon(
                                    DashIconify(icon=icon, width=16),
                                    variant="light", color=color, radius="xl", size="sm"
                                ),
                                dmc.Stack(gap=0, children=[
                                    dmc.Text(title, fw=600, size="sm"),
                                    dmc.Text(subtitle, size="xs", c="dimmed") if subtitle else None
                                ])
                            ]
                        ),
                        dmc.Group(gap=6, align="center", children=right if right else [])
                    ]
                )
            ),
            dmc.AccordionPanel(children=panel_children)
        ]
    )


# ==================== ХЕЛПЕРЫ СТАТИСТИКИ ====================

def _series_streak_above_median(series: pd.Series):
    if series.empty:
        return 0, [], float("nan")
    s = series.copy()
    med = float(s.median())
    mask = s > med
    max_len = cur = 0
    best_end_idx = None
    for i, v in enumerate(mask.values):
        if v:
            cur += 1
            if cur > max_len:
                max_len = cur
                best_end_idx = i
        else:
            cur = 0
    dates = []
    if max_len > 0 and best_end_idx is not None:
        idx = s.index
        start_i = best_end_idx - max_len + 1
        dates = [pd.to_datetime(d).date() for d in idx[start_i:best_end_idx + 1]]
    return int(max_len), dates, med

def _fmt_dom(n: int) -> str:
    return f"{n} число"

def _concentration_top_p_details(day_sum: pd.Series, p=0.2):
    if day_sum.empty or day_sum.sum() == 0:
        return float("nan"), 0, 0
    n = len(day_sum)
    k = max(1, ceil(n * p))
    share = day_sum.sort_values(ascending=False).head(k).sum() / day_sum.sum()
    return float(share * 100), int(k), int(n)

def _find_anomalies(day_sum: pd.Series, z=2.5, limit=5):
    if len(day_sum) < 6:
        return [], []
    v = day_sum.values.astype(float)
    mu, sigma = v.mean(), v.std(ddof=0)
    if sigma == 0:
        return [], []
    zscores = (v - mu) / sigma
    dfz = pd.DataFrame({"date": day_sum.index, "value": v, "z": zscores})
    pos = dfz[dfz["z"] >= z].sort_values("z", ascending=False).head(limit)
    neg = dfz[dfz["z"] <= -z].sort_values("z", ascending=True).head(limit)
    pos_list = [(r["date"], float(r["value"]), float(r["z"])) for _, r in pos.iterrows()]
    neg_list = [(r["date"], float(r["value"]), float(r["z"])) for _, r in neg.iterrows()]
    return pos_list, neg_list

def _gini(x: pd.Series | np.ndarray) -> float:
    s = np.asarray(pd.Series(x).dropna().astype(float))
    if s.size == 0:
        return float("nan")
    if np.all(s == 0):
        return 0.0
    s = np.sort(s)
    n = s.size
    cum = np.cumsum(s)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)

def _rolling_7d_change(day_series: pd.Series):
    s = day_series.dropna()
    if len(s) < 14:
        return float("nan"), float("nan")
    last7 = float(s.iloc[-7:].sum())
    prev7 = float(s.iloc[-14:-7].sum())
    chg = ((last7 / prev7 - 1) * 100) if prev7 != 0 else None
    y = s.iloc[-14:].values.astype(float)
    x = np.arange(len(y), dtype=float)
    slope = float(polyfit(x, y, 1)[1])
    return (chg if chg is not None else float("nan")), slope


# ==================== UI-БРИКИ ====================

def _kpi_card(title, value, sub=None, *, is_dark=False, hint=None, value_color=None):
    # bg = "rgba(0,0,0,0.35)" if is_dark else "#ffffff"
    head = dmc.Group(
        justify="space-between",
        align="center",
        children=[
            dmc.Text(title, size="sm", c="dimmed"),
            _tooltip(
                dmc.ActionIcon(dmc.Text("i", size="xs"), variant="subtle", size="sm", radius="xl"), hint
            ) if hint else None
        ]
    )
    return dmc.Card(
        withBorder=True, radius="lg", shadow="sm", 
        children=[
            head,
            dmc.Text(value, fw=800, size="lg", c=value_color or "inherit"),
            dmc.Text(sub, size="xs", c="dimmed") if sub else None
        ]
    )

def _anoms_table(title, rows, color):
    if not rows:
        return dmc.Text(f"{title}: нет ярко выраженных аномалий", size="sm", c="dimmed")
    head = dmc.TableThead(children=dmc.TableTr(children=[
        dmc.TableTh("Дата"),
        dmc.TableTh("Сумма"),
        dmc.TableTh(_tooltip(dmc.Text("Отклонение, σ", style={"display":"inline-block"}),
                             "Насколько день отличается от среднего по периоду (в σ)."))
    ]))
    body = dmc.TableTbody(children=[
        dmc.TableTr(children=[
            dmc.TableTd(_fmt_dt(d)),
            dmc.TableTd(_fmt_rub(v)),
            dmc.TableTd(f"{z:.2f}")
        ])
        for (d, v, z) in rows
    ])
    return dmc.Card(
        withBorder=True, radius="md", shadow="sm",
        children=[
            dmc.Text(title, fw=600, c=color, size="sm"),
            dmc.Table(
                withTableBorder=True, striped=True, highlightOnHover=True,
                horizontalSpacing="xs", verticalSpacing="xs",
                children=[head, body]
            )
        ]
    )

def _promo_block(title, lines, color="grape"):
    return dmc.Card(
        withBorder=True, radius="md", shadow="sm",
        children=[
            dmc.Text(title, fw=800, c=color),
            dmc.List(size="sm", spacing="xs", children=[dmc.ListItem(dmc.Text(t)) for t in lines])
        ]
    )


# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

def _build_heatmap_insights(df_raw: pd.DataFrame, start, end, *, weekdays: bool, is_dark: bool):
    """
    Сжатая витрина KPI (6 карточек) + 'Подробнее':
    - Итого, Месяц к месяцу, Динамика 7д, Концентрация топ-20%, Лучший месяц,
    - 1 режимная карточка (дни недели или день месяца),
    - Остальное (Gini, медиана/серии, сезонность, аномалии) — в Accordion с иконками/бейджами.
    """
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    if df.empty:
        return dmc.Spoiler("Нет данных", maxHeight=0)

    # ===== база
    day = df.groupby(df["date"].dt.normalize())["amount"].sum().sort_index()
    total = float(day.sum())
    mon = day.groupby(day.index.to_period("M")).sum()

    if len(mon) >= 2:
        last, prev = float(mon.iloc[-1]), float(mon.iloc[-2])
        mom = ((last / prev - 1) * 100) if prev != 0 else None
        mom_text = "нет базы" if prev == 0 else _pct(mom)
    else:
        mom, mom_text = None, "—"

    conc20, topk, total_days = _concentration_top_p_details(day, 0.2)
    streak_len, streak_dates, median_val = _series_streak_above_median(day)
    gini_val = _gini(day)
    r7_chg_pct, r7_slope = _rolling_7d_change(day)
    hi_anoms, lo_anoms = _find_anomalies(day, z=2.5, limit=3)

    # ===== режимные метрики
    regime_cards = []
    promo = None
    season_blocks = []

    if weekdays:
        df["eom"]  = (df["date"] + pd.offsets.MonthEnd(0)).dt.normalize()
        df["wday"] = df["date"].dt.day_name("ru_RU.UTF-8")

        m = (df.groupby(["wday", "eom"], observed=False)["amount"].sum()
               .groupby(level=0).mean().reindex(RU_WDAYS))

        best_wd = m.idxmax()
        worst_wd = m.idxmin()
        best_wd_val, worst_wd_val = float(m.max()), float(m.min())
        wknd = _safe_mean(m.loc[["Суббота", "Воскресенье"]]) if {"Суббота", "Воскресенье"}.issubset(m.index) else float("nan")
        wkdy = _safe_mean(m.loc[RU_WDAYS[:5]]) if set(RU_WDAYS[:5]).issubset(m.index) else float("nan")
        diff = ((wknd / wkdy - 1) * 100) if (wkdy and not np.isnan(wkdy)) else None

        # KPI режима (одна карточка): "Лучший день недели"
        regime_cards = [
            _kpi_card(
                "Лучший день недели",
                WD_SHORT.get(best_wd, best_wd),
                sub=f"ср. {_fmt_rub(best_wd_val)}; слабый: {WD_SHORT.get(worst_wd, worst_wd)} (ср. {_fmt_rub(worst_wd_val)})",
                is_dark=is_dark,
                hint="Средние по месяцам: какой день недели сильнее/слабее."
            )
        ]

        # промо-подсказки
        promo_lines = []
        if not np.isnan(wknd) and not np.isnan(wkdy):
            promo_lines.append(
                ("Weekend сильнее будней — " if diff and diff > 0 else "Weekend слабее будней — ") +
                ("ставьте ключевые промо на выходные." if (diff and diff > 0) else "усильте выходные: спец-офферы, витрины.")
            )
        promo_lines.append(
            f"Ставьте премиальные офферы в {WD_SHORT.get(best_wd, best_wd)}. "
            f"Для {WD_SHORT.get(worst_wd, worst_wd)} — локальная механика и контроль наличия."
        )
        promo = _promo_block("Когда делать промо", promo_lines, color="violet")

        # сезонность (индексы по дням недели)
        mean_w = float(m.mean()) if len(m.dropna()) else float("nan")
        if not np.isnan(mean_w) and mean_w:
            rows = [
                (WD_SHORT.get(w, w), (float(m.loc[w]) / mean_w - 1) * 100)
                for w in RU_WDAYS if w in m.index
            ]
            season_table = dmc.Table(
                withTableBorder=True, striped=True, highlightOnHover=True,
                horizontalSpacing="xs", verticalSpacing="xs",
                children=[
                    dmc.TableThead(children=dmc.TableTr(children=[
                        dmc.TableTh("День"), dmc.TableTh("Индекс, %")
                    ])),
                    dmc.TableTbody(children=[
                        dmc.TableTr(children=[dmc.TableTd(lbl), dmc.TableTd(_pct(val))]) for lbl, val in rows
                    ])
                ]
            )
            season_blocks = [dmc.Card(
                withBorder=True, radius="md", shadow="sm",
                children=[dmc.Text("Индекс сезонности (дни недели)", fw=600, size="sm"), season_table]
            )]

    else:
        df["eom"] = (df["date"] + pd.offsets.MonthEnd(0)).dt.normalize()
        df["dom"] = df["date"].dt.day
        dom_month = df.groupby(["dom", "eom"], observed=False)["amount"].sum().unstack("eom")
        dom_avg = dom_month.mean(axis=1)
        best_dom = int(dom_avg.idxmax())
        best_dom_val = float(dom_avg.max())
        worst_dom = int(dom_avg.idxmin())
        worst_dom_val = float(dom_avg.min())
        first3 = dom_avg.loc[[d for d in range(1, 4) if d in dom_avg.index]].mean() if len(dom_avg) else float("nan")
        last3  = dom_avg.loc[[d for d in range(29, 32) if d in dom_avg.index]].mean() if len(dom_avg) else float("nan")
        eom_eff = ((last3 / first3 - 1) * 100) if (pd.notna(first3) and first3 not in [0, None]) else None

        # KPI режима (одна карточка): "Лучший день месяца"
        regime_cards = [
            _kpi_card(
                "Лучший день месяца",
                _fmt_dom(best_dom),
                sub=f"в среднем: {_fmt_rub(best_dom_val)} • слабее всего: {_fmt_dom(worst_dom)} ({_fmt_rub(worst_dom_val)})",
                is_dark=is_dark,
                hint=("Смотрим не одно наблюдение, а средние по месяцам с равным весом. "
                      "Лучший — день месяца с наибольшей средней выручкой; слабый — с наименьшей.")
            )
        ]

        promo_lines = [
            f"Основные промо — около {best_dom} числа (пик среднего).",
            ("Усиливайте последние 2–3 дня месяца." if eom_eff and eom_eff > 0 else
             "Фокус на начале месяца — конец слабее.") if eom_eff is not None else ""
        ]
        promo = _promo_block("Когда делать промо", [p for p in promo_lines if p], color="grape")

    # ===== верхние KPI (ровно 6 карт)
    top_cards = []
    top_cards.append(_kpi_card("Итого за период", _fmt_rub(total), is_dark=is_dark))

    top_cards.append(_kpi_card(
        "Месяц к месяцу (посл./пред.)",
        f"{_arrow(mom)} {_pct(mom)}" if mom is not None else mom_text,
        is_dark=is_dark,
        value_color=_color_by_delta(mom),
        hint="Сумма последнего полного месяца vs предыдущий, %."
    ))

    top_cards.append(_kpi_card(
        "Динамика 7 дней",
        f"{_arrow(r7_chg_pct)} {_pct(r7_chg_pct)}" if not np.isnan(r7_chg_pct) else "—",
        sub=f"скорость: {(_fmt_rub(r7_slope) if not np.isnan(r7_slope) else '—')}/день",
        is_dark=is_dark,
        value_color=_color_by_delta(r7_chg_pct),
        hint="Сумма последних 7 дней vs предыдущих 7; скорость изменения — ₽/день."
    ))

    top_cards.append(_kpi_card(
        "Концентрация топ-20%",
        _pct(conc20) if not np.isnan(conc20) else "—",
        sub=f"{topk} из {total_days} дней",
        is_dark=is_dark,
        hint="Какую долю выручки делают самые сильные ~20% дней."
    ))

    if len(mon):
        best_mon = mon.idxmax().strftime("%b %Y")
        best_mon_val = float(mon.max())
        top_cards.append(_kpi_card("Лучший месяц", best_mon, sub=_fmt_rub(best_mon_val), is_dark=is_dark))
    else:
        top_cards.append(_kpi_card("Лучший месяц", "—", sub="нет данных", is_dark=is_dark))

    top_cards += regime_cards[:1]  # 6-я карточка

    # ===== подробности (Accordion) — говорящие секции
    details_items = []

    # Распределение и концентрация
    gini_text = "—" if np.isnan(gini_val) else f"{gini_val:.2f}"
    conc_text = "—" if np.isnan(conc20) else f"{conc20:.0f}%"
    dist_right = [
        _metric_badge(f"Gini {gini_text}", color="indigo"),
        _metric_badge(f"Топ-20%: {conc_text}", color="grape")
    ]

    
    dist_panel = dmc.Stack(
        gap="xs",
        children=[
            dmc.Text("Коэффициент Джини — показатель неравномерности распределения выручки по дням.", size="sm"),
            dmc.Blockquote(
                f"G = 1 - (2 / n) × Σᵢ ((n + 1 - i) × yᵢ) / Σᵢ yᵢ",
                cite="— Формула Джини",
                color="grape",
                style={"fontSize": "0.85rem", "marginTop": "-2px"}
            ),
            dmc.Text(
                f"• Значение **0** — идеальное равенство (каждый день приносит одинаковую выручку).", 
                size="sm", c="dimmed"
            ),
            dmc.Text(
                f"• Значение **1** — вся выручка сосредоточена в одном дне (максимальная концентрация).", 
                size="sm", c="dimmed"
            ),
            dmc.Text(
                f"Текущий Gini = **{gini_text}**. Это означает, что распределение выручки "
                f"{'близко к равномерному' if gini_val < 0.3 else 'умеренно концентрировано' if gini_val < 0.6 else 'высоко концентрировано'}.",
                size="sm"
            ),
            dmc.Divider(variant="dashed", size="xs"),
            dmc.Text(
                f"Топ-20% дней ({topk} из {total_days}) обеспечили {(_pct(conc20) if not np.isnan(conc20) else '—')} выручки.",
                size="sm"
            ),
            dmc.Text(
                "Этот показатель помогает понять, насколько продажи зависят от небольшого числа «пиковых» дней. "
                "Чем выше Gini и доля топ-20%, тем менее стабилен поток выручки.", 
                size="sm", c="dimmed"
            ),
        ]
    )

    details_items.append(_acc_section(
        value="dist",
        icon="lucide:pie-chart",
        title="Распределение и концентрация",
        subtitle="Насколько выручка сосредоточена в части дней",
        right=dist_right,
        color="grape",
        panel_children=dist_panel
    ))


   

    # ===== Стабильность и серии
    cv_val = _safe_cv(day)  # σ/μ по дневным суммам
    cv_text = "—" if np.isnan(cv_val) else f"{cv_val*100:.0f}%"

    # Цвет бейджа по степени волатильности
    if np.isnan(cv_val):
        cv_color = "gray"
    elif cv_val <= 0.15:
        cv_color = "teal"      # стабильный поток
    elif cv_val <= 0.35:
        cv_color = "yellow"    # умеренная волатильность
    else:
        cv_color = "red"       # высокая волатильность

    # Мини-график (sparkline)
    sparkline_fig = go.Figure()
    sparkline_fig.add_trace(go.Scatter(
        x=day.index,
        y=day.values,
        mode="lines",
        line=dict(width=1.6),
        hoverinfo="skip"
    ))
    sparkline_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=40,
        width=120,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    sparkline = dcc.Graph(
        figure=sparkline_fig,
        config={"displayModeBar": False},
        style={"height": "40px", "width": "120px", "marginTop": "2px"}
    )

    # Правый блок с метриками и мини-графиком
    series_right = [
        _metric_badge(f"Серия: {streak_len} дн.", color="cyan"),
        _metric_badge(f"Медиана: {_fmt_rub(median_val)}", color="gray"),
        _metric_badge(f"CV: {cv_text}", color=cv_color),
        sparkline,
    ]

    series_dates_str = _format_date_range(streak_dates)
    series_panel = dmc.Stack(
        gap="xs",
        children=[
            dmc.Text(
                "Оценивает устойчивость дневной выручки и продолжительность положительных серий.",
                size="sm"
            ),
            dmc.Blockquote(
                "CV = σ / μ  — коэффициент вариации (чем ниже, тем стабильнее).\n"
                "Медиана — значение, выше/ниже которого половина всех дней.",
                cite="— Метрики стабильности",
                color="cyan",
                style={"fontSize": "0.85rem", "marginTop": "-2px"}
            ),
            dmc.Text(
                f"• Серия дней выше медианы: **{streak_len}** дн."
                + (f" ({series_dates_str})" if series_dates_str else ""),
                size="sm"
            ),
            dmc.Text(
                f"• Коэффициент вариации (CV): **{cv_text}** — "
                + (
                    "стабильный поток (≤15%)" if (not np.isnan(cv_val) and cv_val <= 0.15)
                    else "умеренная волатильность (16–35%)" if (not np.isnan(cv_val) and cv_val <= 0.35)
                    else "высокая волатильность (>35%)" if (not np.isnan(cv_val))
                    else "недостаточно данных"
                ),
                size="sm"
            ),
            dmc.Divider(variant="dashed", size="xs"),
            dmc.Text(
                "Зачем это нужно: длинные серии > медианы и низкий CV означают устойчивый поток продаж "
                "и предсказуемый cash flow. Рост CV и короткие серии — сигнал волатильности: "
                "усиливайте промо и балансируйте периоды спада.",
                size="sm",
                c="dimmed"
            ),
        ]
    )

    # Сам AccordionItem
    details_items.append(_acc_section(
        value="stability",
        icon="lucide:activity",
        title="Стабильность и серии",
        subtitle="Медиана, серия выше медианы и волатильность (CV)",
        right=series_right,
        color="cyan",
        panel_children=series_panel
    ))


   
    
    # ===== СЕЗОННОСТЬ =====
    if not weekdays:
        # индексы по дням месяца
        df["eom"] = (df["date"] + pd.offsets.MonthEnd(0)).dt.normalize()
        df["dom"] = df["date"].dt.day

        dom_month = df.groupby(["dom", "eom"], observed=False)["amount"].sum().unstack("eom")
        dom_avg = dom_month.mean(axis=1)                         # ср. выручка для каждого числа месяца
        dom_mean = float(dom_avg.mean()) if len(dom_avg.dropna()) else float("nan")  # ср. по всем дням месяца

        if not np.isnan(dom_mean) and dom_mean:
            # индекс дня = (avg_day / mean_all - 1) * 100
            idx = [(int(d), (float(v) / dom_mean - 1) * 100) for d, v in dom_avg.items()]
            top5 = sorted(idx, key=lambda t: t[1], reverse=True)[:5]
            bot5 = sorted(idx, key=lambda t: t[1])[:5]

            # Таблица с зелёно/красными стрелками и подсказками-расчётами
            season_table_rows = []
            for d, v in top5 + bot5:
                day_val = float(dom_avg.loc[d])
                hint = _fmt_month_day_idx_hint(day_val, dom_mean, v)
                season_table_rows.append(
                    dmc.TableTr(children=[
                        dmc.TableTd(f"{d}"),
                        dmc.TableTd(_index_arrow_cell(v, hint))
                    ])
                )

            season_table = dmc.Table(
                withTableBorder=True, striped=True, highlightOnHover=True,
                horizontalSpacing="xs", verticalSpacing="xs",
                children=[
                    dmc.TableThead(children=dmc.TableTr(children=[
                        dmc.TableTh("День месяца"),
                        dmc.TableTh(_tooltip(dmc.Text("Индекс, %"), "Отклонение от среднего по всем дням месяца"))
                    ])),
                    dmc.TableTbody(children=season_table_rows)
                ]
            )

            # Пример расчёта (берём лучший и худший день)
            best_d, best_idx = top5[0]
            worst_d, worst_idx = bot5[0]
            best_val = float(dom_avg.loc[best_d])
            worst_val = float(dom_avg.loc[worst_d])

            example_block = dmc.Stack(gap=4, children=[
                dmc.Text("Как читается индекс:", size="sm"),
                dmc.Blockquote(
                    "Индекс дня = (средняя выручка этого числа / средняя выручка по всем дням месяца − 1) × 100%",
                    cite="— Формула индекса сезонности по дню месяца",
                    color="violet",
                    style={"fontSize": "0.85rem", "marginTop": "-2px"}
                ),
                dmc.Text(
                    f"Пример (лучший день: {best_d}): "
                    f"({_fmt_rub(best_val)} / {_fmt_rub(dom_mean)} − 1) × 100% = {_pct(best_idx)}",
                    size="sm"
                ),
                dmc.Text(
                    f"Пример (слабый день: {worst_d}): "
                    f"({_fmt_rub(worst_val)} / {_fmt_rub(dom_mean)} − 1) × 100% = {_pct(worst_idx)}",
                    size="sm"
                ),
            ])

            # Верхнее объяснение секции
            season_header = dmc.Stack(gap="xs", children=[
                dmc.Text(
                    "Индексы показывают, насколько конкретные числа месяца сильнее или слабее среднего уровня продаж.",
                    size="sm"
                ),
                dmc.Text(
                    "Положительный индекс ( " + "↑".replace("↑", "") + " ) — день сильнее среднего; "
                    "отрицательный — слабее. Значения помогают планировать промо и прогнозировать пики/провалы.",
                    size="sm", c="dimmed"
                ),
            ])

            details_items.append(_acc_section(
                value="season-dom",
                icon="lucide:calendar-range",
                title="Сезонность по дням месяца",
                subtitle="Топ-5 и анти-5 индексов с расчётами",
                right=[_metric_badge("Дни месяца", color="violet"),
                    _metric_badge(f"Среднее: {_fmt_rub(dom_mean)}", color="gray")],
                color="violet",
                panel_children=dmc.Stack(gap="sm", children=[
                    season_header,
                    dmc.Divider(variant="dashed", size="xs"),
                    season_table,
                    example_block
                ])
            ))
    else:
        # улучшим и вариант по дням недели — тоже со стрелками
        if season_blocks:
            # season_blocks[0] у тебя уже собран как карточка с таблицей; переделаем кратко на стрелки
            df["eom"]  = (df["date"] + pd.offsets.MonthEnd(0)).dt.normalize()
            df["wday"] = df["date"].dt.day_name("ru_RU.UTF-8")
            m = (df.groupby(["wday","eom"], observed=False)["amount"].sum()
                .groupby(level=0).mean().reindex(RU_WDAYS))
            mean_w = float(m.mean()) if len(m.dropna()) else float("nan")

            if not np.isnan(mean_w) and mean_w:
                rows = []
                for w in RU_WDAYS:
                    if w in m.index and not np.isnan(m.loc[w]):
                        idx_pct = (float(m.loc[w]) / mean_w - 1) * 100
                        hint = _fmt_month_day_idx_hint(float(m.loc[w]), mean_w, idx_pct).replace("дню", "дню недели")
                        rows.append(dmc.TableTr(children=[
                            dmc.TableTd(WD_SHORT.get(w, w)),
                            dmc.TableTd(_index_arrow_cell(idx_pct, hint))
                        ]))

                wd_table = dmc.Table(
                    withTableBorder=True, striped=True, highlightOnHover=True,
                    horizontalSpacing="xs", verticalSpacing="xs",
                    children=[
                        dmc.TableThead(children=dmc.TableTr(children=[
                            dmc.TableTh("День недели"),
                            dmc.TableTh(_tooltip(dmc.Text("Индекс, %"), "Отклонение от среднего по дням недели"))
                        ])),
                        dmc.TableTbody(children=rows)
                    ]
                )

                details_items.append(_acc_section(
                    value="season-wd",
                    icon="lucide:calendar-days",
                    title="Сезонность по дням недели",
                    subtitle="Средние по месяцам + индексы к среднему",
                    right=[_metric_badge("Дни недели", color="violet"),
                        _metric_badge(f"Среднее: {_fmt_rub(mean_w)}", color="gray")],
                    color="violet",
                    panel_children=dmc.Stack(gap="sm", children=[
                        dmc.Text("Положительный индекс — день недели сильнее среднего; отрицательный — слабее.", size="sm"),
                        wd_table
                    ])
                ))


   # # Аномалии

    # ===== ХЕЛПЕРЫ ДЛЯ АНОМАЛИЙ (стрелки + тултипы) =====
  

    def _td_span(child, n: int):
        return dmc.TableTd(child, td={"colSpan": n})

    def _z_arrow_cell(x_val: float, mu: float, sigma: float):
        """Ячейка с стрелкой, % к среднему и тултипом с формулой z-score и подстановкой чисел."""
        if sigma in [0, None] or np.isnan(sigma) or np.isnan(mu) or np.isnan(x_val):
            return dmc.Text("—", size="sm", c="dimmed")

        z = (x_val - mu) / sigma
        pct = (x_val / mu - 1) * 100

        color = "green" if z > 0 else ("red" if z < 0 else "gray")
        icon  = "lucide:arrow-up-right" if z > 0 else ("lucide:arrow-down-right" if z < 0 else "lucide:arrow-right")

        cell = dmc.Group(
            gap=6, align="center",
            children=[
                dmc.ThemeIcon(DashIconify(icon=icon, width=14), color=color, variant="light", radius="xl", size="sm"),
                dmc.Text(_pct(pct), size="sm", c=color, fw=600),
                dmc.Badge(f"z = {z:.2f}", color=color, variant="light", radius="sm", size="sm"),
            ]
        )

        hint = (
            "z = (x − μ) / σ\n"
            f"x = {_fmt_rub(x_val)}\n"
            f"μ = {_fmt_rub(mu)},  σ = {_fmt_rub(sigma)}\n"
            f"% к среднему = {_pct(pct)}"
        )
        return _tooltip(cell, hint)

    def _badge_tip(label: str, value: str, *, color="gray", tip: str | None = None):
        """Бейдж с подсказкой, чтобы пользователю было понятно без документации."""
        badge = _metric_badge(f"{label}: {value}", color=color)
        return _tooltip(badge, tip) if tip else badge


    # ===== АНОМАЛИИ =====
    mu = float(day.mean()) if len(day) else float("nan")
    sigma = float(day.std(ddof=0)) if len(day) else float("nan")

    peaks = hi_anoms or []  # список кортежей: (date, value, z)
    dips  = lo_anoms or []

    anoms_count = len(peaks) + len(dips)

    # Понятные бейджи справа + подробные тултипы
    mu_text    = _fmt_rub(mu)    if not np.isnan(mu)    else "—"
    sigma_text = _fmt_rub(sigma) if not np.isnan(sigma) else "—"

    anoms_right = [
        _badge_tip(
            "Среднее", mu_text, color="gray",
            tip="Средняя дневная выручка (μ). От неё считаем отклонения и z-score."
        ),
        _badge_tip(
            "Разброс (σ)", sigma_text, color="gray",
            tip="Стандартное отклонение — характерная амплитуда колебаний вокруг среднего."
        ),
        _badge_tip(
            "Пики", str(len(peaks)), color="green",
            tip="Количество дней с существенно выше среднего (обычно z ≥ 2.5)."
        ),
        _badge_tip(
            "Провалы", str(len(dips)), color="red",
            tip="Количество дней существенно ниже среднего (обычно z ≤ −2.5)."
        ),
    ]

    if anoms_count:
        # Таблица пиков
        peaks_rows = [
            dmc.TableTr(children=[
                dmc.TableTd(_fmt_dt(d)),
                dmc.TableTd(_fmt_rub(v)),
                dmc.TableTd(_z_arrow_cell(v, mu, sigma))
            ])
            for (d, v, _z) in peaks
        ]
        peaks_table = dmc.Card(
            withBorder=True, radius="md", shadow="sm",
            children=[
                dmc.Text("Пиковые дни", fw=600, c="green", size="sm"),
                dmc.Table(
                    withTableBorder=True, striped=True, highlightOnHover=True,
                    horizontalSpacing="xs", verticalSpacing="xs",
                    children=[
                        dmc.TableThead(children=dmc.TableTr(children=[
                            dmc.TableTh("Дата"), dmc.TableTh("Сумма"), dmc.TableTh("Отклонение")
                        ])),
                        dmc.TableTbody(children=peaks_rows if peaks_rows else [
                            dmc.TableTr(children=[_td_span(dmc.Text("Нет пиков", c="dimmed", size="sm"), 3)])
                        ])
                    ]
                )
            ]
        )

        # Таблица провалов
        dips_rows = [
            dmc.TableTr(children=[
                dmc.TableTd(_fmt_dt(d)),
                dmc.TableTd(_fmt_rub(v)),
                dmc.TableTd(_z_arrow_cell(v, mu, sigma))
            ])
            for (d, v, _z) in dips
        ]
        dips_table = dmc.Card(
            withBorder=True, radius="md", shadow="sm",
            children=[
                dmc.Text("Провальные дни", fw=600, c="red", size="sm"),
                dmc.Table(
                    withTableBorder=True, striped=True, highlightOnHover=True,
                    horizontalSpacing="xs", verticalSpacing="xs",
                    children=[
                        dmc.TableThead(children=dmc.TableTr(children=[
                            dmc.TableTh("Дата"), dmc.TableTh("Сумма"), dmc.TableTh("Отклонение")
                        ])),
                        dmc.TableTbody(children=dips_rows if dips_rows else [
                            dmc.TableTr(children=[_td_span(dmc.Text("Нет провалов", c="dimmed", size="sm"), 3)])
                        ])
                    ]
                )
            ]
        )

        anoms_panel = dmc.Stack(
            gap="sm",
            children=[
                dmc.Text(
                    "Выявляем статистически значимые отклонения дневной выручки от среднего уровня.",
                    size="sm"
                ),
                dmc.Blockquote(
                    "z = (x − μ) / σ — стандартный балл (z-score).\n"
                    "Ориентиры: |z| ≥ 2.0 — заметно; ≥ 2.5 — сильно; ≥ 3.0 — экстремально.",
                    cite="— Оценка аномалий",
                    color="red",
                    style={"fontSize": "0.85rem", "marginTop": "-2px"}
                ),
                dmc.SimpleGrid(cols=2, spacing="sm", children=[peaks_table, dips_table]),
                dmc.Alert(
                    title="Как использовать",
                    color="gray", variant="light",
                    children=dmc.Text(
                        "Пики — удачные дни/промо (масштабируйте практики). "
                        "Провалы — риск-слоты (усильте наличие, промо, персонал, каналы).",
                        size="sm"
                    )
                )
            ]
        )
    else:
        anoms_panel = _empty_block(
            "Аномалий не обнаружено",
            "В выбранном периоде дни не выделяются статистически значимо."
        )

    details_items.append(_acc_section(
        value="anoms",
        icon="lucide:signal-high",
        title="Аномалии (пики/провалы)",
        subtitle="Стандартные отклонения (z-score) и % к среднему",
        right=anoms_right,
        color="red",
        panel_children=anoms_panel
    ))

    # Пересобираем аккордеон (если это конец секции)
    details = dmc.Accordion(
        variant="separated", multiple=True, radius="md", chevronPosition="right",
        children=details_items
    )

    # ===== сборка
    content = dmc.Stack(
        gap="sm",
        children=[
          
            dmc.SimpleGrid(cols=3, spacing="sm", children=top_cards),
            promo if promo else _empty_block("Нет рекомендаций по промо", "Недостаточно данных для подсказок."),
            dmc.Divider(label="Подробнее", labelPosition="center"),
            details,
            dmc.Space(h=12)
        ]
    )

    return dmc.Spoiler(
        showLabel=dmc.Group(
            gap="xs", align="center",
            children=[
                dmc.ThemeIcon(
                    DashIconify(icon="lucide:lightbulb", width=18),
                    color="teal", variant="light", radius="xs", size="sm"
                ),
                dmc.Badge(
                    "Показать авто-выводы",
                    color="teal", variant="light", radius="xs", size="lg",
                    style={"cursor": "pointer", "fontWeight": 500}
                ),
            ]
        ),
        hideLabel=dmc.Group(
            gap="xs", align="center",
            children=[
                dmc.ThemeIcon(
                    DashIconify(icon="lucide:eye-off", width=18),
                    color="teal", variant="light", radius="xl", size="sm"
                ),
                dmc.Badge(
                    "Скрыть авто-выводы",
                    color="teal", variant="light", radius="xs", size="lg",
                    style={"cursor": "pointer", "fontWeight": 500}
                ),
            ]
        ),
        transitionDuration=400,
        maxHeight=220,
        children=content
    )






def get_days_heatmap(start='2024-10-01', end='2025-10-31', store=None, is_dark=False, weekdays=False):
    start = pd.to_datetime(start) + pd.offsets.MonthBegin(-1)
    start = start.strftime('%Y-%m-%d')
    end_dt = pd.to_datetime(end)

    # фильтры по магазину
    if store:
        store_title = ', '.join(store)
        stores_clause = ','.join(f"'{s}'" for s in store)
        stores = f'and sg.name in ({stores_clause})'
    else:
        store_title = 'Все магазины'
        stores = ''

    # заголовки
    agg_title = "Средняя выручка по дням недели" if weekdays else "Выручка по дням месяца"
    mode_hint = "(среднее)" if weekdays else "(сумма)"
    title = f"{store_title} • {agg_title} {mode_hint}"

    q = f"""
    SELECT 
        s.date,
        sum(s.dt - s.cr) as amount,
        sg.name as store_gr_name
    FROM sales_salesdata as s
    left join corporate_stores as st on st.id = s.store_id
    left join corporate_storegroups as sg on sg.id = st.gr_id
    where date between '{start}' and '{end}'
    {stores}
    group by date, sg.name
    """

    df_raw = pd.read_sql(q, ENGINE)
    df = df_raw.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['eom'] = df['date'] + pd.offsets.MonthEnd(0)
    df['eom'] = df['eom'].dt.normalize()
    df['day'] = df['date'].dt.day
    df['wday'] = df['date'].dt.day_name('ru_RU.UTF-8')

    # 📊 главное отличие: сначала группировка по day + wday + eom
    df = (
        df.groupby(['day', 'wday', 'eom'], as_index=False, observed=False)['amount']
        .sum()
    )

    # потом pivot как в оригинале
    if weekdays:
        df_pivot = df.pivot_table(
            index='wday',
            columns='eom',
            values='amount',
            aggfunc='mean',
            observed=False
        ).fillna(0).reindex(RU_WDAYS)
        y = df_pivot.index.to_list()
    else:
        df_pivot = df.pivot_table(
            index='day',
            columns='eom',
            values='amount',
            aggfunc='sum',
            observed=False
        ).fillna(0)
        y = df_pivot.index.to_list()

    z = df_pivot.to_numpy()
    x = [d.strftime("%b %y") for d in df_pivot.columns]

    # 🎨 оформление
    if is_dark:
        bg_color = '#1e1e1e'
        text_color = '#ffffff'
        grid_color = '#333333'
        colorscale = 'Cividis'
        template = 'plotly_dark'
    else:
        bg_color = '#ffffff'
        text_color = '#000000'
        grid_color = '#e0e0e0'
        colorscale = 'Blues'
        template = 'plotly_white'

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            coloraxis="coloraxis",
            text=z,
            texttemplate="%{text:,.0f}",
            textfont={"size": 8},
        )
    )

    fig.update_xaxes(side="top", gridcolor=grid_color)
    fig.update_yaxes(tickmode='linear', dtick=1, autorange='reversed', gridcolor=grid_color)
    fig.update_layout(
        template=template,
        height=900,
        margin=dict(l=60, r=20, t=60, b=60),
        coloraxis=dict(
            colorscale=colorscale,
            colorbar=dict(
                tickcolor=text_color,
                tickfont=dict(color=text_color),
            ),
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
    )

    # 🧠 авто-выводы остаются
    spoiler = _build_heatmap_insights(
        df_raw[['date', 'amount', 'store_gr_name']],
        start, end_dt,
        weekdays=weekdays,
        is_dark=is_dark
    )

    return fig, title, spoiler
