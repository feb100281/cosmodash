# pages/cats/analysis.py
"""Сравнение двух месяцев по группам, категориям и подкатегориям + текстовая справка."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .queries import VALS_DICT, get_df

METRICS = ["amount", "dt", "cr", "quant"]
LEVELS = {
    "parent": ["parent"],
    "cat": ["parent", "cat"],
    "subcat": ["parent", "cat", "subcat"],
}
MONTHS_NOM = ["январь", "февраль", "март", "апрель", "май", "июнь",
              "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
MONTHS_PREP = ["январе", "феврале", "марте", "апреле", "мае", "июне",
               "июле", "августе", "сентябре", "октябре", "ноябре", "декабре"]


MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня",
              "июля", "августа", "сентября", "октября", "ноября", "декабря"]


def month_gen(ts) -> str:
    return f"{MONTHS_GEN[ts.month - 1]} {ts.year}"


def month_nom(ts) -> str:
    return f"{MONTHS_NOM[ts.month - 1]} {ts.year}"


def month_prep(ts) -> str:
    return f"{MONTHS_PREP[ts.month - 1]} {ts.year}"


def month_bounds(start: str, end: str):
    """Сравниваемые месяцы: первый месяц периода (база) и последний (текущий)."""
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    ref = (start_dt.replace(day=1), start_dt + pd.offsets.MonthEnd(0))
    cur = (end_dt.replace(day=1), end_dt)
    return ref, cur


def load_compare(start: str, end: str) -> dict:
    ref, cur = month_bounds(start, end)
    frames = []
    for tp, (a, b) in (("ref", ref), ("cur", cur)):
        d = get_df(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"))
        d["tp"] = tp
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for c in ("parent", "cat", "subcat"):
        df[c] = df[c].fillna("Без категории" if c != "subcat" else "Нет подкатегории").astype(str)
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0.0)
    return {
        "df": df,
        "ref_label": month_nom(ref[0]),
        "ref_gen": month_gen(ref[0]),
        "cur_label": month_nom(cur[0]),
        "ref_prep": month_prep(ref[0]),
        "cur_prep": month_prep(cur[0]),
    }


def level_table(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Строка на элемент уровня: <metric>_cur, <metric>_ref, <metric>_var, <metric>_pct, share."""
    keys = LEVELS[level]
    piv = df.pivot_table(index=keys, columns="tp", values=METRICS, aggfunc="sum", fill_value=0.0)
    piv.columns = [f"{m}_{tp}" for m, tp in piv.columns]
    piv = piv.reset_index()
    for m in METRICS:
        for tp in ("cur", "ref"):
            if f"{m}_{tp}" not in piv.columns:
                piv[f"{m}_{tp}"] = 0.0
        piv[f"{m}_var"] = piv[f"{m}_cur"] - piv[f"{m}_ref"]
        piv[f"{m}_pct"] = np.where(piv[f"{m}_ref"] != 0, piv[f"{m}_var"] / piv[f"{m}_ref"], np.nan)
    total_cur = piv["amount_cur"].sum()
    piv["share"] = piv["amount_cur"] / total_cur if total_cur else 0.0
    piv["ret_share_cur"] = np.where(piv["dt_cur"] != 0, piv["cr_cur"] / piv["dt_cur"], np.nan)
    piv["ret_share_ref"] = np.where(piv["dt_ref"] != 0, piv["cr_ref"] / piv["dt_ref"], np.nan)
    return piv.sort_values("amount_cur", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Текст
# ---------------------------------------------------------------------------

def fmt_value(v: float, val: str) -> str:
    v = float(v or 0)
    if val == "quant":
        return f"{v:,.0f} шт.".replace(",", " ")
    a = abs(v)
    if a >= 1_000_000:
        return f"{v / 1_000_000:.1f} млн ₽".replace(".", ",")
    if a >= 1_000:
        return f"{v / 1_000:.0f} тыс. ₽"
    return f"{v:,.0f} ₽".replace(",", " ")


def fmt_delta(v: float, val: str) -> str:
    return ("+" if v >= 0 else "−") + fmt_value(abs(v), val)


def fmt_pct(p, new_text: str = "новая") -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return new_text
    if p >= 1:
        return f"в {1 + p:.1f} раза".replace(".", ",")
    return f"{p * 100:+.0f}%".replace("-", "−")


def pct1(p) -> str:
    return f"{p * 100:.1f}%".replace(".", ",")


def _names(frame: pd.DataFrame, col: str, val: str, n: int = 3, with_parent: bool = False) -> str:
    parts = []
    for r in frame.head(n).itertuples():
        name = f"«{getattr(r, col)}»"
        if with_parent:
            name += f" в «{getattr(r, 'cat')}»"
        parts.append(f"{name} ({fmt_delta(getattr(r, val + '_var'), val)}, {fmt_pct(getattr(r, val + '_pct'))})")
    return ", ".join(parts)


def build_narrative(data: dict, val: str = "amount") -> list[dict]:
    """
    Справка обычным языком.
    Возвращает разделы: [{"title": str, "items": [str, ...]}].
    """
    df = data["df"]
    if df.empty:
        return [{"title": "Нет данных", "items": ["За выбранные месяцы продаж нет."]}]

    metric = VALS_DICT.get(val, val).lower()
    groups = level_table(df, "parent")
    cats = level_table(df, "cat")
    subs = level_table(df, "subcat")
    c, r = f"{val}_cur", f"{val}_ref"

    total_cur, total_ref = groups[c].sum(), groups[r].sum()
    total_var = total_cur - total_ref
    total_pct = total_var / total_ref if total_ref else np.nan
    sections = []

    # 1. Общая картина
    if total_ref == 0:
        head = (f"{metric.capitalize()} в {data['cur_prep']} — {fmt_value(total_cur, val)}; "
                f"в {data['ref_prep']} продаж не было.")
    else:
        trend = "рост" if total_var >= 0 else "снижение"
        head = (f"{metric.capitalize()} в {data['cur_prep']} — {fmt_value(total_cur, val)}, "
                f"в {data['ref_prep']} — {fmt_value(total_ref, val)}: {trend} на "
                f"{fmt_value(abs(total_var), val)} ({fmt_pct(total_pct)}).")
    items = [head]
    cur_end = pd.to_datetime(df.loc[df["tp"] == "cur", "date"]).max() if "date" in df.columns else None
    today = pd.Timestamp.today().normalize()
    month_cur = pd.Timestamp(year=today.year, month=today.month, day=1)
    if cur_end is not None and pd.notna(cur_end) and cur_end >= month_cur:
        items.append(f"Внимание: {data['cur_label']} ещё не закончился, поэтому неполный месяц "
                     f"сравнивается с полным — рост будет занижен, падение завышено.")
    top3 = cats.head(3)
    if len(cats) >= 3 and cats["amount_cur"].sum():
        items.append(
            f"Три крупнейшие категории по выручке — {', '.join('«' + x + '»' for x in top3['cat'])} — "
            f"дают {top3['share'].sum():.0%} выручки в {data['cur_prep']}."
        )
    sections.append({"title": "Общая картина", "items": items})

    # 2. Группы
    g = groups.copy()
    items = []
    growth = g[g[f"{val}_var"] > 0].sort_values(f"{val}_var", ascending=False)
    fall = g[g[f"{val}_var"] < 0].sort_values(f"{val}_var")
    if total_var > 0 and not growth.empty:
        lead = growth.iloc[0]
        items.append(f"Главный вклад в рост дала группа «{lead['parent']}»: {fmt_delta(lead[f'{val}_var'], val)} "
                     f"({fmt_pct(lead[f'{val}_pct'])}).")
    if total_var < 0 and not fall.empty:
        lead = fall.iloc[0]
        items.append(f"Сильнее всего просела группа «{lead['parent']}»: {fmt_delta(lead[f'{val}_var'], val)} "
                     f"({fmt_pct(lead[f'{val}_pct'])}).")
    if not growth.empty:
        items.append(f"Выросли: {_names(growth, 'parent', val, 5)}.")
    if not fall.empty:
        items.append(f"Снизились: {_names(fall, 'parent', val, 5)}.")
    sections.append({"title": "Группы", "items": items or ["Изменений по группам нет."]})

    # 3. Категории
    items = []
    cg = cats[cats[f"{val}_var"] > 0].sort_values(f"{val}_var", ascending=False)
    cf = cats[cats[f"{val}_var"] < 0].sort_values(f"{val}_var")
    if not cg.empty:
        items.append(f"Больше всего выросли: {_names(cg, 'cat', val)}.")
    if not cf.empty:
        items.append(f"Больше всего снизились: {_names(cf, 'cat', val)}.")
    new_c = cats[(cats[r] == 0) & (cats[c] > 0)]
    gone_c = cats[(cats[r] > 0) & (cats[c] == 0)]
    if not new_c.empty:
        items.append(f"Появились продажи в категориях: {', '.join('«' + x + '»' for x in new_c['cat'].head(8))}.")
    if not gone_c.empty:
        items.append(f"Не было продаж в {data['cur_prep']} (а в {data['ref_prep']} были): "
                     f"{', '.join('«' + x + '»' for x in gone_c['cat'].head(8))}.")
    sections.append({"title": "Категории", "items": items or ["Изменений по категориям нет."]})

    # 4. Подкатегории
    items = []
    sg = subs[subs[f"{val}_var"] > 0].sort_values(f"{val}_var", ascending=False)
    sf = subs[subs[f"{val}_var"] < 0].sort_values(f"{val}_var")
    if not sg.empty:
        items.append(f"Лидеры роста: {_names(sg, 'subcat', val, 5, with_parent=True)}.")
    if not sf.empty:
        items.append(f"Сильнее всего упали: {_names(sf, 'subcat', val, 5, with_parent=True)}.")
    new_s = subs[(subs[r] == 0) & (subs[c] > 0)]
    gone_s = subs[(subs[r] > 0) & (subs[c] == 0)]
    if len(new_s) or len(gone_s):
        items.append(f"Новых подкатегорий с продажами — {len(new_s)}, пропавших из продаж — {len(gone_s)}.")
    sections.append({"title": "Подкатегории", "items": items or ["Изменений по подкатегориям нет."]})

    # 5. Возвраты
    dt_cur, dt_ref = groups["dt_cur"].sum(), groups["dt_ref"].sum()
    cr_cur, cr_ref = groups["cr_cur"].sum(), groups["cr_ref"].sum()
    items = []
    if dt_cur:
        rc = cr_cur / dt_cur
        rr = cr_ref / dt_ref if dt_ref else None
        line = f"Возвраты в {data['cur_prep']} — {fmt_value(cr_cur, 'cr')}, это {pct1(rc)} от продаж"
        if rr is not None:
            line += f" (в {data['ref_prep']} — {pct1(rr)})"
        items.append(line + ".")
        high = cats[(cats["dt_cur"] > 0) & (cats["ret_share_cur"] >= max(rc * 2, 0.05))] \
            .sort_values("cr_cur", ascending=False)
        if not high.empty:
            items.append("Высокая доля возвратов: " + ", ".join(
                f"«{x.cat}» ({x.ret_share_cur:.0%})" for x in high.head(5).itertuples()) + ".")
    sections.append({"title": "Возвраты", "items": items or ["Возвратов нет."]})

    # 6. Что стоит сделать
    items = []
    big_fall = cf[(cf[r] > 0) & (cf[f"{val}_pct"] <= -0.2)].head(3)
    if not big_fall.empty:
        items.append("Разобрать просевшие категории "
                     + ", ".join("«" + x + "»" for x in big_fall["cat"])
                     + ": проверить наличие товара и остатки в магазинах, цены и выкладку.")
    if not cg.empty:
        items.append("Для растущих категорий "
                     + ", ".join("«" + x + "»" for x in cg["cat"].head(3))
                     + " проверить запас по ходовым позициям, чтобы рост не упёрся в дефицит.")
    if not gone_s.empty:
        items.append(f"По {len(gone_s)} подкатегориям продажи прекратились — уточнить, выведены ли они "
                     f"из ассортимента или просто закончился товар.")
    if dt_cur and dt_ref and (cr_cur / dt_cur) > (cr_ref / dt_ref) * 1.3:
        items.append("Доля возвратов заметно выросла — разобрать причины возвратов по категориям с высокой долей.")
    sections.append({"title": "Что стоит сделать", "items": items or ["Критичных изменений нет."]})
    return sections
