# barcode_details.py
import pandas as pd
import dash_mantine_components as dmc
from dash import dcc
from data import ENGINE


cell_style = {"fontSize": "13px"}

def fetch_barcode_breakdown(engine, item_id: int, start: str, end: str) -> pd.DataFrame:
    """
    Детализация по штрихкодам для выбранной номенклатуры (item_id).

    ВАЖНО:
    В твоей БД sales_salesdata НЕ содержит barcode_id (судя по ошибке).
    Поэтому тут:
    - продажи агрегируются по item_id за период
    - штрихкоды подтягиваются из corporate_items_barcode -> corporate_barcode
    - amount/quant будут одинаковыми для каждого штрихкода (это честное ограничение данных)

    start/end — строки 'YYYY-MM-DD'
    """
    q = """
    select
        coalesce(b.barcode,'нет штрихкода') as barcode, -- если не нужен b.barcode только
        sum(t.dt-t.cr) as amount,
        sum(t.quant_dt - t.quant_cr) as quant

        from sales_salesdata as t
        left join corporate_barcode as b on b.id = t.barcode_id
        WHERE t.item_id = %(item_id)s
          AND LAST_DAY(t.date) BETWEEN %(start)s AND %(end)s -- and b.barcode is not null добавить если не нужен без штрихкода
        group by b.barcode
        ORDER BY b.barcode       
   
    """

    df = pd.read_sql(q, engine, params={"item_id": int(item_id), "start": start, "end": end})

    if not df.empty:
        total = df["amount"].sum()
        df["share"] = (df["amount"] / total) if total else 0

    return df


# def render_barcode_panel(df: pd.DataFrame, title: str):
#     if df is None or df.empty:
#         return dmc.Stack(
#             [
#                 dmc.Text(title, fw=700),
#                 dmc.Alert("По выбранному периоду нет данных для отображения.", color="gray"),
#             ],
#             gap="sm",
#         )

#     def fmt_int(x):
#         try:
#             return f"{float(x):,.0f}".replace(",", " ")
#         except Exception:
#             return "—"

#     def fmt_float(x):
#         try:
#             return f"{float(x):,.2f}".replace(",", " ")
#         except Exception:
#             return "—"

#     rows = []
#     for _, r in df.iterrows():
#         barcode = r.get("barcode")
#         amount = r.get("amount", 0)
#         quant = r.get("quant", 0)
#         share = r.get("share", 0)

#         rows.append(
#             dmc.TableTr(
#                 [
#                     dmc.TableTd(str(barcode) if pd.notna(barcode) else "—"),
#                     dmc.TableTd(fmt_int(amount)),
#                     dmc.TableTd(fmt_float(quant)),
#                     dmc.TableTd(f"{float(share) * 100:.1f}%"),
#                 ]
#             )
#         )

#     table = dmc.Table(
#         striped=True,
#         highlightOnHover=True,
#         withTableBorder=True,
#         withColumnBorders=False,
#         children=[
#             dmc.TableThead(
#                 dmc.TableTr(
#                     [
#                         dmc.TableTh("Штрихкод"),
#                         dmc.TableTh("Выручка"),
#                         dmc.TableTh("Кол-во"),
#                         dmc.TableTh("Доля"),
#                     ]
#                 )
#             ),
#             dmc.TableTbody(rows),
#         ],
#     )

#     return dmc.Stack(
#         [
#             dmc.Text(title, fw=700),
#             # dmc.Alert(
#             #     "Внимание: разрез по штрихкодам показан как список штрихкодов товара. "
#             #     "Продажи агрегируются по item_id (в sales_salesdata нет barcode_id).",
#             #     color="yellow",
#             #     variant="light",
#             # ),
#             table,
#         ],
#         gap="sm",
#     )



def render_barcode_panel(df: pd.DataFrame, title_name: str, subtitle: str):
    # ----- empty
    if df is None or df.empty:
        return dmc.Stack(
            [
                dmc.Stack(
                    [
                        dmc.Text(title_name, fw=700, size="lg"),
                        dmc.Text(subtitle, size="sm", c="dimmed"),
                    ],
                    gap=2,
                ),
                dmc.Divider(),
                dmc.Alert("По выбранному периоду нет данных для отображения.", color="gray", variant="light"),
            ],
            gap="md",
        )

    NBSP = "\u00A0"  # неразрывный пробел

    # ----- formatters
    def fmt_int(x):
        try:
            return f"{float(x):,.0f}".replace(",", NBSP)
        except Exception:
            return "—"

    def fmt_float(x):
        try:
            return f"{float(x):,.2f}".replace(",", NBSP)
        except Exception:
            return "—"

    # ----- typography
    body_style = {
        "fontSize": "13px",
        "fontVariantNumeric": "tabular-nums",  # ровные цифры
    }
    head_style = {"fontSize": "12px"}
    head_text_props = dict(fw=600, c="dimmed")

    def small(text, **kwargs):
        return dmc.Text(text, style=body_style, **kwargs)

    def head(text, **kwargs):
        return dmc.Text(text, style=head_style, **head_text_props, **kwargs)

    # ----- totals
    total_amount = float(df["amount"].sum()) if "amount" in df.columns else 0.0
    total_quant = float(df["quant"].sum()) if "quant" in df.columns else 0.0
    n_barcodes = int(df.shape[0])

    # ----- share
    if "share" not in df.columns:
        df = df.copy()
        df["share"] = (df["amount"] / total_amount) if total_amount else 0

    # ---- meta row
    meta = dmc.Group(
        [
            dmc.Text(f"Штрихкодов: {n_barcodes}", size="sm", c="dimmed"),
            dmc.Text(f"Выручка: {fmt_int(total_amount)}", size="sm", c="dimmed"),
            dmc.Text(f"Кол-во: {fmt_float(total_quant)}", size="sm", c="dimmed"),
        ],
        gap="lg",
        wrap="wrap",
    )

    # ---- rows
    rows = []
    for _, r in df.iterrows():
        barcode = r.get("barcode")
        amount = r.get("amount", 0)
        quant = r.get("quant", 0)
        share = float(r.get("share", 0) or 0)

        barcode_str = str(barcode) if pd.notna(barcode) else "—"

        barcode_cell = dmc.Group(
            [
                dmc.Text(
                    barcode_str,
                    style={
                        "whiteSpace": "nowrap",
                        "fontFamily": "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
                        "fontSize": "13px",
                        "lineHeight": 1.15,
                    },
                ),
                dcc.Clipboard(
                    content=barcode_str if barcode_str != "—" else "",
                    title="Скопировать",
                    style={
                        "cursor": "pointer",
                        "display": "inline-block",
                        "opacity": 0.55,
                        "padding": "0 6px",
                        "transform": "translateY(1px)",
                    },
                ),
            ],
            gap="xs",
            wrap="nowrap",
        )

        rows.append(
            dmc.TableTr(
                [
                    dmc.TableTd(barcode_cell, style={"whiteSpace": "nowrap"}),
                    dmc.TableTd(small(fmt_int(amount), ta="right")),
                    dmc.TableTd(small(fmt_float(quant), ta="right")),
                    dmc.TableTd(small(f"{share*100:.1f}%".replace(" ", NBSP), ta="right")),
                ]
            )
        )

    table = dmc.Table(
        striped=True,
        highlightOnHover=True,
        withTableBorder=True,
        withColumnBorders=False,
        horizontalSpacing="sm",
        verticalSpacing="sm",
        # ключ: таблица по контенту, без растяжки
        style={
            "width": "max-content",
            "tableLayout": "auto",
        },
        children=[
            dmc.TableThead(
                dmc.TableTr(
                    [
                        dmc.TableTh(head("Штрихкод"), style={"whiteSpace": "nowrap"}),
                        dmc.TableTh(head("Выручка", ta="right"), style={"whiteSpace": "nowrap"}),
                        dmc.TableTh(head("Кол-во", ta="right"), style={"whiteSpace": "nowrap"}),
                        dmc.TableTh(head("Доля", ta="right"), style={"whiteSpace": "nowrap"}),
                    ]
                )
            ),
            dmc.TableTbody(rows),
        ],
    )

    return dmc.Stack(
        [
            dmc.Stack(
                [
                    dmc.Text(title_name, fw=700, size="lg"),
                    dmc.Text(subtitle, size="sm", c="dimmed"),
                ],
                gap=2,
            ),
            meta,
            dmc.Divider(),
            dmc.ScrollArea(
                h=380,
                offsetScrollbars=True,
                children=dmc.Box(
                    table,
                    # ключ: контейнер не растягивает таблицу
                    style={"display": "inline-block", "width": "max-content"},
                ),
            ),
        ],
        gap="md",
    )