# barcode_details.py
import pandas as pd
import dash_mantine_components as dmc


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
    WITH sales AS (
        SELECT
            s.item_id,
            SUM(s.dt - s.cr) AS amount,
            SUM(s.quant_dt - s.quant_cr) AS quant
        FROM sales_salesdata s
        WHERE s.item_id = %(item_id)s
          AND LAST_DAY(s.date) BETWEEN %(start)s AND %(end)s
        GROUP BY s.item_id
    )
    SELECT
        b.barcode AS barcode,
        sales.amount AS amount,
        sales.quant AS quant
    FROM sales
    JOIN corporate_items_barcode t ON t.items_id = sales.item_id
    JOIN corporate_barcode b ON b.id = t.barcode_id
    ORDER BY b.barcode
    """

    df = pd.read_sql(q, engine, params={"item_id": int(item_id), "start": start, "end": end})

    if not df.empty:
        total = df["amount"].sum()
        df["share"] = (df["amount"] / total) if total else 0

    return df


def render_barcode_panel(df: pd.DataFrame, title: str):
    if df is None or df.empty:
        return dmc.Stack(
            [
                dmc.Text(title, fw=700),
                dmc.Alert("По выбранному периоду нет данных для отображения.", color="gray"),
            ],
            gap="sm",
        )

    def fmt_int(x):
        try:
            return f"{float(x):,.0f}".replace(",", " ")
        except Exception:
            return "—"

    def fmt_float(x):
        try:
            return f"{float(x):,.2f}".replace(",", " ")
        except Exception:
            return "—"

    rows = []
    for _, r in df.iterrows():
        barcode = r.get("barcode")
        amount = r.get("amount", 0)
        quant = r.get("quant", 0)
        share = r.get("share", 0)

        rows.append(
            dmc.TableTr(
                [
                    dmc.TableTd(str(barcode) if pd.notna(barcode) else "—"),
                    dmc.TableTd(fmt_int(amount)),
                    dmc.TableTd(fmt_float(quant)),
                    dmc.TableTd(f"{float(share) * 100:.1f}%"),
                ]
            )
        )

    table = dmc.Table(
        striped=True,
        highlightOnHover=True,
        withTableBorder=True,
        withColumnBorders=False,
        children=[
            dmc.TableThead(
                dmc.TableTr(
                    [
                        dmc.TableTh("Штрихкод"),
                        dmc.TableTh("Выручка"),
                        dmc.TableTh("Кол-во"),
                        dmc.TableTh("Доля"),
                    ]
                )
            ),
            dmc.TableTbody(rows),
        ],
    )

    return dmc.Stack(
        [
            dmc.Text(title, fw=700),
            dmc.Alert(
                "Внимание: разрез по штрихкодам показан как список штрихкодов товара. "
                "Продажи агрегируются по item_id (в sales_salesdata нет barcode_id).",
                color="yellow",
                variant="light",
            ),
            table,
        ],
        gap="sm",
    )
