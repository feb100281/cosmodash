# matrix/export_excel.py
from __future__ import annotations

from io import BytesIO
from typing import Dict, Optional, List

import pandas as pd


def _fetch_barcodes_for_items(engine, item_ids: List[int], start: str, end: str) -> pd.DataFrame:
    """
    Возвращает разрез по штрихкодам СРАЗУ по всем item_id.
    Группировка: item_id + barcode.
    """
    if not item_ids:
        return pd.DataFrame(columns=["item_id", "barcode", "amount", "quant", "share"])

    # Под IN-clause (аккуратно, через params)
    placeholders = ", ".join([f"%(id_{i})s" for i in range(len(item_ids))])
    params = {f"id_{i}": int(v) for i, v in enumerate(item_ids)}
    params.update({"start": start, "end": end})

    q = f"""
        select
            t.item_id as item_id,
            coalesce(b.barcode,'нет штрихкода') as barcode,
            sum(t.dt-t.cr) as amount,
            sum(t.quant_dt - t.quant_cr) as quant
        from sales_salesdata as t
        left join corporate_barcode as b on b.id = t.barcode_id
        where t.item_id in ({placeholders})
          and LAST_DAY(t.date) between %(start)s and %(end)s
        group by t.item_id, b.barcode
        order by t.item_id, b.barcode
    """

    df = pd.read_sql(q, engine, params=params)

    if df.empty:
        df["share"] = []
        return df

    # доля штрихкода внутри item_id по выручке
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    totals = df.groupby("item_id")["amount"].transform("sum")
    df["share"] = df["amount"] / totals.replace({0: 1})

    return df


def build_matrix_excel_bytes(
    engine,
    df_matrix: pd.DataFrame,
    start: str,
    end: str,
) -> bytes:
    """
    Делает XLSX в байтах:
      - Sheet 'Matrix' (матрица)
      - Sheet 'Barcodes' (детализация по штрихкодам по всем item_id из матрицы)
    """
    df_matrix_export = df_matrix.copy()

    # ВАЖНО: если у тебя в матрице столбец называется иначе — поправь тут
    if "item_id" not in df_matrix_export.columns:
        raise ValueError("В df_matrix нет колонки 'item_id' — экспорт штрихкодов невозможен.")

    item_ids = (
        df_matrix_export["item_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    df_barcodes = _fetch_barcodes_for_items(engine, item_ids=item_ids, start=start, end=end)

    # Для удобства добавим "Номенклатура" в лист Barcodes, если есть в матрице
    if "fullname" in df_matrix_export.columns:
        map_fullname = (
            df_matrix_export[["item_id", "fullname"]]
            .drop_duplicates()
            .set_index("item_id")["fullname"]
            .to_dict()
        )
        df_barcodes["fullname"] = df_barcodes["item_id"].map(map_fullname)

        # чуть красивее порядок колонок
        cols = ["item_id", "fullname", "barcode", "amount", "quant", "share"]
        df_barcodes = df_barcodes[[c for c in cols if c in df_barcodes.columns]]

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_matrix_export.to_excel(writer, index=False, sheet_name="Matrix")
        df_barcodes.to_excel(writer, index=False, sheet_name="Barcodes")

        # авто-ширина (простая, без фанатизма)
        for sheet_name, df in [("Matrix", df_matrix_export), ("Barcodes", df_barcodes)]:
            ws = writer.sheets[sheet_name]
            for i, col in enumerate(df.columns, start=1):
                max_len = max([len(str(col))] + [len(str(v)) for v in df[col].head(200).tolist()])
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(max_len + 2, 55)

    return output.getvalue()
