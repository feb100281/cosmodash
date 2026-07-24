# pages/matrix/stock_export.py

from __future__ import annotations

import json
import re
from io import BytesIO
from typing import Any, Iterable

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


UNALLOCATED_LABEL = "НЕ РАСПРЕДЕЛЕНО В ИСТОЧНИКЕ"
EPS = 1e-9


SUMMARY_COLUMNS = [
    ("cat_name", "Категория"),
    ("sc_name", "Подкатегория"),
    ("manu", "Производитель"),
    ("fullname", "Номенклатура"),
    ("article", "Артикул"),
    ("barcode", "Штрихкоды товара"),
    ("stock_available", "Доступный остаток"),
    ("stock_ordered", "Заказано"),
    ("stock_total", "Всего с заказами"),
    ("stock_status", "Статус запаса"),
    ("barcode_stocks_display", "Остатки по штрихкодам"),
]


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _to_float(value: Any) -> float:
    value = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).fillna(0).iloc[0]
    return float(value)


def _parse_qty_list(value: Any) -> dict[str, float]:
    """
    Разбирает детализацию остатков вида:

        ["Европарк - 2 шт.", "ОСНОВНОЙ склад - 1364 шт."]

    или:

        ["2000000001944 - 23 шт.", "2000000001944 - 984 шт."]

    Повторяющиеся склады / штрихкоды суммируются.
    """
    if _is_empty(value):
        return {}

    raw_values = None

    if isinstance(value, (list, tuple)):
        raw_values = list(value)
    else:
        text = str(value).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                raw_values = parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            raw_values = None

    result: dict[str, float] = {}

    if raw_values is not None:
        for item in raw_values:
            match = re.match(
                r"^\s*(.+?)\s*-\s*(-?\d+(?:[.,]\d+)?)\s*шт\.?\s*$",
                str(item),
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            name = match.group(1).strip().strip("'\"").strip()
            if not name:
                continue

            try:
                qty = float(match.group(2).replace(",", "."))
            except (TypeError, ValueError):
                continue

            result[name] = result.get(name, 0.0) + qty

        return result

    matches = re.findall(
        r"""["']?([^"'\[\],]+?)["']?\s*-\s*
            (-?\d+(?:[.,]\d+)?)\s*шт\.?""",
        str(value),
        flags=re.IGNORECASE | re.VERBOSE,
    )

    for name, qty in matches:
        name = str(name).strip().strip("'\"").strip()
        if not name:
            continue

        try:
            qty_value = float(str(qty).replace(",", "."))
        except (TypeError, ValueError):
            continue

        result[name] = result.get(name, 0.0) + qty_value

    return result


def _dynamic_columns(
    columns: Iterable[str],
    prefix: str,
) -> list[str]:
    return sorted(
        [
            str(column)
            for column in columns
            if str(column).startswith(prefix)
        ],
        key=lambda value: value.split("::", 1)[-1].lower(),
    )


def _base_row(row: pd.Series) -> dict[str, Any]:
    return {
        "Категория": row.get("cat_name", ""),
        "Подкатегория": row.get("sc_name", ""),
        "Производитель": row.get("manu", ""),
        "Номенклатура": row.get("fullname", ""),
        "Артикул": row.get("article", ""),
    }


def _prepare_summary_df(df_matrix: pd.DataFrame) -> pd.DataFrame:
    if df_matrix is None or df_matrix.empty:
        return pd.DataFrame(
            columns=[title for _, title in SUMMARY_COLUMNS]
        )

    df = df_matrix.copy()

    warehouse_columns = _dynamic_columns(df.columns, "stock_wh::")
    ordered_warehouse_columns = _dynamic_columns(df.columns, "ordered_wh::")

    base_columns = [
        source
        for source, _ in SUMMARY_COLUMNS
        if source in df.columns
    ]

    result = df[
        base_columns
        + warehouse_columns
        + ordered_warehouse_columns
    ].copy()

    rename_map = {
        source: title
        for source, title in SUMMARY_COLUMNS
        if source in result.columns
    }

    for column in warehouse_columns:
        warehouse = column.split("::", 1)[1]
        rename_map[column] = f"Магазин | {warehouse}"

    for column in ordered_warehouse_columns:
        warehouse = column.split("::", 1)[1]
        rename_map[column] = f"Заказ | {warehouse}"

    return result.rename(columns=rename_map)


def _warehouse_breakdown(
    row: pd.Series,
    warehouse_columns: list[str],
) -> tuple[list[tuple[str, float]], float]:
    values: list[tuple[str, float]] = []
    total = 0.0

    for column in warehouse_columns:
        qty = _to_float(row.get(column, 0))

        if abs(qty) <= EPS:
            continue

        warehouse = column.split("::", 1)[1]
        values.append((warehouse, qty))
        total += qty

    return values, total


def _barcode_breakdown(
    row: pd.Series,
) -> tuple[list[tuple[str, float]], float]:
    parsed = _parse_qty_list(
        row.get("barcode_stocks")
    )

    values: list[tuple[str, float]] = []
    total = 0.0

    for barcode, qty in sorted(
        parsed.items(),
        key=lambda item: str(item[0]),
    ):
        qty = float(qty)

        if abs(qty) <= EPS:
            continue

        values.append((str(barcode), qty))
        total += qty

    return values, total


def _prepare_warehouse_df(
    df_matrix: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "Категория",
        "Подкатегория",
        "Производитель",
        "Номенклатура",
        "Артикул",
        "Магазин",
        "Остаток",
    ]

    if df_matrix is None or df_matrix.empty:
        return pd.DataFrame(columns=columns)

    warehouse_columns = _dynamic_columns(
        df_matrix.columns,
        "stock_wh::",
    )

    rows: list[dict[str, Any]] = []

    for _, row in df_matrix.iterrows():
        base = _base_row(row)

        available = _to_float(
            row.get("stock_available", 0)
        )

        breakdown, detailed_total = _warehouse_breakdown(
            row,
            warehouse_columns,
        )

        for warehouse, qty in breakdown:
            rows.append(
                {
                    **base,
                    "Магазин": warehouse,
                    "Остаток": qty,
                }
            )

        difference = available - detailed_total

        if abs(difference) > EPS:
            rows.append(
                {
                    **base,
                    "Магазин": UNALLOCATED_LABEL,
                    "Остаток": difference,
                }
            )

    return pd.DataFrame(rows, columns=columns)


def _prepare_barcode_df(
    df_matrix: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "Категория",
        "Подкатегория",
        "Производитель",
        "Номенклатура",
        "Артикул",
        "Штрихкод",
        "Остаток",
    ]

    if df_matrix is None or df_matrix.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []

    for _, row in df_matrix.iterrows():
        base = _base_row(row)

        available = _to_float(
            row.get("stock_available", 0)
        )

        breakdown, detailed_total = _barcode_breakdown(row)

        for barcode, qty in breakdown:
            rows.append(
                {
                    **base,
                    "Штрихкод": barcode,
                    "Остаток": qty,
                }
            )

        difference = available - detailed_total

        if abs(difference) > EPS:
            rows.append(
                {
                    **base,
                    "Штрихкод": UNALLOCATED_LABEL,
                    "Остаток": difference,
                }
            )

    return pd.DataFrame(rows, columns=columns)


def _prepare_control_df(
    df_matrix: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "Категория",
        "Подкатегория",
        "Производитель",
        "Номенклатура",
        "Артикул",
        "Доступный остаток",
        "По магазинам",
        "Расхождение магазины",
        "По штрихкодам",
        "Расхождение штрихкоды",
    ]

    if df_matrix is None or df_matrix.empty:
        return pd.DataFrame(columns=columns)

    warehouse_columns = _dynamic_columns(
        df_matrix.columns,
        "stock_wh::",
    )

    rows: list[dict[str, Any]] = []

    for _, row in df_matrix.iterrows():
        available = _to_float(
            row.get("stock_available", 0)
        )

        _, warehouse_total = _warehouse_breakdown(
            row,
            warehouse_columns,
        )

        _, barcode_total = _barcode_breakdown(row)

        rows.append(
            {
                **_base_row(row),
                "Доступный остаток": available,
                "По магазинам": warehouse_total,
                "Расхождение магазины": (
                    available - warehouse_total
                ),
                "По штрихкодам": barcode_total,
                "Расхождение штрихкоды": (
                    available - barcode_total
                ),
            }
        )

    result = pd.DataFrame(rows, columns=columns)

    result["_problem_size"] = (
        result["Расхождение магазины"].abs()
        + result["Расхождение штрихкоды"].abs()
    )

    return (
        result
        .sort_values(
            "_problem_size",
            ascending=False,
        )
        .drop(columns="_problem_size")
        .reset_index(drop=True)
    )


def _style_sheet(
    ws,
    *,
    title: str,
    stock_date: str | None = None,
) -> None:
    """
    Единый строгий стиль для всех листов выгрузки остатков.

    Что делаем:
    - строка 1: общий заголовок листа;
    - строка 2: шапка таблицы;
    - freeze F3: фиксируем строки 1-2 и первые 5 колонок;
    - тонкая светло-серая сетка по всем ячейкам;
    - лёгкая зебра по строкам данных;
    - автофильтр;
    - без стандартной Excel-сетки;
    - числовые колонки выравниваем вправо;
    - текстовые ключевые колонки — влево;
    - строки "НЕ РАСПРЕДЕЛЕНО В ИСТОЧНИКЕ" выделяем мягким предупреждением;
    - расхождения на листе "Контроль" визуально выделяем.
    """
    max_col = max(ws.max_column, 1)
    max_row = max(ws.max_row, 2)

    # ------------------------------------------------------------------
    # Палитра
    # ------------------------------------------------------------------
    title_fill = PatternFill(
        fill_type="solid",
        fgColor="1F2937",
    )

    header_fill = PatternFill(
        fill_type="solid",
        fgColor="E9EEF3",
    )

    zebra_fill = PatternFill(
        fill_type="solid",
        fgColor="F8FAFC",
    )

    warning_fill = PatternFill(
        fill_type="solid",
        fgColor="FFF7E6",
    )

    warning_font = Font(
        name="Helvetica Light",
        size=10,
        color="9A6700",
        bold=True,
    )

    error_fill = PatternFill(
        fill_type="solid",
        fgColor="FDECEC",
    )

    error_font = Font(
        name="Helvetica Light",
        size=10,
        color="B42318",
        bold=True,
    )

    grid_side = Side(
        style="thin",
        color="D9DEE5",
    )

    cell_border = Border(
        left=grid_side,
        right=grid_side,
        top=grid_side,
        bottom=grid_side,
    )

    # ------------------------------------------------------------------
    # 1. Общий заголовок листа
    # ------------------------------------------------------------------
    ws.merge_cells(
        start_row=1,
        start_column=1,
        end_row=1,
        end_column=max_col,
    )

    title_cell = ws.cell(
        row=1,
        column=1,
    )

    if stock_date:
        title_cell.value = f"{title} — на {stock_date}"
    else:
        title_cell.value = title

    title_cell.font = Font(
        name="Helvetica Light",
        size=12,
        bold=True,
        color="FFFFFF",
    )

    title_cell.fill = title_fill

    title_cell.alignment = Alignment(
        horizontal="left",
        vertical="center",
    )

    title_cell.border = cell_border

    ws.row_dimensions[1].height = 26

    # Чтобы merged-заголовок визуально имел единый контур/заливку.
    for col_idx in range(1, max_col + 1):
        cell = ws.cell(
            row=1,
            column=col_idx,
        )
        cell.fill = title_fill
        cell.border = cell_border

    # ------------------------------------------------------------------
    # 2. Шапка таблицы
    # ------------------------------------------------------------------
    for cell in ws[2]:
        cell.font = Font(
            name="Helvetica Light",
            size=10,
            bold=True,
            color="1F2937",
        )

        cell.fill = header_fill

        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

        cell.border = cell_border

    ws.row_dimensions[2].height = 36

    # ------------------------------------------------------------------
    # 3. Основные строки + зебра + тонкая сетка
    # ------------------------------------------------------------------
    for row_idx in range(3, max_row + 1):
        use_zebra = (row_idx - 3) % 2 == 1

        for col_idx in range(1, max_col + 1):
            cell = ws.cell(
                row=row_idx,
                column=col_idx,
            )

            cell.font = Font(
                name="Helvetica Light",
                size=10,
                color="1F2937",
            )

            cell.alignment = Alignment(
                vertical="center",
            )

            cell.border = cell_border

            if use_zebra:
                cell.fill = zebra_fill

        ws.row_dimensions[row_idx].height = 20

    # ------------------------------------------------------------------
    # 4. Словарь заголовков -> номер колонки
    # ------------------------------------------------------------------
    header_to_col = {
        str(
            ws.cell(
                row=2,
                column=col,
            ).value
            or ""
        ): col
        for col in range(
            1,
            max_col + 1,
        )
    }

    # ------------------------------------------------------------------
    # 5. Ширины колонок
    # ------------------------------------------------------------------
    widths = {
        "Категория": 22,
        "Подкатегория": 24,
        "Производитель": 22,
        "Номенклатура": 48,
        "Артикул": 18,
        "Штрихкоды товара": 28,
        "Доступный остаток": 18,
        "Заказано": 14,
        "Всего с заказами": 18,
        "Статус запаса": 28,
        "Остатки по штрихкодам": 44,
        "Магазин": 34,
        "Штрихкод": 30,
        "Остаток": 14,
        "По магазинам": 16,
        "Расхождение магазины": 21,
        "По штрихкодам": 17,
        "Расхождение штрихкоды": 22,
    }

    for header, width in widths.items():
        col = header_to_col.get(
            header
        )

        if col:
            ws.column_dimensions[
                get_column_letter(col)
            ].width = width

    # Динамические колонки магазинов / заказов.
    for col in range(
        1,
        max_col + 1,
    ):
        header = str(
            ws.cell(
                row=2,
                column=col,
            ).value
            or ""
        )

        if (
            header.startswith("Магазин | ")
            or header.startswith("Заказ | ")
        ):
            ws.column_dimensions[
                get_column_letter(col)
            ].width = 21

    # ------------------------------------------------------------------
    # 6. Числовые форматы
    # ------------------------------------------------------------------
    numeric_headers = {
        "Доступный остаток",
        "Заказано",
        "Всего с заказами",
        "Остаток",
        "По магазинам",
        "Расхождение магазины",
        "По штрихкодам",
        "Расхождение штрихкоды",
    }

    for col in range(
        1,
        max_col + 1,
    ):
        header = str(
            ws.cell(
                row=2,
                column=col,
            ).value
            or ""
        )

        is_qty = (
            header in numeric_headers
            or header.startswith("Магазин | ")
            or header.startswith("Заказ | ")
        )

        if is_qty:
            for row_idx in range(
                3,
                max_row + 1,
            ):
                cell = ws.cell(
                    row=row_idx,
                    column=col,
                )

                # Остатки у тебя фактически целые единицы.
                cell.number_format = '#,##0'

                cell.alignment = Alignment(
                    horizontal="right",
                    vertical="center",
                )

    # ------------------------------------------------------------------
    # 7. Текстовые колонки
    # ------------------------------------------------------------------
    left_aligned_headers = {
        "Категория",
        "Подкатегория",
        "Производитель",
        "Номенклатура",
        "Артикул",
        "Штрихкоды товара",
        "Статус запаса",
        "Остатки по штрихкодам",
        "Магазин",
        "Штрихкод",
    }

    for header in left_aligned_headers:
        col = header_to_col.get(
            header
        )

        if not col:
            continue

        for row_idx in range(
            3,
            max_row + 1,
        ):
            ws.cell(
                row=row_idx,
                column=col,
            ).alignment = Alignment(
                horizontal="left",
                vertical="center",
                wrap_text=(
                    header
                    in {
                        "Номенклатура",
                        "Остатки по штрихкодам",
                    }
                ),
            )

    # ------------------------------------------------------------------
    # 8. Выделение нераспределённых остатков
    # ------------------------------------------------------------------
    for row_idx in range(
        3,
        max_row + 1,
    ):
        row_values = [
            str(
                ws.cell(
                    row=row_idx,
                    column=col_idx,
                ).value
                or ""
            )
            for col_idx in range(
                1,
                max_col + 1,
            )
        ]

        has_unallocated = any(
            UNALLOCATED_LABEL in value
            for value in row_values
        )

        if has_unallocated:
            for col_idx in range(
                1,
                max_col + 1,
            ):
                cell = ws.cell(
                    row=row_idx,
                    column=col_idx,
                )
                cell.fill = warning_fill
                cell.font = warning_font
                cell.border = cell_border

    # ------------------------------------------------------------------
    # 9. Лист "Контроль": выделяем реальные расхождения
    # ------------------------------------------------------------------
    difference_headers = {
        "Расхождение магазины",
        "Расхождение штрихкоды",
    }

    for header in difference_headers:
        col = header_to_col.get(
            header
        )

        if not col:
            continue

        for row_idx in range(
            3,
            max_row + 1,
        ):
            cell = ws.cell(
                row=row_idx,
                column=col,
            )

            try:
                value = float(
                    cell.value or 0
                )
            except (
                TypeError,
                ValueError,
            ):
                value = 0.0

            if abs(value) > EPS:
                cell.fill = error_fill
                cell.font = error_font

    # ------------------------------------------------------------------
    # 10. Freeze / filter / view
    # ------------------------------------------------------------------

    # F3:
    # - фиксируем строки 1-2;
    # - фиксируем A:E:
    #   Категория / Подкатегория / Производитель /
    #   Номенклатура / Артикул.
    ws.freeze_panes = "F3"

    ws.auto_filter.ref = (
        f"A2:"
        f"{get_column_letter(max_col)}"
        f"{max_row}"
    )

    # Убираем стандартную сетку Excel:
    # вместо неё используется наша тонкая светло-серая сетка.
    ws.sheet_view.showGridLines = False

    # Масштаб чуть комфортнее для широких таблиц.
    ws.sheet_view.zoomScale = 90

    # Активная ячейка после открытия.
    ws.sheet_view.selection[0].activeCell = "F3"
    ws.sheet_view.selection[0].sqref = "F3"


def _get_stock_date_label(
    df_matrix: pd.DataFrame,
) -> str | None:
    """
    Возвращает дату остатков для заголовка Excel.

    Если в выгрузке одна дата — показываем:
        24.07.2026

    Если по каким-то причинам дат несколько —
    показываем диапазон:
        23.07.2026–24.07.2026
    """
    if (
        df_matrix is None
        or df_matrix.empty
        or "stock_date" not in df_matrix.columns
    ):
        return None

    dates = (
        pd.to_datetime(
            df_matrix["stock_date"],
            errors="coerce",
        )
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )

    if dates.empty:
        return None

    first_date = dates.iloc[0].strftime("%d.%m.%Y")
    last_date = dates.iloc[-1].strftime("%d.%m.%Y")

    if first_date == last_date:
        return first_date

    return f"{first_date}–{last_date}"


def build_stock_excel_bytes(
    df_matrix: pd.DataFrame,
) -> bytes:
    """
    Формирует Excel с тремя листами:

    - Остатки
    - По магазинам
    - По штрихкодам

    Дата остатков выводится в верхнем заголовке каждого листа,
    поэтому отдельная колонка "Дата остатков" не нужна.

    На листах детализации остаток, которого нет в детализации
    источника, выводится отдельной строкой
    "НЕ РАСПРЕДЕЛЕНО В ИСТОЧНИКЕ".

    Поэтому сумма листа всегда совпадает с stock_available.
    """
    stock_date = _get_stock_date_label(
        df_matrix
    )

    summary_df = _prepare_summary_df(
        df_matrix
    )

    warehouse_df = _prepare_warehouse_df(
        df_matrix
    )

    barcode_df = _prepare_barcode_df(
        df_matrix
    )

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:
        summary_df.to_excel(
            writer,
            sheet_name="Остатки",
            index=False,
            startrow=1,
        )

        warehouse_df.to_excel(
            writer,
            sheet_name="По магазинам",
            index=False,
            startrow=1,
        )

        barcode_df.to_excel(
            writer,
            sheet_name="По штрихкодам",
            index=False,
            startrow=1,
        )

        _style_sheet(
            writer.book["Остатки"],
            title="Текущие остатки ассортиментной матрицы",
            stock_date=stock_date,
        )

        _style_sheet(
            writer.book["По магазинам"],
            title="Остатки по магазинам",
            stock_date=stock_date,
        )

        _style_sheet(
            writer.book["По штрихкодам"],
            title="Остатки по штрихкодам",
            stock_date=stock_date,
        )

    output.seek(0)

    return output.getvalue()
