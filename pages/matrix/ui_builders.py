# matrix/ui_builders.py
from __future__ import annotations

from typing import Tuple

import dash_mantine_components as dmc
from dash import dcc


def build_help(
    *,
    open_btn_id: str,
    modal_id: str,
    icon_text: str,
    legend_text: str,
    modal_title: str,
    markdown_text: str,
    icon_variant: str = "subtle",
) -> Tuple[dmc.Group, dmc.Modal]:
    """
    Возвращает (legend, modal) для Fieldset:
    - legend: dmc.Group с ActionIcon + подписью
    - modal: dmc.Modal с Markdown-текстом
    """
    open_btn = dmc.ActionIcon(
        dmc.Text(icon_text, fw=700),
        id=open_btn_id,
        variant=icon_variant,
        style={"cursor": "pointer"},
    )

    legend = dmc.Group(
        [open_btn, dmc.Text(legend_text, size="sm")],
        gap=6,
        wrap="nowrap",
    )

    modal = dmc.Modal(
        id=modal_id,
        title=modal_title,
        children=dcc.Markdown(markdown_text, className="markdown-25"),
        opened=False,
        size="lg",
        centered=True,
    )

    return legend, modal
