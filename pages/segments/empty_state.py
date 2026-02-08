# segments/empty_state.py
import dash_mantine_components as dmc
from dash_iconify import DashIconify


def render_segments_empty_state() -> dmc.Container:
    return dmc.Container(
        fluid=True,
        children=[
            dmc.Center(
                style={"minHeight": 560},
                children=dmc.Stack(
                    gap="md",
                    align="center",
                    style={"maxWidth": 760, "textAlign": "center"},
                    children=[
                        dmc.ThemeIcon(
                            size=72,
                            radius="xl",
                            variant="light",
                            children=DashIconify(icon="mdi:layers-search", width=34),
                        ),
                        dmc.Title("Сегментный анализ ещё не сформирован", order=3),
                        dmc.Text(
                            "Выберите период сверху, затем отметьте нужные группы, бренды или производителей слева — "
                            "после выбора появится аналитика справа.",
                            size="sm",
                            c="dimmed",
                            style={"lineHeight": 1.4},
                        ),

                        dmc.Divider(w="100%"),

                        # Шаги
                        dmc.Stack(
                            gap="sm",
                            align="flex-start",
                            style={"width": "100%"},
                            children=[
                                dmc.Group(
                                    gap="sm",
                                    align="center",
                                    wrap="nowrap",
                                    children=[
                                        dmc.Badge("1", variant="light", radius="xl"),
                                        dmc.Text("Выберите период на слайдере сверху", size="sm"),
                                    ],
                                ),
                                dmc.Group(
                                    gap="sm",
                                    align="center",
                                    wrap="nowrap",
                                    children=[
                                        dmc.Badge("2", variant="light", radius="xl"),
                                        dmc.Text("Выберите тип группировки: Группа / Бренд / Производитель", size="sm"),
                                    ],
                                ),
                                dmc.Group(
                                    gap="sm",
                                    align="center",
                                    wrap="nowrap",
                                    children=[
                                        dmc.Badge("3", variant="light", radius="xl"),
                                        dmc.Text("При необходимости используйте поиск по номенклатуре", size="sm"),
                                    ],
                                ),
                                dmc.Group(
                                    gap="sm",
                                    align="center",
                                    wrap="nowrap",
                                    children=[
                                        dmc.Badge("4", variant="light", radius="xl"),
                                        dmc.Text("Отметьте позиции в дереве — и получите аналитику справа", size="sm"),
                                    ],
                                ),
                            ],
                        ),

                        # Сноска
                        dmc.Stack(
                            gap=6,
                            style={"width": "100%"},
                            children=[
                                dmc.Divider(variant="dashed"),
                                dmc.Text(
                                    "Подсказка: можно выбирать как верхние узлы (группы/категории), так и конкретные SKU. "
                                    "Данные пересчитываются только после выбора позиций.",
                                    size="xs",
                                    c="dimmed",
                                    style={"lineHeight": 1.35},
                                ),
                            ],
                        ),

                        dmc.Space(h=8),

                        # Подсказка-алерт
                        # dmc.Alert(
                        #     title="Подсказка",
                        #     icon=DashIconify(icon="tabler:info-circle", width=18),
                        #     variant="light",
                        #     children=dmc.Text(
                        #         "Клик по строке в таблице «Выбранные позиции» откроет детальную карточку по товару.",
                        #         size="sm",
                        #     ),
                        # ),
                    ],
                ),
            )
        ],
    )
