# planning/empty_state.py
import dash_mantine_components as dmc
from dash_iconify import DashIconify


def render_planning_empty_state() -> dmc.Container:
    return dmc.Container(
        fluid=True,
        children=[
            dmc.Center(
                style={"minHeight": 520},
                children=dmc.Stack(
                    gap="md",
                    align="center",
                    style={"maxWidth": 760, "textAlign": "center"},
                    children=[
                        dmc.ThemeIcon(
                            size=72,
                            radius="xl",
                            variant="light",
                            children=DashIconify(icon="mdi:calendar-month-outline", width=34),
                        ),
                        dmc.Title("План ещё не рассчитан", order=3),
                        dmc.Text(
                            "Выберите дату горизонта планирования — после этого появятся графики, таблица и метрика точности.",
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
                                        dmc.Text("Укажите дату горизонта планирования", size="sm"),
                                    ],
                                ),
                                dmc.Group(
                                    gap="sm",
                                    align="center",
                                    wrap="nowrap",
                                    children=[
                                        dmc.Badge("2", variant="light", radius="xl"),
                                        dmc.Text("При необходимости задайте «Текущую дату» и «Исторические данные»", size="sm"),
                                    ],
                                ),
                                dmc.Group(
                                    gap="sm",
                                    align="center",
                                    wrap="nowrap",
                                    children=[
                                        dmc.Badge("3", variant="light", radius="xl"),
                                        dmc.Text("Настройте сезонности и параметры тренда", size="sm"),
                                    ],
                                ),
                                dmc.Group(
                                    gap="sm",
                                    align="center",
                                    wrap="nowrap",
                                    children=[
                                        dmc.Badge("4", variant="light", radius="xl"),
                                        dmc.Text("Проверьте MAPE и сравнение План/Факт", size="sm"),
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
                                    "MAPE — метрика точности прогноза: чем ниже значение, тем ближе план к факту. "
                                    "Сезонность и чувствительность тренда помогают точнее учитывать циклы продаж и изменения спроса.",
                                    size="xs",
                                    c="dimmed",
                                    style={"lineHeight": 1.35},
                                ),
                            ],
                        ),

                        dmc.Space(h=8),

                        # Подсказка
                        dmc.Alert(
                            title="Подсказка",
                            icon=DashIconify(icon="tabler:info-circle", width=18),
                            variant="light",
                            children=dmc.Text(
                                "Если прогноз выглядит «рваным» — уменьшите чувствительность к изменениям тренда "
                                "(changepoint_prior_scale) или сократите число изломов.",
                                size="sm",
                            ),
                        ),
                    ],
                ),
            )
        ],
    )
