# matrix/empty_state.py
import dash_mantine_components as dmc
from dash_iconify import DashIconify


def render_matrix_empty_state() -> dmc.Container:
    return dmc.Container(
        fluid=True,
        children=[
            dmc.Center(
                style={"minHeight": 520},
                children=dmc.Stack(
                    gap="md",
                    align="center",
                    style={"maxWidth": 720, "textAlign": "center"},
                    children=[
                        dmc.ThemeIcon(
                            size=72,
                            radius="xl",
                            variant="light",
                            children=DashIconify(icon="mynaui:rocket-solid", width=34),
                        ),
                        dmc.Title("Матрица ещё не рассчитана", order=3),
                        dmc.Text(
                            "Выберите параметры слева и нажмите «Рассчитать». "
                            "После расчёта справа появится таблица с ABC/XYZ, статистикой и расчётом ROP/SS.",
                            c="dimmed",
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
                                        dmc.Text("Выберите период на ползунке сверху", size="sm"),
                                    ],
                                ),
                                dmc.Group(
                                    gap="sm",
                                    align="center",
                                    wrap="nowrap",
                                    children=[
                                        dmc.Badge("2", variant="light", radius="xl"),
                                        dmc.Text("Настройте пороги ABC / XYZ и параметры ROP/SS*", size="sm"),
                                    ],
                                ),
                                dmc.Group(
                                    gap="sm",
                                    align="center",
                                    wrap="nowrap",
                                    children=[
                                        dmc.Badge("3", variant="light", radius="xl"),
                                        dmc.Text(
                                            "При необходимости задайте фильтр по группам/категориям",
                                            size="sm",
                                        ),
                                    ],
                                ),
                                dmc.Group(
                                    gap="sm",
                                    align="center",
                                    wrap="nowrap",
                                    children=[
                                        dmc.Badge("4", variant="light", radius="xl"),
                                        dmc.Text("Нажмите «Рассчитать» — и получите таблицу", size="sm"),
                                    ],
                                ),
                            ],
                        ),

                        # Сноска-объяснение (мелко, серым, под чертой)
                        dmc.Stack(
                            gap=6,
                            style={"width": "100%"},
                            children=[
                                dmc.Divider(variant="dashed"),
                                dmc.Text(
                                    "* ABC — ранжирование по вкладу в выручку; XYZ — оценка стабильности спроса. "
                                    "ROP/SS — расчёт точки заказа и страхового запаса, чтобы снизить риск дефицита "
                                    "при колебаниях спроса и сроках поставки.",
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
                                "Вы можете кликнуть по строке таблицы — откроется детализация по штрихкодам.",
                                size="sm",
                            ),
                        ),
                    ],
                ),
            )
        ],
    )

