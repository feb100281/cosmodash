# planning_tab.py
# -*- coding: utf-8 -*-

from dash import html, dcc, Dash
import dash_mantine_components as dmc
from dash_iconify import DashIconify


def planning_layout() -> html.Div:
    """
    Разметка вкладки «Планирование арендного дохода».
    Только UI: инпуты, сценарии, плейсхолдеры. Без расчётной логики.
    """
    # ---------- ЛЕВАЯ КОЛОНКА: Assumptions ----------
    left_assumptions = dmc.Stack(
        gap="md",
        children=[
            dmc.Title("Параметры планирования", order=4),
            dmc.Text("Задайте год, ставки по типам и параметры вакантности.", size="sm", c="dimmed"),

            dmc.Select(
                id="pln_year_select",
                label="Год планирования",
                placeholder="Выберите год",
                data=[{"value": str(y), "label": str(y)} for y in range(2025, 2032)],
                value="2026",
                clearable=False,
                size="sm",
                leftSection=DashIconify(icon="mdi:calendar-range", width=18),
            ),

            dmc.Divider(label="Ставки (план), ₽/м²/мес (офис/склад) и ₽/место/мес (парковка)", labelPosition="left"),

            dmc.Grid(
                gutter="md",
                children=[
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="pln_rate_office",
                            label="Офисы",
                            value=1800,
                            min=0,
                            step=50,
                            size="sm",
                            thousandSeparator=" ",
                            leftSection=DashIconify(icon="mdi:office-building", width=16),
                        ), span=12
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="pln_rate_warehouse",
                            label="Склады",
                            value=900,
                            min=0,
                            step=25,
                            size="sm",
                            thousandSeparator=" ",
                            leftSection=DashIconify(icon="mdi:warehouse", width=16),
                        ), span=12
                    ),
                    dmc.GridCol(
                        dmc.NumberInput(
                            id="pln_rate_parking",
                            label="Парковка (за место)",
                            value=6000,
                            min=0,
                            step=100,
                            size="sm",
                            thousandSeparator=" ",
                            leftSection=DashIconify(icon="mdi:parking", width=16),
                        ), span=12
                    ),
                ],
            ),

            dmc.Divider(label="Вакантность и заселение", labelPosition="left"),

            dmc.SegmentedControl(
                id="pln_vacancy_scope",
                value="global",
                data=[
                    {"label": "Глобально", "value": "global"},
                    {"label": "По типам", "value": "by_type"},
                ],
                size="sm",
            ),

            # глобальный коэффициент вакантности
            dmc.Stack(
                id="pln_vacancy_global_block",
                gap=6,
                children=[
                    dmc.Text("Коэффициент вакантности (глобально), %", size="sm"),
                    dmc.Slider(
                        id="pln_vacancy_global",
                        value=10,
                        min=0,
                        max=50,
                        step=1,
                        marks=[{"value": v, "label": str(v)} for v in (0, 10, 20, 30, 40, 50)],
                        size="sm",
                    ),
                ],
            ),

            # по типам — скрытый блок, покажем через callback позже
            dmc.Stack(
                id="pln_vacancy_bytype_block",
                gap="sm",
                style={"display": "none"},
                children=[
                    dmc.Text("Коэффициент вакантности по типам, %", size="sm"),
                    dmc.NumberInput(id="pln_vac_office", label="Офисы", value=10, min=0, max=100, step=1, size="sm"),
                    dmc.NumberInput(id="pln_vac_warehouse", label="Склады", value=12, min=0, max=100, step=1, size="sm"),
                    dmc.NumberInput(id="pln_vac_parking", label="Парковка", value=5, min=0, max=100, step=1, size="sm"),
                ],
            ),

            dmc.Group(gap="md", grow=False, align="end", children=[
                dmc.NumberInput(
                    id="pln_lease_lag_months",
                    label="Лаг заселения (мес.)",
                    value=2, min=0, max=12, step=1, size="sm",
                    leftSection=DashIconify(icon="mdi:timer-sand", width=16),
                ),
                dmc.NumberInput(
                    id="pln_free_rent_months",
                    label="Free rent (мес.)",
                    value=0, min=0, max=12, step=1, size="sm",
                    leftSection=DashIconify(icon="mdi:gift-open", width=16),
                ),
            ]),

            dmc.Divider(label="Индексация", labelPosition="left"),
            dmc.Group(gap="md", grow=False, children=[
                dmc.NumberInput(
                    id="pln_indexation_pct",
                    label="Индексация, % годовых (по умолчанию)",
                    value=0, min=0, max=100, step=0.5, size="sm",
                    leftSection=DashIconify(icon="mdi:chart-line", width=16),
                ),
                dmc.Select(
                    id="pln_indexation_month",
                    label="Месяц индексации (по умолчанию)",
                    data=[{"value": str(m), "label": str(m)} for m in range(1, 13)],
                    value="1",
                    size="sm",
                ),
            ]),

            dmc.Divider(label="Формат отображения", labelPosition="left"),
            dmc.Switch(
                id="pln_vat_switch",
                label="Показывать суммы с НДС",
                checked=False,
                size="sm",
                onLabel="С НДС",
                offLabel="Без НДС",
            ),

            dmc.Divider(),

            # Сохранение/загрузка пресетов (только UI)
            dmc.Group(
                gap="sm",
                children=[
                    dmc.TextInput(
                        id="pln_preset_name",
                        placeholder="Название пресета",
                        size="sm",
                        leftSection=DashIconify(icon="mdi:content-save-edit-outline", width=16),
                    ),
                    dmc.Button("Сохранить пресет", id="pln_preset_save_btn", size="sm", variant="outline"),
                    dmc.Button("Загрузить пресет", id="pln_preset_load_btn", size="sm", variant="subtle"),
                ],
            ),
        ],
    )

    # ---------- ПРАВАЯ КОЛОНКА: Сценарии + Вывод ----------
    scenarios_block = dmc.Card(
        withBorder=True, radius="md", shadow="xs", padding="md",
        children=[
            dmc.Group(
                justify="space-between",
                children=[
                    dmc.Title("Сценарии", order=4),
                    dmc.SegmentedControl(
                        id="pln_scenario_mode",
                        value="single",
                        data=[
                            {"label": "Один", "value": "single"},
                            {"label": "Сравнение (Base/Best/Worst)", "value": "compare"},
                        ],
                        size="sm",
                    ),
                ],
            ),
            dmc.Space(h=10),

            # Панель пресетов Base / Best / Worst
            dmc.Group(
                id="pln_scenario_quickset",
                gap="sm",
                children=[
                    dmc.Button("Base", id="pln_btn_base", size="xs", variant="light"),
                    dmc.Button("Best", id="pln_btn_best", size="xs", variant="light"),
                    dmc.Button("Worst", id="pln_btn_worst", size="xs", variant="light"),
                ],
            ),

            dmc.Alert(
                "В режиме сравнения будут показаны дельты по доходу и графики по каждому сценарию.",
                title="Подсказка",
                color="blue", variant="light", mt="sm",
                icon=DashIconify(icon="mdi:information-outline", width=18),
            ),
        ],
    )

    summary_cards = dmc.SimpleGrid(
        id="pln_summary_cards",
        cols=3,
        spacing="md",
        children=[
            dmc.Card(
                withBorder=True, radius="md", shadow="xs",
                children=[
                    dmc.Text("Итого за год", size="sm", c="dimmed"),
                    dmc.Title(id="pln_kpi_total_year", order=3, children="—"),
                    dmc.Text(id="pln_kpi_total_year_note", size="xs", c="dimmed"),
                ],
            ),
            dmc.Card(
                withBorder=True, radius="md", shadow="xs",
                children=[
                    dmc.Text("Средняя загрузка", size="sm", c="dimmed"),
                    dmc.Title(id="pln_kpi_occupancy_avg", order=3, children="—"),
                    dmc.Text(id="pln_kpi_occupancy_note", size="xs", c="dimmed"),
                ],
            ),
            dmc.Card(
                withBorder=True, radius="md", shadow="xs",
                children=[
                    dmc.Text("Вакантная площадь (конец года)", size="sm", c="dimmed"),
                    dmc.Title(id="pln_kpi_vacant_eoy", order=3, children="—"),
                    dmc.Text(id="pln_kpi_vacant_note", size="xs", c="dimmed"),
                ],
            ),
        ],
    )

    chart_block = dmc.Card(
        withBorder=True, radius="md", shadow="xs", padding="md",
        children=[
            dmc.Group(
                justify="space-between",
                children=[
                    dmc.Title("Динамика по месяцам", order=4),
                    dmc.SegmentedControl(
                        id="pln_chart_filter_type",
                        value="all",
                        data=[
                            {"label": "Все типы", "value": "all"},
                            {"label": "Офисы", "value": "office"},
                            {"label": "Склады", "value": "warehouse"},
                            {"label": "Парковка", "value": "parking"},
                        ],
                        size="xs",
                    ),
                ],
            ),
            dmc.Space(h=10),
            dcc.Graph(id="pln_monthly_chart", figure={}),  # плейсхолдер
        ],
    )

    # Вкладки под графиком: сводка, завершающиеся, overrides, ассумпшены
    tables_tabs = dmc.Tabs(
        value="summary",
        variant="default",
        radius="md",
        children=[
            dmc.TabsList(
                children=[
                    dmc.TabsTab("Сводка по месяцам", value="summary",
                                leftSection=DashIconify(icon="mdi:table", width=16)),
                    dmc.TabsTab("Завершающиеся договоры", value="ending",
                                leftSection=DashIconify(icon="mdi:calendar-end", width=16)),
                    dmc.TabsTab("Overrides вакансий", value="overrides",
                                leftSection=DashIconify(icon="mdi:tune", width=16)),
                    dmc.TabsTab("Ассумпшены", value="assumptions",
                                leftSection=DashIconify(icon="mdi:clipboard-text-outline", width=16)),
                ],
                grow=True,
            ),

            dmc.TabsPanel(
                children=[
                    dmc.Space(h=10),
                    dmc.Alert("Здесь появится таблица помесячной сводки (тип/месяц/сумма).",
                              color="gray", variant="outline"),
                    html.Div(id="pln_table_monthly_summary"),
                    dmc.Group(
                        justify="flex-end",
                        children=[
                            dmc.Button("Экспорт в Excel", id="pln_export_excel_btn", variant="outline", size="xs"),
                            dmc.Button("Экспорт в PDF", id="pln_export_pdf_btn", variant="light", size="xs"),
                        ],
                    ),
                ],
                value="summary",
            ),

            dmc.TabsPanel(
                children=[
                    dmc.Space(h=10),
                    dmc.Alert("Здесь появится таблица договоров, завершающихся в выбранном году.",
                              color="gray", variant="outline"),
                    html.Div(id="pln_table_ending_contracts"),
                ],
                value="ending",
            ),

            dmc.TabsPanel(
                children=[
                    dmc.Space(h=10),
                    dmc.Alert("Здесь можно будет задать индивидуальные ставки/лаги по конкретным вакансиям.",
                              color="gray", variant="outline"),
                    html.Div(id="pln_table_overrides"),
                ],
                value="overrides",
            ),

            dmc.TabsPanel(
                children=[
                    dmc.Space(h=10),
                    dmc.Alert("Снимок всех предпосылок для отчёта/аудита.", color="gray", variant="outline"),
                    html.Div(id="pln_assumptions_snapshot"),
                ],
                value="assumptions",
            ),
        ],
    )

    right_content = dmc.Stack(
        gap="md",
        children=[
            scenarios_block,
            summary_cards,
            chart_block,
            tables_tabs,
        ],
    )

    # ---------- СБОРКА СТРАНИЦЫ ----------
    page = dmc.Container(
        fluid=True,
        px="md",
        children=[
            dmc.Title("Планирование арендного дохода", order=3),
            dmc.Space(h=10),
            dmc.Grid(
                gutter="lg",
                align="stretch",
                children=[
                    dmc.GridCol(left_assumptions, span=12,  ),
                    dmc.GridCol(right_content, span=12, ),
                ],
            ),
            # внутренние служебные хранилища данных (для будущих callback-ов)
            dcc.Store(id="pln_store_presets"),
            dcc.Store(id="pln_store_monthly_result"),
            dcc.Store(id="pln_store_scenarios_compare"),
        ],
    )

    return page


# Удобный контейнер, если хочешь просто импортировать переменную:
planning_container = planning_layout()

app = Dash(__name__)
app.layout = dmc.MantineProvider(planning_container)

if __name__ == "__main__":
    app.run(debug=True)

