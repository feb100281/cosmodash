import pandas as pd
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import json
import dash_ag_grid as dag

from dash import dcc, Input, Output, State, no_update

from .forecast import SEASONS_OPTIONS, forecast
from components import NoData

import locale

locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")


class PlaningPage:
    def __init__(self):

        self.horizon_date_picker_id = "horizon_date_picker_id"
        self.init_date_picker_id = "forecast_init_date_picker_id"
        self.hustorical_cut_date_picker_id = "forecast_current_date_picker_id"
        self.resultes_container_id = "forecast_detailed_container_id"

        self.yearly_season_checkbox_id = "yearly_season_checkbox_id"
        self.weekly_season_checkbox_id = "weekly_season_checkbox_id"

        self.seasons_type_select_id = "seasons_type_select_id"

        self.changepoint_prior_scale_id = "changepoint_prior_scale_id"
        self.changepoint_range_id = "changepoint_range_id"
        self.n_changepoints_id = "n_changepoints_id"
        
        self.empty_hint = dmc.Alert(
            title="Нужно выбрать параметры",
            children="Выберите горизонт планирования (дату) — после этого появится прогноз.",
            color="teal",
            variant="light",
            radius="md",
            withCloseButton=False,
            icon=DashIconify(icon="mdi:calendar-check", width=18),
        )


        self.dates_fieldsets = dmc.Fieldset(
            [
                dmc.DatesProvider(
                    [
                        dmc.DateInput(
                            placeholder="Введите дату",
                            label="Дата горизонта планирования",
                            variant="default",
                            size="sm",
                            radius="sm",
                            withAsterisk=True,
                            disabled=False,
                            clearable=True,
                            valueFormat="D MMMM YYYY (dddd)",
                            id=self.horizon_date_picker_id,
                        ),
                        dmc.DateInput(
                            placeholder="Введите дату",
                            label="Текущая дата",
                            variant="default",
                            size="sm",
                            radius="sm",
                            withAsterisk=False,
                            disabled=False,
                            clearable=True,
                            valueFormat="D MMMM YYYY (dddd)",
                            id=self.init_date_picker_id,
                        ),
                        dmc.DateInput(
                            placeholder="Введите дату начала",
                            label="Исторические данные",
                            variant="default",
                            size="sm",
                            radius="sm",
                            withAsterisk=False,
                            disabled=False,
                            clearable=True,
                            valueFormat="D MMMM YYYY (dddd)",
                            id=self.hustorical_cut_date_picker_id,
                        ),
                    ],
                    settings={"locale": "ru"},
                ),
            ],
            legend="Даты и горизонт планирования",
        )

        self.seasons_fieldset = dmc.Fieldset(
            [
                dmc.Stack(
                    [
                        dmc.Checkbox(
                            "Годовая сезонность",
                            id=self.yearly_season_checkbox_id,
                            checked=True,
                        ),
                        dmc.Checkbox(
                            "Недельная сезонность",
                            id=self.weekly_season_checkbox_id,
                            checked=True,
                        ),
                        dmc.Select(
                            data=SEASONS_OPTIONS,
                            value="additive",
                            id=self.seasons_type_select_id,
                        ),
                    ]
                )
            ],
            legend="Сезонность",
        )

        self.trend_fieldset = dmc.Fieldset(
            [
                dmc.Stack(
                    [
                        dmc.NumberInput(
                            id=self.n_changepoints_id,
                            label="Максимальное кол-во изломов",
                            value=25,
                            max=100,
                            min=1,
                            allowDecimal=False,
                        ),
                        dmc.NumberInput(
                            id=self.changepoint_prior_scale_id,
                            label="Чувствительность к изменениям",
                            value=0.05,
                            max=0.9,
                            min=0.001,
                            allowDecimal=True,
                            step=0.001,
                        ),
                        dmc.NumberInput(
                            id=self.changepoint_range_id,
                            label="Исторические данные",
                            value=1.0,
                            max=1,
                            min=0.1,
                            allowDecimal=True,
                            step=0.1,
                        ),
                    ]
                )
            ],
            legend="Параметры тренда",
        )

        self.planning_memo = """
# Планирование продаж (инструкция)

Прогноз продаж формируется на основе исторических данных с использованием библиотеки **Facebook Prophet** — современной модели анализа временных рядов, признанной отраслевым стандартом для бюджетного и операционного планирования.

## Создание базовых планов

Чтобы построить прогноз продаж, достаточно указать **горизонт расчёта бюджета** — модель автоматически рассчитает план выручки от текущей даты до конца выбранного периода.

## Настройка начальной даты

При необходимости можно изменить **начальную дату прогноза**, указав её в соответствующем поле.

## Параметры прогнозной модели

- **Исторические данные** — можно ограничить период анализа, задав начальную дату временных данных.  
- **Режим сезонности**  
  - *Additive* — при стабильной выручке (амплитуда колебаний постоянна).  
  - *Multiplicative* — при росте выручки (сезонные колебания масштабируются).  
- **Годовая сезонность** — учитывает циклы продаж в течение года (например, рост в декабре, спад летом).  
- **Недельная сезонность** — отражает различия между днями недели (например, спад в выходные, пик в будни).  
- **Чувствительность к изменениям тренда** — регулирует плавность изменения тренда (0.01–0.5).  
- **Процент данных для поиска изломов** — доля исторических данных, где ищутся точки изменения тренда (например, 0.8 = первые 80%).  
- **Максимальное количество изломов тренда** — количество потенциальных точек изменения тренда (по умолчанию 25).

## Сохранение и сценарии бюджета

После расчёта прогнозов модель позволяет **сохранить бюджет** в базе данных для дальнейшей работы.  
Рекомендуется следующий подход:

- **Годовой бюджет** — формируется в конце года в трёх сценариях: *базовый*, *оптимистичный*, *консервативный*.  
- **Квартальный (rolling) бюджет** — обновляется в конце каждого квартала с учётом фактических данных.  
- **Ad-hoc бюджет** — может быть рассчитан для любых дат и горизонтов и хранится под отдельной категорией.
"""

    def layout(self):
        return dmc.Container(
            children=[
                dmc.Title("Планирование продаж и доходной части", order=1, c="teal"),
                dmc.Text(
                    "В данном разделе осуществляется планирование продаж и создание бюджетов доходной части",
                    size="xs",
                ),
                dmc.Spoiler(
                    dcc.Markdown(self.planning_memo, className="planing-memo"),
                    showLabel="Показать",
                    hideLabel="Скрыть",
                    maxHeight=50,
                ),
                dmc.Grid(
                    [
                        dmc.GridCol(
                            [
                                dmc.Container(
                                    [
                                        dmc.Stack(
                                            [
                                                self.dates_fieldsets,
                                                self.seasons_fieldset,
                                                self.trend_fieldset,
                                            ]
                                        )
                                    ],
                                    fluid=True,
                                ),
                            ],
                            span=4,
                        ),
                        dmc.GridCol(
                            [
                                dcc.Loading(
                                    [
                                        dmc.Container(
                                            children=[self.empty_hint],
                                            id=self.resultes_container_id,
                                            fluid=True,
                                        )
                                    ],
                                    type='cube'
                                )
                            ],
                            span=8,
                        ),
                    ]
                ),
            ],
            fluid=True
        )

    def registered_callbacks(self, app):

        conteiner = self.resultes_container_id
        horizon = self.horizon_date_picker_id
        current_date = self.init_date_picker_id
        cut_off_historical = self.hustorical_cut_date_picker_id
        year_season = self.yearly_season_checkbox_id
        week_season = self.weekly_season_checkbox_id
        season_type = self.seasons_type_select_id
        changepaints = self.n_changepoints_id
        proprsacale = self.changepoint_prior_scale_id
        data_samle = self.changepoint_range_id

        @app.callback(
            Output(conteiner, "children"),
            Input(horizon, "value"),
            Input(current_date, "value"),
            Input(cut_off_historical, "value"),
            Input(year_season, "checked"),
            Input(week_season, "checked"),
            Input(season_type, "value"),
            Input(changepaints, "value"),
            Input(proprsacale, "value"),
            Input(data_samle, "value"),
            prevent_initial_call=True,
        )
        def update_forecast_conteiner(
            horizon,
            cur_date,
            cut_off,
            year_season,
            week_season,
            saeson_type,
            changepoins,
            proprsacale,
            data_samle,
        ):
            def content():
                date = pd.to_datetime(horizon).strftime("%Y-%m-%d")
                data, yearly_season_chart, total_mape, table = forecast(
                    horizon=date,
                    current_date=cur_date,
                    historical_cut_off=cut_off,
                    yearly_seasonality=year_season,
                    weekly_seasonality=week_season,
                    seasonality_mode=saeson_type,
                    changepoint_prior_scale=proprsacale,
                    changepoint_range=data_samle,
                    n_changepoints=changepoins,
                )

                data["eom"] = pd.to_datetime(data["ds"]) + pd.offsets.MonthEnd(0)
                bc_data = (
                    data.pivot_table(
                        index="eom", columns="type", values="y", aggfunc="sum"
                    )
                    .reset_index()
                    .sort_values("eom")
                    .fillna(0)
                    .tail(24)
                )
                bc_data.columns = bc_data.columns.get_level_values(-1)
                bc_data["eom"] = (
                    pd.to_datetime(bc_data["eom"]).dt.strftime("%b %y").str.capitalize()
                )

                plan_act_chart = dmc.BarChart(
                    h=300,
                    data=bc_data.to_dict("records"),
                    dataKey="eom",
                    series=[
                        {
                            "name": "План",
                            "color": "cyan",
                        },
                        {
                            "name": "Факт",
                            "color": "teal",
                        },
                    ],
                )
                return plan_act_chart, yearly_season_chart, total_mape, table

            if horizon:
                plan_act_chart, yerly_season_chart, total_mape, table = content()
                return dmc.Stack(
                    [
                        dmc.Title(
                            f"Результаты планирования - MAPE = {total_mape:,.2f}%",
                            order=2,
                        ),
                        dmc.Space(h=10),
                        dmc.Center(table),
                        dmc.Space(h=10),
                        plan_act_chart,
                        dmc.Space(h=10),
                        yerly_season_chart,
                    ]
                )
            else:
                return NoData().component
