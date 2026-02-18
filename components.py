# Тут функции для создания компонентов что бы все были одинаковые
import pandas as pd
import numpy as np
import uuid

from dash import (
    dcc,
    html,
)
import dash_ag_grid as dag
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import locale

locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")

DATES = pd.date_range("2021-01-31", periods=30 * 12, freq="ME")


BASE_COLORS = [
    "blue",
    "cyan",
    "teal",
    "green",
    "lime",
    "yellow",
    "orange",
    "red",
    "pink",
    "grape",
    "violet",
    "indigo",
    "gray",
    "dark",
    "brand",
]

SHADES = ["6", "3", "9"]


# 1) По оттенкам: сначала все .3, потом .6, потом .9
def colors_by_shade(base=BASE_COLORS, shades=SHADES):
    return [f"{c}.{s}" for s in shades for c in base]


# 2) По цветам: для каждого цвета три оттенка подряд (blue.3, blue.6, blue.9, затем next color)
def colors_by_color(base=BASE_COLORS, shades=SHADES):
    return [f"{c}.{s}" for c in base for s in shades]


# Примеры готовых списков
COLORS_BY_SHADE = colors_by_shade()
COLORS_BY_COLOR = colors_by_color()


class MonthSlider(dmc.RangeSlider):
    def __init__(
        self, id, min_date="2021-01-31", max_date=None, defaul_period=12, **kwargs #ЗДЕСЬ МЕНЯТЬ СЛАЙДЕР
    ):
        dates = DATES
        if max_date is None:
            today = pd.Timestamp.today()
            max_date = today + pd.offsets.MonthEnd(0)
            max_date = max_date.strftime("%Y-%m-%d")
        else:
            max_date = pd.to_datetime(max_date) + pd.offsets.MonthEnd(0)
            max_date = max_date.strftime("%Y-%m-%d")

        # собираем DataFrame
        df = pd.DataFrame({"month_id": range(len(dates)), "date": dates})
        df["month_name"] = (
            df["date"].dt.strftime("%b").str.capitalize()
            + "\u202f"
            + df["date"].dt.strftime("%y")
        )
        df["month_name"].str.capitalize()

        min_id = df.loc[df["date"] == pd.Timestamp(min_date), "month_id"].iloc[0]
        max_id = df.loc[df["date"] == pd.Timestamp(max_date), "month_id"].iloc[0]

        df = df[df["month_id"] <= max_id]

        month_marks = (
            df[["month_id", "month_name"]]
            .sort_values("month_id")
            .assign(value=lambda x: x["month_id"], label=lambda x: x["month_name"])[
                ["value", "label"]
            ]
            .to_dict("records")
        )
        short_marks = [mark.copy() for mark in month_marks]  # Копируем month_marks
        diff = max_id - min_id
        keep_indices = []
        if diff <= 12:
            keep_indices = [min_id, max_id]
        elif diff <= 24:
            # 3 точки: начало, середина, конец
            keep_indices = np.linspace(min_id, max_id, 3, dtype=int).tolist()
        elif diff <= 36:
            # 4 точки: начало, 1/3, 2/3, конец
            keep_indices = np.linspace(min_id, max_id, 4, dtype=int).tolist()
        elif diff <= 42:
            # 5 точек для больших диапазонов
            keep_indices = np.linspace(min_id, max_id, 5, dtype=int).tolist()
        else:
            keep_indices = np.linspace(min_id, max_id, 6, dtype=int).tolist()

        for i in range(max_id):
            if i not in keep_indices:
                short_marks[i]["label"] = ""

        super().__init__(
            id=id,
            value=[max_id - defaul_period, max_id],
            marks=short_marks,
            mb=35,
            min=min_id,
            max=max_id,
            minRange=1,
            labelAlwaysOn=True,
            size=10,
            mt="xl",
            styles={"thumb": {"borderWidth": 2, "padding": 3}},
            color="red",
            thumbSize=42,
            thumbChildren=[
                DashIconify(icon="mdi:arrow-right-circle", width=22),  # mdi:heart
                DashIconify(icon="mdi:arrow-left-circle", width=22),
            ],
            label={
                "function": "formatMonthLabel",
                "options": {
                    "monthDict": {
                        month["value"]: month["label"] for month in month_marks
                    }
                },
            },
            **kwargs,
        )


class ValuesRadioGroups(dmc.RadioGroup):
    def __init__(self, id_radio, options_dict: dict, grouped=True, val=None, **kwargs):

        if val is None and options_dict:
            val = next(iter(options_dict.keys()))

        container = dmc.Group if grouped else dmc.Stack

        children = container(
            [dmc.Radio(label=v, value=k, color="blue") for k, v in options_dict.items()]
        )

        super().__init__(
            children=children,
            my=5,
            value=val,
            size="xs",
            mt=5,
            id=id_radio,
            **kwargs,
        )


class InDevNotice:
    def __init__(self):
        self.in_dev_conteines = dmc.Container(
            dmc.Center(
                dmc.Stack(
                    [
                        dmc.Space(h=20),
                        dmc.Center(
                            DashIconify(icon="lucide:hammer", width=80, color="gray"),
                        ),
                        dmc.Center(
                            dmc.Title("Раздел в разработке", order=3, c="gray"),
                        ),
                        dmc.Center(
                            dmc.Text(
                                "Я активно работаю над этим разделом. Скоро здесь появятся новые возможности.",
                                size="md",
                                c="dimmed",
                            ),
                        ),
                        dmc.Space(h=20),
                    ]
                )
            ),
            fluid=True,
            px="xl",
        )









class LoadingScreen:
    """
    Стендэлон-блок 'данные подгружаются'.
    Плейсхолдер без собственного dcc.Loading.
    """
    def __init__(
        self,
        title: str = "Загрузка данных…",
        subtitle: str = "Подождите немного, мы подтягиваем свежие данные",
    ):
        self.component = dmc.Container(
            dmc.Center(
                dmc.Stack(
                    [
                        dmc.Space(h=30),
                        DashIconify(icon="line-md:loading-loop", width=90, color="gray"),
                        dmc.Title(title, order=3, c="gray"),
                        dmc.Text(subtitle, size="md", c="dimmed"),
                        dmc.Space(h=12),
                    ],
                    gap="sm",
                    align="center",
                ),
            ),
            fluid=True,
            px="xl",
            style={"minHeight": "40vh"},
        )




class LoadingWrap:
    """
    Универсальная обёртка вокруг любого children.
    Пока children грузится, показывается cube-лоадер + свой мини-экран.
    """
    def __init__(
        self,
        children,
        title: str = "Загрузка…",
        subtitle: str = "Подождите немного, данные подгружаются",
        cube_color: str = None,  # можно передать цвет, если нужно
    ):
        loading_overlay = dmc.Center(
            dmc.Stack(
                [
                    DashIconify(icon="line-md:hourglass", width=54, color="gray"),
                    dmc.Text(title, fw=600, size="lg"),
                    dmc.Text(subtitle, c="dimmed"),
                ],
                align="center",
                gap="xs",
            ),
            style={"minHeight": 220},
        )

        self.component = dcc.Loading(
            id="loading-wrap",
            type="cube",
            color=cube_color,   # если None — возьмёт тему
            fullscreen=False,   # можно переключить на True для полноэкранного спиннера
            children=html.Div(
                children=[
                    # Когда загрузка идёт — dcc.Loading покажет кубики + этот overlay.
                    dmc.Paper(
                        loading_overlay,
                        withBorder=False,
                        shadow="sm",
                        radius="lg",
                        style={"minHeight": 220},
                    ),
                    # Когда загрузка завершится — покажется реальный контент.
                    html.Div(children, id="loading-wrap-content"),
                ]
            ),
        )


class NoData:
    def __init__(self):
        self.component = dmc.Container(
            dmc.Center(
                dmc.Stack(
                    [
                        dmc.Space(h=20),
                        dmc.Center(
                            DashIconify(
                                icon="carbon:data-error", width=80, color="gray"
                            ),
                        ),
                        dmc.Center(
                            dmc.Title("Нет данных", order=3, c="gray"),
                        ),
                        dmc.Center(
                            dmc.Text(
                                "Пока нет данных для данного раздела",
                                size="md",
                                c="dimmed",
                            ),
                        ),
                        dmc.Space(h=20),
                    ]
                )
            ),
            fluid=True,  # ✅ у Container это допустимо
            px="xl",
        )


MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
    # если нужны русские месяцы
    "Янв": 1,
    "Фев": 2,
    "Мар": 3,
    "Апр": 4,
    "Май": 5,
    "Июн": 6,
    "Июл": 7,
    "Авг": 8,
    "Сен": 9,
    "Окт": 10,
    "Ноя": 11,
    "Дек": 12,
}


def month_str_to_date(s: str) -> str:
    """
    Превращает строку 'Авг 23' или 'Aug 23' в '2023-08-01'
    """
    try:
        mon_str, year_str = s.split()
        month = MONTHS[mon_str]
        year = int(year_str)
        if year < 100:  # двухзначный год
            year += 2000
        return f"{year:04d}-{month:02d}-01"
    except Exception as e:
        raise ValueError(f"Невозможно распарсить '{s}': {e}")




from dash_iconify import DashIconify
import dash_mantine_components as dmc


class ClickOnNotice:
    def __init__(
        self,
        notice="Кликните на график, чтобы просмотреть отчет по заказам за выбранный месяц",
        icon="streamline-ultimate:task-finger-show",     # ← можно поменять на любую, например "solar:cursor-click-bold-duotone"
        color="#007BFF",
        icon_width=50,
    ):
        self.notice = notice
        self.icon = icon
        self.color = color
        self.icon_width = icon_width

    @property
    def component(self):
        return dmc.Alert(
            icon=DashIconify(icon=self.icon, width=50, color=self.color),
            title=dmc.Text("Детализация доступна", fw=600, c=self.color),
            color=self.color,
            radius="md",
            variant="light",
            withCloseButton=False,
            style={
                "background": "rgba(0, 123, 255, 0.05)",
                "border": f"1px solid {self.color}33",
                "padding": "10px 16px",
            },
            children=dmc.Text(self.notice, size="sm", mt=4),
        )






class DownLoadMenu:
    def __init__(
        self,
        pdf_disable=False,
        pdf_id_type=None,
        xls_disable=False,
        xls_id_type=None,
        html_disable=False,
        html_id_type=None,
    ):

        self.pdf_disable = pdf_disable
        self.pdf_id = {
            "type": pdf_id_type if pdf_id_type else str(uuid.uuid4()),
            "index": "1",
        }

        self.xls_disable = xls_disable
        self.xls_id = {
            "type": xls_id_type if xls_id_type else str(uuid.uuid4()),
            "index": "1",
        }

        self.html_disable = html_disable
        self.html_id = {
            "type": html_id_type if html_id_type else str(uuid.uuid4()),
            "index": "1",
        }

        self.dnl_menu = dmc.Menu(
            shadow="md",
            width=260,
            position="bottom-start",
            withArrow=True,
            children=[
                dmc.MenuTarget(
                    dmc.Button(
                    "Скачать",
                    leftSection=DashIconify(icon="tabler:download", width=20),
                    variant="outline",
                    color="blue",
                    size="md",
                )),
                dmc.MenuDropdown([
                    dmc.MenuLabel("Доступные загрузки"),
                    dmc.MenuDivider(),
                    dmc.Button(
                        "PDF отчёт",
                        id=self.pdf_id,
                        disabled=self.pdf_disable,
                        leftSection=DashIconify(
                            icon="material-icon-theme:pdf", width=18
                        ),
                        variant="subtle",
                        color="blue",
                        fullWidth=True,
                        size="xs",
                        justify="flex-start",
                    ),
                    dmc.Button(
                        "Excel",
                        id=self.xls_id,
                        disabled=self.xls_disable,
                        leftSection=DashIconify(
                            icon="vscode-icons:file-type-excel", width=18
                        ),
                        variant="subtle",
                        color="blue",
                        fullWidth=True,
                        size="xs",
                        justify="flex-start",
                    ),
                    dmc.Button(
                        "HTML",
                        id=self.html_id,
                        disabled=self.html_disable,
                        leftSection=DashIconify(icon="mdi:language-html5", width=18),
                        variant="subtle",
                        color="blue",
                        fullWidth=True,
                        size="xs",
                        justify="flex-start",
                    ),]
                ),
            ],
        )

    @property
    def menu(self):
        return dmc.Stack(
            justify="flex-start",
            gap="sm",
            children=[
                dcc.Loading(
                    type='circle',
                    delay_show=250,
                    color="#1f77b4",
                    style={"display": "inline-block", "verticalAlign": "middle"},
                    children=[
                        dmc.Box(id='loading-indicator')
                    ]
                    
                ),
                self.dnl_menu,
            ],
        )


class CsvAGgridDownloader():
    def __init__(self, id):
        self.bnt_icon = "vscode-icons:file-type-excel2"
        self.title = "Загрузить CSV"
        self.id = id
        
    
    @property
    def dnl_button(self):
        return dmc.Button(
                        self.title,
                        id=self.id,                        
                        leftSection=DashIconify(icon=self.bnt_icon, width=18),
                        variant="subtle",
                        color="blue",
                        fullWidth=True,
                        size="xs",
                        justify="flex-start",
                    )