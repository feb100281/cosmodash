import pandas as pd
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import json
import dash_ag_grid as dag
import math

from dash import dcc, Input, Output, State, no_update

# from .forecast import SEASONS_OPTIONS, forecast
from components import NoData, MonthSlider, DATES

import locale

locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")

from .data import fletch_cats, matrix_calculation


def id_to_months(start, end):
    return DATES[start].strftime("%Y-%m-%d"), DATES[end].strftime("%Y-%m-%d")


# Делаем контроллеры матрицы
class LeftSection:
    def __init__(self):

        self.a_score_id = "a_score_id"
        self.b_score_id = "b_score_id"
        self.c_score_id = "c_score_id"

        self.x_score_id = "x_score_id"
        self.y_score_id = "y_score_id"
        self.z_score_id = "z_score_id"
        
        self.lead_time_id = "lead_time_id_for_matix"
        self.servis_ratio_id = "servis_ratio_id_for_matrix"

        self.gr_multyselect_id = "gr_multyselect_id_for_matrix"
        self.cat_multyselect_id = "cat_multyselect_id_for_matrix"

        self.groupby_sc_id = "groupby_sc_id_for_matrix"

        self.launch_batton_id = "launch_batton_id_for_matrix"

        
        # --------------------------
        # Ховеры для помощи в настройках
        # --------------------------
        
        abc_help = """
        #### Ранкирование по выручки
        Здесь устанавливаются процент выручки от каждого SKU для ранкирования ABC.
        По умолчанию:
        - рейтинг A присваевается товаром которые делают 50% общей выручки
        - рейтинг B присваевается товаром которые делают следующие 25% до общей выручки
        - рейтинг C присваевается товаром которые делают последние 25% до общей выручки 
                
        """
        abc_ranking_hover = dmc.HoverCard(
            withArrow=True,
            width=600,
            shadow="md",
            children=[
                dmc.HoverCardTarget(dmc.Text("Параметры для ABC расчетов    🤷🏻‍♀️")),
                dmc.HoverCardDropdown(
                    dcc.Markdown(abc_help,className='markdown-25')
                ),
            ],
        )
        
        xyz_help = """
        #### Ранкирование по спросу
        Здесь устанавливаются коэффициент вариации для каждого товара.
        Коэффициент вариации _cv_ это отношение стандарного отклонения _σ_ к среднему значению продаж _μ_.
        __Чем ниже данный коэффициент, тем стабильнее спрос__
        По умолчанию:
        - рейтинг X присваевается товаром c _cv_ ≦ 0.8;
        - рейтинг Y присваевается товаром c _cv_ > 0.8 и ≦ 1.8
        - рейтинг Z присваевается товаром c _cv_ <  1.8 (Рваный и непостоянный спрос)
                
        """
        xyz_ranking_hover = dmc.HoverCard(
            withArrow=True,
            width=600,
            shadow="md",
            children=[
                dmc.HoverCardTarget(dmc.Text("Параметры для XYZ расчетов    🤷‍♂️")),
                dmc.HoverCardDropdown(
                    dcc.Markdown(xyz_help,className='markdown-25')
                ),
            ],
        )
        
        rob_help = """
        #### Ранкирование по спросу
        Здесь устанавливаются коэффициент вариации для каждого товара.
        Коэффициент вариации _cv_ это отношение стандарного отклонения _σ_ к среднему значению продаж _μ_.
        __Чем ниже данный коэффициент, тем стабильнее спрос__
        По умолчанию:
        - рейтинг X присваевается товаром c _cv_ ≦ 0.8;
        - рейтинг Y присваевается товаром c _cv_ > 0.8 и ≦ 1.8
        - рейтинг Z присваевается товаром c _cv_ <  1.8 (Рваный и непостоянный спрос)
                
        """
        rob_hover = dmc.HoverCard(
            withArrow=True,
            width=600,
            shadow="md",
            children=[
                dmc.HoverCardTarget(dmc.Text("Параметры для XYZ расчетов    🤷‍♂️")),
                dmc.HoverCardDropdown(
                    dcc.Markdown(xyz_help,className='markdown-25')
                ),
            ],
        )
                
        
        
        # --------------------------
        # Прописываем компоненты
        # --------------------------

        # Кнопки управления ABC
        a_acore_number_imput = dmc.NumberInput(
            value=50,
            min=35,
            max=98,
            step=1,
            allowDecimal=False,
            suffix="%",
            leftSection=DashIconify(
                icon="mynaui:letter-a-waves-solid", color="red", width=24
            ),
            w=80,
            size="xs",
            id=self.a_score_id,
        )
        b_acore_number_imput = dmc.NumberInput(
            value=25,
            min=1,
            max=64,
            step=1,
            allowDecimal=False,
            suffix="%",
            leftSection=DashIconify(
                icon="mynaui:letter-b-waves-solid", color="blue", width=24
            ),
            w=75,
            size="xs",
            id=self.b_score_id,
        )
        c_acore_number_imput = dmc.NumberInput(
            value=25,
            min=1,
            max=64,
            step=1,
            allowDecimal=False,
            disabled=True,
            suffix="%",
            leftSection=DashIconify(
                icon="mynaui:letter-c-waves-solid", color="gray", width=24
            ),
            w=80,
            size="xs",
            id=self.c_score_id,
        )
        abc_fieldset = dmc.Fieldset(
            children=[
                dmc.Group(
                    [a_acore_number_imput, b_acore_number_imput, c_acore_number_imput]
                )
            ],
            radius="sm",
            legend=abc_ranking_hover,
        )

        # Кнопки управления XYZ
        x_acore_number_imput = dmc.NumberInput(
            value=0.8,
            min=0.1,
            max=2,
            step=0.1,
            allowDecimal=True,
            prefix="≤",
            leftSection=DashIconify(
                icon="mynaui:letter-x-diamond-solid", color="red", width=24
            ),
            w=80,
            size="xs",
            id=self.x_score_id,
        )
        y_acore_number_imput = dmc.NumberInput(
            value=1.8,
            min=0.25,
            max=3,
            step=0.1,
            allowDecimal=True,
            leftSection=DashIconify(
                icon="mynaui:letter-y-diamond-solid", color="teal", width=24
            ),
            w=75,
            size="xs",
            id=self.y_score_id,
        )
        z_acore_number_imput = dmc.NumberInput(
            value=1.8,
            min=0.5,
            max=100,
            step=0.1,
            allowDecimal=True,
            prefix=">",
            leftSection=DashIconify(
                icon="mynaui:letter-z-diamond-solid", color="gray", width=24
            ),
            w=80,
            size="xs",
            id=self.z_score_id,
            disabled=True,
        )
        xyz_fieldset = dmc.Fieldset(
            children=[
                dmc.Group(
                    [x_acore_number_imput, y_acore_number_imput, z_acore_number_imput]
                )
            ],
            radius="sm",
            legend=xyz_ranking_hover,
        )

        # Мультиселекты по группам и категориям

        self.cats_df = fletch_cats()

        gr_data = (
            self.cats_df[["gr_id", "gr_name"]]
            .dropna(subset=["gr_id", "gr_name"])
            .drop_duplicates()
            .assign(gr_id=lambda x: x["gr_id"].astype(str))
            .rename(columns={"gr_id": "value", "gr_name": "label"})
            .to_dict(orient="records")
        )

        cat_data = (
            self.cats_df[["cat_id", "cat_name"]]
            .dropna(subset=["cat_id", "cat_name"])
            .drop_duplicates()
            .assign(cat_id=lambda x: x["cat_id"].astype(str))
            .rename(columns={"cat_id": "value", "cat_name": "label"})
            .to_dict(orient="records")
        )

        gr_mulyselect = dmc.MultiSelect(
            id=self.gr_multyselect_id,
            label="Группы",
            placeholder="Выберите группу",
            data=gr_data,
            w="100%",
            radius=0,
            clearable=True,
            searchable=True,
            leftSection=DashIconify(icon="tabler:building-store"),
        )

        cat_mulyselect = dmc.MultiSelect(
            id=self.cat_multyselect_id,
            label="Магазин",
            placeholder="Выберите категорию",
            data=[],
            w="100%",
            radius=0,
            clearable=True,
            searchable=True,
            leftSection=DashIconify(icon="tabler:building-store"),
        )

        cats_ms_fieldset = dmc.Fieldset(
            children=[gr_mulyselect, cat_mulyselect],
            radius="sm",
            legend="Фильтр групп и категорий",
        )

        # Групировки

        sc_groupby_switch = dmc.Switch(
            onLabel="ON",
            offLabel="OFF",
            radius="sm",
            labelPosition="right",
            label="Групировать по подкатегориям",
            checked=False,
            id=self.groupby_sc_id,
        )

        groupby_sc_fieldset = dmc.Fieldset(
            children=[
                sc_groupby_switch,
            ],
            radius="sm",
            legend="Групировки номенклатур",
        )
        
        # Параметры ROB и SS        
        lt_number_imput = dmc.NumberInput(
            value=2,
            min=0.5,
            max=24,
            step=1,
            allowDecimal=True,
            suffix=" мес.",
            leftSection=DashIconify(
                icon="mdi:tool-time", color="red", width=24
            ),
            w=120,
            size="xs",
            id=self.lead_time_id,
        )
        sration_number_imput = dmc.NumberInput(
            value=90,
            min=70,
            max=100,
            step=1,
            allowDecimal=False,
            suffix="%",
            leftSection=DashIconify(
                icon="medical-icon:interpreter-services", color="blue", width=24
            ),
            w=120,
            size="xs",
            id=self.servis_ratio_id,
        )
        
        rob_fieldset = dmc.Fieldset(
            children=[
                dmc.Group(
                    [lt_number_imput, sration_number_imput]
                )
            ],
            radius="sm",
            legend=abc_ranking_hover,
        )


        # Кнопка запуска

        launch_btn = dmc.Button(
            "Рассчитать",
            id=self.launch_batton_id,
            leftSection=DashIconify(icon="mynaui:rocket-solid", width=24),
        )

        # --------------------------
        # Финальный Layout
        # --------------------------

        self.left_section_layout = dmc.Container(
            children=[
                dmc.Title("Настройки матрицы", order=4),
                dmc.Space(h=20),
                abc_fieldset,
                dmc.Space(h=20),
                xyz_fieldset,
                dmc.Space(h=20),
                rob_fieldset,
                dmc.Space(h=20),
                cats_ms_fieldset,
                dmc.Space(h=20),
                groupby_sc_fieldset,
                dmc.Space(h=20),
                launch_btn,
            ],
            fluid=True,
        )

    def register_callbacks(self, app):

        # фильтр категорий при выбранной группе
        @app.callback(
            Output(self.cat_multyselect_id, "data"),
            Input(self.gr_multyselect_id, "value"),
            prevent_initial_call=True,
        )
        def filter_cat_ms(gr_list):
            gr_list_int = [int(x) for x in gr_list]
            df = self.cats_df[self.cats_df["gr_id"].isin(gr_list_int)]

            return (
                df[["cat_id", "cat_name"]]
                .dropna(subset=["cat_id", "cat_name"])
                .drop_duplicates()
                .assign(cat_id=lambda x: x["cat_id"].astype(str))
                .rename(columns={"cat_id": "value", "cat_name": "label"})
                .to_dict(orient="records")
            )

        # автопересчет abc
        @app.callback(
            Output(self.b_score_id, "value"),
            Output(self.c_score_id, "value"),
            Output(self.b_score_id, "max"),
            Output(self.c_score_id, "max"),
            Input(self.a_score_id, "value"),
            prevent_initial_call=True,
        )
        def split_bc(a_val):
            r = 100 - a_val
            b = math.ceil(r / 2)
            c = 100 - b - a_val
            return b, c, r - 1, r - 1

        @app.callback(
            Output(self.c_score_id, "value", allow_duplicate=True),
            Input(self.b_score_id, "value"),
            State(self.a_score_id, "value"),
            prevent_initial_call=True,
        )
        def adjust_c(b_val, a_val):
            c = 100 - b_val - a_val
            return c

        # автопересчет xyz
        @app.callback(
            Output(self.y_score_id, "value"),
            Output(self.y_score_id, "min"),
            Output(self.z_score_id, "value"),
            Input(self.x_score_id, "value"),
            State(self.y_score_id, "value"),
            prevent_initial_call=True,
        )
        def set_yz(x_val, y_val):
            y_min = x_val + 1
            z = 0
            if y_val > y_min:
                z = y_val
            else:
                z = y_min

            return z, y_min, z

        @app.callback(
            Output(self.z_score_id, "value", allow_duplicate=True),
            Input(self.y_score_id, "value"),
            prevent_initial_call=True,
        )
        def set_z(y_val):
            return y_val


# Панель с самой матрицей
class RightSection:
    def __init__(self):
        
        # ID компонентов
        self.right_conteiner_id = "right_conteiner_id_for_matrix"
        
        self.matrix_dag_id = "matrix-ag-greed-id"
        
        
        
        # Инициируем пустую layout 
        self.layout = dmc.Container(children=[], id=self.right_conteiner_id, fluid=True)
    
    #Метод по получению мартицы
    def get_matrix(self, start, end, cat, threholds)->pd.DataFrame:
        return matrix_calculation(start, end, cat, threholds)
    
    #Метод для построения ag-grid
    def matrix_ag_grid(self,df:pd.DataFrame,rrgrid_className):
        
        #Это список всех полей
        df_columns_list = ['item_id', 'amount', 'date_json', 'quant_json', 'article', 
                           'fullname', 'cat_id', 'cat_name', 'subcat_id', 'sc_name', 
                           'share', 'cum_share', 'abc', 'mean_month', 'std_month', 'cv', 
                           'months_count', 'xyz']
        
        #Спецификация полей dag
        matrix_dag_cols_spec = [
            {
                        "headerName": "item_id",
                        "field": "item_id",
                        # "minWidth": 20,
                        # "type": "leftAligned",
                        # "cellClass": "ag-firstcol-bg",
                        # "headerClass": "ag-center-header",
                        # "pinned": "left",
                        "hide": True,
                    },
            
            {
                "headerName": "Рейтинги",
                "groupId": "ratings",
                "minWidth": 50,
                "marryChildren": True,
                "headerClass": "ag-center-header",
                "children": [
                    {
                        "headerName": "ABC",
                        "field": "abc",
                        "width": 90, 
                        # "minWidth": 10,
                        "type": "leftAligned",
                        "cellClass": "ag-firstcol-bg",
                        "headerClass": "ag-center-header",
                        "pinned": "left",
                        
                    },
            
            {
                        "headerName": "XYZ",
                        "field": "xyz",
                        "width": 90, 
                        # "minWidth": 10,
                        "type": "leftAligned",
                        "cellClass": "ag-firstcol-bg",
                        "headerClass": "ag-center-header",
                        "pinned": "left",
                        
                    },
                ]
            },
            
            
            
            
            {
                "headerName": "Номенклатура",
                "groupId": "product",
                "marryChildren": True,
                "headerClass": "ag-center-header",
                "children": [
                    {
                        "headerName": "Номенклатура",
                        "field": "fullname",
                        "minWidth": 220,
                        "type": "leftAligned",
                        "cellClass": "ag-firstcol-bg",
                        "headerClass": "ag-center-header",
                        #  "pinned": "left",
                    },
                    {
                        "headerName": "Категория",
                        "field": "cat_name",
                        "minWidth": 220,
                        "type": "leftAligned",
                        "cellClass": "ag-firstcol-bg",
                        "headerClass": "ag-center-header",
                        #  "pinned": "left",
                    },
                    {
                        "headerName": "Категория",
                        "field": "sc_name",
                        "minWidth": 220,
                        "type": "leftAligned",
                        "cellClass": "ag-firstcol-bg",
                        "headerClass": "ag-center-header",
                        #  "pinned": "left",
                    },
                ]
            },
            
            {
                "headerName": "Статистика",
                "groupId": "stats",
                "marryChildren": True,
                "headerClass": "ag-center-header",
                "children": [
                    {
                        "headerName": "Выручка",
                        "field": "amount",
                        "valueFormatter": {"function": "RUB(params.value)"},
                        "cellClass": "ag-firstcol-bg",
                    },
                    {
                        "headerName": "Доля выручка",
                        "field": "share",
                        "valueFormatter": {"function": "FormatPercent(params.value)"},
                        "cellClass": "ag-firstcol-bg",
                         
                    },
                    {
                        "headerName": "Ср мес продажи(шт)",
                        "field": "mean_month",
                        "minWidth": 100,
                        "type": "leftAligned",
                        "cellClass": "ag-firstcol-bg",
                        "headerClass": "ag-center-header",
                    },
                    {
                        "headerName": "Ст отклонение",
                        "field": "std_month",
                        "minWidth": 100,
                        "type": "leftAligned",
                        "cellClass": "ag-firstcol-bg",
                        "headerClass": "ag-center-header",
                    },
                    {
                        "headerName": "Коэф вариации",
                        "field": "cv",
                        "minWidth": 100,
                        "type": "leftAligned",
                        "cellClass": "ag-firstcol-bg",
                        "headerClass": "ag-center-header",
                    },
                ]
            }           
            
        ]
        
        RowData = df.to_dict("records")
                
        return dag.AgGrid(
                    id=self.matrix_dag_id,
                    rowData=RowData,
                    columnDefs=matrix_dag_cols_spec,
                    defaultColDef={"sortable": True, "filter": True, "resizable": True},
                    dashGridOptions={
                    "rowSelection": "single", 
                    "pagination": True, 
                    "paginationPageSize": 20,
                    "suppressRowClickSelection": False,
                    #"enableCellTextSelection": True,
                    "ensureDomOrder": True,
                    #"onRowDoubleClicked": {"function": "function(params) { window.dashAgGridFunctions.onRowDoubleClick(params); }"}
                },
                    
                # getRowId="function(params) { return params.data.fullname + '_' + params.data.init_date; }",
                    style={"height": "600px", "width": "100%"},
                    className=rrgrid_className,
                    dangerously_allow_code=True,
                )
           
        
    
        
    
    
    
    
    #Метод делаем layout c компонентыми после расчетов матрицы !!!! Вот сдесь свистелки / перделки можно делать
    def maxrix_layout(self, start, end, cat, threholds,rrgrid_className) ->dmc.Container:
        
        # Загружаем df с матрицей
        df = self.get_matrix(start, end, cat, threholds)
        
        matrix_dag = self.matrix_ag_grid(df,rrgrid_className)
        
        return dmc.Container(
            [
            dmc.Title("Расчет ассортиментной матрицы",order=3),
            dmc.Space(h=20),
            matrix_dag,
            dmc.Space(h=40)
            ],
            fluid=True            
        )
        
        
         
    


# Соединяем все вместе в единый layout
class MainWindow:

    def __init__(self):
        self.ls = LeftSection()
        self.rs = RightSection()
        self.mslider_id = "mslider-id-for-matrix-calculations"
        self.mslider = MonthSlider(id=self.mslider_id)

    def layout(self):

        return dmc.Container(
            children=[
                dmc.Title(
                    "Создание и анализ ассортиментой матрицы", order=1, c="indigo"
                ),
                dmc.Text(
                    "В данном разделе можно создавать и анализировать ассортиментные матрицы",
                    size="xs",
                ),
                dmc.Space(h=40),
                self.mslider,
                dmc.Space(h=20),
                dmc.Grid(
                    [
                        dmc.GridCol([self.ls.left_section_layout], span=3),
                        dmc.GridCol([self.rs.layout], span=9),
                    ]
                ),
            ],
            fluid=True,
        )

    def register_callbacks(self, app):
        self.ls.register_callbacks(app)

        @app.callback(
            Output(self.rs.right_conteiner_id, "children"),
            Input(self.ls.launch_batton_id, "n_clicks"),
            State(self.ls.a_score_id, "value"),
            State(self.ls.b_score_id, "value"),
            State(self.ls.c_score_id, "value"),
            State(self.ls.x_score_id, "value"),
            State(self.ls.y_score_id, "value"),
            State(self.ls.z_score_id, "value"),
            State(self.ls.gr_multyselect_id, "value"),
            State(self.ls.cat_multyselect_id, "value"),
            State(self.mslider_id, "value"),
            State("theme_switch", "checked"),
            prevent_initial_call=True,
        )
        def get_matrix(nclicks, a, b, c, x, y, z, grs, cats, ms, theme):

            def fined_cats_if_gr():

                gr_list_int = [int(x) for x in grs]
                df = self.ls.cats_df[self.ls.cats_df["gr_id"].isin(gr_list_int)]
                return df["cat_id"].to_list()

            if nclicks:
                threholds = {"a": a, "b": b, "c": c, "x": x, "y": y, "z": z}
                start, end = id_to_months(ms[0], ms[1])
                gr = None if not grs else ",".join(grs)
                cat = None if not cats else ",".join(cats)
                
                rrgrid_className = "ag-theme-alpine-dark" if theme else "ag-theme-alpine"

                if gr and not cat:
                    cat = ",".join(map(str, fined_cats_if_gr()))                

            return self.rs.maxrix_layout(start,end,cat,threholds,rrgrid_className)
            
