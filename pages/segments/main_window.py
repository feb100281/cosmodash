import pandas as pd
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import json
import dash_ag_grid as dag


from data import (load_columns_df, load_df_from_redis, delete_df_from_redis, save_df_to_redis)
from components import MonthSlider, DATES
from dash import (
    dcc,
    Input,
    Output,
    State,
    no_update
)
from .db_queries import get_items

COLS = [
    'date',
    'eom',
    'init_date',
    'parent_cat',
    'parent_cat_id',
    'cat',
    'cat_id',
    'subcat',
    'subcat_id',
    'item_id',
    'fullname',
    'brend',
    'manu',
    'amount',
    'quant'
]

def id_to_months(start, end):
    return DATES[start].strftime("%Y-%m-%d"), DATES[end].strftime("%Y-%m-%d")

class SegmentMainWindow:
    def __init__(self):
        
        self.title = dmc.Title("Сегментный аналих", order=1, c="blue")
        self.memo = dmc.Text("Данный раздел предоставляет аналитику по номенклатурам продукции", size="xs")
        
        # self.mslider_id = {"type":"segment_analisys_monthslider", "index":'1'}
        self.mslider_id = "segment_analisys_monthslider"
        self.tree_conteiner_id = "segment_analisys_tree_container_very_unique_id"
        self.details_conteiner_id = "segment_analisys_details_container_very_unique_id"
        self.mslider = MonthSlider(id=self.mslider_id)        
        
        self.tree_id = 'segments_tree_id'    
        self.tree = dmc.Tree(
            id = self.tree_id,
            data = [],
            expandedIcon=DashIconify(icon="line-md:chevron-right-circle",width=20),
            collapsedIcon=DashIconify(icon="line-md:arrow-up-circle",width=20),
            checkboxes=True,
            
        )
        
        self.df_store_id = "df_segment_store"
        self.df_store = dcc.Store(id=self.df_store_id, storage_type="session")
        
        self.last_update_lb_id = "last_update_segments_lb"
        self.last_update_lb = dcc.Loading(
            dmc.Badge(size="md", variant="light", radius="xs", 	color="red", id=self.last_update_lb_id)
        )
        
    def update_ag(self, df, rrgrid_className):
        df:pd.DataFrame = df
        df['amount'] = df.dt - df.cr
        df['quant'] = df['quant_dt'] - df['quant_cr']
        df['ret_ratio'] = df.amount / df.dt * 100        
        
        cols = [
            {"headerName": "Номенклатура", 
             "field": "fullname"
             },#, "cellClass": "ag-firstcol-bg",     "pinned": "left",},
            {
            "headerName": "Дата инициализации",
            "field": "init_date",
            "valueFormatter": {"function": "RussianDate(params.value)"}
            },
            {
            "headerName": "Последняя продажа",
            "field": "last_sales_date",
            "valueFormatter": {"function": "RussianDate(params.value)"}
            },
            {
            "headerName": "Выручка",
            "field": "amount",
            "valueFormatter": {"function": "RUB(params.value)"}, "cellClass": "ag-firstcol-bg",
            },
            {
            "headerName": "Всего продано",
            "field": "quant",
            "valueFormatter": {
                "function": "FormatWithUnit(params.value,'ед')"
            },
            },
            {
            "headerName": "Процент возвратов",
            "field": "quant",
            "valueFormatter": {
                "function": "FormatWithUnit(params.value,'%')"
            },
            },
            {"headerName": "Бренд", 
             "field": "brend"
             },
            {"headerName": "Производитель", 
             "field": "manu"
             },
        ]
            
        return dmc.Stack([
                dmc.Space(h=5),
                dmc.Title(f"Выбранные позиции",order=4),
                dmc.Space(h=10),
                dag.AgGrid(            
                id="orders-grid",
                rowData=df.to_dict("records"),
                columnDefs=cols,
                defaultColDef={
                    "sortable": True,
                    "filter": True,
                    "resizable": True,
                },
                dashGridOptions={
                    "rowSelection": "single",
                    "pagination": True,
                    "paginationPageSize": 20,
                },
                style={"height": "600px", "width": "100%"},
                className=rrgrid_className,
                dangerously_allow_code=True            
            ),
                
            ]
            )
        
        
        
    def data(self,start_eom,end_eom):        
        df = load_columns_df(COLS,start_eom,end_eom)
        df['parent_cat_id'] = df['parent_cat_id'].fillna(10_000_000)
        
        df['cat_id'] = df['cat_id'].fillna(10_000_000)
        df['subcat_id'] = df['subcat_id'].fillna(10_000_000)
        df['item_id'] = df['item_id'].fillna(10_000_001)
        
        df = df.pivot_table(
            index=['parent_cat_id','parent_cat','cat_id','cat','subcat_id','subcat','fullname','item_id'],
            # index='fullname',
            values=['amount','quant'],
            aggfunc={
                'amount':'sum',
                'quant':'sum',
                
            }
        ).reset_index()
        df['fullname'] = df['fullname'].apply(lambda x: x if len(x) <= 50 else x[:50] + '...')
        
        return  df
    
    def maketree(self, df):
        tree = []

        def find_or_create(lst, value, label):
            """Находит или создаёт узел"""
            for node in lst:
                if node["value"] == str(value):
                    return node
            node = {
                "value": str(value),
                "label": str(label),
                "children": [],
                "_count": 0  # внутренний счётчик
            }
            lst.append(node)
            return node

        for _, row in df.iterrows():
            pid, pname = row['parent_cat_id'], row['parent_cat']
            cid, cname = row['cat_id'], row['cat']
            sid, sname = row['subcat_id'], row['subcat']
            fullname = (row['item_id'], row['fullname'])

            # Преобразуем 10_000_000 обратно в None
            cid = None if cid == 10_000_000 else cid
            sid = None if sid == 10_000_000 else sid

            # 1 уровень — parent
            parent_node = find_or_create(tree, pid, pname)

            # 2 уровень — cat
            if cid is not None:
                cat_node = find_or_create(parent_node["children"], cid, cname)
            else:
                cat_node = parent_node

            # 3 уровень — subcat
            if sid is not None:
                subcat_node = find_or_create(cat_node["children"], sid, sname)
            else:
                subcat_node = cat_node

            # 4 уровень — fullname
            subcat_node["children"].append({
                "value": str(fullname[0]),
                "label": str(fullname[1])
            })

            # Увеличиваем счётчики на всех уровнях
            parent_node["_count"] += 1
            if cat_node is not parent_node:
                cat_node["_count"] += 1
            if subcat_node not in (parent_node, cat_node):
                subcat_node["_count"] += 1

        # Финальный проход для добавления (N) в label
        def finalize_labels(lst):
            for node in lst:
                count = node.get("_count", 0)
                if count > 0:
                    node["label"] = f"{node['label']} ({count})"
                # только если есть дети
                if "children" in node and node["children"]:
                    finalize_labels(node["children"])
                # удаляем внутренний ключ
                node.pop("_count", None)

        finalize_labels(tree)
        
        return tree

        
    def layout(self):
        return dmc.Container(
            children=[
                self.title,
                self.memo,
                dmc.Space(h=10),
                self.mslider,
                self.last_update_lb,
                dmc.Space(h=10),
                dmc.Grid(
                    [
                        dmc.GridCol(
                            children=[
                                dmc.Container(
                                    id=self.tree_conteiner_id,
                                    children=[
                                        self.tree,
                                        ],
                                    fluid=True
                                )
                            ],
                            span=6
                        ),
                        dmc.GridCol(
                            children=[
                                dmc.Container(
                                    id=self.details_conteiner_id,
                                    children=['это детали'],
                                    fluid=True
                                )
                            ],
                            span=6
                        ),
                        
                    ]
                ),
                dcc.Store(id="dummy_imputs_for_segment_slider"),
                dcc.Store(id="dummy_imputs_for_segment_render"),
            ],
            fluid=True
        )
    
    def register_callbacks(self, app):
        @app.callback(
            Output(self.df_store_id, "data"),
            Output(self.last_update_lb_id, "children"),  
            Input(self.mslider_id, "value"),
            Input("dummy_imputs_for_segment_render", "data"),
            State(self.df_store_id, "data"),
            prevent_initial_call=False,
        )
        def update_df(slider_value, dummy, store_data):
            start, end = id_to_months(slider_value[0], slider_value[1])

            if store_data and "df_id" in store_data:
                if store_data["start"] == start and store_data["end"] == end:
                    df = load_df_from_redis(store_data["df_id"])
                    if df is not None:  # ключ ещё живой в Redis
                        min_date = pd.to_datetime(start)
                        max_date = pd.to_datetime(end)
                        notification = f"{min_date.strftime('%b %y')} - {max_date.strftime('%b %y')}"
                        return no_update, notification

                delete_df_from_redis(store_data["df_id"])

            df = self.data(start_eom=start, end_eom=end)

            df_id = save_df_to_redis(df, expire_seconds=1200)
            
            store_dict = {
                "df_id": df_id,
                "start": start,
                "end": end,
                "slider_val": slider_value,
            }
            
            nnoms = df.fullname.nunique()

            min_date = pd.to_datetime(start)
            max_date = pd.to_datetime(end)

            notificattion = (
                f"{min_date.strftime('%b %y')} - {max_date.strftime('%b %y')} ВСЕГО: {nnoms:.0f} НОМЕНКЛАТУР"
            )

            return store_dict, notificattion
        
        @app.callback(
            Output(self.tree_id, "data"),
            Input(self.df_store_id, "data"),
        )
        def update_tabs(store_data):            
            id_df = store_data["df_id"]
            df = load_df_from_redis(id_df)
            
             
            return self.maketree(df)
        
        @app.callback(
            Output(self.details_conteiner_id,"children"),
            Input(self.tree_id, "checked"),
            State("theme_switch", "checked"),
            prevent_initial_call=True             
        )
        def get_data(checked,theme):
            code = json.dumps( checked, indent=4)
            if not checked:
                return dmc.Paper("Нет выбранных элементов") 
            
            rrgrid_className = "ag-theme-alpine-dark" if theme else "ag-theme-alpine"
            md = get_items(checked)
            
            
            return self.update_ag(md,rrgrid_className)

           
# a = SegmentMainWindow()
# d = a.data('2025-01-31','2025-09-30')

# # d = d[d['cat']=='Столы']
# print(d['parent_cat',])