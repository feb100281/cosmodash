import dash_mantine_components as dmc
from .db_queries import fleching_cats, assign_cat
from dash import dcc, Input, Output, State, no_update, MATCH
import pandas as pd

ACTIONS = [
    {"parent_cat": "Назначить категорию"},
    {"brend": "Назначить бренд"},
    {"manu": "Назначить производителя"},
]

from components import NoData


class CatManagementDrawer:

    def __init__(self):

        self.drawer_id = "CatManagementDrawer"
        self.drawer_conteiner_id = "CatManagementDrawer_conteiner"
        self.group_select_id = {
            "type": "CatManagementDrawer_group_select_id",
            "index": "1",
        }
        self.cat_select_id = {"type": "CatManagementDrawer_cat_select_id", "index": "1"}
        self.subcat_select_id = {
            "type": "CatManagementDrawer_subcat_select_id",
            "index": "1",
        }
        self.assign_button_id = {
            "type": "CatManagementDrawer_assign_button_id",
            "index": "1",
        }
        self.ids_store_id = {"type": "CatManagementDrawer_ids_stores_id", "index": "1"}
        self.df_store_id = {"type": "CatManagementDrawer_df_stores_id", "index": "1"}
        self.progeres_memo_id = {"type": "CatManagementDrawer_progressmemo_id", "index": "1"}
        
        self.title = "Управление категориями"
        

        

    def make_drawler(self):
        return dmc.Drawer(
            children=[
                dmc.Container(
                    id=self.drawer_conteiner_id, children=[NoData().component]
                )
            ],
            id=self.drawer_id,
            padding="md",
        )

    def update_drawer(self, ids):
        data = fleching_cats()
        data_gr = data[["parent_cat_id", "parent_cat"]].drop_duplicates()
        gr_data = []
        for _, row in data_gr.iterrows():
            gr_data.append({"value": str(row["parent_cat_id"]), "label": row["parent_cat"]})

        return dmc.Stack(
            children=[
                dmc.Title(self.title, order=4, c="indigo"),
                dmc.Space(h=20),
                dmc.Select(
                    id=self.group_select_id,
                    label="Выберите группу",
                    placeholder="Группа",
                    data=gr_data,
                    size="sm",
                    withAsterisk=True,
                ),
                dmc.Space(h=5),
                dmc.Select(
                    id=self.cat_select_id,
                    label="Выберите категорию",
                    placeholder="Категория",
                    data=[],
                    size="sm",
                    withAsterisk=True,
                ),
                dmc.Space(h=5),
                dmc.Select(
                    id=self.subcat_select_id,
                    label="Выберите подкатегорию",
                    placeholder="Подкатегория",
                    data=[],
                    size="sm",
                ),
                dmc.Space(h=20),
                dmc.Button(
                    "Применить",
                    id=self.assign_button_id,
                    disabled=True
                ),
                dmc.Space(h=10),
                dmc.Text(id=self.progeres_memo_id),
                dcc.Store(
                    id=self.df_store_id,
                    data=data.to_dict("records"),
                    storage_type="memory",
                ),
                dcc.Store(id=self.ids_store_id, data=ids, storage_type="memory"),
            ]
        )

    def register_callbacks(self, app):     
        
        cat_selct = self.cat_select_id['type']
        gr_select = self.group_select_id['type']
        sc_select = self.subcat_select_id['type']
        button = self.assign_button_id['type']
        memo = self.progeres_memo_id['type']
        items_ids = self.ids_store_id['type']
    
        @app.callback(
            Output({'type':cat_selct,'index':MATCH},'data'),
            Output({'type':sc_select,'index':MATCH},'data'),
            Output({"type":button,"index":MATCH},'disabled'),
            Input({'type':gr_select,'index':MATCH},'value'),
            Input({'type':cat_selct,'index':MATCH},'value'),
            prevent_initial_call=True,
        )
        def update_filters(gr_val,cat_val):    
            df = fleching_cats()
            
            if gr_val:
                df = df[df['parent_cat_id'] == int(gr_val)]
                cat = df[['cat_id', 'cat']].drop_duplicates()
                cat_data = [{"value": str(row["cat_id"]), "label": row["cat"]} for _, row in cat.iterrows()]
            else:
                cat_data = []

            if cat_val:
                df_cat = df[df['cat_id'] == int(cat_val)]
                subcat = df_cat[['subcat_id', 'subcat']].drop_duplicates()
                subcat_data = [
                    {"value": str(row["subcat_id"]), "label": str(row["subcat"]) if pd.notna(row["subcat"]) else ""}
                    for _, row in subcat.iterrows()
                ]
            else:
                subcat_data = []
            
            da = True
            if gr_val and cat_val:
               da = False 
            return cat_data, subcat_data, da
                
        @app.callback(
            Output({'type':memo,'index':MATCH},'children'),
            Input({'type':button,'index':MATCH},'n_clicks'),
            State({'type':cat_selct,'index':MATCH},'value'),
            State({'type':sc_select,'index':MATCH},'value'),
            State({'type':items_ids,'index':MATCH},'data'),
            prevent_initial_call=True,
        )
        def setcat(nclicks, cat_id, subcat_id, ids):
            assign_cat(cat_id=cat_id,subcat_id=subcat_id,ids=ids)
            return f"Номенелатуры обновлены. Перегрузите страницу"
         
                
                
CATS_MANAGEMENT = CatManagementDrawer()
