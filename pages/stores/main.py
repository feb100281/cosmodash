from dash_iconify import DashIconify
import dash_mantine_components as dmc
from dash import dcc, Input, Output, State, no_update
from components import MonthSlider, DATES, LoadingScreen
import pandas as pd
from data import load_df_from_redis, save_df_to_redis, delete_df_from_redis, load_sql_df
from pages.dinamix.stores.layouts import StoresComponents
SC_CALLBACKS = StoresComponents()

def id_to_months(start, end):
    return DATES[start].strftime("%Y-%m-%d"), DATES[end].strftime("%Y-%m-%d")


class StoreWindow:
    def __init__(self):
        self.title = dmc.Title("Магазины и каналы продаж", order=1, c="blue")
        self.memo = dmc.Text("Данный раздел предоставляет аналитику по магазинам и каналам продаж.", size="xs")
        
        self.mslider_id = "store_monthslider"
        self.mslider = MonthSlider(id=self.mslider_id)

        # store для ханнения df по слайдеру
        self.df_store_id = "store_df_store"
        self.store_df_store = dcc.Store(id=self.df_store_id, storage_type="session")

        # lable для хранения дат c учетом последнего обновления
        self.last_update_lb_id = "store_last_update_lb"
        self.last_update_lb = dcc.Loading(
            dmc.Badge(size="md", variant="light", radius="xs", 	color="red", id=self.last_update_lb_id)
        )
        
        self.data_conteiner_id = 'store_data_conteirer_id'
        
    def make_layout(self):
        return dmc.Container(
            children=[
                self.title,
                self.memo,
                self.mslider,
                self.last_update_lb,
                dmc.Container(
                    id = self.data_conteiner_id,
                    fluid=True,
                    children=[LoadingScreen().component]
                    ),
                dcc.Store(id="store_dummy_imputs_for_slider"),
                dcc.Store(id="store_dummy_imputs_for_render"),
                
            ],
            fluid=True,
        )
    
    def reistered_callbacks(self,app):
        # Обновляем df и пешем в redis по ключу
        @app.callback(
            Output(self.df_store_id, "data"),
            Output(self.last_update_lb_id, "children"),            
            Input(self.mslider_id, "value"),
            Input("store_dummy_imputs_for_render", "id"),
            Input("store_dummy_imputs_for_slider", "data"),
            State(self.df_store_id, "data"),
            prevent_initial_call=False,
        )
        def update_df(slider_value, dummy1, dummy2,  store_data):
            start, end = id_to_months(slider_value[0], slider_value[1])

           
            if store_data and "df_id" in store_data:
                if store_data["start"] == start and store_data["end"] == end:
                    df = load_df_from_redis(store_data["df_id"])
                    if df is not None:  # ключ ещё живой в Redis
                        min_date = pd.to_datetime(df["date"].min())
                        max_date = pd.to_datetime(df["date"].max())
                        notification = f"{min_date.strftime('%d %b %y')} - {max_date.strftime('%d %b %y')}"
                        return no_update, notification

                delete_df_from_redis(store_data["df_id"])

            df = load_sql_df(start_eom=start, end_eom=end)

            df_id = save_df_to_redis(df, expire_seconds=1200)

            store_dict = {
                "df_id": df_id,
                "start": start,
                "end": end,
                "slider_val": slider_value,
            }

            min_date = pd.to_datetime(df["date"].min())
            max_date = pd.to_datetime(df["date"].max())

            notificattion = (
                f"{min_date.strftime('%d %b %y')} - {max_date.strftime('%d %b %y')}"
            )

            return store_dict, notificattion
        
        @app.callback(
            Output(self.data_conteiner_id,'children'),
            Input(self.df_store_id,'data'),
            Input("store_dummy_imputs_for_render", "id"),
            
        )
        def update_data_conteiner(data,dummy):
            id_df = data["df_id"]
            return StoresComponents(df_id=id_df).tab_layout()
            
        SC_CALLBACKS.register_callbacks(app)
        
        

        