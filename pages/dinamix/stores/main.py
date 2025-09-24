# Файл для сбора layout и callbacks по табу Магазины


from .layouts import StoresComponents
def layout(df_id=None):
    return StoresComponents(df_id).tab_layout()

callbacks = StoresComponents()
