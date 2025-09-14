import pandas as pd
import numpy as np


filenew = '/Users/pavelustenko/Downloads/v3.xlsx'
fileold = '/Users/pavelustenko/Downloads/09092025.xlsx'



df_new = pd.read_excel(filenew, skiprows=3, skipfooter=1)
df_old = pd.read_excel(fileold, skiprows=3, skipfooter=1)

df_new = df_new.loc[:, ~df_new.columns.str.startswith("Unnamed")]
df_old = df_old.loc[:, ~df_old.columns.str.startswith("Unnamed")]

new_collumns = df_new.columns.to_list()
old_columns = df_old.columns.to_list()

new_collumns = set(new_collumns)
old_columns = set(old_columns)

print(len(new_collumns))
print(len(old_columns))

new_only = new_collumns - old_columns

old_only = old_columns - new_collumns

common = new_collumns & old_columns

all_columns = new_collumns | old_columns

different = new_collumns ^ old_columns

print("Новые колонки:", new_only)
print("Удалённые колонки:", old_only)
print("Общие колонки:", common)
print("Различия:", different)

# df_new['Номер Заказа клиента'] = np.where(
#     df_new['Номер Заказа клиента'].isna(),
#     df_new['Регистратор'],
#     df_new['Номер Заказа клиента']
# )

df_new['Номер Заказа клиента'] = df_new['Номер Заказа клиента'].fillna(df_new['Регистратор'])
df_new.drop(columns=['Регистратор'], inplace=True)



print(df_new[['Дата документа','Заказ клиента','Номер Заказа клиента']])