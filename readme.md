# 📊 Cosmodash — Руководство по установке и работе

## 🚀 Установка проекта

1. Перейти в папку, где будут храниться проекты:
   ```bash
   cd /path/to/projects
   ```

2. Проверить путь:
   ```bash
   pwd
   ```

3. Клонировать проект:
   ```bash
   git clone https://github.com/feb100281/cosmodash.git
   ```

4. Перейти в папку проекта:
   ```bash
   cd cosmodash
   ```

5. Создать виртуальное окружение (Python 3.12):
   ```bash
   python3.12 -m venv venv
   ```

6. Активировать виртуальное окружение:
   ```bash
   source venv/bin/activate
   ```

7. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```

8. Создаем файл окружения и копируем в него переменные из присланного env.txt:
   ```bash
   touch .env
   ```

---

## 📂 Структура проекта

```plaintext
cosmodash/
│
├── app.py              # точка входа (локальный запуск)
├── wsgi.py             # серверная версия
├── run_dash.py         # тестовый запуск
├── components.py       # классы компонентов для реюзабилити
├── data.py             # обработчик данных
├── update_redis.py     # обновление Redis на сервере
│
├── assets/             # css, js, png для проекта
├── static/assets/      # копия (не используется)
└── pages/              # файлы страниц (рабочая директория)
```

---

## 🛠️ Workflow

- Разработка ведётся в папке **`pages/`** (открывать в VS Code).  
- Для проверки локально - запуск приложения:
  ```bash
  python app.py
  ```

### Git-flow

```bash
git switch feature/page_name     # пример: page_stores или page_cat
git add .
git commit -m "Комментарий"
git push origin feature/page_name
```

---

## 🗄️ Работа с данными

- Все данные берутся из **MySQL на сервере**  
- Redis подключается к серверу (локально запускать не нужно)  
- Обновление БД выполняется через админку сайта  
- Обновление Redis для отчетов:
  ```bash
  python update_redis.py
  ```

---

## 🔑 Структура хранения в Redis

- Данные разбиты на **чанки по месяцам (end of month, eom)**  
- Каждый чанк — колонка `pandas.Series`  

**Пример ключей:**

```
mydf:brend:2025-09-30
mydf:brend_origin:2025-09-30
mydf:subcat_id:2025-09-30
mydf:subcat:2025-09-30
mydf:store:2025-09-30
mydf:chanel:2025-09-30
mydf:store_gr_name:2025-09-30
mydf:store_region:2025-09-30
```

Где:  
- `mydf` → идентификатор набора данных  
- `store_gr_name` → название колонки  
- `2025-09-30` → месяц (eom)  

---

## 📦 Методы работы с данными (`data.py`)

### 1. `load_columns_df(columns, start_eom, end_eom)`

Загрузка данных по диапазону месяцев.

**Параметры:**
- `columns` — список колонок  
- `start_eom` — начальный месяц (`pd.to_datetime`)  
- `end_eom` — конечный месяц (`pd.to_datetime`)  

**Пример:**
```python
from data import load_columns_df
import pandas as pd

cols = ["eom", "agent_name", "amount"]
start = pd.to_datetime("2025-01-31")
end = pd.to_datetime("2025-08-31")

df = load_columns_df(cols, start, end)
```

---

### 2. `load_columns_dates(columns, dates)`

Загрузка данных за конкретные месяцы.

**Параметры:**
- `columns` — список колонок  
- `dates` — список дат (обязательно список, даже если одна дата)  

**Пример:**
```python
from data import load_columns_dates

cols = ["eom", "store_gr_name", "dt", "store_gr_name_dt_ytd"]
dates = ["2024-08-31", "2025-08-31"]

df = load_columns_dates(cols, dates)
```

---

## 📝 Примечания

- Авто-офсет на конец месяца пока не реализован → указываем вручную  
  ```python
  + pd.offsets.MonthEnd(0)
  ```
- `dates` всегда список, даже для одной даты:
  ```python
  ["2024-04-30"]
  ```

---

## 🎨 Код-стайлинг
# 📐 Рекомендации по структуре кода в Dash + DMC

1. **Объединяем компоненты страницы в классы**
   Каждая страница — это отдельный класс (например, `StorePage`).  
   Такой подход:  
   - изолирует логику,  
   - не мешает другим страницам,  
   - облегчает тестирование и поддержку.  

2. **В `__init__` класса прописываем только id компонентов**
   - Данные и компоненты не создаём сразу.  
   - Id храним в виде dict:  
     ```python
     {"type": "bar_chart_for_store_very_unique", "index": "1"}
     ```
   - Это исключает пересечения и упрощает колбэки.  

3. **Компоненты создаём методами класса**
   - Каждый метод отвечает за конкретный компонент (`make_table`, `make_chart`, `make_text`).  
   - Это уменьшает дублирование и упрощает переиспользование.  

4. **Layout и callbacks внутри класса**
   - `layout()` возвращает всю страницу.  
   - `register_callbacks(app)` регистрирует колбэки для этой страницы.  
   Это делает код самодостаточным и переносимым и не мучаемся.  

5. **Все страницы собираем в одном модуле**
   - Например, `mypage.py` содержит все классы страници.  
   - В `app.py` просто импортируем страницы и добавляем их в приложение.  

6. **Повторяющиеся компоненты переопределяем в `components.py`**
   - Например: `Monthslider`, `RadioOtionGroup`.  
   - Убирает дублирование кода и делает UI единообразным.  

7. **JS-функции для DMC и Ag-Grid — в отдельных файлах**
   - Думаю проще их в разные файлы засунуть custom.js и dag.js  
   - Подключаются как `assets/*.js`.  
   - Пишем комменты к каждой функции

8. **CSS также разделяем на файлы**
   - Например: `theme.css`, `charts.css`, `tables.css`.  
   - Повышает читаемость и упрощает поддержку.  Но это желательно. Пока все в одной помойке

9. **При установке нового пакета обновляем `requirements.txt`**
   - После `pip install ...` выполняем:
     ```bash
     pip freeze > requirements.txt
     ```
   - Будет гиммор с git но зато все пакеты везде работчии. 

---


# ❌ Пример плохого кода

```python
df: pd.DataFrame = load_df_from_redis(
    argument
)  # -> df постоянно находиться в памяти - так не делаем

data = df.to_dict("records")  # -> дублируемый df в памяти как JSON

bar_chart = dmc.BarChart(
    dataKey="eom",
    data=data,  # -> здесь третий раз хранятся одни и те же данные
    series=[{"name": "eom", "color": "blue"}, {"name": "revenue", "color": "cyan"}],
    id="bar_chart",
)

text = dmc.Text("Здесь отображается дата", id="data_display")

layout = dmc.MantineProvider([dmc.Container([bar_chart, text], id="page_conteiner")])

def register_callbacks(app):
    @app.callback(
        Output("data_display", "children"),  # -> нужно помнить все id вручную
        Input("bar_chart", "clickdata"),
    )
    def show_data(clickdata):
        return clickdata
```

### 🚫 Минусы
- **Данные дублируются** в памяти трижды.  
- Жёстко прописанные `id`, которые легко перепутать.  
- Нет инкапсуляции — всё свалено в один файл.  
- Такой код тяжело переносить и масштабировать.  

---

# ✅ Пример хорошего кода

```python
import dash_mantine_components as dmc
from dash import Output, Input, MATCH
import pandas as pd

class BarChartPage:
    def __init__(self, argument=None):
        """
        Класс страницы с графиком.
        :param argument: аргумент для загрузки данных (например, ключ для Redis).
        """
        self.argument = argument

        # Определяем id компонентов динамически
        self.bar_chart_id = {"type": "bar_chart_for_mypage_unique", "index": "1"}
        self.bar_text_id = {"type": "text_for_bar_chart_for_mypage_unique", "index": "1"}
        self.page_container_id = {"type": "page_container_for_mypage_unique", "index": "1"}

    # --- Методы для компонентов ---

    def create_page_container(self):
        """Контейнер-заглушка (если нет аргументов)."""
        return dmc.Container(["Нет данных"], id=self.page_container_id, fluid=True)

    def make_bar_chart(self):
        """Создаёт график на основе данных."""
        if not self.argument:
            return None

        df: pd.DataFrame = load_df_from_redis(self.argument)
        data = df.to_dict("records")

        return dmc.BarChart(
            dataKey="eom",
            data=data,  # ✅ данные хранятся только здесь!
            series=[
                {"name": "eom", "color": "blue"},
                {"name": "revenue", "color": "cyan"},
            ],
            id=self.bar_chart_id,
        )

    def make_bar_chart_text(self):
        """Подпись к графику (обновляется по клику)."""
        return dmc.Text("Кликните по графику", id=self.bar_text_id)

    # --- Layout страницы ---

    def layout(self):
        """Возвращает layout страницы."""
        if not self.argument:
            return self.create_page_container()

        return dmc.Container(
            [
                dmc.Title("Отчёт по выручке", order=2, c="blue"),
                dmc.Space(h=10),
                self.make_bar_chart(),
                dmc.Space(h=10),
                self.make_bar_chart_text(),
            ]
        )

    # --- Callbacks ---

    def register_callbacks(self, app):
        """Регистрирует все колбэки страницы."""

        bar_chart = self.bar_chart_id["type"]
        bar_chart_text = self.bar_text_id["type"]

        @app.callback(
            Output({"type": bar_chart_text, "index": MATCH}, "children"),
            Input({"type": bar_chart, "index": MATCH}, "clickdata"),
        )
        def show_data(clickdata):
            return f"Вы кликнули: {clickdata}"
```

---

# 🚀 Как использовать в приложении

```python
from dash import Dash
import dash_mantine_components as dmc
from bar_chart_page import BarChartPage

# Инициализируем Dash
app = Dash(__name__, suppress_callback_exceptions=True)

# Создаём страницу
bar_page = BarChartPage(argument="revenue_key")

# Регистрируем её колбэки
bar_page.register_callbacks(app)

# Layout приложения
app.layout = dmc.MantineProvider(
    dmc.Container(
        [
            dmc.Header("Demo Dashboard", height=60),
            bar_page.layout(),
        ]
    )
)

if __name__ == "__main__":
    app.run_server(debug=True)
```

---

# 📌 Почему это лучше
1. **Нет дублирования данных** — датафрейм живёт только внутри метода.  
2. **Удобные id** — не надо помнить строки, всё через dict.  
3. **Класс = модульность** — каждая страница изолирована.  
4. **Callbacks внутри страницы** — легко переносить между проектами.  