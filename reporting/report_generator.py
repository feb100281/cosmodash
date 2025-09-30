import pandas as pd
import base64
import plotly.io as pio
import markdown
from jinja2 import Environment, FileSystemLoader
import os
from pathlib import Path
from typing import Literal

BASE_DIR = Path(__file__).parent

TEMPLATE_DIR = BASE_DIR / "templates"
BS_DIR = TEMPLATE_DIR / "bs"
CSS_DIR = TEMPLATE_DIR / 'css'
ING_DIR = TEMPLATE_DIR / "img"
ICONS_DIR = TEMPLATE_DIR / "icons"
FONT_DIR = TEMPLATE_DIR / "fonts"

SIZES = {
    'xs':'8pt',
    'sm':'10pt',
    'ns':'12pt',
    'lg':'14pt',
    'xl':'16pt'
}

from bs import THEMES

THEMS_LIST = ['brite', 'cerulean', 'cosmo', 'cyborg', 'darkly', 'flatly', 'litera', 'lumen', 'lux', 'materia', 'minty', 'morph', 'pulse', 'quartz', 'sandstone', 'simplex', 'slate', 'solar', 'spacelab', 'superhero', 'united', 'vapor', 'yeti', 'zephyr']


class ReportComponent:
    def render(self) -> str:
        raise NotImplementedError

class MarkdownBlock(ReportComponent):
    def __init__(  self, 
                    text, 
                    id_element = None,
                    font_size: Literal['xs','sm','ns','lg','xl']='ns',                  
                    color_class: Literal[
                        "text-primary",
                        "text-secondary",
                        "text-success",
                        "text-danger",
                        "text-warning",
                        "text-info",
                        "text-body",
                        "text-body-secondary",
                        "text-body-tertiary",
                    ] = "text-body"
                    ):
                 
        self.text = text
        self.id = f'id = "{id_element}"' if id_element else ''
        self.font_size = SIZES[font_size]
        self.color_class = color_class

    def render(self):
        html = markdown.markdown(self.text, extensions=["tables"])
        return f'<div class="{self.color_class}" {self.id} style="font-size:{self.font_size};">{html}</div>'

class DataTable(ReportComponent):
    def __init__(self, df, font_size="14px", table_classes=None):
        self.df = df
        self.font_size = font_size
        self.table_classes = table_classes or "table table-striped table-hover table-sm align-middle w-auto"

    def render(self):
        html = self.df.to_html(classes=self.table_classes, border=0, escape=False)
        html = html.replace("<th>", '<th class="table-light text-center">')
        html = html.replace("<td>", f'<td class="text-center" style="font-size:{self.font_size};">')
        return f'<div class="table-responsive">{html}</div>'

class PlotlyFigure(ReportComponent):
    def __init__(self, fig, format="png", css_class="img-fluid"):
        self.fig = fig
        self.format = format
        self.css_class = css_class

    def render(self):
        img_bytes = pio.to_image(self.fig, format=self.format)
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f'<img src="data:image/{self.format};base64,{b64}" class="{self.css_class}"/>'
    
class ReportGenerator:
    def __init__(
        self, 
        title, 
        template_name="base_report.html", 
        bootswatch_theme: Literal['brite', 'cerulean', 'cosmo', 'cyborg', 'darkly', 'flatly', 'litera', 'lumen', 'lux', 'materia', 'minty', 'morph', 'pulse', 'quartz', 'sandstone', 'simplex', 'slate', 'solar', 'spacelab', 'superhero', 'united', 'vapor', 'yeti', 'zephyr'] = "spacelab", 
        date=None
        ):
        self.date = pd.to_datetime(date) if date else pd.Timestamp.today()
        self.date = self.date.strftime("%-d %B %Y")
        self.title = title
        self.template_name = template_name
        self.bootswatch_theme = bootswatch_theme
        self.bootswatch_theme_link = str(BS_DIR / bootswatch_theme) + ".css"

        self.components: list[ReportComponent] = []

        self.env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        self.template = self.env.get_template(self.template_name)

    def add_component(self, component: ReportComponent):
        self.components.append(component)

    def render_report(self):
        html_content = "\n".join([c.render() for c in self.components])
        return self.template.render(
            
            title=self.title,
            date=self.date,
            fontface = THEMES[self.bootswatch_theme]['fonts'],
            content=html_content,
            bootswatch_theme=self.bootswatch_theme_link,
        )

    # --- Рендер отчета ---
    
    def to_pdf(self, filename="report.pdf"):
        from weasyprint import HTML
        html = self.render_report1()
        HTML(string=html, base_url=os.getcwd()).write_pdf(filename)
        return filename

    # --- Dash download (возвращает контент для dcc.Download) ---
    def for_dash_download(self, as_pdf=True):
        if as_pdf:
            from weasyprint import HTML
            html = self.render_report1()
            pdf_bytes = HTML(string=html, base_url=os.getcwd()).write_pdf()
            b64 = base64.b64encode(pdf_bytes).decode("utf-8")
            return dict(content=b64, filename="report.pdf", type="application/pdf", base64=True)
        else:
            html = self.render_report1()
            b64 = base64.b64encode(html.encode("utf-8")).decode("utf-8")
            return dict(content=b64, filename="report.html", type="text/html", base64=True)


rg = ReportGenerator(
    title='Тестовый отчет',
    bootswatch_theme='lux'
    
)

text = """
# Заголовок 1 уровня

Это пример *текста* для теста отчета. Здесь **можно** писать _любой_ текст, он `нужен` только для ***проверки*** рендеринга Markdown в HTML.

## Заголовок 2 уровня

- Пункт списка 1
- Пункт списка 2
  - Вложенный ***пункт 2.1***
  - Вложенный *пункт 2.2*
- Пункт списка 3

### Заголовок 3 уровня

1. Нумерованный пункт 1
2. Нумерованный пункт 2
3. Нумерованный пункт 3

---

## Таблица для теста

| Имя       | Возраст | Город        |
|-----------|--------|--------------|
| Иван      | 25     | Москва       |
| Мария     | 30     | Санкт-Петербург |
| Алексей   | 22     | Новосибирск  |
| Ольга     | 28     | Екатеринбург |
| Сергей    | 35     | Казань       |

---

Дополнительный текст для проверки переносов и длинных строк. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

- Еще один список для проверки:
  - Подпункт А
  - Подпункт Б
  - Подпункт В

---

Заключение: этот текст нужен только для проверки генерации отчета из Markdown в HTML с таблицами, списками и разными заголовками.


"""

para1 = MarkdownBlock(text)
para2 = MarkdownBlock(text=text,font_size='ns',color_class='text-danger')


rg.add_component(para1)
rg.add_component(para2)


a = rg.render_report()
print(a)
from weasyprint import HTML
html_content = rg.render_report()
HTML(string=html_content, base_url=str(Path(__file__).parent)).write_pdf("test.pdf")