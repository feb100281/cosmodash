import pandas as pd
import numpy as np
import base64
import plotly.io as pio
import markdown
from jinja2 import Environment, FileSystemLoader
import os
from pathlib import Path
from typing import Literal
from dash import html

from .bs import THEMES
from .icons import Streamline, ColorEmoji, Solar
from .bscomponents import Badge, ProgressBar, ProgressBarRelative

BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "templates"
BS_DIR = TEMPLATE_DIR / "bs"
CSS_DIR = TEMPLATE_DIR / "css"
IMG_DIR = TEMPLATE_DIR / "img"
ICONS_DIR = TEMPLATE_DIR / "icons"
FONT_DIR = TEMPLATE_DIR / "fonts"

SIZES = {"xs": "8pt", "sm": "10pt", "md": "12pt", "lg": "14pt", "xl": "16pt"}



class Icon:
    Streamline = Streamline
    ColorEmoji = ColorEmoji
    Solar = Solar
    
class BS:
    Badge = Badge
    ProgressBar = ProgressBar
    ProgressBarRelative = ProgressBarRelative


class ReportComponent:
    def render(self) -> str:
        raise NotImplementedError


class MarkdownBlock(ReportComponent):
    def __init__(
        self,
        text,
        id_element=None,
        tow_columns=False,
        font_size: Literal["xs", "sm", "md", "lg", "xl"] | None = None,
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
        ] = "text-body",
    ):

        self.text = text
        self.id = f'id = "{id_element}"' if id_element else ""
        self.font_size = f'style="font-size: {SIZES[font_size]};"' if font_size else ""
        self.color_class = color_class
        self.two_columns = "two-columns" if tow_columns else ""

    def render(self):
        html = markdown.markdown(self.text, extensions=["tables"])
        html = html.replace("<table>", '<table class="table table-hover">')

        return f'<section class="{self.color_class} {self.two_columns}" {self.id} {self.font_size} >{html}</section>'


class DataTable(ReportComponent):
    def __init__(self, df, font_size="14px", table_classes=None):
        self.df = df
        self.font_size = font_size
        self.table_classes = (
            table_classes
            or "table table-striped table-hover table-sm align-middle w-auto"
        )

    def render(self):
        html = self.df.to_html(classes=self.table_classes, border=0, escape=False)
        html = html.replace("<th>", '<th class="table-light text-center">')
        html = html.replace(
            "<td>", f'<td class="text-center" style="font-size:{self.font_size};">'
        )
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
        template_css="base_report.css",
        pdf_name = 'report',
        bootswatch_theme: Literal[
            "brite",
            "cerulean",
            "cosmo",
            "cyborg",
            "darkly",
            "flatly",
            "journal",
            "litera",
            "lumen",
            "lux",
            "materia",
            "minty",
            "morph",
            "pulse",
            "quartz",
            "sandstone",
            "simplex",
            "slate",
            "solar",
            "spacelab",
            "superhero",
            "united",
            "vapor",
            "yeti",
            "zephyr",
        ] = "cosmo",
        date=None,
        fontsize: Literal["xs", "sm", "md", "lg", "xl"] = "md",
    ):
        self.date = pd.to_datetime(date) if date else pd.Timestamp.today()
        self.date = self.date.strftime("%-d %B %Y")
        self.title = title
        self.template_name = template_name
        self.bootswatch_theme = bootswatch_theme
        self.bootswatch_theme_link = "file://" + str(BS_DIR / bootswatch_theme) + ".css"
        self.fontsize = SIZES[fontsize]
        self.components: list[ReportComponent] = []
        self.env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        self.template = self.env.get_template(self.template_name)
        self.template_css = (
            f'<link rel="stylesheet" href="file://{TEMPLATE_DIR}/{template_css}">'
        )
        self.pdf_name = pdf_name

    def add_component(self, component: ReportComponent):
        self.components.append(component)

    def render_report(self):
        html_content = "\n".join([c.render() for c in self.components])
        return self.template.render(
            title=self.title,
            date=self.date,
            fontface=THEMES[self.bootswatch_theme]["fonts"],
            fontfamily=THEMES[self.bootswatch_theme]["fontfamaly"],
            logo=THEMES[self.bootswatch_theme]["logo"],
            content=html_content,
            bootswatch_theme=self.bootswatch_theme_link,
            fontsize=self.fontsize,
            template_css=self.template_css,
        )

    # --- Рендер отчета ---

    def to_pdf(self, filename="report.pdf"):
        from weasyprint import HTML

        html = self.render_report()
        HTML(string=html, base_url=os.getcwd()).write_pdf(filename)
        return filename

    # --- Dash download (возвращает контент для dcc.Download) ---
    def for_dash_download(self, as_pdf=True):
        if as_pdf:
            from weasyprint import HTML

            html = self.render_report()
            pdf_bytes = HTML(string=html, base_url=os.getcwd()).write_pdf()
            b64 = base64.b64encode(pdf_bytes).decode("utf-8")
            return dict(
                content=b64, filename=f"{self.pdf_name}.pdf", type="application/pdf", base64=True
            )
        else:
            html = self.render_report()
            b64 = base64.b64encode(html.encode("utf-8")).decode("utf-8")
            return dict(
                content=b64, filename=f"{self.pdf_name}.html", type="text/html", base64=True
            )
    
    def return_embet(self):
        dl = self.for_dash_download(as_pdf=True)
        b64 = dl["content"]
        src = "data:application/pdf;base64," + b64
        return html.Embed(
                            id="pdf_file",
                            src=src,
                            type="application/pdf",
                            style={"width": "100%", "height": "80vh"},
                        )
    
    def return_iframe(self):
        dl = self.for_dash_download(as_pdf=True)
        b64 = dl["content"]
        src = "data:application/pdf;base64," + b64
        return html.Iframe(
            src=src,
            style={"width": "100%", "height": "80vh"},
        )
    
    def change_theme(self,theme):
        self.bootswatch_theme = theme
        self.bootswatch_theme_link = "file://" + str(BS_DIR / theme) + ".css"
        
    def change_fontsize(self,fontsize):
        self.fontsize = SIZES[fontsize]


# rg = ReportGenerator(title="Тестовый отчет", bootswatch_theme='yeti', fontsize='xl')

# df = pd.DataFrame(
#     {
#         "Магазин": [f"Магазин {i+1}" for i in range(6)],
#         "Выручка": np.random.randint(1000, 5000, size=6),
#         "Отклонение": np.round(np.random.uniform(-500, 500, size=6), 2),
#     }
# )

# df["Отклонение"] = np.where(
#     df["Отклонение"] > 0,
#     Icon.Streamline.arrow_double_up.render(size='sm') + " " + df["Отклонение"].astype(str),
#     Icon.Streamline.arrow_double_down_1.render() + " " + df["Отклонение"].astype(str),
# )

# md_table = df.to_markdown(index=False, colalign=("left", "center", "left"))


# text = f"""
# # Заголовок 1 уровня

# <svg width="12" height="12" viewBox="0 0 512 512" fill="red" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle;">
#     <path d="M256 8C119 8 8 119 8 256s111 248 248 248 248-111 248-248S393 8 256 8zm0 48c110.3 0 200 89.7 200 200S366.3 456 256 456 56 366.3 56 256 145.7 56 256 56zm0 80c-11 0-20 9-20 20v100c0 11 9 20 20 20s20-9 20-20V156c0-11-9-20-20-20zm0 200c-13.3 0-24 10.7-24 24s10.7 24 24 24 24-10.7 24-24-10.7-24-24-24z"/>
#   </svg> Главная тема

# **Это** пример *текста* для теста отчета. Здесь **можно** писать _любой_ текст, он `нужен` только для **проверки** рендеринга **Markdown в HTML**.

# ## Заголовок 2 уровня

# - Пункт списка 1
# - Пункт списка 2 {BS.Badge.badge_rounded_success.text(f'{Icon.ColorEmoji.beer_mug.render()} для пробы')}
#   - Вложенный ***пункт 2.1***
#   - Вложенный *пункт 2.2*
# - Пункт списка 3

# ### Заголовок 3 уровня

# 1. Нумерованный пункт 1
# 2. Нумерованный пункт 2
# 3. Нумерованный пункт 3

# {BS.Badge.badge_danger.text(f'{Icon.ColorEmoji.beer_mug.render()} для пробы','xs')} {BS.Badge.badge_danger.text(f'{Icon.ColorEmoji.beer_mug.render()} для пробы')}

# ### Тест иконок для проверки алгоритма

# {Icon.Streamline.attachment.render()} - Первый размер иконки

# {Icon.Streamline.analytics_bars_3d.render()} - Большой размер иконки

# {Icon.Streamline.ghost.render()}  - Маленький размер

# {Icon.Streamline.validation_1.render()}  - Огромный размер

# {Icon.Streamline.analytics_board_graph_line.render('xs')}  - Малый размер


# {Icon.Streamline.dangerous_chemical_lab.render(size=50)} - same shit

# ### Таблица с иконками
# ---
# {md_table}



# {Icon.Solar.bar_chair_outline.render('md','cyan')} - это стул


# Это прогрес бар {BS.ProgressBar(value=25,label='Магазин 1').render}


# ## Таблица для теста

# | Имя       | Возраст | Город        |
# |-----------|--------|--------------|
# | <svg width="16" height="16" viewBox="0 0 512 512" fill="red" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle;"><path d="M256 8C119 8 8 119 8 256s111 248 248 248 248-111 248-248S393 8 256 8zm0 48c110.3 0 200 89.7 200 200S366.3 456 256 456 56 366.3 56 256 145.7 56 256 56zm0 80c-11 0-20 9-20 20v100c0 11 9 20 20 20s20-9 20-20V156c0-11-9-20-20-20zm0 200c-13.3 0-24 10.7-24 24s10.7 24 24 24 24-10.7 24-24-10.7-24-24-24z"/></svg>Иван      | {BS.ProgressBar(value=25,label='Магазин 1').render}     | Москва       |
# | Мария     | {BS.ProgressBar(value=30,label='Магазин 1').render}     | Санкт-Петербург |
# | Алексей   | {BS.ProgressBar(value=40,label='Магазин 1').render}     | Новосибирск  |
# | Ольга     | {BS.ProgressBar(value=25,label='Магазин 1').render}     | Екатеринбург |
# | Сергей    | {BS.ProgressBar(value=25,label='Магазин 1').render}     | Казань       |


# **Дополнительный текст** для проверки *переносов* и длинных строк. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

# - Еще один список для проверки:
#   - Подпункт А
#   - Подпункт Б
#   - Подпункт В


# Заключение: этот текст нужен только для проверки генерации отчета из Markdown в HTML с таблицами, списками и разными заголовками.


# """

# para1 = MarkdownBlock(text,tow_columns=True)
# para2 = MarkdownBlock(text=text, color_class="text-danger")


# rg.add_component(para1)
# rg.add_component(para2)


# html_content = rg.render_report()
# from weasyprint import HTML

# # Сохраняем как файл
# html_file = Path("test.html")
# html_file.write_text(html_content, encoding="utf-8")
# print(f"HTML сохранён: {html_file.resolve()}")

# # Дополнительно — сразу делаем PDF
# HTML(string=html_content, base_url=str(html_file.parent)).write_pdf("test.pdf")
# print("PDF сгенерирован: test.pdf")
