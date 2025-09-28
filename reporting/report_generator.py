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

FONT_THEME = {
    'font'
}



class ReportComponent:
    def render(self) -> str:
        raise NotImplementedError

class MarkdownBlock(ReportComponent):
    def __init__(self, text, font_size="16px", color_class="text-primary"):
        self.text = text
        self.font_size = font_size
        self.color_class = color_class

    def render(self):
        html = markdown.markdown(self.text, extensions=["tables"])
        return f'<div class="{self.color_class}" style="font-size:{self.font_size};">{html}</div>'

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
    def __init__(self, title, template_name="base.html", bootswatch_theme="litera", date=None):
        self.date = pd.to_datetime(date) if date else pd.Timestamp.today()
        self.date = self.date.strftime("%-d %B %Y")
        self.title = title
        self.template_name = template_name
        self.bootswatch_theme = bootswatch_theme + ".css"

        self.bs = (BS_PATH / "bootstrap.min.css").resolve().as_uri()
        self.bw = (BS_PATH / self.bootswatch_theme).resolve().as_uri()
        self.rs = (BASE_DIR / "styles/report.css").resolve().as_uri()
        self.ap = ASSETS_PATH

        self.components: list[ReportComponent] = []

        self.env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
        self.template = self.env.get_template(self.template_name)

    def add_component(self, component: ReportComponent):
        self.components.append(component)

    def render_report1(self):
        html_content = "\n".join([c.render() for c in self.components])
        return self.template.render(
            bs_path=self.bs,
            bw_path=self.bw,
            rs_path=self.rs,
            title=self.title,
            date=self.date,
            ap_path=self.ap,
            content=html_content,
            bootswatch_theme=self.bootswatch_theme,
        )

    # --- Рендер отчета ---
    def render_report(self):
        html_content = "\n".join(self.content)
        html = self.template.render(
            bs_path = self.bs,
            bw_path = self.bw,
            rs_path = self.rs,
            title=self.title,
            date=self.date,
            ap_path = self.ap,
            content=html_content,
            bootswatch_theme=self.bootswatch_theme,
            
        )
        return html

    def switch_them(self,theme):
        self.bootswatch_theme = theme + ".css"
        self.bw = (BS_PATH / self.bootswatch_theme).resolve().as_uri()
    
    # --- PDF через WeasyPrint ---
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
    "Тестовый отчет"
)

text = """
### Это параграф основного текста

Здесь будем давать комментарии. А, теперь понятно 😅 — проблема в том, что @bottom-center не может выходить за пределы @page, и margin-bottom просто сдвигает контент внутри доступной области, но нижняя граница страницы остаётся нулевой (0cm). То есть нельзя просто "поднять" блок за пределы нижнего края через margin.

- Нижний отступ страницы **(margin-bottom)** создаёт место, куда @bottom-center может вставиться.
- В примере 2cm — *это расстояние* от нижнего края страницы до номера.


"""

para1 = MarkdownBlock(text,font_size='12px')


rg.add_component(para1)


print(rg.render_report1())