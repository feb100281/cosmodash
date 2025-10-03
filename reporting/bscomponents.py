from enum import StrEnum
from typing import Literal

SIZES = {"xs": "6pt", "sm": "8pt", "md": "10pt", "lg": "12pt", "xl": "14pt"}
PADDINGS = {"xs": "1pt", "sm": "2pt", "md": "3pt", "lg": "4pt", "xl": "5pt"}


class Badge(StrEnum):
    # Rounded badges
    badge_rounded_bg_primary = '<span class="badge rounded-pill bg-primary"{style}>{text}</span>'
    badge_rounded_secondary = '<span class="badge rounded-pill bg-secondary"{style}>{text}</span>'
    badge_rounded_success = '<span class="badge rounded-pill bg-success"{style}>{text}</span>'
    badge_rounded_danger = '<span class="badge rounded-pill bg-danger"{style}>{text}</span>'
    badge_rounded_warning = '<span class="badge rounded-pill bg-warning"{style}>{text}</span>'
    badge_rounded_info = '<span class="badge rounded-pill bg-info"{style}>{text}</span>'
    badge_rounded_light = '<span class="badge rounded-pill bg-light"{style}>{text}</span>'
    badge_rounded_dark = '<span class="badge rounded-pill bg-dark"{style}>{text}</span>'

    # Simple badges
    badge_primary = '<span class="badge bg-primary"{style}>{text}</span>'
    badge_secondary = '<span class="badge bg-secondary"{style}>{text}</span>'
    badge_success = '<span class="badge bg-success"{style}>{text}</span>'
    badge_danger = '<span class="badge bg-danger"{style}>{text}</span>'
    badge_warning = '<span class="badge bg-warning"{style}>{text}</span>'
    badge_info = '<span class="badge bg-info"{style}>{text}</span>'
    badge_light = '<span class="badge bg-light"{style}>{text}</span>'
    badge_dark = '<span class="badge bg-dark"{style}>{text}</span>'

    def text(
        self,
        text: str = 'Add Text',
        va: Literal['top','middle','bottom']='top',
        size: Literal['xs','sm','md','lg','xl'] | None = None,
        padding: Literal['xs','sm','md','lg','xl'] | None = None,
        extra_style: str = 'vertical-align:top'
    ) -> str:
        """
        Генерирует HTML для бейджа.
        :param text: текст внутри бейджа
        :param size: размер шрифта (по SIZES)
        :param padding: padding (по SIZES)
        :param extra_style: любые дополнительные CSS свойства
        :return: корректный HTML
        """
        styles = []
        if size:
            styles.append(f'font-size:{SIZES[size]};')
        if padding:
            styles.append(f'padding:{PADDINGS[padding]};')
        if extra_style:
            styles.append(extra_style.strip(';') + ';')  # добавляем дополнительные стили
        
        style_attr = f' style="{" ".join(styles)}"' if styles else ''
        return self.value.format(style=style_attr, text=text)

class ProgressBar:
    def __init__(self,
                 color: Literal[
                     "text-bg-success",
                     "text-bg-info",
                     "text-bg-warning",
                     "text-bg-danger"
                 ] | None = None,
                 label = '',
                 value = None,    
                 value_max = 100,
                 value_min = 0,
                 units = "%",
                 ):
        self.color = color if color else ""
        self.label = label
        self.value = value if value else 0
        self.units = units
        self.vlabel = f"{value}{units}"
        self.vmin = value_min
        self.vmax = value_max
    @property
    def render(self):
        return f'<div class="progress" role="progressbar" aria-label="{self.label}" aria-valuenow="{self.value}" aria-valuemin="{self.vmin}" aria-valuemax="{self.vmax}"> <div class="progress-bar {self.color}" style="width: {self.value}%">{self.vlabel}</div></div>'

