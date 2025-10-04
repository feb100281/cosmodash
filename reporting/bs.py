from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from typing import Literal

BASE_DIR = Path(__file__).parent

TEMPLATE_DIR = BASE_DIR / "templates"
BS_DIR = TEMPLATE_DIR / "bs"
CSS_DIR = TEMPLATE_DIR / 'css'
IMG_DIR = TEMPLATE_DIR / "img"
ICONS_DIR = TEMPLATE_DIR / "icons"
FONT_DIR = TEMPLATE_DIR / "fonts"

brite_ff = f"""
@font-face {{
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/Roboto-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Roboto';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/Roboto-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/Roboto-Bold.ttf") format('truetype');
}}

"""

cerulean_ff = f"""
@font-face {{
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/Roboto-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Roboto';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/Roboto-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/Roboto-Bold.ttf") format('truetype');
}}
"""

litera_ff = f"""
@font-face {{
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/Roboto-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Roboto';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/Roboto-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/Roboto-Bold.ttf") format('truetype');
}}
"""

pulse_ff = f"""
@font-face {{
  font-family: 'NotoSans';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/NotoSans-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'NotoSans';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/NotoSans-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'NotoSans';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/NotoSans-Bold.ttf") format('truetype');
}}
"""

quartz_ff = f"""
@font-face {{
  font-family: 'NotoSans';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/NotoSans-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'NotoSans';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/NotoSans-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'NotoSans';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/NotoSans-Bold.ttf") format('truetype');
}}
"""

slate_ff = f"""
@font-face {{
  font-family: 'NotoSans';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/NotoSans-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'NotoSans';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/NotoSans-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'NotoSans';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/NotoSans-Bold.ttf") format('truetype');
}}
"""

cosmo_ff = f"""
@font-face {{
  font-family: 'Source Sans Pro';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/SourceSans3-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Source Sans Pro';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/SourceSans3-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Source Sans Pro';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/SourceSans3-Bold.ttf") format('truetype');
}}
"""

cyborg_ff = f"""
@font-face {{
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/Roboto-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Roboto';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/Roboto-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/Roboto-Bold.ttf") format('truetype');
}}

"""

darkly_ff = f"""
@font-face {{
  font-family: 'Lato';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/Lato-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Lato';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/Lato-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Lato';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/Lato-Bold.ttf") format('truetype');
}}

"""

flatly_ff = f"""
@font-face {{
  font-family: 'Lato';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/Lato-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Lato';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/Lato-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Lato';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/Lato-Bold.ttf") format('truetype');
}}


"""

journal_ff = f"""

@font-face {{
  font-family: 'News Cycle';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/NewsCycle-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'News Cycle';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}//NewsCycle-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'News Cycle';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/NewsCycle-Bold.ttf") format('truetype');
}}

"""

lumen_ff = f"""
@font-face {{
  font-family: 'Source Sans Pro';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/SourceSans3-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Source Sans Pro';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/SourceSans3-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Source Sans Pro';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/SourceSans3-Bold.ttf") format('truetype');
}}

"""

lux_ff = f"""
@font-face {{
  font-family: 'Nunito Sans';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/Nunito-Regular.ttf") format('truetype');
}}

@font-face {{
  font-family: 'Nunito Sans';
  font-style: italic;
  font-weight: 400;
  src: url("file://{FONT_DIR}/Nunito-LightItalic.ttf") format('truetype');
}}

/* Жирный */
@font-face {{
  font-family: 'Nunito Sans';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/Nunito-Bold.ttf") format('truetype');
}}

"""

materia_ff = f"""
@font-face {{
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/Roboto-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Roboto';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/Roboto-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/Roboto-Bold.ttf") format('truetype');
}}
"""

minty_ff = f"""
@font-face {{
  font-family: 'Montserrat';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/Montserrat-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Montserrat';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/Montserrat-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Montserrat';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/Montserrat-Bold.ttf") format('truetype');
}}


"""

morph_ff = f"""
@font-face {{
  font-family: 'Nunito';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/Nunito-Regular.ttf") format('truetype');
}}

@font-face {{
  font-family: 'Nunito';
  font-style: italic;
  font-weight: 400;
  src: url("file://{FONT_DIR}/Nunito-LightItalic.ttf") format('truetype');
}}

/* Жирный */
@font-face {{
  font-family: 'Nunito';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/Nunito-Bold.ttf") format('truetype');
}}

"""

sandstone_ff = f"""
@font-face {{
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/Roboto-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Roboto';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/Roboto-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/Roboto-Bold.ttf") format('truetype');
}}
"""

simplex_ff = f"""
@font-face {{
  font-family: 'Open Sans';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/OpenSans-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Open Sans';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/OpenSans-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Open Sans';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/OpenSans-Bold.ttf") format('truetype');
}}

"""

solar_ff = f"""
@font-face {{
  font-family: 'Source Sans Pro';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/SourceSans3-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Source Sans Pro';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/SourceSans3-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Source Sans Pro';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/SourceSans3-Bold.ttf") format('truetype');
}}


"""

spacelab_ff = f"""
@font-face {{
  font-family: 'Open Sans';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/OpenSans-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Open Sans';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/OpenSans-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Open Sans';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/OpenSans-Bold.ttf") format('truetype');
}}

"""

superhero_ff = f"""
@font-face {{
  font-family: 'Lato';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/Lato-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Lato';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/Lato-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Lato';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/Lato-Bold.ttf") format('truetype');
}}
"""

united_ff = f"""
@font-face {{
  font-family: 'Ubuntu';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/Ubuntu-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Ubuntu';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/Ubuntu-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Ubuntu';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/Ubuntu-Bold.ttf") format('truetype');
}}
"""

vapor_ff = f"""
@font-face {{
  font-family: 'Lato';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/Lato-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Lato';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/Lato-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Lato';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/Lato-Bold.ttf") format('truetype');
}}

"""

yeti_ff = f"""
@font-face {{
  font-family: 'Open Sans';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/OpenSans-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Open Sans';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/OpenSans-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Open Sans';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/OpenSans-Bold.ttf") format('truetype');
}}
"""

zephyr_ff = f"""
@font-face {{
  font-family: 'Inter';
  font-style: normal;
  font-weight: 400;
  src: url("file://{FONT_DIR}/Inter_28pt-Regular.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Inter';
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url("file://{FONT_DIR}/Inter_28pt-Italic.ttf") format('truetype');
}}
@font-face {{
  font-family: 'Inter';
  font-style: normal;
  font-weight: 700;
  src: url("file://{FONT_DIR}/Inter_28pt-Bold.ttf") format('truetype');
}}

"""

THEMES = {
    "brite":{
        "fonts": brite_ff,
        "fontfamaly": "Roboto",
        "logo":f"file://{IMG_DIR}/logo_black.png"
    },
    "cerulean": {
        "fonts": cerulean_ff,
        "fontfamaly": "Roboto",
        "logo":f"file://{IMG_DIR}/logo_black.png"
    },
    "cosmo":{
        "fonts": cosmo_ff,
        "fontfamaly": "Source Sans Pro",
        "logo":f"file://{IMG_DIR}/logo_black.png"
    },
    "cyborg":{
        "fonts": cyborg_ff,
        "fontfamaly": "Roboto",
        "logo":f"file://{IMG_DIR}/logo_white.png"
    },
    "darkly":{
        "fonts": darkly_ff,
        "fontfamaly": "Lato",
        "logo":f"file://{IMG_DIR}/logo_white.png"
    },
    "flatly":{
        "fonts": flatly_ff,
        "fontfamaly": "Lato",
        "logo":f"file://{IMG_DIR}/logo_white.png"
    },
    "journal":{
      "fonts": journal_ff,
      "fontfamaly": "News Cycle",
      "logo":f"file://{IMG_DIR}/logo_black.png"      
    },
    
    "litera": {
        "fonts": litera_ff,
        "fontfamaly": "Roboto",
        "logo":f"file://{IMG_DIR}/logo_black.png"
    },
    
    "lumen": {
        "fonts" : lumen_ff,
        "fontfamaly": "Source Sans Pro",
        "logo":f"file://{IMG_DIR}/logo_black.png"    
    },
    
    "lux": {
        "fonts":lux_ff,
        "fontfamaly": "Nunito Sans",
        "logo":f"file://{IMG_DIR}/logo_white.png"    
    },
    "materia": {
        "fonts": materia_ff,
        "fontfamaly": "Roboto",
        "logo":f"file://{IMG_DIR}/logo_white.png"
    },
    "minty": {
        "fonts": minty_ff,
        "fontfamaly": "Montserrat",
        "logo":f"file://{IMG_DIR}/logo_black.png"
    },
    "morph": {
        "fonts": morph_ff,
        "fontfamaly": "Nunito",
        "logo":f"file://{IMG_DIR}/logo_black.png"
    },
    "pulse": {
        "fonts": pulse_ff,
        "fontfamaly": "NotoSans",
        "logo":f"file://{IMG_DIR}/logo_white.png"
    },
    "quartz": {
        "fonts": quartz_ff,
        "fontfamaly": "NotoSans",
        "logo":f"file://{IMG_DIR}/logo_black.png"
    },
    "sandstone": {
        "fonts": sandstone_ff,
        "fontfamaly": "Roboto",
        "logo":f"file://{IMG_DIR}/logo_white.png"
    },
    "simplex": {
        "fonts": simplex_ff,
        "fontfamaly": "Open Sans",
        "logo":f"file://{IMG_DIR}/logo_white.png"     
    },
    "slate": {
        "fonts": slate_ff,
        "fontfamaly": "NotoSans",
        "logo":f"file://{IMG_DIR}/logo_white.png"
    },
    "solar": {
        "fonts": solar_ff,
        "fontfamaly": "Source Sans Pro",
        "logo":f"file://{IMG_DIR}/logo_white.png"
    },
    "spacelab": {
        "fonts": spacelab_ff,
        "fontfamaly": "Open Sans",
        "logo":f"file://{IMG_DIR}/logo_white.png"
    },
    "superhero": {
        "fonts": superhero_ff,
        "fontfamaly": "Lato",
        "logo":f"file://{IMG_DIR}/logo_white.png"
    },
    "united": {
        "fonts":united_ff,
        "fontfamaly": "Ubuntu",
        "logo":f"file://{IMG_DIR}/logo_white.png"
    },
    "vapor": {
        "fonts": vapor_ff,
        "fontfamaly": "Lato",
        "logo":f"file://{IMG_DIR}/logo_white.png"
    },
    "yeti": {
        "fonts": yeti_ff,
        "fontfamaly": "Open Sans",
        "logo":f"file://{IMG_DIR}/logo_white.png"
    },
    "zephyr": {
        "fonts": zephyr_ff,
        "fontfamaly": "Inter",
        "logo":f"file://{IMG_DIR}/logo_white.png"
    }    
}



