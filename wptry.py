from weasyprint import HTML, CSS 

# Сохранить в домашнюю директорию
HTML('').write_pdf('/Users/pavelustenko/cosmodash/website.pdf')

# # или прямо рядом со скриптом
# HTML('http://127.0.0.1:8050/').write_pdf('website.pdf')