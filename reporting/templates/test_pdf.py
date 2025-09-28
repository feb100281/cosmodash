from weasyprint import HTML

HTML("base_report.html").write_pdf("test.pdf")
print("PDF сгенерирован: test.pdf")