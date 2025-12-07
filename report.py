import os
from reportlab.pdfgen import canvas

REPORTS_DIR = "reports"

def generate_report(metrics):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "rapor.pdf")
    
    c = canvas.Canvas(report_path)
    c.setFont("Helvetica", 12)

    c.drawString(50, 800, "Derin Pekistirmeli Ogrenme ile Finansal Portfoy Yonetimi")
    c.drawString(50, 785, "Model Performans Raporu")

    y = 750
    for k, v in metrics.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 20

    c.drawString(50, y-20, "Grafikler plots/ klasorunde kaydedildi.")
    c.showPage()
    c.save()
    
    print(f"Rapor olusturuldu: {report_path}")
