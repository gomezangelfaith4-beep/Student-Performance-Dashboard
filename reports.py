from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def build_pdf_report(df):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=40, bottomMargin=36)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCustom", parent=styles["Title"], textColor=colors.HexColor("#243b80"), fontSize=18, leading=22)
    story = [Paragraph("Student Performance Monitoring Report", title), Spacer(1,8),
             Paragraph("Synthetic demonstration data only — not actual school records.", styles["Italic"]), Spacer(1,12)]
    total = len(df)
    avg = float(df["general_average"].mean()) if total else 0
    attendance = float(df["attendance_rate"].mean()) if total else 0
    high = int((df["risk_level"]=="High").sum()) if total else 0
    story.append(Paragraph(f"Students included: {total} &nbsp;&nbsp; Average grade: {avg:.2f} &nbsp;&nbsp; Average attendance: {attendance:.2f}% &nbsp;&nbsp; High risk: {high}", styles["BodyText"]))
    story.append(Spacer(1,14))
    counts = df["risk_level"].value_counts().to_dict()
    data = [["Risk level","Number of students"],["Low",counts.get("Low",0)],["Moderate",counts.get("Moderate",0)],["High",counts.get("High",0)]]
    table = Table(data, colWidths=[220,180])
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#243b80")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                               ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#d6d9e2")),("PADDING",(0,0),(-1,-1),8)]))
    story.append(table)
    story.append(Spacer(1,14))
    story.append(Paragraph("This report is a decision-support summary. Human review and authorized school data are required before real student interventions.", styles["BodyText"]))
    doc.build(story)
    return buffer.getvalue()
