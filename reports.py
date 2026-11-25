
from fpdf import FPDF


def build_pdf_report(df, filename="student_performance_report.pdf"):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 12, "Student Performance Report", ln=True)

    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"Total Records: {len(df)}", ln=True)

    pdf.ln(5)

    pdf.set_font("Arial", "B", 9)

    for column in df.columns:
        pdf.cell(0, 7, str(column)[:80], ln=True)

        pdf.set_font("Arial", size=9)

        for value in df[column].head(20):
            safe_value = str(value).encode(
                "latin-1", "replace"
            ).decode("latin-1")

            pdf.multi_cell(0, 6, safe_value)

        pdf.ln(2)
        pdf.set_font("Arial", "B", 9)

    pdf.output(filename)

    return filename
