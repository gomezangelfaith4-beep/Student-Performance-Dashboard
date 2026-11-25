from typing import Any

import pandas as pd
from fpdf import FPDF


REPORT_COLUMNS = (
    ("student_code", "Student"),
    ("year_level", "Year"),
    ("section", "Section"),
    ("general_average", "Average"),
    ("attendance_rate", "Attendance"),
    ("failed_subjects", "Failed"),
    ("risk_level", "Risk"),
    ("recommended_action", "Suggested support"),
)


def _pdf_text(value: Any, max_length: int = 72) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text.encode("latin-1", errors="replace").decode("latin-1")[:max_length]


def build_pdf_report(df: pd.DataFrame) -> bytes:
    """Build a compact PDF summary in memory for Streamlit downloads."""
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.set_margins(10, 12, 10)
    pdf.add_page()
    pdf.set_title("Student performance report")
    pdf.set_author("Student Performance Dashboard")

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Student performance report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(
        0,
        7,
        f"Records included: {len(df):,}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    if "general_average" in df.columns and not df.empty:
        average = pd.to_numeric(df["general_average"], errors="coerce").mean()
        if pd.notna(average):
            pdf.cell(
                0,
                7,
                f"Average general grade: {average:.1f}",
                new_x="LMARGIN",
                new_y="NEXT",
            )
    if "risk_level" in df.columns:
        high_risk = int(
            df["risk_level"].astype("string").str.casefold().eq("high").sum()
        )
        pdf.cell(
            0,
            7,
            f"High-risk records: {high_risk:,}",
            new_x="LMARGIN",
            new_y="NEXT",
        )
    pdf.ln(3)

    columns = [(key, label) for key, label in REPORT_COLUMNS if key in df.columns]
    if columns:
        base_widths = [30, 15, 20, 23, 24, 16, 20, 129]
        selected_widths = base_widths[: len(columns)]
        scale = 277 / sum(selected_widths)
        widths = [width * scale for width in selected_widths]
        header_height = 8
        row_height = 7

        def draw_header() -> None:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(230, 239, 250)
            for (_, label), width in zip(columns, widths):
                pdf.cell(width, header_height, _pdf_text(label), border=1, fill=True)
            pdf.ln(header_height)
            pdf.set_font("Helvetica", size=7)

        draw_header()
        for _, row in df.iterrows():
            if pdf.get_y() + row_height > pdf.page_break_trigger:
                pdf.add_page()
                draw_header()
            for (key, _), width in zip(columns, widths):
                value = row.get(key, "")
                if key in {"general_average", "attendance_rate"}:
                    number = pd.to_numeric(value, errors="coerce")
                    value = f"{number:.1f}" if pd.notna(number) else ""
                pdf.cell(width, row_height, _pdf_text(value, 34), border=1)
            pdf.ln(row_height)

    output = pdf.output()
    return bytes(output)
