"""PDF report generation service using ReportLab."""
import os
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

NAVY   = colors.HexColor("#1a4a7a")
TEAL   = colors.HexColor("#0d9488")
GREEN  = colors.HexColor("#10b981")
AMBER  = colors.HexColor("#f59e0b")
RED    = colors.HexColor("#ef4444")
LIGHT  = colors.HexColor("#f0f4f8")

def _risk_color(risk: str):
    return {"Low": GREEN, "Moderate": AMBER, "High": RED}.get(risk, AMBER)


def generate_health_report(user_name: str, user_email: str, predictions: list,
                            profile: dict = None) -> bytes:
    """Generate a PDF health report and return as bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    # â”€â”€ Title â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    title_style = ParagraphStyle("title", parent=styles["Title"],
                                  textColor=NAVY, fontSize=22, spaceAfter=6, alignment=TA_CENTER)
    sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                  textColor=TEAL, fontSize=11, spaceAfter=4, alignment=TA_CENTER)
    story.append(Paragraph("ðŸ¥ HealthGuard AI", title_style))
    story.append(Paragraph("Health Risk Assessment Report", sub_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y %I:%M %p')}", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=TEAL))
    story.append(Spacer(1, 0.4*cm))

    # â”€â”€ Patient info â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=NAVY, fontSize=13, spaceBefore=8)
    normal = styles["Normal"]
    story.append(Paragraph("Patient Information", h2))
    info_data = [["Name:", user_name], ["Email:", user_email]]
    if profile:
        info_data += [
            ["Age:", str(profile.get("age", "—"))],
            ["Gender:", str(profile.get("gender", "—"))],
            ["BMI:", str(profile.get("bmi", "—"))],
            ["Smoking:", "Yes" if profile.get("smoking") else "No"],
            ["Physical Activity:", str(profile.get("physical_activity", "—"))],
        ]
    t = Table(info_data, colWidths=[4*cm, 10*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), LIGHT),
        ("TEXTCOLOR",  (0,0), (0,-1), NAVY),
        ("FONTNAME",   (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, LIGHT]),
        ("GRID",       (0,0), (-1,-1), 0.3, colors.lightgrey),
        ("LEFTPADDING",(0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # â”€â”€ Prediction Results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Paragraph("Disease Risk Assessment Results", h2))

    disease_map = {"diabetes": "Diabetes", "heart": "Heart Disease",
                   "hypertension": "Hypertension"}

    if not predictions:
        story.append(Paragraph("No assessments recorded yet.", normal))
    else:
        # Group by disease, take latest
        seen = {}
        for p in sorted(predictions, key=lambda x: x.get("prediction_date",""), reverse=True):
            d = p.get("disease_type","")
            if d not in seen:
                seen[d] = p

        for disease, p in seen.items():
            risk   = p.get("risk_category", "—")
            score  = p.get("model_score", 0) or 0
            model  = p.get("best_model_name", "—")
            date_s = (p.get("prediction_date") or "")[:19]
            rc = _risk_color(risk)
            story.append(Paragraph(f"â—† {disease_map.get(disease, disease.title())}", h2))
            pred_data = [
                ["Risk Category:", risk],
                ["Model Score:",   f"{score:.1%}"],
                ["Algorithm:",     model],
                ["Assessment Date:", date_s],
            ]
            pt = Table(pred_data, colWidths=[5*cm, 9*cm])
            pt.setStyle(TableStyle([
                ("BACKGROUND",   (0,0), (0,-1), LIGHT),
                ("TEXTCOLOR",    (0,0), (0,-1), NAVY),
                ("FONTNAME",     (0,0), (0,-1), "Helvetica-Bold"),
                ("FONTSIZE",     (0,0), (-1,-1), 10),
                ("GRID",         (0,0), (-1,-1), 0.3, colors.lightgrey),
                ("LEFTPADDING",  (0,0), (-1,-1), 6),
                ("TOPPADDING",   (0,0), (-1,-1), 4),
                ("BOTTOMPADDING",(0,0), (-1,-1), 4),
                ("BACKGROUND",   (1,0), (1,0), rc),
                ("TEXTCOLOR",    (1,0), (1,0), colors.white),
                ("FONTNAME",     (1,0), (1,0), "Helvetica-Bold"),
            ]))
            story.append(pt)
            story.append(Spacer(1, 0.3*cm))

    # â”€â”€ General Recommendations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Paragraph("General Health Recommendations", h2))
    recs = [
        "Maintain at least 150 minutes of moderate aerobic exercise per week.",
        "Follow a balanced diet rich in vegetables, whole grains, and lean protein.",
        "Monitor blood pressure and blood glucose regularly.",
        "Avoid smoking and limit alcohol consumption.",
        "Aim for 7“9 hours of quality sleep per night.",
        "Schedule regular check-ups with your healthcare provider.",
        "Manage stress through mindfulness, meditation, or yoga.",
    ]
    for rec in recs:
        story.append(Paragraph(f"â€¢ {rec}", normal))
        story.append(Spacer(1, 0.15*cm))

    # Disclaimer 
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=AMBER))
    disc_style = ParagraphStyle("disc", parent=styles["Normal"],
                                 textColor=colors.HexColor("#9a3412"),
                                 fontSize=8, spaceBefore=6, backColor=colors.HexColor("#fff7ed"))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI-powered educational tool and is NOT "
        "a medical diagnosis. The risk assessments are based on statistical machine learning "
        "models and have not been clinically validated. This report should NOT be used as a "
        "substitute for professional medical advice, diagnosis, or treatment. Always consult a "
        "qualified healthcare professional for medical concerns.", disc_style))

    doc.build(story)
    return buf.getvalue()

