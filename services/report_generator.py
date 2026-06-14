from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet


def generate_report(record, output_path):

    doc = SimpleDocTemplate(output_path)

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "NeuroLis AI MRI Analysis Report",
            styles["Title"]
        )
    )

    content.append(Spacer(1, 20))

    content.append(
        Paragraph(
            f"Prediction: {record.prediction}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Confidence: {record.confidence}%",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Explanation: {record.explanation}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Date: {record.upload_date}",
            styles["Normal"]
        )
    )

    doc.build(content)