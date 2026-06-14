def generate_explanation(
    prediction
):

    explanations = {

        "Glioma":
        "Patterns similar to glioma-class MRI scans were detected. This result should be reviewed by a medical professional.",

        "Meningioma":
        "Patterns similar to meningioma-class MRI scans were detected. This result is intended for informational purposes only.",

        "Pituitary":
        "Patterns similar to pituitary tumor MRI scans were detected. Professional evaluation is recommended.",

        "No Tumor":
        "No significant tumor-like patterns were identified in the uploaded MRI image."
    }

    return explanations.get(
        prediction,
        "Analysis completed."
    )