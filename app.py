import os

from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    session
)
from flask import request
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from authlib.integrations.flask_client import OAuth
from werkzeug.middleware.proxy_fix import ProxyFix
from extensions import db

from models.user import User
from models.mri_record import MRIRecord
from models.symptom import Symptom
from services.predictor import predict_mri

from services.explanation import (
    generate_explanation
)
from services.gradcam import (
    generate_gradcam
)
from flask import send_file

from services.report_generator import (
    generate_report
)
from services.predictor import (
    predict_mri,
    CLASS_NAMES
)
# Load environment variables
load_dotenv()


# Create Flask App
app = Flask(__name__)

app.wsgi_app = ProxyFix(
    app.wsgi_app,
    x_proto=1,
    x_host=1
)

# App Configuration
app.config["SECRET_KEY"] = os.getenv(
    "SECRET_KEY",
    "neurolis_development_secret_key"
)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///neurolis.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


# Initialize Database
db.init_app(app)


# OAuth Setup
oauth = OAuth(app)

google = oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={
        "scope": "openid email profile"
    }
)


# Create Database Tables
with app.app_context():
    db.create_all()


# Home Page
@app.route("/")
def home():
    return render_template("home.html")


# Google Login
# @app.route("/login")
# def login():
#     return google.authorize_redirect(
#         url_for(
#             "google_callback",
#             _external=True
#         )
#     )
# @app.route("/login")
# def login():

#     callback_url = "https://web-production-3a3a9.up.railway.app/auth/google/callback"

#     return google.authorize_redirect(
#         callback_url
#     )
@app.route("/login")
def login():
    return google.authorize_redirect(
        url_for(
            "google_callback",
            _external=True,
            _scheme="https"
        )
    )

# Google Callback
@app.route("/auth/google/callback")
def google_callback():

    token = google.authorize_access_token()

    user_info = token["userinfo"]

    google_id = user_info["sub"]
    email = user_info["email"]
    name = user_info["name"]
    picture = user_info.get("picture")

    user = User.query.filter_by(
        google_id=google_id
    ).first()

    if not user:

        user = User(
            google_id=google_id,
            email=email,
            name=name,
            profile_picture=picture
        )

        db.session.add(user)
        db.session.commit()

    session["user_id"] = user.id
    session["user_name"] = user.name

    return redirect("/dashboard")


# Dashboard
@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        return redirect("/login")

    records = MRIRecord.query.filter_by(
        user_id=session["user_id"]
    ).all()

    total_uploads = len(records)

    latest_prediction = (
        records[-1].prediction
        if records
        else "No Records"
    )

    average_confidence = (
        round(
            sum(r.confidence for r in records) /
            len(records),
            2
        )
        if records
        else 0
    )

    highest_confidence = (
        max(r.confidence for r in records)
        if records
        else 0
    )

    recent_record = (
        records[-1]
        if records
        else None
    )

    return render_template(
        "dashboard/dashboard.html",
        user_name=session["user_name"],
        total_uploads=total_uploads,
        latest_prediction=latest_prediction,
        average_confidence=average_confidence,
        highest_confidence=highest_confidence,
        recent_record=recent_record
    )

    if "user_id" not in session:
        return redirect("/login")

    records = MRIRecord.query.filter_by(
        user_id=session["user_id"]
    ).order_by(
        MRIRecord.upload_date.desc()
    ).all()

    total_uploads = len(records)

    latest_prediction = (
        records[0].prediction
        if records
        else "No Analysis Yet"
    )

    average_confidence = (
        round(
            sum(
                r.confidence
                for r in records
            ) / len(records),
            2
        )
        if records
        else 0
    )

    return render_template(
        "dashboard/dashboard.html",
        user_name=session["user_name"],
        total_uploads=total_uploads,
        latest_prediction=latest_prediction,
        average_confidence=average_confidence
    )


# Upload MRI
@app.route("/upload", methods=["GET", "POST"])
def upload():

    if "user_id" not in session:
        return redirect("/login")

    if request.method == "POST":

        image = request.files["mri_image"]

        notes = request.form.get("notes")

        if image:
            filename = secure_filename(
                image.filename
            )

            save_path = os.path.join(
                "static",
                "uploads",
                "mri",
                filename
            )

            image.save(save_path)
            heatmap_path = os.path.join(
                "static",
                "uploads",
                "heatmaps",
                filename
            )

            os.makedirs(
                "static/uploads/heatmaps",
                exist_ok=True
            )

            try:
                generate_gradcam(
                    save_path,
                    heatmap_path
                )

                print("GradCAM Generated")

            except Exception as e:
                print("GRADCAM ERROR:")
                print(e)

            prediction, confidence, probabilities = predict_mri(
                save_path
            )

            explanation = generate_explanation(
                prediction
            )

            record = MRIRecord(
                user_id=session["user_id"],
                image_path=save_path,
                heatmap_path=heatmap_path,
                notes=notes,
                prediction=prediction,
                confidence=confidence,
                explanation=explanation
            )
            db.session.add(record)
            db.session.commit()

            return redirect(
                url_for(
                    "result",
                    record_id=record.id
                )
            )
    return render_template(
        "mri/upload.html"
    )


# Timeline
@app.route("/timeline")
def timeline():

    if "user_id" not in session:
        return redirect("/login")

    records = MRIRecord.query.filter_by(
        user_id=session["user_id"]
    ).order_by(
        MRIRecord.upload_date.desc()
    ).all()

    return render_template(
        "timeline/timeline.html",
        records=records
    )
@app.route("/result/<int:record_id>")
def result(record_id):

    if "user_id" not in session:
        return redirect("/login")

    record = MRIRecord.query.get_or_404(
        record_id
    )

    prediction, confidence, probs = predict_mri(
        record.image_path
    )

    probability_data = zip(
        CLASS_NAMES,
        probs
    )

    # Risk Assessment

    if record.confidence >= 90:
        risk_level = "High Attention"

    elif record.confidence >= 75:
        risk_level = "Moderate Attention"

    else:
        risk_level = "Needs Further Evaluation"

    # Specialist Recommendations

    prediction_lower = prediction.lower()

    if prediction_lower == "glioma":

        specialists = [
            "Neuro-Oncologist",
            "Neurosurgeon",
            "Radiologist"
        ]

    elif prediction_lower == "meningioma":

        specialists = [
            "Neurologist",
            "Neurosurgeon",
            "Radiologist"
        ]

    elif prediction_lower == "pituitary":

        specialists = [
            "Endocrinologist",
            "Neurologist",
            "Radiologist"
        ]

    else:

        specialists = [
            "Neurologist",
            "General Physician"
        ]

    return render_template(
        "mri/result.html",
        record=record,
        probability_data=probability_data,
        risk_level=risk_level,
        specialists=specialists
    )

@app.route("/reports")
def reports():

    if "user_id" not in session:
        return redirect("/login")

    records = MRIRecord.query.filter_by(
        user_id=session["user_id"]
    ).order_by(
        MRIRecord.upload_date.desc()
    ).all()

    return render_template(
        "reports/reports.html",
        records=records
    )

@app.route("/download-report/<int:record_id>")
def download_report(record_id):

    if "user_id" not in session:
        return redirect("/login")

    record = MRIRecord.query.get_or_404(
        record_id
    )

    os.makedirs(
        "static/reports",
        exist_ok=True
    )

    pdf_path = (
        f"static/reports/report_{record.id}.pdf"
    )

    generate_report(
        record,
        pdf_path
    )

    return send_file(
        pdf_path,
        as_attachment=True
    )

# Profile
@app.route("/profile")
def profile():

    if "user_id" not in session:
        return redirect("/login")

    user = User.query.get(
        session["user_id"]
    )

    total_uploads = MRIRecord.query.filter_by(
        user_id=session["user_id"]
    ).count()

    symptom_count = Symptom.query.filter_by(
        user_id=session["user_id"]
    ).count()

    return render_template(
        "profile/profile.html",
        user=user,
        total_uploads=total_uploads,
        symptom_count=symptom_count
    )

#resources
@app.route("/resources")
def resources():

    hospitals = [

        {
            "name": "Apollo Hospitals",
            "speciality": "Neurology & Neurosurgery",
            "location": "Hyderabad",
            "contact": "+91 XXXXX XXXXX"
        },

        {
            "name": "Yashoda Hospitals",
            "speciality": "Neurosciences",
            "location": "Hyderabad",
            "contact": "+91 XXXXX XXXXX"
        },

        {
            "name": "KIMS Hospitals",
            "speciality": "Neurology",
            "location": "Hyderabad",
            "contact": "+91 XXXXX XXXXX"
        },

        {
            "name": "AIG Hospitals",
            "speciality": "Neurosciences",
            "location": "Hyderabad",
            "contact": "+91 XXXXX XXXXX"
        },

        {
            "name": "NIMS",
            "speciality": "Government Neuro Center",
            "location": "Hyderabad",
            "contact": "+91 XXXXX XXXXX"
        }

    ]

    return render_template(
        "resources/resources.html",
        hospitals=hospitals
    )
# Logout
@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")


# Run App
if __name__ == "__main__":
    app.run(debug=True)

