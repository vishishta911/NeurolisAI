from extensions import db
from datetime import datetime

class MRIRecord(db.Model):

    __tablename__ = "mri_records"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id")
    )

    image_path = db.Column(db.String(500))

    prediction = db.Column(db.String(100))

    confidence = db.Column(db.Float)

    heatmap_path = db.Column(db.String(500))

    explanation = db.Column(db.Text)

    notes = db.Column(db.Text)

    upload_date = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )
    heatmap_path = db.Column(
    db.String(500)
)