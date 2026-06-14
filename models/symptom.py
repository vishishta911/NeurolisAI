from extensions import db
from datetime import datetime

class Symptom(db.Model):

    __tablename__ = "symptoms"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id")
    )

    symptom_name = db.Column(db.String(100))

    notes = db.Column(db.Text)

    severity = db.Column(db.String(20))

    symptom_date = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )