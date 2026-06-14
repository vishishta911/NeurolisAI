from extensions import db

class User(db.Model):

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)

    google_id = db.Column(
        db.String(255),
        unique=True
    )

    name = db.Column(db.String(255))

    email = db.Column(
        db.String(255),
        unique=True
    )

    profile_picture = db.Column(db.String(500))

    mri_records = db.relationship(
        "MRIRecord",
        backref="user",
        lazy=True
    )

    symptoms = db.relationship(
        "Symptom",
        backref="user",
        lazy=True
    )