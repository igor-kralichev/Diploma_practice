# app/models.py
from app.extensions import db

class TourismData(db.Model):
    __tablename__ = 'tourism_data'
    id                = db.Column(db.Integer, primary_key=True)
    date_of_arrival   = db.Column(db.DateTime, nullable=False)
    days_cnt          = db.Column(db.Numeric)
    visitors_cnt      = db.Column(db.Numeric)
    spent             = db.Column(db.Numeric)
    home_region       = db.Column(db.Text)
    home_city         = db.Column(db.Text)
    home_country      = db.Column(db.Text)
    age               = db.Column(db.String(32))
    gender            = db.Column(db.String(16))
    income            = db.Column(db.String(32))
    month_start       = db.Column(db.DateTime)
    avg_spent_per_visitor = db.Column(db.Numeric)
