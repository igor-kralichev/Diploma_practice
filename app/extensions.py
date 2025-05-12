# app/extensions.py
from flask_sqlalchemy import SQLAlchemy
from flask_migrate  import Migrate
from flask_restx import Api
from flask_caching import Cache

db      = SQLAlchemy()
migrate = Migrate()
api     = Api(
    title="Tourism Indicators API",
    version="1.0",
    description="CRUD и аналитика по показателям туризма",
    doc="/docs"
)
cache = Cache()