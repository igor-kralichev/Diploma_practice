import os
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    SSL_CERT_PATH = os.path.join(PROJECT_ROOT, 'data', 'root.crt')

    DATABASE_URL = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        f"?sslmode=verify-full&sslrootcert={SSL_CERT_PATH}&target_session_attrs=read-write"
    )

    # Добавляем обязательный параметр для Flask-SQLAlchemy
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    JSON_FILE_PATH = os.path.join(PROJECT_ROOT, 'output', 'indicators.json')
    DATA_CSV_PATH  = os.path.join(PROJECT_ROOT, 'data', 'final.csv')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')

    REDIS_URL = os.getenv('REDIS_URL', os.getenv('CELERY_BROKER_URL'))
    CACHE_TYPE = 'RedisCache'
    CACHE_REDIS_URL = REDIS_URL
    CACHE_DEFAULT_TIMEOUT = 300

    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', REDIS_URL)



