# Для запуска celery -A app.celery.celery_app.celery worker --pool=solo --loglevel=info
from celery import Celery
from app.config import Config

celery = Celery(
    __name__,
    broker=Config.CELERY_BROKER_URL,
    backend=Config.CELERY_RESULT_BACKEND
)

celery.conf.update(
    task_serializer   = 'json',
    result_serializer = 'json',
    accept_content    = ['json'],
    timezone          = 'Europe/Moscow',
    enable_utc        = False,
)
