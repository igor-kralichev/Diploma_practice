# app/tasks.py
import os

from celery.utils.log import get_task_logger
from app.celery.celery_app import celery
from app.config import Config
from app.services.load_data import load_data_to_db

task_logger = get_task_logger(__name__)

@celery.task(bind=True, name='process_csv_task')
def process_csv_task(self, csv_path: str):
    """
    Фоновая задача: загрузить CSV в БД и пересчитать JSON.
    """
    # импортируем create_app только когда задача стартует,
    # чтобы не было circular import при загрузке модуля
    from app import create_app

    app = create_app()
    with app.app_context():
        try:
            # загружаем и пересчитываем
            load_data_to_db(
                database_url=Config.DATABASE_URL,
                csv_path=csv_path
            )
        except Exception as e:
            task_logger.exception(f"Ошибка в задаче обработки CSV {csv_path}: {e}")
            # пробуем ещё 3 раза с задержкой
            raise self.retry(exc=e, countdown=10, max_retries=3)
        else:
            # если всё прошло без исключений — удаляем временный файл
            try:
                os.remove(csv_path)
                task_logger.info(f"Временный файл удалён: {csv_path}")
            except OSError as e:
                task_logger.warning(f"Не удалось удалить временный файл {csv_path}: {e}")
        return {"status": "ok", "json_path": Config.JSON_FILE_PATH}