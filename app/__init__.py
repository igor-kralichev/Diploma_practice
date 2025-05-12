# app/__init__.py
from flask import Flask, jsonify
from app.config import Config
from app.services.load_data import load_data_to_db
from app.utils.logger import setup_logger, logger
from app.extensions import db, migrate, api, cache
from app.routes.routes import ns as indicators_ns
from sqlalchemy import create_engine
import os


def process_data_load(csv_path=None):
    """
    Всегда загружает данные и пересчитывает показатели (перезаписывает JSON).
    Если csv_path=None, возьмёт путь в Config.py
    """
    path = csv_path or Config.DATA_CSV_PATH
    try:
        load_data_to_db(
            Config.DATABASE_URL,
            csv_path=path
        )
    except FileNotFoundError:
        logger.error(f"CSV-файл не найден: {path}")
    except Exception as e:
        logger.error(f"Ошибка загрузки CSV ({path}): {e}")

def check_db_connection(database_url):
    """
    Проверяет подключение к базе данных.
    """
    try:
        engine = create_engine(database_url)
        with engine.connect() as connection:
            logger.info("Успешное подключение к базе данных.")
    except Exception as e:
        logger.error(f"Ошибка подключения к базе данных: {e}")
        raise


def init_app(app):
    """
    Выполняет начальную загрузку данных при запуске приложения.
    """
    try:
        # Проверка подключения к БД
        check_db_connection(Config.DATABASE_URL)

        # Вычисляем абсолютный путь до JSON-файла, взяв путь из config.py
        json_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), Config.JSON_FILE_PATH)
        )

        # Если файл уже есть — пропускаем загрузку
        if os.path.exists(json_path):
            logger.info(f"Файл показателей найден ({json_path}), начальная загрузка пропущена.")
            return

        # Иначе — запускаем процесс загрузки
        process_data_load()

    except Exception as e:
        logger.error(f"Ошибка при начальной загрузке данных: {e}")


def create_app():
    app = Flask(__name__)

    # Загружаем конфигурацию
    app.config.from_object('app.config.Config')

    # Отключаем строгую привязку к слэшу для всех маршрутов
    app.url_map.strict_slashes = False

    # Отключаем "did you mean…" в 404 API-ответах
    app.config['RESTX_ERROR_404_HELP'] = False

    # Инициализация расширения
    db.init_app(app)
    migrate.init_app(app, db)
    api.init_app(app)
    cache.init_app(app)

    # Создание папки для логов
    setup_logger(log_dir=Config.LOG_DIR)

    # Регистрируем маршруты
    api.add_namespace(indicators_ns, path='/indicators')


    @app.errorhandler(404)
    def handle_404(e):
        return jsonify({"error": "Ресурс не найден"}), 404

    @app.errorhandler(405)
    def handle_405(e):
        return jsonify({"error": "Метод не поддерживается"}), 405

    @app.errorhandler(Exception)
    def handle_500(e):
        # сюда попадут все необработанные исключения
        app.logger.exception("Необработанная ошибка")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500

    # Инициализация и загрузка данных
    with app.app_context():
        init_app(app)

    return app
