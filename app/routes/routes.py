import io
import os
import json
from datetime import datetime

from flask import send_file, abort
from flask_restx import Namespace, Resource, fields
from werkzeug.datastructures import FileStorage

from app.config import Config
from app.celery.celery_app import celery
from app.extensions import cache
from app.utils.logger import logger

# Создаём namespace
ns = Namespace('indicators', description='Показатели туризма')

# Описываем модель ответа
indicator_model = ns.model('Indicator', {
    '1_total_tourists': fields.Integer(
        description="Общее число туристов"
    ),
    '2_tourists_per_month': fields.Raw(
        description="Словарь месяц→туристов"
    ),
    '3_territorial_distribution': fields.Raw(
        description="Территориальное распределение"
    ),
    '4_demographic_distribution': fields.Raw(
        description="Демографическое распределение"
    ),
    '5_profitable_categories': fields.Raw(
        description="Выгодные категории туристов"
    ),
    '6_average_tourist_profile': fields.Raw(
        description="Средний профиль туриста"
    ),
})



# Парсер для GET-параметров
get_parser = ns.parser()
get_parser.add_argument('from', type=str, help='начало периода YYYY-MM')
get_parser.add_argument('to', type=str, help='конец периода YYYY-MM')

# Парсер для POST-файла
upload_parser = ns.parser()
upload_parser.add_argument(
    'file',
    location='files',
    type=FileStorage,
    required=True,
    help='CSV-файл с данными'
)

# Эндпоинты
@ns.route('/async')
class IndicatorsAsync(Resource):

    @ns.expect(upload_parser)
    @ns.response(202, 'Задача принята')
    @ns.response(400, 'Неправильный запрос')
    def post(self):
        """Фоновая загрузка CSV (возвращает task_id)"""
        from app.celery.tasks import process_csv_task

        args = upload_parser.parse_args()
        csv_file = args['file']
        if not csv_file.filename.lower().endswith('.csv'):
            abort(400, "Нужен CSV-файл")

        temp_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),  # текущая директория файла
                os.pardir,  # поднимаемся на уровень выше
                'temp'  # создаем папку temp
            )
        )
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, csv_file.filename)
        csv_file.save(temp_path)

        task = process_csv_task.apply_async(args=[temp_path])
        return {'task_id': task.id}, 202

@ns.route('/status/<string:task_id>')
class TaskStatus(Resource):
    @ns.response(200, 'OK')
    @ns.response(202, 'В процессе')
    @ns.response(404, 'Не найдено')
    @ns.response(500, 'Ошибка выполнения')
    def get(self, task_id):
        """Статус фоновой задачи по task_id"""
        res = celery.AsyncResult(task_id)
        if res.state == 'PENDING':
            return {'state': 'PENDING'}, 202
        if res.state == 'STARTED':
            return {'state': 'STARTED'}, 202
        if res.state == 'SUCCESS':
            # Задача завершена — сразу отдаем содержимое indicators.json
            file_path = Config.JSON_FILE_PATH
            try:
                with open(file_path, encoding='utf-8') as f:
                    data = json.load(f)
            except FileNotFoundError:
                abort(404, "Файл с результатом не найден")
            except json.JSONDecodeError:
                abort(500, "Невозможно прочитать JSON")
            return data, 200
        if res.state in ('FAILURE', 'RETRY'):
            return {'state': res.state, 'error': str(res.result or res.traceback)}, 500
        return {'state': res.state}, 200

@ns.route('/')
class Indicators(Resource):

    @cache.cached(
        timeout=300,
        key_prefix=lambda: f"indicators_all_{get_parser.parse_args().get('from')}_{get_parser.parse_args().get('to')}"
    )
    @ns.doc('get_indicators')
    @ns.expect(get_parser)
    @ns.marshal_with(indicator_model)
    def get(self):
        """Получить набор рассчитанных показателей"""
        args = get_parser.parse_args()
        start    = args.get('from')
        end      = args.get('to')

        # Загрузка JSON
        file_path = Config.JSON_FILE_PATH
        logger.info(f"Загружаю данные из {file_path}")
        if not os.path.exists(file_path):
            abort(404, "Файл не найден, запустите пересчёт.")
        try:
            with open(file_path, encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            abort(404, "Файл не найден, запустите пересчёт.")
        except json.JSONDecodeError:
            abort(500, "Файл повреждён.")

        # Валидация YYYY-MM
        for val, name in [(start, 'from'), (end, 'to')]:
            if val:
                try:
                    datetime.strptime(val, '%Y-%m')
                except ValueError:
                    abort(400, f"Параметр '{name}' должен быть в формате YYYY-MM")

        # Фильтрация
        if start or end:
            raw = data.get('2_tourists_per_month', {})
            data['2_tourists_per_month'] = {
                period: cnt
                for period, cnt in raw.items()
                if (not start or period >= start)
                   and (not end   or period <= end)
            }

        return data

    @ns.route('/download')
    class IndicatorsDownload(Resource):
        @ns.doc('download_indicators')
        @ns.response(200, 'Файл сформирован')
        @ns.response(404, 'Файл не найден')
        @ns.response(500, 'Ошибка формирования файла')
        def get(self):
            """Скачать indicators.json как файл"""
            file_path = Config.JSON_FILE_PATH
            logger.info(f"Готовлю к скачиванию {file_path}")

            if not os.path.exists(file_path):
                abort(404, "Файл не найден, запустите пересчёт.")

            # Читаем готовый JSON и упаковываем в поток
            with open(file_path, encoding='utf-8') as f:
                data = json.load(f)
            blob = json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')

            # Отдаём через send_file, без маршалинга
            return send_file(
                io.BytesIO(blob),
                mimetype='application/octet-stream',
                as_attachment=True,
                download_name='indicators.json'
            )

    @ns.doc('upload_indicators')
    @ns.expect(upload_parser)
    @ns.response(201, 'Данные успешно загружены')
    @ns.response(400, 'Неправильный запрос')
    @ns.response(422, 'Ошибка формата/валидации CSV')
    def post(self):
        """Загрузить CSV и пересчитать показатели"""
        from app import process_data_load
        args = upload_parser.parse_args()
        csv_file = args['file']  # это FileStorage

        # сохраняем временно
        temp_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), 'temp'))
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, csv_file.filename)
        csv_file.save(temp_path)
        logger.info(f"Временный файл: {temp_path}")

        try:
            process_data_load(csv_path=temp_path)
        except FileNotFoundError as e:
            abort(400, str(e))
        except (ValueError, KeyError) as e:
            abort(422, f"Ошибка обработки CSV: {e}")
        except IOError:
            abort(500, "Проблема с чтением или сохранением файла")
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning(f"Не удалось удалить временный файл: {temp_path}")

        return {'message': 'Данные обновлены'}, 201
