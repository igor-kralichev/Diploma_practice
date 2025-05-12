# app/utils/logger.py

import logging
import os
import sys
import re
from logging.handlers import RotatingFileHandler

# Создаём логгер на уровне модуля
logger = logging.getLogger('my_app')

def setup_logger(log_dir):
    """
    Настраивает логгер для приложения.
    Логи записываются в консоль (с цветами) и в файл (без ANSI-кодов).
    """
    # Создаем директорию logs, если она не существует
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Создаем логгер
    logger = logging.getLogger('my_app')
    logger.setLevel(logging.INFO)

    # Проверяем, чтобы не добавлять обработчики повторно
    if not logger.handlers:
        # Форматтер для логов
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Обработчик для консоли (с цветами)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Обработчик для файла (без цветовых кодов, с ротацией)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, 'app.log'),
            maxBytes=1024*1024,  # 1 MB
            backupCount=5  # Хранить до 5 файлов резервных копий
        )
        file_handler.setFormatter(formatter)

        # Фильтр для удаления ANSI-кодов из файла
        class NoAnsiFilter(logging.Filter):
            def filter(self, record):
                record.msg = self.remove_ansi(record.msg)
                return True

            def remove_ansi(self, message):
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                return ansi_escape.sub('', str(message))

        file_handler.addFilter(NoAnsiFilter())
        logger.addHandler(file_handler)

    return logger