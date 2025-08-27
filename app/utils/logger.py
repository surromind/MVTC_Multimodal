import logging
import os
from logging.handlers import RotatingFileHandler


class LoggerHandler:
    """싱글톤 패턴으로 동작하는 로깅 핸들러 클래스"""

    _instance = None
    initialized = False
    logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialize()
        return cls._instance

    @classmethod
    def _initialize(cls):
        if not cls.initialized:
            # 로거 이름 설정
            cls.logger_name = "MVTC"
            cls.log_level = logging.INFO

            # 로거 생성 및 설정
            logger = logging.getLogger(cls.logger_name)
            logger.setLevel(cls.log_level)

            # 기존 핸들러 제거
            while logger.hasHandlers():
                logger.removeHandler(logger.handlers[0])

            # 로그 포맷 설정
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # 콘솔 핸들러 추가
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(cls.log_level)
            logger.addHandler(console_handler)

            # 로그 디렉토리 생성
            os.makedirs("logs", exist_ok=True)

            # 파일 핸들러 추가 (10MB 단위로 로테이션, 최대 5개 파일 유지)
            file_handler = RotatingFileHandler(
                os.path.join("logs", "MVTC.log"),
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",  # 10MB
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(cls.log_level)
            logger.addHandler(file_handler)

            cls.logger = logger
            cls.initialized = True


def get_logger():
    """로거 인스턴스를 반환"""
    return LoggerHandler().logger


# 전역 로거 객체 생성
logger = get_logger()
