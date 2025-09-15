import logging
import os
from logging.handlers import RotatingFileHandler

from config.path import LOGGING_FILE_PATH


class CustomLogger:
    """
    커스텀 로깅 클래스
    - 콘솔과 파일에 동시에 로그 출력
    - 로그 포맷 및 날짜 포맷 설정
    - 기본 로그 레벨 INFO
    - 로그 파일 저장 경로 커스텀 가능
    """

    LOG_FORMAT = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(process)d "
        "--- [%(threadName)15s] %(name)-7s : %(message)s"
    )
    DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
    LOG_LEVEL = logging.INFO  # 기본 로그 레벨

    def __init__(
        self,
        name: str,
        log_filename: str = "app.log",
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
    ):
        """
        :param name: 로거 이름
        :param log_filename: 로그 파일 이름 (기본값: app.log)
        :param max_bytes: 로그 파일 최대 크기 (기본 5MB)
        :param backup_count: 최대 백업 파일 개수 (기본 5개)
        """
        logger_name = None
        if isinstance(name, type):  # 클래스 이름인 경우
            logger_name = name.__name__
        elif isinstance(name, str) and os.path.isfile(name):  # 파일 경로인 경우
            logger_name = os.path.splitext(os.path.basename(name))[0]
        else:
            logger_name = str(name)

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(self.LOG_LEVEL)
        self.logger.propagate = False  # 부모 로거 전파 방지

        # 로그 디렉토리 설정 및 생성
        self.log_dir = LOGGING_FILE_PATH
        self.log_filepath = os.path.join(self.log_dir, log_filename)
        os.makedirs(self.log_dir, exist_ok=True)  # 로그 디렉토리가 없으면 생성

        # 핸들러가 중복 추가되지 않도록 초기화
        if not self.logger.hasHandlers():
            self._add_console_handler()
            self._add_file_handler(self.log_filepath, max_bytes, backup_count)

    def _add_console_handler(self):
        """콘솔 출력 핸들러 추가"""
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self._get_formatter())
        self.logger.addHandler(console_handler)

    def _add_file_handler(self, log_filepath, max_bytes, backup_count):
        """파일 출력 핸들러 추가 (로그 파일 회전)"""
        file_handler = RotatingFileHandler(
            log_filepath, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(self._get_formatter())
        self.logger.addHandler(file_handler)

    def _get_formatter(self):
        """로그 포맷터 반환"""
        return logging.Formatter(self.LOG_FORMAT, datefmt=self.DATE_FORMAT)

    def get_logger(self):
        """로거 인스턴스 반환"""
        return self.logger

    def _log(self, level, msg, *args, **kwargs):
        """로깅 시 호출한 파일 경로를 포함"""
        # 현재 호출 스택의 프레임을 가져옵니다
        self.logger.log(level, msg, *args)
