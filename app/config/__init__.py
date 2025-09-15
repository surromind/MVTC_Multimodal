from app.config.logger import CustomLogger

_custom_logger = CustomLogger("app", "app.log")
logger = _custom_logger.get_logger()
