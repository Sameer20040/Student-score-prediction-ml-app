import sys
from src.logger import logger
from src.exception import CustomException

try:
    a = 1 / 0
except Exception as e:
    logger.error("Division by zero error caught", exc_info=True)
    raise CustomException(e, sys)
