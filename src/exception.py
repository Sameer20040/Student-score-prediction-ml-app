import sys
from src.logger import logger
def error_message_detail(error, error_details: sys):## Function to extract error details
    _, _, exc_tb = error_details.exc_info()## Get the traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename ## Get the filename where the exception occurred
    error_message = f"Error occurred in python script: {file_name} at line number: {exc_tb.tb_lineno} with message: {str(error)}"
    return error_message

class CustomException(Exception):### Custom exception class inheriting from the base Exception class
    def __init__(self, error_message, error_details: sys):## Constructor to initialize the exception
        super().__init__(error_message)## Initialize the base Exception class
        self.error_message = error_message_detail(error_message, error_details)

    def __str__(self): ## String representation of the exception
        return self.error_message
if __name__ == "__main__":
    try:
        a = 1 / 0 ## Example to raise a ZeroDivisionError
    except Exception as e:
        logger.info("An exception occurred")## Log the occurrence of an exception
        raise CustomException(e, sys) ## Raise the custom exception with details