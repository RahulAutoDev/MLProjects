import sys 
import logging
from src.logger import logging

def error_message_details(error, error_details):
    """
    Constructs a detailed error message with file name, line number, and error description.
    """
    _, _, exc_tb = error_details
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in script [{}] at line [{}]: [{}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_details):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_details)

    def __str__(self):
        return self.error_message


