# importing system library for runtime environment
# and accessibility
import sys

# a function to store the error message information
def err_msg_info(err, err_info: sys):
    _, _, exc_tb = err_info.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"error occurred in python script name [{file_name}] , \n line number [{exc_tb.tb_lineno}],  \n exception info [{str(err)}]"
    return error_message


# a class to define the functions of error handling
class CustomException(Exception):
    def __init__(self, error_message, err_info: sys):
        super.__init__(error_message)
        self.error_message = err_msg_info(error_message, err_info=err_info)

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return self.error_message