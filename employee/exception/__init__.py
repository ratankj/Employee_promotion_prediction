import os
import sys


#**Exceptions**

#Exceptions are errors that occur during the execution of a program. In a machine learning pipeline, exceptions can occur 
# at different stages, such as data preprocessing, model training, or model evaluation.

#exec_tb is a traceback object created when an exception occurs.

#tb_frame is an attribute of the traceback object that contains information about the current frame in the call stack at the point 
# where the exception occurred.

#f_code is an attribute of the frame object that contains information about the code being executed in the frame.

#co_filename is an attribute of the code object that contains the name of the file from which the code was loaded.

#The exec_tb variable contains a traceback object, which represents the call stack at the point where the exception was raised. 
# The traceback object contains information about the functions and modules that were called leading up to the exception, 
# as well as the line numbers where the exception occurred.

#In Python, exc_info() is a function from the sys module that returns a tuple representing the current exception. 
# The tuple contains three values: the exception type, the exception value, and the traceback object.



# error message , #error details
class CustomException(Exception):

    #error_details from sys module , and error_messafe from exception class

    def __init__(self,error_message:Exception,error_details:sys):
        self.error_message = CustomException.get_detailed_error_message(error_message=error_message,error_details=error_details)

    @staticmethod
    def get_detailed_error_message(error_message:Exception,error_details:sys)->str:
        _, _, exec_tb= error_details.exc_info()

#error details

        #we need exception like in which the error is occured
        exception_block_line_number=exec_tb.tb_frame.f_lineno
 
        #for try block
        try_block_line_number=exec_tb.tb_lineno

        # in which file we get error and which code we get issue
        file_name=exec_tb.tb_frame.f_code.co_filename

# error message

        error_message = f"""
        Error occured in script: 
        [ {file_name} ] at 
        try block line number: [{try_block_line_number}]
        and exception block line number: [{exception_block_line_number}] 
        error message: [{error_message}]
        """
        return error_message
    
#The __str__() method returns a human-readable, or informal, string representation of an object. 
# This method is called by the built-in print(), str(), and format() functions.
    def __str__(self):
        return self.error_message

#The __repr__() method returns a more information-rich, or official, string representation of an object.
#  This method is called by the built-in repr() function.
#  If possible, the string returned should be a valid Python expression that can be used to recreate the object.

    def __repr__(self) -> str:
        return CustomException.__name__.str()



#In this case, the __repr__ method is defined to return a string representing the name of the CustomException class. 
# The __name__ attribute is used to retrieve the name of the class, 
# and the str() function is called to convert the name to a string.

