import os
import logging
from datetime import datetime


# logging.basicconfig()
#logging.basicConfig() is a function provided by the Python logging module that allows you to quickly configure logging 
# in a basic way without having to create a logger manually.

#Some of the commonly used parameters for basicConfig() are the following:
#level: The root logger will be set to the specified severity level.
#filename: This specifies the file.
#filemode: If filename is given, the file is opened in this mode. The default is a, which means append.
#format: This is the format of the log message.


LOG_DIR= "logs"

CURRENT_TIME_STEMP= f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

log_file_name= f"log{CURRENT_TIME_STEMP}.log"

#making directory
os.makedirs(LOG_DIR, exist_ok=True)

log_file_path= os.path.join(LOG_DIR,log_file_name)

logging.basicConfig(filename=log_file_path,
filemode="w",
format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
level=logging.INFO
)
