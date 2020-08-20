from logging import basicConfig, getLogger, StreamHandler, FileHandler, Formatter, INFO
import datetime

now = datetime.datetime.now()
log_file = "../log/" + now.strftime('%Y%m%d_%H%M%S')  + ".log"

f_fmt='%(asctime)s - %(levelname)s - %(funcName)s- %(message)s'

basicConfig(
    filename=log_file,
    filemode='w', # Default is 'a'
    format=f_fmt, 
    level=INFO)


# define a new Handler to log to console as well
console = StreamHandler()
# optional, set the logging level
console.setLevel(INFO)
# set a format which is the same for console use
formatter = Formatter('%(asctime)s - %(levelname)s - %(funcName)s- %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logger = getLogger(__name__).addHandler(console)
print(__name__)

