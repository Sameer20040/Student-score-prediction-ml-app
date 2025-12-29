import logging # records application events
import os # handles file and directory path
from datetime import datetime # create time based log files name


# creates log directory  
LOG_DIR = os.path.join(os.getcwd(), "logs")
# creates folder if dont exist and avoids error if it already exists
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH, # logs are saved
    level=logging.INFO, #log info,warning,error and critical
    format="[%(asctime)s] %(levelname)s - %(message)s" # log message structured
)

logger = logging.getLogger() # retrieves the root logger
# can be used across all project files

if __name__ == "__main__":
    logger.info("Logger is set up and ready to use.")