import os
import logging


LOG_DIR = "./logs/"
LOG_FILE = "logs.log"

def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)

    log_file = os.path.join(LOG_DIR, LOG_FILE)

    # Configure root logger
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])

    # Create logger for BiotechAgent
    logger = logging.getLogger('BiotechAgent')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to prevent duplication
    logger.handlers = []
    
    # Add FileHandler and StreamHandler to this logger
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Log the start of the session and the log file location
    logger.info(f"Logging session started. Log file: {log_file}")
    
    return logger

class FeedbackSystem:
    def __init__(self, logger):
        self.logger = logger

    async def collect_feedback(self, user_id: str, query: str, response: str, rating: int, comment: str = None):
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "query": query,
            "response": response,
            "rating": rating,
            "comment": comment
        }
        self.logger.info(f"User feedback: {feedback}")