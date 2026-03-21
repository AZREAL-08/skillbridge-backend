import logging
import sys

def setup_logging():
    """
    Configures structured logging for the application.
    Sets levels for noisy libraries and ensures output is directed to stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # Suppress noise from deep learning libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("skillNer").setLevel(logging.ERROR)
    
    # Enable our app logs
    logging.getLogger("app").setLevel(logging.INFO)
    
    logging.info("Logging initialized.")
