import gevent.subprocess as subprocess
import logging
import sys 

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("main")

args = sys.argv
from_stack = False
if len(args) > 1:
    if args[1] == "from_stack":
        from_stack = True

def start_server():
    """
    Boots the NLP Server for API calls.
    """
    log.info("Starting Vision Server...")
    subprocess.call(["python", "start_server.py"])


if __name__ == "__main__":
    start_server()
