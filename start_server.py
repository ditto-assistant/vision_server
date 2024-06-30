from server import Server
from gevent.pywsgi import WSGIServer
import logging

log = logging.getLogger("start_server")
logging.basicConfig(level=logging.DEBUG)


class devnull:
    write = lambda _: None


def start_server():
    server = Server()
    http_server = WSGIServer(("0.0.0.0", 42030), server.app, log=devnull)
    log.info("Vision Server started on port 42030]")
    http_server.serve_forever()


if __name__ == "__main__":
    start_server()
