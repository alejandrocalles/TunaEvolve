import time
import sys
import logging
import subprocess
import atexit
import argparse
import psutil

logger = logging.getLogger(__name__)

class VLLMServer:
    def __init__(self, model_path: str, model_name: str, host: str = "0.0.0.0", port: int = 8000):
        self.model_path = model_path
        self.model_name = model_name
        self.host = host
        self.port = port
        self.process = None

    def start(self) -> None:
        """Start this server"""
        if (self._is_port_in_use(verbose=True)):
            return

        command = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--served-model-name", self.model_name,
            "--port", f"{self.port}",
            "--host", self.host,
            "--trust-remote-code",
            "--dtype", "auto"
        ]

        logger.info(f"Starting vLLM server: '{' '.join(command)}'")

        # TODO consider using ProcessWithLogging from ./local.py, just need to handle termination properly
        self.process = subprocess.Popen(command)

        self._wait_for_server_start()

    def stop(self, _subprocess=subprocess) -> None:
        """Stop this server"""
        if self.process is None:
            return

        logger.info(f"Stopping vLLM server at port {self.port}.")
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except _subprocess.TimeoutExpired:
            self.process.kill()
        self.process = None

    def __enter__(self):
        atexit.register(self.stop)
        self.start()
        return self

    def __exit__(self, exit_type, exit_value, exit_traceback):
        self.stop()

    def _is_port_in_use(self, verbose: bool = False) -> bool:
        connections = [
            connection for connection in psutil.net_connections(kind="inet")
            if connection.laddr == (self.host, self.port)
        ]
        if not connections:
            return False
        
        if verbose:
            connection = connections[0]
            logger.info(f"Port {self.host}:{self.port} is already in use.")
            if connection.status == psutil.CONN_LISTEN and self.process:
                logger.info(
                    f"The port was found in {psutil.CONN_LISTEN} status,"
                    f"and this class already has an active process, "
                    f"consider whether this server has already been started."
                )
        return True

    def _wait_for_server_start(self, timeout=120) -> bool:
        """Wait for server to start using the port."""
        start_time = time.time()
        logger.info(f"Waiting for vLLM server to start on {self.host}:{self.port}")

        while time.time() - start_time < timeout:
            if self._is_port_in_use():
                logger.info(f"vLLM server is live at {self.host}:{self.port}")
                return True
            time.sleep(1)

        logger.warning("Timeout waiting for vLLM server to start")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str
    )

    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000
    )
    args = parser.parse_args()

    with VLLMServer(model_path=args.model, model_name=args.model, port=args.port):
        logger.info("Press Ctrl+C to stop the vLLM server.")
        try:
            while True:
                time.sleep(86400)
        except KeyboardInterrupt:
            logger.info("vLLM server stopped manually. Exiting...")

if __name__=="__main__":
    main()
