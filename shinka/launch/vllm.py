import time
import sys
import logging
import subprocess
import atexit
import argparse

logger = logging.getLogger(__name__)

class VLLMServer:
    def __init__(self, model_path: str, model_name: str, port: int = 8000):
        self.model_path = model_path
        self.model_name = model_name
        self.port = port
        self.process = None

    def start(self):
        """Start this server"""

        # TODO: if the port is already open, fail or ignore
        command = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--served-model-name", self.model_name,
            "--port", f"{self.port}",
            "--trust-remote-code",
            "--dtype", "auto"
        ]

        logger.info(f"Starting vLLM server: '{' '.join(command)}'")

        self.process = subprocess.Popen(command)

        # TODO: verify that the port is indeed open (wait if necessary)

    def stop(self, _subprocess=subprocess):
        """Stop this server"""
        if self.process is not None:
            logger.info("Stopping vLLM server.")
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str
    )
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

    with VLLMServer(model_path=args.path, model_name=args.model, port=args.port):
        logger.info("Press Ctrl+C to stop the server.")
        try:
            while True:
                time.sleep(86400)
        except KeyboardInterrupt:
            logger.info("Server stopped manually. Exiting...")

if __name__=="__main__":
    main()
