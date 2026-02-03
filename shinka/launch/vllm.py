import time
import sys
import logging
import subprocess
import atexit
import argparse
import signal
import enum
import pathlib
from typing import Optional, IO

import psutil

logger = logging.getLogger(__name__)

class VLLMServer:

    class PortStatus(enum.Enum):
        IN_USE_BY_OTHER_PROCESS = 0
        IN_USE_BY_THIS_VLLM_PROCESS = 1
        NOT_IN_USE = 2

    def __init__(
        self,
        model_path_or_id: str,
        served_model_name: Optional[str],
        host: str = "0.0.0.0",
        port: int = 8000,
        gpu_memory_utilization: float = 0.9,
        log_dir: Optional[pathlib.Path] = None,
    ):
        # vLLM
        self.model_path_or_id = model_path_or_id
        self.served_model_name = served_model_name or model_path_or_id
        self.gpu_memory_utilization = gpu_memory_utilization
        self.host = host
        self.port = port

        # handles
        self.process: Optional[subprocess.Popen] = None
        self.log_dir = log_dir
        self._stdout_file: Optional[IO] = None
        self._stderr_file: Optional[IO] = None

    def __enter__(self):
        self.start(timeout=300)
        return self

    def __exit__(self, exit_type, exit_value, exit_traceback):
        self.stop(timeout=15)

    def start(self, timeout: int = 300) -> None:
        """Start this server"""

        # handle cases where the port is already in use
        match self._query_port_status():
            case VLLMServer.PortStatus.IN_USE_BY_THIS_VLLM_PROCESS:
                logger.warning(
                    f"attempted to start vLLM server at {self.address}, but port is already in use "
                    f"by this server."
                )
                return None
            case VLLMServer.PortStatus.IN_USE_BY_OTHER_PROCESS:
                logger.warning(
                    f"attempted to start vLLM server at {self.address}, but port is already in use "
                    f"by another process."
                )
                return None
        
        # otherwise we can launch the server
        command = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path_or_id,
            "--served-model-name", self.served_model_name,
            "--port", f"{self.port}",
            "--host", self.host,
            "--gpu-memory-utilization", f"{self.gpu_memory_utilization}",
            "--trust-remote-code",
            "--dtype", "auto"
        ]

        if self.log_dir is not None:
            # ensure the log directory exists
            self.log_dir.mkdir(parents=True, exist_ok=True)

            self._stdout_file = open(self.log_dir / "vllm.out", "w", buffering=1)
            self._stderr_file = open(self.log_dir / "vllm.err", "w", buffering=1)

        logger.info(f"Starting vLLM server: '{' '.join(command)}'")

        atexit.register(self.stop)
        self.process = subprocess.Popen(
            command,
            stdout=self._stdout_file or subprocess.DEVNULL,
            stderr=self._stderr_file or subprocess.DEVNULL,
            text=True,
        )

        logger.info(f"Waiting for vLLM server to start at {self.address}...")

        if not self._wait_for_server_to_start(timeout=timeout):
            logger.error("Timeout waiting for vLLM server to start, exiting...")
            self.stop(timeout=15)
            raise TimeoutError(f"vLLM server failed to start in {timeout}s")
        
        logger.info(f"vLLM server started successfully at {self.address}")


    def stop(self, timeout: Optional[int] = 15) -> None:
        """Stop this server."""
        for handle in [self._stdout_file, self._stderr_file]:
            if handle is not None and not handle.closed:
                handle.close()
        if self.process is None:
            return
        logger.info(f"Stopping vLLM server at {self.address}...")
        kill_process_tree(pid=self.process.pid, include_parent=True, timeout=timeout)
        self.process = None

    def _query_port_status(self) -> PortStatus:
        connections = [c for c in psutil.net_connections() if c.laddr == (self.host, self.port)]
        if not connections:
            return VLLMServer.PortStatus.NOT_IN_USE
        connection_pid = connections[0].pid
        if self.process is not None and connection_pid == self.process.pid:
            return VLLMServer.PortStatus.IN_USE_BY_THIS_VLLM_PROCESS
        return VLLMServer.PortStatus.IN_USE_BY_OTHER_PROCESS

    def _wait_for_server_to_start(self, timeout: int = 300) -> bool:
        """Wait for server to start using the port.
        
        Returns:
            `True` if the server started successfully, `False` otherwise.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._query_port_status() == VLLMServer.PortStatus.IN_USE_BY_THIS_VLLM_PROCESS:
                return True
            time.sleep(1)
        return False

    @property
    def address(self) -> str:
        return f"http://{self.host}:{self.port}"

def kill_process_tree(
    pid: int,
    include_parent: bool = False,
    timeout: Optional[int] = None
) -> None:
    """Recursively kills all the children of the given pid."""
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.send_signal(signal.SIGTERM)
    if include_parent:
        parent.send_signal(signal.SIGTERM)
    
    _, alive = psutil.wait_procs(children + ([parent] if parent else []), timeout=timeout)
    for process in alive:
        process.send_signal(signal.SIGKILL)


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
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9
    )
    args = parser.parse_args()

    vllm_server = VLLMServer(
        model_path_or_id=args.model,
        served_model_name=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        port=args.port,
    )
    with vllm_server:
        logger.info("Press Ctrl+C to stop the vLLM server")
        try:
            while True:
                time.sleep(86400)
        except KeyboardInterrupt:
            logger.info("vLLM server stopped manually. Exiting...")

if __name__=="__main__":
    main()
