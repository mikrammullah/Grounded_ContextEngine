import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
PORTAL_FILE = Path(__file__).with_name("dev_portal.html")
VENV_DIR = ROOT / ".venv"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
API_URL = "http://127.0.0.1:8000"
ACTIONS = {
    "prepare": "Create .venv and install project requirements",
    "start": "Build and start the API and PostgreSQL services",
    "stop": "Stop the Compose services without deleting database data",
    "test": "Run the complete pytest suite",
    "smoke": "Exercise readiness, sample ingestion, and query APIs",
    "validate": "Start services, wait for readiness, run tests, and smoke test",
}


def validate_action(action):
    if not isinstance(action, str) or action not in ACTIONS:
        raise ValueError("Unsupported action")
    return action


def _api_key_for_demo():
    api_keys = os.environ.get("API_KEYS", "")
    env_file = ROOT / ".env"
    if not api_keys and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "API_KEYS":
                api_keys = value.strip().strip("\"'")
                break
    for entry in api_keys.split(","):
        tenant, separator, secret = entry.strip().partition("=")
        if separator and tenant == "demo" and secret:
            return secret
    return None


def _request_json(path, method="GET", payload=None, headers=None, timeout=8):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request_headers = dict(headers or {})
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    request = Request(f"{API_URL}{path}", data=data, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else None
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            details = json.loads(body)
        except json.JSONDecodeError:
            details = {"detail": body}
        return error.code, details


def run_smoke_test(log):
    headers = {"X-Tenant-ID": "demo"}
    api_key = _api_key_for_demo()
    if api_key:
        headers["X-API-Key"] = api_key

    status, readiness = _request_json("/readyz")
    if status != 200:
        raise RuntimeError(f"API/database readiness failed ({status}): {readiness}")
    log("API and database readiness passed.")

    sample_path = ROOT / "app" / "static" / "sample_policy.txt"
    sample = sample_path.read_bytes()
    boundary = f"----RAGPortal{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="sample_policy.txt"\r\n'
        "Content-Type: text/plain\r\n\r\n"
    ).encode() + sample + f"\r\n--{boundary}--\r\n".encode()
    upload_request = Request(
        f"{API_URL}/api/v1/rag/documents",
        data=body,
        headers={**headers, "Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    document_id = None
    try:
        with urlopen(upload_request, timeout=20) as response:
            upload = json.loads(response.read().decode("utf-8"))
            document_id = upload["document_id"]
        log(f"Sample document uploaded ({upload['chunks_created']} chunks).")

        status, result = _request_json(
            "/api/v1/rag/query",
            method="POST",
            payload={"query": "What is the critical incident initial response target?", "top_k": 3},
            headers=headers,
            timeout=30,
        )
        if status != 200:
            raise RuntimeError(f"Query failed ({status}): {result}")
        answer = result.get("answer", "")
        if "one-hour" not in answer.lower() and "one hour" not in answer.lower():
            raise RuntimeError("Query completed but its answer did not contain the expected one-hour target")
        if not result.get("sources"):
            raise RuntimeError("Query completed without any cited sources")
        log("Sample query returned the expected target and at least one citation.")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Sample API request failed ({error.code}): {detail}") from error
    finally:
        if document_id:
            delete_headers = dict(headers)
            request = Request(
                f"{API_URL}/api/v1/rag/documents/{document_id}",
                headers=delete_headers,
                method="DELETE",
            )
            try:
                with urlopen(request, timeout=10):
                    log("Temporary smoke-test document was deleted.")
            except (HTTPError, URLError) as error:
                log(f"Warning: could not remove smoke-test document: {error}")


class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs = {}
        self._active_job_id = None

    def start(self, action):
        action = validate_action(action)
        with self._lock:
            if self._active_job_id is not None:
                active = self._jobs[self._active_job_id]
                if active["status"] in {"queued", "running"}:
                    raise RuntimeError("Another portal task is already running")
            job_id = uuid.uuid4().hex
            job = {"id": job_id, "action": action, "status": "queued", "logs": [], "started_at": time.time()}
            self._jobs[job_id] = job
            self._active_job_id = job_id
            thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
            thread.start()
            return {key: value for key, value in job.items() if key != "logs"} | {"logs": []}

    def get(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {**job, "logs": list(job["logs"])}

    def _log(self, job_id, line):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job["logs"].append(line.rstrip("\r\n"))
                job["logs"] = job["logs"][-2000:]

    def _set_status(self, job_id, status):
        with self._lock:
            self._jobs[job_id]["status"] = status
            if status in {"completed", "failed"}:
                self._jobs[job_id]["finished_at"] = time.time()

    def _run_command(self, job_id, command, label):
        self._log(job_id, f"$ {label}")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )

        def collect_output():
            for line in process.stdout:
                self._log(job_id, line)

        output_thread = threading.Thread(target=collect_output, daemon=True)
        output_thread.start()
        try:
            return_code = process.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            self._log(job_id, "Command exceeded the 30-minute limit and was terminated.")
            raise RuntimeError("Command timed out")
        finally:
            output_thread.join(timeout=5)
        if return_code != 0:
            raise RuntimeError(f"Command failed with exit code {return_code}")

    def _prepare(self, job_id):
        if not VENV_PYTHON.exists():
            self._run_command(
                job_id,
                [sys.executable, "-m", "venv", str(VENV_DIR)],
                f'"{sys.executable}" -m venv .venv',
            )
        self._run_command(
            job_id,
            [str(VENV_PYTHON), "-m", "pip", "install", "-r", "requirements.txt"],
            ".venv Python -m pip install -r requirements.txt",
        )

    def _python(self):
        return str(VENV_PYTHON if VENV_PYTHON.exists() else sys.executable)

    def _wait_for_readiness(self, job_id):
        self._log(job_id, "Waiting up to 90 seconds for API and database readiness...")
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                status, result = _request_json("/readyz", timeout=3)
                if status == 200:
                    self._log(job_id, "API and database are ready.")
                    return
                last_error = f"HTTP {status}: {result}"
            except (URLError, TimeoutError) as error:
                last_error = str(error)
            time.sleep(2)
        raise RuntimeError(f"Services did not become ready in time: {last_error}")

    def _execute(self, job_id, action):
        if action == "prepare":
            return self._prepare(job_id)
        if action == "start":
            return self._run_command(job_id, ["docker", "compose", "up", "--build", "-d"], "docker compose up --build -d")
        if action == "stop":
            return self._run_command(job_id, ["docker", "compose", "down"], "docker compose down")
        if action == "test":
            return self._run_command(job_id, [self._python(), "-m", "pytest", "-q"], f'"{self._python()}" -m pytest -q')
        if action == "smoke":
            return run_smoke_test(lambda line: self._log(job_id, line))
        if action == "validate":
            self._execute(job_id, "start")
            self._wait_for_readiness(job_id)
            self._execute(job_id, "test")
            return self._execute(job_id, "smoke")
        raise ValueError("Unsupported action")

    def _run(self, job_id):
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "running"
            action = job["action"]
        self._log(job_id, ACTIONS[action])
        try:
            self._execute(job_id, action)
        except Exception as error:
            self._log(job_id, f"ERROR: {error}")
            self._set_status(job_id, "failed")
        else:
            self._log(job_id, "Task completed successfully.")
            self._set_status(job_id, "completed")


jobs = JobManager()


def _probe(url):
    try:
        with urlopen(url, timeout=2) as response:
            return {"available": response.status == 200, "status": response.status}
    except (HTTPError, URLError, TimeoutError) as error:
        return {"available": False, "detail": str(error)}


def get_status():
    docker_path = shutil.which("docker")
    compose = {"available": bool(docker_path), "containers": [], "detail": None}
    if docker_path:
        try:
            result = subprocess.run(
                [docker_path, "compose", "ps", "--format", "json"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=6,
                check=False,
                shell=False,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    try:
                        container = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    compose["containers"].append(
                        {"name": container.get("Name"), "state": container.get("State"), "health": container.get("Health")}
                    )
            else:
                compose["detail"] = result.stderr.strip() or f"docker compose exited {result.returncode}"
        except (OSError, subprocess.TimeoutExpired) as error:
            compose["detail"] = str(error)
    return {
        "compose": compose,
        "api": _probe(f"{API_URL}/healthz"),
        "database": _probe(f"{API_URL}/readyz"),
        "active_job": next(
            (job for job in [jobs.get(jobs._active_job_id)] if job and job["status"] in {"queued", "running"}),
            None,
        ),
    }


class PortalHandler(BaseHTTPRequestHandler):
    server_version = "RAGDeveloperPortal"

    def _send(self, status, content, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status, value):
        self._send(status, json.dumps(value).encode("utf-8"), "application/json; charset=utf-8")

    def _is_local_request(self):
        if self.client_address[0] not in {"127.0.0.1", "::1"}:
            return False
        host = self.headers.get("Host", "")
        allowed_hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        return host in allowed_hosts

    def _is_same_origin_post(self):
        origin = self.headers.get("Origin")
        if not origin:
            return False
        parsed = urlsplit(origin)
        return (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost"}
            and parsed.port == self.server.server_port
            and parsed.path == ""
            and not parsed.query
            and not parsed.fragment
        )

    def do_GET(self):
        if not self._is_local_request():
            return self._send_json(403, {"error": "Loopback requests only"})
        if self.path == "/":
            return self._send(200, PORTAL_FILE.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/status":
            return self._send_json(200, get_status())
        if self.path.startswith("/api/jobs/"):
            job_id = self.path.removeprefix("/api/jobs/")
            if not job_id.isalnum():
                return self._send_json(400, {"error": "Invalid job ID"})
            job = jobs.get(job_id)
            return self._send_json(200 if job else 404, job or {"error": "Job not found"})
        return self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if not self._is_local_request() or not self._is_same_origin_post():
            return self._send_json(403, {"error": "Loopback same-origin requests only"})
        if self.path != "/api/actions":
            return self._send_json(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._send_json(400, {"error": "Invalid content length"})
        if length <= 0 or length > 1024:
            return self._send_json(413, {"error": "Invalid request size"})
        try:
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict) or set(body) != {"action"}:
                raise ValueError("Expected an object with only an action field")
            action = validate_action(body["action"])
            job = jobs.start(action)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            return self._send_json(400, {"error": str(error)})
        except RuntimeError as error:
            return self._send_json(409, {"error": str(error)})
        return self._send_json(202, job)

    def log_message(self, format_string, *args):
        return


def main():
    parser = argparse.ArgumentParser(description="Local developer portal for the RAG service")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), PortalHandler)
    print(f"Developer portal: http://127.0.0.1:{args.port}", flush=True)
    print("Bound to loopback only. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()