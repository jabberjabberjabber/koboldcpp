import os
import json
import random
import threading
import shutil
import subprocess
import urllib.request
import urllib.error
import queue
from . import state

class MCPStdioClient:
    def resolve_command(self, command):
        resolved = shutil.which(command)
        if resolved:
            return resolved
        return command

    def __init__(self, command, largs, env=None, cwd=None):
        if isinstance(command, str):
            command = self.resolve_command(command)
            cmd = [command]
        else:
            cmd = list(command)
        if largs:
            cmd.extend(largs)
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            env=full_env,
            cwd=cwd
        )
        self.lock = threading.Lock()
        self.stderr_buffer = []
        self.stderr_limit = 20
        self.alive = True
        self.stderr_thread = threading.Thread(
            target=self._read_stderr,
            daemon=True
        )
        self.stderr_thread.start()

        self._pending = {}
        self._pending_lock = threading.Lock()
        self.stdout_thread = threading.Thread(
            target=self._read_stdout,
            daemon=True
        )
        self.stdout_thread.start()

    def _read_stderr(self):
        try:
            for line in self.process.stderr:
                if not line:
                    break
                line = line.rstrip()
                self.stderr_buffer.append(line)
                if len(self.stderr_buffer) > self.stderr_limit:
                    self.stderr_buffer.pop(0)
        finally:
            self.alive = False

    def _read_stdout(self):  # notifications (no id) are silently dropped
        try:
            for line in self.process.stdout:
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                msg_id = msg.get("id")
                if msg_id is not None:
                    with self._pending_lock:
                        q = self._pending.get(msg_id)
                    if q:
                        q.put(msg)
                    else:
                        print(f"[MCP] Unexpected response id: {msg_id}")
        finally:
            self.alive = False
            with self._pending_lock:
                for q in self._pending.values():
                    q.put(None)

    def send(self, message: dict, await_response=True) -> dict:
        line = json.dumps(message)
        msg_id = message.get("id")

        if await_response and msg_id is None:
            raise ValueError("Cannot await response for a message without an 'id' field")

        response_q = queue.Queue()

        try:
            with self._pending_lock:
                if await_response and msg_id is not None:
                    self._pending[msg_id] = response_q
                with self.lock:
                    if self.process.stdin.closed:
                        raise RuntimeError("MCP server stdin is closed")
                    self.process.stdin.write(line + "\n")
                    self.process.stdin.flush()
        except Exception:
            if await_response and msg_id is not None:
                with self._pending_lock:
                    self._pending.pop(msg_id, None)
            raise

        if not await_response:
            return None
        try:
            response = response_q.get(timeout=180)
        except queue.Empty:
            raise RuntimeError("MCP server timed out (no response in 180s)")
        finally:
            with self._pending_lock:
                self._pending.pop(msg_id, None)
        if response is None:
            errmsg = "\n".join(self.stderr_buffer[-10:])
            print(f"[MCP Server Error!]\n{errmsg}")
            raise RuntimeError("MCP server closed stdout")
        return response

    def notify(self, message: dict) -> None:
        line = json.dumps(message)
        with self.lock:
            if self.process.stdin.closed:
                raise RuntimeError("MCP server stdin is closed")
            self.process.stdin.write(line + "\n")
            self.process.stdin.flush()

    def terminate(self):
        self.process.terminate()


class MCPHTTPClient:
    def __init__(self, url, headers=None, timeout=180.0):
        self.url = url
        self.headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if headers:
            self.headers.update(headers)
        self.timeout = timeout
        ssl_cert_dir = os.environ.get('SSL_CERT_DIR')
        if not ssl_cert_dir and not state.nocertify and os.name != 'nt':
            os.environ['SSL_CERT_DIR'] = '/etc/ssl/certs'

    def _read_sse(self, response) -> bytes:
        json_events = []
        buf = []
        for raw in response:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if not line:
                if buf:
                    payload = "\n".join(buf)
                    if payload and payload[0] in "{[":
                        json_events.append(payload)
                    buf = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                buf.append(line[5:].lstrip())
        if buf:
            payload = "\n".join(buf)
            if payload and payload[0] in "{[":
                json_events.append(payload)
        if not json_events:
            raise RuntimeError("MCP HTTP server returned no JSON SSE response")
        return json_events[-1].encode("utf-8")

    def send(self, message: dict, await_response=True) -> dict:
        data = json.dumps(message).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                sid = response.headers.get("MCP-Session-Id", "92604d65-d82c-468a-96e9-cf4463ba68fc")
                if sid:
                    self.headers["MCP-Session-Id"] = sid
                ctype = response.headers.get("Content-Type", "")
                body = self._read_sse(response) if "text/event-stream" in ctype else response.read()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"MCP HTTP error {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"MCP HTTP connection failed: {e.reason}") from e
        if not await_response:
            return None
        if not body:
            raise RuntimeError("MCP HTTP server returned empty response")
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"MCP HTTP server returned invalid JSON: {body!r}") from e

    def notify(self, message: dict) -> None:
        data = json.dumps(message).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout):
                pass
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"MCP HTTP notification failed ({e.code}): {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"MCP HTTP notification connection failed: {e.reason}") from e


def load_mcp_async(args):
    filepath = os.path.abspath(args.mcpfile)
    if not filepath.lower().endswith(".json"):
        filepath += ".json"
        args.mcpfile += ".json"
    try:
        print(f"MCP start loading json file at '{filepath}'...")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("MCP config must be a JSON object")
            servers = loaded.get("mcpServers")
            if not isinstance(servers, dict):
                serversVsCode = loaded.get("servers")
                if isinstance(serversVsCode, dict):
                    servers = serversVsCode
                else:
                    raise ValueError("MCP config missing 'mcpServers' object")
            for name, cfg in servers.items():
                try:
                    print(f"Connecting to MCP Server {name}...")
                    if not isinstance(cfg, dict):
                        raise ValueError(f"MCP server '{name}' must be an object")
                    mcpurl = cfg.get("url", "")
                    mcpcmd = cfg.get("command", "")
                    if mcpcmd and not mcpurl:
                        mcpargs = cfg.get("args", [])
                        mcpenv = cfg.get("env", {})
                        client = MCPStdioClient(command=mcpcmd, largs=mcpargs, env=mcpenv)
                    elif mcpurl:
                        mcp_ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
                        headers = cfg.get("headers", {})
                        headers.setdefault('User-Agent', mcp_ua)
                        client = MCPHTTPClient(url=mcpurl, headers=headers)
                    else:
                        raise ValueError(f"MCP server '{name}' missing 'command' and 'url'")
                    with state.mcp_lock:
                        state.mcp_connections.append({"client": client, "tools": [], "name": name})
                except Exception as e:
                    print(f"MCP Init Error: {e}")
            for conn in list(state.mcp_connections):
                try:
                    init_payload = {
                        "jsonrpc": "2.0",
                        "id": random.randint(100000, 999999),
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "koboldcpp", "version": "1.0.0"}
                        }
                    }
                    notif_payload = {
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized"
                    }
                    toolget_payload = {
                        "jsonrpc": "2.0",
                        "id": random.randint(100000, 999999),
                        "method": "tools/list",
                        "params": {}
                    }

                    resp1 = conn["client"].send(init_payload)
                    if "result" not in resp1:
                        continue

                    conn["client"].send(notif_payload, await_response=False)
                    resp2 = conn["client"].send(toolget_payload)

                    if "result" not in resp2 or "tools" not in resp2["result"]:
                        continue
                    with state.mcp_lock:
                        conn["tools"] = resp2["result"]["tools"]
                except Exception as e:
                    print(f"MCP Setup Error: {e}")
            print(f"Completed load of MCP json file at '{filepath}'.")
    except Exception as e:
        print(f"Failed to parse MCP json file at '{filepath}': {e}")
