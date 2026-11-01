"""fake_server.py - a stand-in OpenAI-compatible server for dev/test.sh.

usage: fake_server.py PORT ENGINE [REASONING] [MODEL_PATH]
  ENGINE     llama.cpp | vllm
  REASONING  1: replies carry reasoning_content
  MODEL_PATH what /props or /v1/models reports as the model file
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port, engine = int(sys.argv[1]), sys.argv[2]
reasoning = len(sys.argv) > 3 and sys.argv[3] == "1"
model_path = sys.argv[4] if len(sys.argv) > 4 else "/models/test.gguf"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = self.path.rstrip("/")
        if engine == "llama.cpp" and p == "/props":
            self.send({"build_info": "b9999-abcdef0", "model_path": model_path,
                       "total_slots": 4, "default_generation_settings": {"n_ctx": 32768}})
        elif engine == "vllm" and p == "/version":
            self.send({"version": "0.99.0"})
        elif p == "/v1/models":
            self.send({"data": [{"id": "test-model", "owned_by": engine.replace(".cpp", ""),
                                 "root": model_path, "max_model_len": 32768}]})
        else:
            self.send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if req.get("model") != "test-model":
            return self.send({"error": "unknown model"}, 404)
        msg = {"role": "assistant", "content": "2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37."}
        if reasoning:
            msg["reasoning_content"] = "Let me think about primes."
        self.send({"choices": [{"message": msg, "finish_reason": "stop"}]})


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
