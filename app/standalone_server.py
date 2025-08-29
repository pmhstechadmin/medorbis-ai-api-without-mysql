#!/usr/bin/env python3
import json
import os
import sys
import argparse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from io import BytesIO
import cgi


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _read_body(handler: BaseHTTPRequestHandler) -> tuple[bytes, str]:
    length = int(handler.headers.get('Content-Length', '0') or '0')
    body = handler.rfile.read(length) if length > 0 else b''
    ctype = handler.headers.get('Content-Type', '')
    return body, ctype


def _parse_post(handler: BaseHTTPRequestHandler):
    body, ctype = _read_body(handler)
    ctype_lower = ctype.lower()
    if 'application/json' in ctype_lower:
        try:
            return json.loads(body.decode('utf-8') or '{}')
        except Exception:
            raise ValueError("Invalid JSON body")
    if 'application/x-www-form-urlencoded' in ctype_lower:
        qs = body.decode('utf-8')
        return {k: v[0] if isinstance(v, list) else v for k, v in parse_qs(qs, keep_blank_values=True).items()}
    if 'multipart/form-data' in ctype_lower:
        fp = BytesIO(body)
        env = {
            'REQUEST_METHOD': 'POST',
            'CONTENT_TYPE': ctype,
            'CONTENT_LENGTH': str(len(body)),
        }
        form = cgi.FieldStorage(fp=fp, headers=handler.headers, environ=env)
        data = {}
        for key in form.keys() or []:
            field = form[key]
            if isinstance(field, list):
                data[key] = [f.value for f in field]
            else:
                data[key] = field.value
        return data
    raise ValueError("Unsupported Media Type. Use application/json or form-data")


# --- Embeddings using Sentence-Transformers via Hugging Face Inference API (no external deps) ---

def _hf_embed(texts: list[str]) -> list[list[float]] | None:
    model = os.getenv('SENTENCE_MODEL') or os.getenv('HUGGINGFACE_MODEL') or 'sentence-transformers/all-MiniLM-L6-v2'
    endpoint = os.getenv('HUGGINGFACE_API_URL') or f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
    token = os.getenv('HUGGINGFACE_API_KEY')
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f"Bearer {token}"
    payload = json.dumps(texts).encode('utf-8')
    try:
        req = Request(endpoint, data=payload, headers=headers, method='POST')
        with urlopen(req, timeout=30) as resp:
            data = resp.read()
            parsed = json.loads(data)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], (int, float)):
                return [parsed]
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], list):
                return parsed
            return None
    except HTTPError as e:
        sys.stderr.write(f"HF embed HTTPError: {e.code} {e.reason}\n")
        return None
    except URLError as e:
        sys.stderr.write(f"HF embed URLError: {e.reason}\n")
        return None
    except Exception as e:
        sys.stderr.write(f"HF embed error: {e}\n")
        return None


# --- Qdrant search via REST API (no qdrant-client dependency) ---

def _build_qdrant_filter(dept: str, year: str, sem: str) -> dict | None:
    must = []
    if dept:
        must.append({"key": "Department", "match": {"value": dept}})
    if year:
        must.append({"key": "Year", "match": {"value": year}})
    if sem:
        must.append({"key": "Semester", "match": {"value": sem}})
    return {"must": must} if must else None


def _qdrant_search_vec(vec: list[float], q_filter: dict | None = None, limit: int = 3) -> list[dict]:
    url = os.getenv('QDRANT_URL')
    api_key = os.getenv('QDRANT_API_KEY')
    collection = os.getenv('QDRANT_COLLECTION', 'documents')
    vector_name = os.getenv('QDRANT_VECTOR_NAME')
    if not url:
        return []

    body = {
        "limit": limit,
        "with_payload": True,
    }
    if q_filter:
        body["filter"] = q_filter
    if vector_name:
        body["vector"] = {vector_name: vec}
    else:
        body["vector"] = vec

    endpoint = url.rstrip('/') + f"/collections/{collection}/points/search"
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['api-key'] = api_key
    try:
        req = Request(endpoint, data=json.dumps(body).encode('utf-8'), headers=headers, method='POST')
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read() or b'{}')
            res = data.get('result') or []
            return res if isinstance(res, list) else []
    except HTTPError as e:
        sys.stderr.write(f"Qdrant HTTPError: {e.code} {e.reason}\n")
        return []
    except URLError as e:
        sys.stderr.write(f"Qdrant URLError: {e.reason}\n")
        return []
    except Exception as e:
        sys.stderr.write(f"Qdrant error: {e}\n")
        return []


def _qdrant_contexts_from_results(results: list[dict]) -> list[str]:
    contexts: list[str] = []
    for p in results:
        payload = p.get('payload') or {}
        text = payload.get('text') or payload.get('content') or payload.get('chunk') or payload.get('body')
        if isinstance(text, (str, int, float)):
            contexts.append(str(text))
    return contexts


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "MedOrbisSimpleHTTP/1.2"

    def _send_headers(self, status=200, extra_headers=None, content_type='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Credentials', 'true')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()

    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode('utf-8')
        self._send_headers(status=status)
        self.wfile.write(data)

    def do_OPTIONS(self):
        self._send_headers(204)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == '/':
            return self._send_json({"message": "API is running"})
        if path == '/health':
            return self._send_json({"status": "ok"})

        if path in ('/api/v2/chat', '/v2/chat'):
            return self._send_json({
                "endpoint": "/api/v2/chat",
                "method": "POST",
                "content_type": "application/json",
                "example": {"messages": [{"role": "user", "content": "Hello"}]},
            })
        if path in ('/api/v2/chat/echo', '/v2/chat/echo'):
            content = (query.get('content', [''])[0] or '').strip()
            reply_text = f"You said: {content}" if content else "Hello! How can I help you?"
            usage = {
                "input_tokens": _word_count(content),
                "output_tokens": _word_count(reply_text),
                "total_tokens": _word_count(content) + _word_count(reply_text),
            }
            return self._send_json({"reply": reply_text, "usage": usage})

        if path == '/api/v1/chat':
            return self._send_json({
                "endpoint": "/api/v1/chat",
                "method": "POST",
                "content_type": "application/json or multipart/form-data",
                "example": {
                    "user_type": 0,
                    "user_id": "string",
                    "session_id": "string",
                    "user_question": "string",
                    "Department": "Nursing",
                    "Year": "3",
                    "Semester": "1"
                },
                "note": "You can also use user_department, user_year, user_semester as alternative field names."
            })
        if path == '/api/v1/chat/echo':
            content = (query.get('content', [''])[0] or '').strip()
            reply_text = f"You said: {content}" if content else "Hello! How can I help you?"
            usage = {
                "input_tokens": _word_count(content),
                "output_tokens": _word_count(reply_text),
                "total_tokens": _word_count(content) + _word_count(reply_text),
            }
            return self._send_json({"reply": reply_text, "usage": usage})
        if path == '/api/v1/chat/test':
            def _q(name, default=""):
                vals = query.get(name, [default])
                return vals[0] if vals else default
            user_type = int(_q('user_type', '0') or '0')
            payload = {
                "user_type": user_type,
                "user_id": _q('user_id'),
                "session_id": _q('session_id'),
                "user_question": _q('user_question'),
                # Accept both capitalized metadata and legacy fields
                "Department": _q('Department') or _q('user_department'),
                "Year": _q('Year') or _q('user_year'),
                "Semester": _q('Semester') or _q('user_semester'),
                "model": _q('model', None),
            }
            return self._send_json(self._generate_v1_reply(payload))

        return self._send_json({"detail": "Not Found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in ('/api/v2/chat', '/v2/chat'):
                data = _parse_post(self)
                messages = data.get('messages') or []
                if isinstance(messages, str):
                    try:
                        messages = json.loads(messages)
                    except Exception:
                        raise ValueError("Invalid 'messages'; expected JSON array")
                last_user = None
                for m in reversed(messages):
                    role = (m.get('role') or '').lower()
                    if role == 'user':
                        last_user = m
                        break
                user_text = (last_user.get('content') or '').strip() if last_user else ''
                reply_text = f"You said: {user_text}" if user_text else "Hello! How can I help you?"
                input_tokens = _word_count(" ".join([m.get('content') or '' for m in messages])) if messages else 0
                usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": _word_count(reply_text),
                    "total_tokens": input_tokens + _word_count(reply_text),
                }
                return self._send_json({"reply": reply_text, "usage": usage})

            if path == '/api/v1/chat':
                data = _parse_post(self)
                payload = {
                    "user_type": int(str(data.get('user_type', 0) or '0')),
                    "user_id": data.get('user_id', '') or '',
                    "session_id": data.get('session_id', '') or '',
                    "user_question": data.get('user_question', '') or '',
                    # Accept capitalized metadata; fallback to legacy names
                    "Department": data.get('Department') or data.get('user_department') or '',
                    "Year": data.get('Year') or data.get('user_year') or '',
                    "Semester": data.get('Semester') or data.get('user_semester') or '',
                    "model": data.get('model') if data.get('model') not in (None, '') else None,
                }
                return self._send_json(self._generate_v1_reply(payload))
        except ValueError as e:
            return self._send_json({"detail": str(e)}, status=400)
        except Exception:
            return self._send_json({"detail": "Internal Server Error"}, status=500)

        return self._send_json({"detail": "Not Found"}, status=404)

    def log_message(self, format, *args):
        sys.stdout.write("%s - - [%s] " % (self.client_address[0], self.log_date_time_string()))
        sys.stdout.write(format % args)
        sys.stdout.write("\n")

    def _generate_v1_reply(self, req: dict) -> dict:
        contexts: list[str] = []
        used_vector = False
        dept = str(req.get('Department') or req.get('user_department') or '')
        year = str(req.get('Year') or req.get('user_year') or '')
        sem = str(req.get('Semester') or req.get('user_semester') or '')
        if int(req.get('user_type') or 0) == 0:
            texts = [str(req.get('user_question') or '')]
            vecs = _hf_embed(texts) if texts and texts[0] else None
            if vecs and vecs[0]:
                used_vector = True
                q_filter = _build_qdrant_filter(dept=dept, year=year, sem=sem)
                results = _qdrant_search_vec(vecs[0], q_filter=q_filter, limit=int(os.getenv('QDRANT_TOP_K', '3')))
                contexts = _qdrant_contexts_from_results(results)

        prompt_context = (
            f"user_type: {req.get('user_type')}\n"
            f"user_id: {req.get('user_id')}\n"
            f"session_id: {req.get('session_id')}\n"
            f"Department: {dept}\n"
            f"Year: {year}\n"
            f"Semester: {sem}\n\n"
            f"Question: {req.get('user_question')}\n"
        )
        reply_parts = []
        if contexts:
            reply_parts.append("Relevant information:\n" + "\n---\n".join(contexts))
        reply_parts.append(
            f"Response: Based on your context (Department={dept}, Year={year}, Semester={sem}), "
            f"here's a helpful response to your question: {req.get('user_question')}"
        )
        reply_text = "\n\n".join(reply_parts) if reply_parts else (
            f"[Local] Based on your context (Department={dept}, Year={year}, Semester={sem}), "
            f"here's a helpful response to your question: {req.get('user_question')}"
        )
        tokens_in = _word_count(prompt_context)
        if contexts:
            tokens_in += _word_count(" ".join(contexts))
        tokens_out = _word_count(reply_text)
        usage = {
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out,
        }
        meta = {"used_vector": used_vector, "contexts": len(contexts)}
        return {"reply": reply_text, "usage": usage, "meta": meta}


def run(host: str, port: int):
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, ApiHandler)
    print(f"Serving on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MedOrbis API (stdlib server)')
    parser.add_argument('--host', default=os.getenv('HOST', '0.0.0.0'))
    parser.add_argument('--port', type=int, default=int(os.getenv('PORT', '8000')))
    args = parser.parse_args()
    run(args.host, args.port)
