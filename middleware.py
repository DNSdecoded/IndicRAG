"""Request ID propagation: correlates log lines from the same HTTP request."""

import contextvars
import logging
import uuid

from starlette.middleware.base import BaseHTTPMiddleware

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Injects the current request's ID into every log record as %(request_id)s."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_var.set(req_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = req_id
        return response
