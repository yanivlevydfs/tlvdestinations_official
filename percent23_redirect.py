from starlette.types import ASGIApp, Scope, Receive, Send
from starlette.responses import RedirectResponse

class Percent23RedirectMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            raw_path = scope.get("raw_path", scope["path"].encode("utf-8"))
            if b"%23" in raw_path:
                # Get host from headers
                headers = dict(scope["headers"])
                host = headers.get(b"host", b"localhost").decode("latin-1")
                scheme = scope.get("scheme", "https")
                redirect_url = f"{scheme}://{host}/"
                response = RedirectResponse(url=redirect_url, status_code=301)
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)
