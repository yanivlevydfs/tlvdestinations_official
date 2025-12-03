# middleware/percent23_redirect.py

import socket
from starlette.types import ASGIApp, Scope, Receive, Send
from starlette.responses import RedirectResponse, HTMLResponse

def is_googlebot(ip: str) -> bool:
    """Verify if the IP belongs to a real Googlebot."""
    try:
        # Reverse DNS (PTR)
        hostname, _, _ = socket.gethostbyaddr(ip)
        if not (hostname.endswith('.googlebot.com') or hostname.endswith('.google.com')):
            return False

        # Forward DNS check (A/AAAA)
        resolved_ips = socket.gethostbyname_ex(hostname)[2]
        return ip in resolved_ips
    except Exception:
        return False


class Percent23RedirectMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            raw_path = scope.get("raw_path", scope["path"].encode("utf-8"))

            if b"%23" in raw_path:
                headers = dict(scope["headers"])
                host = headers.get(b"host", b"localhost").decode("latin-1")
                scheme = scope.get("scheme", "https")
                redirect_url = f"{scheme}://{host}/"

                # Identify client
                user_agent = headers.get(b"user-agent", b"").decode("utf-8", errors="replace")
                client_ip = scope.get("client", ("?", 0))[0]

                is_bot = is_googlebot(client_ip)
                bot_flag = "🟢 Verified Googlebot" if is_bot else "❌ Unknown bot"                

                if is_bot:
                    html = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta http-equiv="refresh" content="3; url={redirect_url}">
                        <title>Malformed Anchor</title>
                    </head>
                    <body>
                        <h2>⚠️ Malformed Anchor Detected</h2>
                        <p>This request includes an encoded anchor (<code>%23</code>) in the URL path.</p>
                        <p>Fragments (e.g. <code>#section</code>) should not be encoded and sent to the server.</p>
                        <p>You'll be redirected shortly to the home page.</p>
                        <hr>
                        <small>Redirecting to <a href="{redirect_url}">{redirect_url}</a> in 3 seconds...</small><br>
                        <small>Thanks, bot friend 🤖</small>
                    </body>
                    </html>
                    """
                    response = HTMLResponse(content=html, status_code=200)
                else:
                    response = RedirectResponse(url=redirect_url, status_code=301)

                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)