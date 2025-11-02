from fastapi import FastAPI, Request, Response, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_403_FORBIDDEN
import re

# Suspicious patterns (same as your list)
SUSPICIOUS_PATTERNS = [
    # === PHP / server-side files ===
    r"(^|/)phpinfo(\.php)?$", r"(^|/)(index|test|info|setup|config|env|settings)\.php$", r"\.php$",
    r"wp-config\.php", r"configuration\.php",
    
    # === WordPress scanning ===
    r"(^|/)wp-(admin|login|config|includes|content)(/.*)?(\.php)?$",
    r"(^|/)xmlrpc\.php$", r"(^|/)readme\.html?$", r"(^|/)license\.txt$",
    r"(^|/)trackback/?$", r"(^|/)feed/?$", r"(^|/)wp-json(/.*)?$",
    r"(^|/)index\.php/\?rest_route=.*",
    r"(^|/).+\.php(\.bak|\.old|\.save|~)?$",

    # === Dotfiles / hidden folders ===
    r"(^|/)\.(env|aws|git|svn|hg|idea|vscode|DS_Store|history|cache|npm|editorconfig|pytest_cache|tox)$",
    r"\.swp$", r"\.pid$", r"\.tmp$", r"~$", r"\.bak$", r"\.old$",

    # === Secrets / credentials / tokens / keys ===
    r"(?i)(access|auth|api|private|client|secret|session|vault|token|key|credentials)[_.-]?(id|key|secret)?(\.txt|\.json|\.yml|\.yaml|ini|cfg)?$",
    r"(?i)(stripe|sendgrid|twilio|slack|github|gitlab|heroku|aws|gcp|azure)[-_]?(token|key|secret)?",

    # === Database dumps / backups ===
    r"(?i)(backup|dump|database|db|data|export|site)[-_\.]?(dump|backup)?\.(sql|sqlite|json|db|csv|gz|zip|tar|bz2|bak|old|7z|rar)$",

    # === DevOps / Infrastructure / CI/CD ===
    r"(?i)(terraform|ansible|k8s|kubernetes|docker|jenkins|travis|gitlab|github|helm|vault|argo|vagrant).*",
    r"\.tfstate(\.backup)?$", r"(^|/)(dockerfile|docker-compose\.ya?ml)$",
    r"(cloudbuild|buildspec|azure|bitbucket|gitlab-ci|circleci|appveyor).*\.ya?ml",
    r"(main|deploy|settings|application|config|values|Chart|kustomization)\.(ya?ml|json|ini|properties|xml)$",

    # === Build / dependency files ===
    r"(composer|package(-lock)?|yarn|pnpm-lock)\.json$", r"pyproject\.toml", r"Pipfile(\.lock)?", r"requirements\.txt",
    r"(Gemfile|gradle\.properties|pom\.xml|build\.gradle|Cargo\.toml)$",

    # === ML / AI model & config files ===
    r"(?i)(model|weights|checkpoint|tokenizer|vectorizer|embeddings)\.(pt|pth|h5|onnx|pkl|bin|joblib)$",
    r"(config|params|hyperparams)\.(yaml|yml|json|ini|cfg)$",

    # === Log files ===
    r"(?i)(debug|error|access|application|server|npm[-_]?debug|yarn[-_]?error)\.log$",
    r"\.log$",

    # === Test / CI output ===
    r"(coverage|test-results|report|junit|pytest)\.(xml|json|html)$",
    r"coverage\.xml$", r"\.coverage$", r"junit\.xml$",

    # === Archive / backup / snapshots ===
    r"\.(zip|tar|gz|rar|7z|bak|tmp|swp|log|old|orig|save|backup)$",

]


COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_PATTERNS]

# Middleware to block suspicious paths
BAD_AGENTS = [
    "curl", "wget", "python-requests", "nikto", "fimap", "masscan", "nmap",
    "sqlmap", "dirbuster", "dirb", "wpscan", "acunetix", "netcraft", "curl",
    "HTTrack", "httpx", "fetch", "Go-http-client", "libwww", "Scrapy",
    "Java", "CensysInspect", "scanner", "bot", "spider"
]

class HardenedSecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        user_agent = request.headers.get("user-agent", "").lower()
        client_ip = request.client.host

        # Block by suspicious path
        for pattern in COMPILED_PATTERNS:
            if pattern.search(path):
                return Response(content="🚫 Forbidden", status_code=HTTP_403_FORBIDDEN)

        # Block by suspicious user-agent
        for bad in BAD_AGENTS:
            if bad in user_agent:
                return Response(content="🚫 Forbidden", status_code=HTTP_403_FORBIDDEN)

        # Pass through if clean
        return await call_next(request)

