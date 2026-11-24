"""Interactive OAuth 2.0 helper to generate `credentials/oauth_token.json` for Google Drive/Docs."""

from __future__ import annotations

import argparse
import html
import json
import sys
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx

OAUTH_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
]


def load_client_credentials(client_json_path: Path) -> tuple[str, str, str]:
    """Extract `(client_id, client_secret, client_kind)` from a Google OAuth client JSON."""
    data = json.loads(client_json_path.read_text(encoding="utf-8"))
    for kind in ("installed", "web"):
        if isinstance(data.get(kind), dict):
            section = data[kind]
            cid = str(section.get("client_id", "")).strip()
            csec = str(section.get("client_secret", "")).strip()
            if cid and csec:
                return cid, csec, kind
    raise ValueError(
        f"El archivo '{client_json_path}' no contiene credenciales válidas ('installed' o 'web')."
    )


def build_authorization_url(client_id: str, redirect_uri: str) -> str:
    """Build the Google OAuth 2.0 consent URL requesting offline refresh_token access."""
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
    }
    return f"{OAUTH_AUTH_URL}?{urllib.parse.urlencode(params)}"


def extract_code_from_input(raw_value: str) -> str:
    """Extract authorization `code` from either a raw code string or a full redirect URL."""
    cleaned = raw_value.strip()
    if "code=" in cleaned:
        parsed = urllib.parse.urlparse(cleaned)
        qs = urllib.parse.parse_qs(parsed.query)
        codes = qs.get("code", [])
        if codes and codes[0].strip():
            return codes[0].strip()
    return cleaned


def exchange_code_for_authorized_user(
    *,
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
) -> dict[str, Any]:
    """Exchange an OAuth authorization code for an `authorized_user` token dict."""
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            OAUTH_TOKEN_URL,
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
    if resp.is_error:
        raise RuntimeError(f"Error intercambiando código OAuth ({resp.status_code}): {resp.text}")
    payload = resp.json()
    refresh_token = str(payload.get("refresh_token", "")).strip()
    if not refresh_token:
        raise RuntimeError(
            "Google no devolvió 'refresh_token'. Revoca el acceso previo en myaccount.google.com/permissions "
            "y vuelve a ejecutar este script aceptando los permisos."
        )
    return {
        "type": "authorized_user",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "token_uri": OAUTH_TOKEN_URL,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _wait_for_local_callback(port: int) -> str:
    captured: dict[str, str] = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            if "code" in qs and qs["code"]:
                captured["code"] = qs["code"][0]
                body = (
                    "<html><body><h2>Autorizacion completada</h2>"
                    "<p>Ya puedes cerrar esta ventana y volver a la terminal.</p></body></html>"
                )
                self.send_response(200)
            else:
                err = html.escape((qs.get("error") or ["desconocido"])[0])
                captured["error"] = err
                body = f"<html><body><h2>Error OAuth: {err}</h2></body></html>"
                self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = HTTPServer(("127.0.0.1", port), _Handler)
    try:
        while "code" not in captured and "error" not in captured:
            server.handle_request()
    finally:
        server.server_close()

    if "error" in captured:
        raise RuntimeError(f"Autorización rechazada por Google: {captured['error']}")
    return captured["code"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera credentials/oauth_token.json para Google Drive/Docs.")
    parser.add_argument(
        "--client-secrets",
        type=Path,
        default=Path("credentials/oauth_client.json"),
        help="Ruta al JSON descargado de Google Cloud Console (default: credentials/oauth_client.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("credentials/oauth_token.json"),
        help="Ruta de salida para el token authorized_user (default: credentials/oauth_token.json).",
    )
    parser.add_argument("--port", type=int, default=8765, help="Puerto local para callback OAuth (default: 8765).")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Modo manual (pegar la URL/código de redirección en terminal, útil por SSH).",
    )
    args = parser.parse_args()

    client_id, client_secret, kind = load_client_credentials(args.client_secrets)
    redirect_uri = f"http://localhost:{args.port}"
    auth_url = build_authorization_url(client_id, redirect_uri)

    print(f"Cliente OAuth detectado (tipo={kind}).")
    print("\n1. Abre esta URL en tu navegador e inicia sesión con ceo@nelhealthcoach.com (o julioblogs1998@gmail.com):\n")
    print(auth_url)
    print()

    if args.manual:
        raw = input("2. Tras autorizar, pega aquí la URL completa de la barra del navegador (o el parámetro code): ")
        code = extract_code_from_input(raw)
    else:
        print(f"2. Esperando respuesta automática en {redirect_uri} ...")
        code = _wait_for_local_callback(args.port)

    token_data = exchange_code_for_authorized_user(
        client_id=client_id,
        client_secret=client_secret,
        code=code,
        redirect_uri=redirect_uri,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(token_data, indent=2), encoding="utf-8")
    print(f"\nListo. Token guardado en: {args.output}")
    print("Agrega en tu .env:")
    print(f"GOOGLE_OAUTH_CREDENTIALS=./{args.output.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
