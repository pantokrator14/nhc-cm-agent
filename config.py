"""Configuration loading and validation using project-root-relative paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = BASE_DIR / ".env"
DEFAULT_STATE_PATH = BASE_DIR / "state.json"
DEFAULT_CREDENTIALS_REL = "./credentials/service_account.json"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_WEBHOOK_VERIFY_TOKEN = "nelhealthcoach_verify_2026"
DEFAULT_WEBHOOK_PORT = 8000


class ConfigurationError(ValueError):
    """Raised when required environment configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated application settings loaded from `.env`."""

    gemini_api_key: str
    gemini_model: str

    google_application_credentials: Path
    drive_root_folder_id: str

    telegram_bot_token: str
    telegram_admin_chat_id: int

    meta_access_token: str
    ig_user_id: str
    meta_ad_account_id: str = ""

    webhook_verify_token: str = DEFAULT_WEBHOOK_VERIFY_TOKEN
    webhook_port: int = DEFAULT_WEBHOOK_PORT

    base_dir: Path = BASE_DIR
    state_file_path: Path = DEFAULT_STATE_PATH


def _require_env(name: str, missing: list[str]) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        missing.append(name)
    return value


def resolve_project_path(raw_path: str | Path) -> Path:
    """Resolve relative paths against `BASE_DIR` so systemd CWD does not matter."""
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (BASE_DIR / path).resolve()


def load_settings(env_path: str | Path | None = None) -> Settings:
    """Load and validate environment variables from `.env` in `BASE_DIR`.

    Raises:
        ConfigurationError: If required variables are missing or malformed.
    """
    resolved_env = resolve_project_path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    load_dotenv(dotenv_path=resolved_env, override=False)

    missing: list[str] = []

    gemini_api_key = _require_env("GEMINI_API_KEY", missing)
    gemini_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL

    credentials_raw = (
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS", DEFAULT_CREDENTIALS_REL).strip()
        or DEFAULT_CREDENTIALS_REL
    )
    drive_root_folder_id = _require_env("DRIVE_ROOT_FOLDER_ID", missing)

    telegram_bot_token = _require_env("TELEGRAM_BOT_TOKEN", missing)
    telegram_chat_raw = _require_env("TELEGRAM_ADMIN_CHAT_ID", missing)

    meta_access_token = _require_env("META_ACCESS_TOKEN", missing)
    ig_user_id = _require_env("IG_USER_ID", missing)
    raw_ad_account = os.getenv("META_AD_ACCOUNT_ID", "").strip()
    if raw_ad_account and not raw_ad_account.startswith("act_"):
        meta_ad_account_id = f"act_{raw_ad_account}"
    else:
        meta_ad_account_id = raw_ad_account

    webhook_verify_token = (
        os.getenv("WEBHOOK_VERIFY_TOKEN", DEFAULT_WEBHOOK_VERIFY_TOKEN).strip()
        or DEFAULT_WEBHOOK_VERIFY_TOKEN
    )
    webhook_port_raw = (
        os.getenv("WEBHOOK_PORT", str(DEFAULT_WEBHOOK_PORT)).strip()
        or str(DEFAULT_WEBHOOK_PORT)
    )

    if missing:
        raise ConfigurationError(
            "Faltan variables de entorno obligatorias en .env: " + ", ".join(missing)
        )

    try:
        telegram_admin_chat_id = int(telegram_chat_raw)
    except ValueError as exc:
        raise ConfigurationError(
            "TELEGRAM_ADMIN_CHAT_ID debe ser un número entero válido."
        ) from exc

    try:
        webhook_port = int(webhook_port_raw)
        if not (1 <= webhook_port <= 65535):
            raise ValueError
    except ValueError as exc:
        raise ConfigurationError(
            "WEBHOOK_PORT debe ser un puerto TCP válido entre 1 y 65535."
        ) from exc

    credentials_path = resolve_project_path(credentials_raw)
    if not credentials_path.is_file():
        raise ConfigurationError(
            f"No se encontró el archivo de Service Account en GOOGLE_APPLICATION_CREDENTIALS: {credentials_path}"
        )

    return Settings(
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        google_application_credentials=credentials_path,
        drive_root_folder_id=drive_root_folder_id,
        telegram_bot_token=telegram_bot_token,
        telegram_admin_chat_id=telegram_admin_chat_id,
        meta_access_token=meta_access_token,
        ig_user_id=ig_user_id,
        meta_ad_account_id=meta_ad_account_id,
        webhook_verify_token=webhook_verify_token,
        webhook_port=webhook_port,
        base_dir=BASE_DIR,
        state_file_path=DEFAULT_STATE_PATH,
    )
