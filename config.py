"""Configuration loading and validation using project-root-relative paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = BASE_DIR / ".env"
DEFAULT_STATE_PATH = BASE_DIR / "state.json"
DEFAULT_CREDENTIALS_REL = "./credentials/service_account.json"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_WEBHOOK_VERIFY_TOKEN = "nelhealthcoach_verify_2026"
DEFAULT_WEBHOOK_PORT = 8000
DEFAULT_META_SAVED_AUDIENCE_NAME = "mujeres"
DEFAULT_TIKTOK_PUBLISH_MODE = "DIRECT"
DEFAULT_TIKTOK_PRIVACY_LEVEL = "PUBLIC_TO_EVERYONE"
DEFAULT_TIMEZONE = "America/Caracas"


def resolve_timezone(tz_name: str = DEFAULT_TIMEZONE) -> tzinfo:
    """Resolve an IANA timezone name (defaults to Venezuela `America/Caracas`, UTC-4)."""
    cleaned = (tz_name or DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE
    try:
        return ZoneInfo(cleaned)
    except ZoneInfoNotFoundError as exc:
        if cleaned == DEFAULT_TIMEZONE:
            return timezone(timedelta(hours=-4), name="VET")
        raise ConfigurationError(f"TIMEZONE inválido o no disponible: '{cleaned}'.") from exc


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
    google_drive_impersonate_email: str = ""
    google_oauth_credentials: Path | None = None
    telegram_extra_admin_chat_ids: tuple[int, ...] = ()
    meta_ad_account_id: str = ""
    meta_saved_audience_name: str = DEFAULT_META_SAVED_AUDIENCE_NAME
    meta_app_id: str = ""
    meta_app_secret: str = ""
    meta_location_id_us: str = ""
    meta_location_id_ca: str = ""

    tiktok_client_key: str = ""
    tiktok_client_secret: str = ""
    tiktok_refresh_token: str = ""
    tiktok_access_token: str = ""
    tiktok_publish_mode: str = DEFAULT_TIKTOK_PUBLISH_MODE
    tiktok_privacy_level: str = DEFAULT_TIKTOK_PRIVACY_LEVEL

    webhook_verify_token: str = DEFAULT_WEBHOOK_VERIFY_TOKEN
    webhook_port: int = DEFAULT_WEBHOOK_PORT
    timezone_name: str = DEFAULT_TIMEZONE

    base_dir: Path = BASE_DIR
    state_file_path: Path = DEFAULT_STATE_PATH

    @property
    def tzinfo(self) -> tzinfo:
        """Return resolved `tzinfo` for all scheduling, Pautas, and reporting."""
        return resolve_timezone(self.timezone_name)

    @property
    def telegram_admin_chat_ids(self) -> tuple[int, ...]:
        """Return deduplicated authorized Telegram chat IDs (supports single group or multiple admins)."""
        ordered: list[int] = [self.telegram_admin_chat_id]
        for cid in self.telegram_extra_admin_chat_ids:
            if cid not in ordered:
                ordered.append(cid)
        return tuple(ordered)

    @property
    def tiktok_enabled(self) -> bool:
        """Return True if TikTok organic publishing credentials are configured."""
        has_refresh = bool(
            self.tiktok_refresh_token and self.tiktok_client_key and self.tiktok_client_secret
        )
        return has_refresh or bool(self.tiktok_access_token)


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
    google_drive_impersonate_email = os.getenv("GOOGLE_DRIVE_IMPERSONATE_EMAIL", "").strip()
    google_oauth_raw = os.getenv("GOOGLE_OAUTH_CREDENTIALS", "").strip()
    drive_root_folder_id = _require_env("DRIVE_ROOT_FOLDER_ID", missing)

    telegram_bot_token = _require_env("TELEGRAM_BOT_TOKEN", missing)
    telegram_chat_raw = _require_env("TELEGRAM_ADMIN_CHAT_ID", missing)
    extra_chats_raw = os.getenv("TELEGRAM_EXTRA_ADMIN_CHAT_IDS", "").strip()

    meta_access_token = _require_env("META_ACCESS_TOKEN", missing)
    ig_user_id = _require_env("IG_USER_ID", missing)
    raw_ad_account = os.getenv("META_AD_ACCOUNT_ID", "").strip()
    if raw_ad_account and not raw_ad_account.startswith("act_"):
        meta_ad_account_id = f"act_{raw_ad_account}"
    else:
        meta_ad_account_id = raw_ad_account
    meta_saved_audience_name = (
        os.getenv("META_SAVED_AUDIENCE_NAME", DEFAULT_META_SAVED_AUDIENCE_NAME).strip()
        or DEFAULT_META_SAVED_AUDIENCE_NAME
    )
    meta_app_id = os.getenv("META_APP_ID", "").strip()
    meta_app_secret = os.getenv("META_APP_SECRET", "").strip()
    meta_location_id_us = os.getenv("META_LOCATION_ID_US", "").strip()
    meta_location_id_ca = os.getenv("META_LOCATION_ID_CA", "").strip()

    tiktok_client_key = os.getenv("TIKTOK_CLIENT_KEY", "").strip()
    tiktok_client_secret = os.getenv("TIKTOK_CLIENT_SECRET", "").strip()
    tiktok_refresh_token = os.getenv("TIKTOK_REFRESH_TOKEN", "").strip()
    tiktok_access_token = os.getenv("TIKTOK_ACCESS_TOKEN", "").strip()
    raw_tiktok_mode = (
        os.getenv("TIKTOK_PUBLISH_MODE", DEFAULT_TIKTOK_PUBLISH_MODE).strip().upper()
        or DEFAULT_TIKTOK_PUBLISH_MODE
    )
    tiktok_publish_mode = raw_tiktok_mode if raw_tiktok_mode in {"DIRECT", "INBOX"} else DEFAULT_TIKTOK_PUBLISH_MODE
    tiktok_privacy_level = (
        os.getenv("TIKTOK_PRIVACY_LEVEL", DEFAULT_TIKTOK_PRIVACY_LEVEL).strip().upper()
        or DEFAULT_TIKTOK_PRIVACY_LEVEL
    )

    webhook_verify_token = (
        os.getenv("WEBHOOK_VERIFY_TOKEN", DEFAULT_WEBHOOK_VERIFY_TOKEN).strip()
        or DEFAULT_WEBHOOK_VERIFY_TOKEN
    )
    webhook_port_raw = (
        os.getenv("WEBHOOK_PORT", str(DEFAULT_WEBHOOK_PORT)).strip()
        or str(DEFAULT_WEBHOOK_PORT)
    )
    timezone_name = (
        os.getenv("TIMEZONE", DEFAULT_TIMEZONE).strip()
        or DEFAULT_TIMEZONE
    )
    resolve_timezone(timezone_name)

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

    extra_admin_ids: list[int] = []
    if extra_chats_raw:
        for part in extra_chats_raw.split(","):
            part_clean = part.strip()
            if not part_clean:
                continue
            try:
                extra_admin_ids.append(int(part_clean))
            except ValueError as exc:
                raise ConfigurationError(
                    f"TELEGRAM_EXTRA_ADMIN_CHAT_IDS contiene un ID inválido: '{part_clean}'."
                ) from exc

    try:
        webhook_port = int(webhook_port_raw)
        if not (1 <= webhook_port <= 65535):
            raise ValueError
    except ValueError as exc:
        raise ConfigurationError(
            "WEBHOOK_PORT debe ser un puerto TCP válido entre 1 y 65535."
        ) from exc

    google_oauth_credentials: Path | None = None
    if google_oauth_raw:
        google_oauth_credentials = resolve_project_path(google_oauth_raw)
        if not google_oauth_credentials.is_file():
            raise ConfigurationError(
                f"No se encontró el archivo OAuth de Google en GOOGLE_OAUTH_CREDENTIALS: {google_oauth_credentials}"
            )

    credentials_path = resolve_project_path(credentials_raw)
    if not credentials_path.is_file():
        if google_oauth_credentials is not None:
            credentials_path = google_oauth_credentials
        else:
            raise ConfigurationError(
                f"No se encontró el archivo de credenciales de Google en GOOGLE_APPLICATION_CREDENTIALS: {credentials_path}"
            )

    return Settings(
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        google_application_credentials=credentials_path,
        drive_root_folder_id=drive_root_folder_id,
        google_drive_impersonate_email=google_drive_impersonate_email,
        google_oauth_credentials=google_oauth_credentials,
        telegram_bot_token=telegram_bot_token,
        telegram_admin_chat_id=telegram_admin_chat_id,
        telegram_extra_admin_chat_ids=tuple(extra_admin_ids),
        meta_access_token=meta_access_token,
        ig_user_id=ig_user_id,
        meta_ad_account_id=meta_ad_account_id,
        meta_saved_audience_name=meta_saved_audience_name,
        meta_app_id=meta_app_id,
        meta_app_secret=meta_app_secret,
        meta_location_id_us=meta_location_id_us,
        meta_location_id_ca=meta_location_id_ca,
        tiktok_client_key=tiktok_client_key,
        tiktok_client_secret=tiktok_client_secret,
        tiktok_refresh_token=tiktok_refresh_token,
        tiktok_access_token=tiktok_access_token,
        tiktok_publish_mode=tiktok_publish_mode,
        tiktok_privacy_level=tiktok_privacy_level,
        webhook_verify_token=webhook_verify_token,
        webhook_port=webhook_port,
        timezone_name=timezone_name,
        base_dir=BASE_DIR,
        state_file_path=DEFAULT_STATE_PATH,
    )
