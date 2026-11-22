"""Google Drive integration for hierarchical YYYY/MM folders and local state tracking."""

from __future__ import annotations

import calendar
import html as html_lib
import json
import logging
import os
import re
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Literal, NamedTuple

import google_auth_httplib2
import httplib2
from google.oauth2 import service_account
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaInMemoryUpload, MediaIoBaseDownload

from config import DEFAULT_CREDENTIALS_REL, DEFAULT_STATE_PATH, resolve_project_path

logger = logging.getLogger(__name__)

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
DOCS_SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]
SPANISH_WEEKDAYS = (
    "Lunes",
    "Martes",
    "Miércoles",
    "Jueves",
    "Viernes",
    "Sábado",
    "Domingo",
)
SUPPORTED_MIME_TYPES = {
    "video/mp4",
    "image/jpeg",
    "image/png",
}
SUPPORTED_EXTENSIONS = {".mp4", ".jpg", ".jpeg", ".png"}
GOOGLE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
GOOGLE_DOC_MIME_TYPE = "application/vnd.google-apps.document"
FINAL_SKIP_STATUSES = {
    "APPROVED",
    "PUBLISHED",
    "DISCARDED",
    "ALREADY_IN_GUIDELINES",
}

_STATE_LOCK = threading.Lock()


class DriveServiceError(RuntimeError):
    """Raised when a Google Drive API or state operation fails."""


class TempPublicLink(NamedTuple):
    """Temporary public download link and its Drive permission identifier."""

    url: str
    permission_id: str


class MonthFolderResolution(NamedTuple):
    """Resolved monthly Drive folder ID and whether it was newly created."""

    folder_id: str | None
    created: bool


def _get_state_path(state_path: Path | str | None = None) -> Path:
    if state_path is not None:
        return resolve_project_path(state_path)
    return DEFAULT_STATE_PATH


def _resolve_credentials_path(credentials_path: Path | str | None = None) -> Path:
    raw_creds = (
        str(credentials_path)
        if credentials_path is not None
        else os.getenv("GOOGLE_APPLICATION_CREDENTIALS", DEFAULT_CREDENTIALS_REL).strip()
        or DEFAULT_CREDENTIALS_REL
    )
    creds_path = resolve_project_path(raw_creds)
    if not creds_path.is_file():
        raise DriveServiceError(f"Archivo de credenciales no encontrado: {creds_path}")
    return creds_path


def _get_service_account_email(creds_path: Path) -> str:
    try:
        data = json.loads(creds_path.read_text(encoding="utf-8"))
        return str(data.get("client_email", "")).strip()
    except (OSError, json.JSONDecodeError):
        return ""


DRIVE_HTTP_TIMEOUT_SECONDS = 60


def _get_drive_client(credentials_path: Path | str | None = None) -> Resource:
    creds_path = _resolve_credentials_path(credentials_path)
    credentials = service_account.Credentials.from_service_account_file(
        str(creds_path),
        scopes=DRIVE_SCOPES,
    )
    authorized_http = google_auth_httplib2.AuthorizedHttp(
        credentials,
        http=httplib2.Http(timeout=DRIVE_HTTP_TIMEOUT_SECONDS),
    )
    return build("drive", "v3", http=authorized_http, cache_discovery=False)


def _get_docs_client(credentials_path: Path | str | None = None) -> Resource:
    creds_path = _resolve_credentials_path(credentials_path)
    credentials = service_account.Credentials.from_service_account_file(
        str(creds_path),
        scopes=DOCS_SCOPES,
    )
    authorized_http = google_auth_httplib2.AuthorizedHttp(
        credentials,
        http=httplib2.Http(timeout=DRIVE_HTTP_TIMEOUT_SECONDS),
    )
    return build("docs", "v1", http=authorized_http, cache_discovery=False)


def _execute_with_retry(
    request: Any,
    action_name: str,
    max_attempts: int = 3,
) -> Any:
    """Execute a Google API request with retries on transient socket timeouts."""
    for attempt in range(1, max_attempts + 1):
        try:
            return request.execute(num_retries=2)
        except (TimeoutError, OSError, httplib2.HttpLib2Error) as exc:
            if attempt >= max_attempts:
                raise DriveServiceError(
                    f"Timeout/error de red tras {max_attempts} intentos en '{action_name}': {exc}"
                ) from exc
            wait_seconds = attempt * 2
            logger.warning(
                "Error de red transitorio en '%s' (intento %d/%d): %s. Reintentando en %ds...",
                action_name,
                attempt,
                max_attempts,
                exc,
                wait_seconds,
            )
            time.sleep(wait_seconds)


def _escape_drive_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _empty_state() -> dict[str, Any]:
    return {
        "files": {},
        "media_posts": {},
        "processed_comments": {},
        "published_guideline_headers": {},
        "tiktok_auth": {},
        "meta_auth": {},
        "location_state": {"next_index": 0, "cache": {}},
        "scheduled_sessions": {},
        "performance_memory": {},
    }


def _normalize_state_shape(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return _empty_state()
    return {
        "files": data["files"] if isinstance(data.get("files"), dict) else {},
        "media_posts": data["media_posts"] if isinstance(data.get("media_posts"), dict) else {},
        "processed_comments": (
            data["processed_comments"]
            if isinstance(data.get("processed_comments"), dict)
            else {}
        ),
        "published_guideline_headers": (
            data["published_guideline_headers"]
            if isinstance(data.get("published_guideline_headers"), dict)
            else {}
        ),
        "tiktok_auth": data["tiktok_auth"] if isinstance(data.get("tiktok_auth"), dict) else {},
        "meta_auth": data["meta_auth"] if isinstance(data.get("meta_auth"), dict) else {},
        "location_state": (
            data["location_state"]
            if isinstance(data.get("location_state"), dict)
            else {"next_index": 0, "cache": {}}
        ),
        "scheduled_sessions": (
            data["scheduled_sessions"]
            if isinstance(data.get("scheduled_sessions"), dict)
            else {}
        ),
        "performance_memory": (
            data["performance_memory"]
            if isinstance(data.get("performance_memory"), dict)
            else {}
        ),
    }


def _read_state_unlocked(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DriveServiceError(f"No se pudo leer {path}: {exc}") from exc
    return _normalize_state_shape(data)


def _save_state_unlocked(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def load_state(state_path: Path | str | None = None) -> dict[str, Any]:
    """Load the local state registry (`files`, `media_posts`, `processed_comments`)."""
    path = _get_state_path(state_path)
    with _STATE_LOCK:
        return _read_state_unlocked(path)


def get_scheduled_times_from_state(state_path: Path | str | None = None) -> list[str]:
    """Return ISO scheduled_time strings for APPROVED/PUBLISHED files in `state.json`."""
    state = load_state(state_path)
    slots: list[str] = []
    for entry in state["files"].values():
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status", "")).upper()
        scheduled = str(entry.get("scheduled_time", "")).strip()
        if status in {"APPROVED", "PUBLISHED"} and scheduled:
            slots.append(scheduled)
    return sorted(slots)


def mark_file_state(
    file_id: str,
    status: str,
    metadata: dict[str, Any] | None = None,
    state_path: Path | str | None = None,
) -> None:
    """Persist or update a Drive file's processing status in `state.json`."""
    path = _get_state_path(state_path)
    with _STATE_LOCK:
        try:
            state = _read_state_unlocked(path)
        except DriveServiceError:
            state = _empty_state()

        files = state["files"]
        entry = files.get(file_id, {})
        entry.update(
            {
                "status": status.upper(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if metadata:
            entry.update(metadata)
        files[file_id] = entry
        _save_state_unlocked(state, path)


def _normalize_match_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", ascii_text.lower()).strip()


def mark_guideline_header_published(
    doc_header: str,
    month_folder_id: str = "",
    file_id: str = "",
    state_path: Path | str | None = None,
) -> None:
    """Record a guideline doc header as manually/already published in `state.json`."""
    key = _normalize_match_text(doc_header)
    if not key:
        return
    path = _get_state_path(state_path)
    with _STATE_LOCK:
        try:
            state = _read_state_unlocked(path)
        except DriveServiceError:
            state = _empty_state()

        state["published_guideline_headers"][key] = {
            "header": doc_header.strip(),
            "month_folder_id": month_folder_id,
            "file_id": file_id,
            "marked_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_state_unlocked(state, path)


def get_tiktok_auth_state(state_path: Path | str | None = None) -> dict[str, Any]:
    """Return persisted TikTok OAuth tokens (`access_token`, `refresh_token`, `expires_at`)."""
    state = load_state(state_path)
    auth = state.get("tiktok_auth")
    return dict(auth) if isinstance(auth, dict) else {}


def save_tiktok_auth_state(
    auth_data: dict[str, Any],
    state_path: Path | str | None = None,
) -> None:
    """Persist refreshed TikTok OAuth tokens in `state.json`."""
    path = _get_state_path(state_path)
    with _STATE_LOCK:
        try:
            state = _read_state_unlocked(path)
        except DriveServiceError:
            state = _empty_state()
        state["tiktok_auth"] = dict(auth_data)
        _save_state_unlocked(state, path)


def get_meta_auth_state(state_path: Path | str | None = None) -> dict[str, Any]:
    """Return persisted Meta access token metadata (`access_token`, `expires_at`, `checked_at`)."""
    state = load_state(state_path)
    auth = state.get("meta_auth")
    return dict(auth) if isinstance(auth, dict) else {}


def save_meta_auth_state(
    auth_data: dict[str, Any],
    state_path: Path | str | None = None,
) -> None:
    """Persist refreshed or inspected Meta access token metadata in `state.json`."""
    path = _get_state_path(state_path)
    with _STATE_LOCK:
        try:
            state = _read_state_unlocked(path)
        except DriveServiceError:
            state = _empty_state()
        existing = state.get("meta_auth")
        merged = dict(existing) if isinstance(existing, dict) else {}
        merged.update(auth_data)
        state["meta_auth"] = merged
        _save_state_unlocked(state, path)


def get_effective_meta_token(
    env_token: str,
    state_path: Path | str | None = None,
) -> str:
    """Return the freshest valid Meta access token (persisted in state or fallback to `.env`)."""
    meta_auth = get_meta_auth_state(state_path)
    persisted = str(meta_auth.get("access_token", "")).strip()
    expires_at = int(meta_auth.get("expires_at") or 0)
    if persisted and (expires_at == 0 or expires_at > time.time() + 60):
        return persisted
    return env_token.strip()


LOCATION_ROTATION_OPTIONS: tuple[str, ...] = ("Estados Unidos", "California")


def get_next_location_label(
    *,
    advance: bool = False,
    state_path: Path | str | None = None,
) -> str:
    """Return the next alternating Instagram location label ('Estados Unidos' <-> 'California')."""
    path = _get_state_path(state_path)
    with _STATE_LOCK:
        try:
            state = _read_state_unlocked(path)
        except DriveServiceError:
            state = _empty_state()
        loc_state = state.get("location_state")
        if not isinstance(loc_state, dict):
            loc_state = {"next_index": 0, "cache": {}}
        idx = int(loc_state.get("next_index") or 0) % len(LOCATION_ROTATION_OPTIONS)
        label = LOCATION_ROTATION_OPTIONS[idx]
        if advance:
            loc_state["next_index"] = (idx + 1) % len(LOCATION_ROTATION_OPTIONS)
            if not isinstance(loc_state.get("cache"), dict):
                loc_state["cache"] = {}
            state["location_state"] = loc_state
            _save_state_unlocked(state, path)
        return label


def get_cached_location_id(
    label: str,
    state_path: Path | str | None = None,
) -> str:
    """Read cached Facebook Page location_id for `label` from `state.json`."""
    state = load_state(state_path)
    loc_state = state.get("location_state") or {}
    cache = loc_state.get("cache") if isinstance(loc_state, dict) else {}
    if not isinstance(cache, dict):
        return ""
    return str(cache.get(label, "")).strip()


def save_cached_location_id(
    label: str,
    location_id: str,
    state_path: Path | str | None = None,
) -> None:
    """Cache resolved Facebook Page `location_id` for `label` in `state.json`."""
    if not label.strip() or not location_id.strip():
        return
    path = _get_state_path(state_path)
    with _STATE_LOCK:
        try:
            state = _read_state_unlocked(path)
        except DriveServiceError:
            state = _empty_state()
        loc_state = state.get("location_state")
        if not isinstance(loc_state, dict):
            loc_state = {"next_index": 0, "cache": {}}
        cache = loc_state.get("cache")
        if not isinstance(cache, dict):
            cache = {}
        cache[label.strip()] = location_id.strip()
        loc_state["cache"] = cache
        loc_state.setdefault("next_index", 0)
        state["location_state"] = loc_state
        _save_state_unlocked(state, path)


def save_scheduled_session(
    proposal_id: str,
    session_data: dict[str, Any],
    state_path: Path | str | None = None,
) -> None:
    """Persist an approved `ProposalSession` in `state.json` so it survives restarts."""
    path = _get_state_path(state_path)
    with _STATE_LOCK:
        try:
            state = _read_state_unlocked(path)
        except DriveServiceError:
            state = _empty_state()
        state["scheduled_sessions"][str(proposal_id)] = dict(session_data)
        _save_state_unlocked(state, path)


def remove_scheduled_session(
    proposal_id: str,
    state_path: Path | str | None = None,
) -> None:
    """Remove a completed/discarded `ProposalSession` from `state.json`."""
    path = _get_state_path(state_path)
    with _STATE_LOCK:
        try:
            state = _read_state_unlocked(path)
        except DriveServiceError:
            return
        if str(proposal_id) in state["scheduled_sessions"]:
            state["scheduled_sessions"].pop(str(proposal_id), None)
            _save_state_unlocked(state, path)


def load_scheduled_sessions(
    state_path: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load persisted scheduled sessions for startup rehydration."""
    state = load_state(state_path)
    raw = state.get("scheduled_sessions", {})
    return {
        str(k): dict(v)
        for k, v in raw.items()
        if isinstance(v, dict)
    }


def get_performance_memory(state_path: Path | str | None = None) -> dict[str, Any]:
    """Return learned weekly performance insights stored in `state.json`."""
    state = load_state(state_path)
    mem = state.get("performance_memory")
    return dict(mem) if isinstance(mem, dict) else {}


def save_performance_memory(
    memory_data: dict[str, Any],
    state_path: Path | str | None = None,
) -> None:
    """Persist weekly performance insights in `state.json` to guide future Gemini proposals."""
    path = _get_state_path(state_path)
    with _STATE_LOCK:
        try:
            state = _read_state_unlocked(path)
        except DriveServiceError:
            state = _empty_state()
        existing = state.get("performance_memory")
        merged = dict(existing) if isinstance(existing, dict) else {}
        merged.update(memory_data)
        state["performance_memory"] = merged
        _save_state_unlocked(state, path)


def save_published_media(
    media_id: str,
    has_keyword_cta: bool,
    keyword: str,
    target_link: str,
    public_reply_text: str,
    file_id: str | None = None,
    state_path: Path | str | None = None,
) -> None:
    """Persist published Instagram `media_id` and DM automation metadata in `state.json`."""
    path = _get_state_path(state_path)
    with _STATE_LOCK:
        try:
            state = _read_state_unlocked(path)
        except DriveServiceError:
            state = _empty_state()

        state["media_posts"][str(media_id)] = {
            "file_id": file_id or "",
            "has_keyword_cta": bool(has_keyword_cta),
            "keyword": keyword.strip().upper() if has_keyword_cta else "",
            "target_link": target_link.strip() if has_keyword_cta else "",
            "public_reply_text": public_reply_text.strip() if has_keyword_cta else "",
            "published_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_state_unlocked(state, path)


def get_published_media(
    media_id: str,
    state_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Retrieve stored automation metadata for a published Instagram `media_id`."""
    state = load_state(state_path)
    entry = state["media_posts"].get(str(media_id))
    return dict(entry) if isinstance(entry, dict) else None


def is_comment_processed(
    comment_id: str,
    state_path: Path | str | None = None,
) -> bool:
    """Return True if `comment_id` was already handled by the Instagram webhook."""
    state = load_state(state_path)
    return str(comment_id) in state["processed_comments"]


def mark_comment_processed(
    comment_id: str,
    media_id: str,
    username: str = "",
    dm_sent: bool = False,
    reply_sent: bool = False,
    state_path: Path | str | None = None,
) -> None:
    """Record a handled `comment_id` in `state.json` to prevent duplicate replies/DMs."""
    path = _get_state_path(state_path)
    with _STATE_LOCK:
        try:
            state = _read_state_unlocked(path)
        except DriveServiceError:
            state = _empty_state()

        state["processed_comments"][str(comment_id)] = {
            "media_id": str(media_id),
            "username": username,
            "dm_sent": dm_sent,
            "reply_sent": reply_sent,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_state_unlocked(state, path)


def _is_supported_media(file_meta: dict[str, Any]) -> bool:
    mime_type = (file_meta.get("mimeType") or "").lower()
    name = (file_meta.get("name") or "").lower()
    ext = Path(name).suffix
    return mime_type in SUPPORTED_MIME_TYPES or ext in SUPPORTED_EXTENSIONS


def _find_subfolder_id(
    service: Resource,
    parent_id: str,
    folder_name: str,
) -> str | None:
    escaped_name = _escape_drive_query_value(folder_name)
    query = (
        f"'{parent_id}' in parents "
        f"and name = '{escaped_name}' "
        f"and mimeType = '{GOOGLE_FOLDER_MIME_TYPE}' "
        "and trashed = false"
    )
    response = _execute_with_retry(
        service.files().list(
            q=query,
            spaces="drive",
            corpora="allDrives",
            fields="files(id, name)",
            pageSize=5,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ),
        f"find_subfolder({folder_name})",
    )
    folders = response.get("files", [])
    if not folders:
        return None
    if len(folders) > 1:
        logger.warning(
            "Se encontraron %d subcarpetas llamadas '%s' dentro de %s; usando la primera (%s).",
            len(folders),
            folder_name,
            parent_id,
            folders[0]["id"],
        )
    return str(folders[0]["id"])


def _create_subfolder(
    service: Resource,
    parent_id: str,
    folder_name: str,
) -> str:
    created = _execute_with_retry(
        service.files().create(
            body={
                "name": folder_name,
                "mimeType": GOOGLE_FOLDER_MIME_TYPE,
                "parents": [parent_id],
            },
            fields="id, name",
            supportsAllDrives=True,
        ),
        f"create_subfolder({folder_name})",
    )
    folder_id = str(created["id"])
    logger.info(
        "Subcarpeta '%s' creada automáticamente en Google Drive (id=%s, parent=%s).",
        folder_name,
        folder_id,
        parent_id,
    )
    return folder_id


def ensure_current_month_folder(
    root_folder_id: str,
    year_str: str,
    month_str: str,
    credentials_path: Path | str | None = None,
    create_if_missing: bool = True,
) -> MonthFolderResolution:
    """Resolve `YYYY/MM` inside `NELHEALTHCOACH Publicaciones`, creating missing folders if enabled."""
    normalized_year = year_str.strip()
    normalized_month = (
        month_str.strip().zfill(2) if month_str.strip().isdigit() else month_str.strip()
    )

    try:
        creds_path = _resolve_credentials_path(credentials_path)
        service = _get_drive_client(creds_path)

        try:
            root_meta = _execute_with_retry(
                service.files().get(
                    fileId=root_folder_id,
                    fields="id, name, mimeType, trashed",
                    supportsAllDrives=True,
                ),
                f"get_root_folder({root_folder_id})",
            )
            if root_meta.get("trashed"):
                raise DriveServiceError(
                    f"La carpeta raíz DRIVE_ROOT_FOLDER_ID={root_folder_id} está en la papelera de Google Drive."
                )
        except HttpError as root_exc:
            if root_exc.resp.status in {403, 404}:
                sa_email = _get_service_account_email(creds_path)
                email_hint = f" con '{sa_email}'" if sa_email else " con la Service Account"
                raise DriveServiceError(
                    f"No se puede acceder a DRIVE_ROOT_FOLDER_ID={root_folder_id} "
                    f"(HTTP {root_exc.resp.status}). Verifica que el ID sea correcto y que "
                    f"la carpeta 'NELHEALTHCOACH Publicaciones' esté compartida{email_hint} "
                    "con permiso de Editor."
                ) from root_exc
            raise

        created_any = False
        year_folder_id = _find_subfolder_id(service, root_folder_id, normalized_year)
        if not year_folder_id:
            if not create_if_missing:
                logger.warning(
                    "No existe la carpeta del año '%s' dentro de root_folder_id=%s.",
                    normalized_year,
                    root_folder_id,
                )
                return MonthFolderResolution(folder_id=None, created=False)
            year_folder_id = _create_subfolder(service, root_folder_id, normalized_year)
            created_any = True

        month_folder_id = _find_subfolder_id(service, year_folder_id, normalized_month)
        if not month_folder_id:
            if not create_if_missing:
                logger.warning(
                    "No existe la carpeta del mes '%s' dentro de '%s' (year_folder_id=%s).",
                    normalized_month,
                    normalized_year,
                    year_folder_id,
                )
                return MonthFolderResolution(folder_id=None, created=False)
            month_folder_id = _create_subfolder(service, year_folder_id, normalized_month)
            created_any = True

        logger.debug(
            "Carpeta mensual resuelta: %s/%s -> %s (created=%s)",
            normalized_year,
            normalized_month,
            month_folder_id,
            created_any,
        )
        return MonthFolderResolution(folder_id=month_folder_id, created=created_any)
    except HttpError as exc:
        raise DriveServiceError(
            f"Error de Google Drive resolviendo carpeta {normalized_year}/{normalized_month}: {exc}"
        ) from exc


def get_current_month_folder_id(
    root_folder_id: str,
    year_str: str,
    month_str: str,
    credentials_path: Path | str | None = None,
    create_if_missing: bool = True,
) -> str | None:
    """Resolve (and optionally create) the `YYYY/MM` subfolder ID inside `NELHEALTHCOACH Publicaciones`."""
    return ensure_current_month_folder(
        root_folder_id=root_folder_id,
        year_str=year_str,
        month_str=month_str,
        credentials_path=credentials_path,
        create_if_missing=create_if_missing,
    ).folder_id


def get_guidelines_text(
    doc_id: str,
    credentials_path: Path | str | None = None,
) -> str:
    """Export and return the plain-text content of a Google Doc by ID."""
    try:
        service = _get_drive_client(credentials_path)
        meta = _execute_with_retry(
            service.files().get(
                fileId=doc_id,
                fields="id, name, mimeType",
                supportsAllDrives=True,
            ),
            f"get_guidelines_meta({doc_id})",
        )
        mime_type = meta.get("mimeType", "")
        if mime_type == GOOGLE_DOC_MIME_TYPE:
            content_bytes: bytes = _execute_with_retry(
                service.files().export(fileId=doc_id, mimeType="text/plain"),
                f"export_guidelines({doc_id})",
            )
        else:
            content_bytes = _execute_with_retry(
                service.files().get_media(fileId=doc_id),
                f"download_guidelines({doc_id})",
            )

        text = content_bytes.decode("utf-8-sig").strip()
        if not text:
            raise DriveServiceError(
                f"El documento de pautas '{meta.get('name', doc_id)}' está vacío."
            )
        logger.debug("Pautas leídas correctamente (%d caracteres).", len(text))
        return text
    except HttpError as exc:
        raise DriveServiceError(
            f"Error de Google Drive al leer el documento {doc_id}: {exc}"
        ) from exc


def get_monthly_guidelines(
    month_folder_id: str,
    credentials_path: Path | str | None = None,
) -> str:
    """Find the Google Doc inside `month_folder_id` (most recent if >1) and return its text."""
    try:
        service = _get_drive_client(credentials_path)
        query = (
            f"'{month_folder_id}' in parents "
            f"and mimeType = '{GOOGLE_DOC_MIME_TYPE}' "
            "and trashed = false"
        )
        response = _execute_with_retry(
            service.files().list(
                q=query,
                spaces="drive",
                corpora="allDrives",
                fields="files(id, name, modifiedTime)",
                orderBy="modifiedTime desc",
                pageSize=10,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ),
            f"list_monthly_guidelines({month_folder_id})",
        )
        docs = response.get("files", [])
        if not docs:
            raise DriveServiceError(
                f"No se encontró ningún Google Doc de pautas en la carpeta del mes ({month_folder_id})."
            )

        if len(docs) > 1:
            logger.warning(
                "Se encontraron %d Google Docs en la carpeta del mes %s; "
                "usando el modificado más recientemente: '%s' (%s).",
                len(docs),
                month_folder_id,
                docs[0].get("name"),
                docs[0]["id"],
            )

        doc_id = str(docs[0]["id"])
        doc_name = str(docs[0].get("name", doc_id))
        logger.info(
            "Usando documento de pautas mensual '%s' (%s).",
            doc_name,
            doc_id,
        )
        content_bytes: bytes = _execute_with_retry(
            service.files().export(fileId=doc_id, mimeType="text/plain"),
            f"export_monthly_guidelines({doc_name})",
        )
        text = content_bytes.decode("utf-8-sig").strip()
        if not text:
            raise DriveServiceError(
                f"El documento de pautas '{doc_name}' está vacío."
            )
        return text
    except HttpError as exc:
        raise DriveServiceError(
            f"Error de Google Drive al buscar pautas en carpeta {month_folder_id}: {exc}"
        ) from exc


def _list_folder_children_media(
    service: Resource,
    folder_id: str,
) -> list[dict[str, Any]]:
    query = f"'{folder_id}' in parents and trashed = false"
    children: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        response = _execute_with_retry(
            service.files().list(
                q=query,
                spaces="drive",
                corpora="allDrives",
                fields=(
                    "nextPageToken, "
                    "files(id, name, mimeType, createdTime, modifiedTime, size, appProperties)"
                ),
                orderBy="name asc",
                pageSize=100,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ),
            f"list_folder_children({folder_id})",
        )
        children.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return children


_CAROUSEL_FOLDER_RE = re.compile(
    r"^(?:\[PUBLISHED\]\s*)?(?:\d+\s*[\.\-\)]\s*)?(?:Pro\s*Tip\s+)?\(CARRUSEL\)",
    re.IGNORECASE,
)


def is_carousel_folder_name(name: str) -> bool:
    """Return True only when a Drive subfolder uses the explicit `(CARRUSEL)` naming prefix."""
    return bool(_CAROUSEL_FOLDER_RE.match(name.strip()))


def list_month_media_files(
    month_folder_id: str,
    credentials_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """List single MP4/JPG/PNG files AND `(CARRUSEL)` subfolders (2-10 slides) inside `month_folder_id`."""
    try:
        service = _get_drive_client(credentials_path)
        all_files = _list_folder_children_media(service, month_folder_id)

        results: list[dict[str, Any]] = []
        for item in all_files:
            if _is_supported_media(item):
                results.append(item)
                continue

            if item.get("mimeType") == GOOGLE_FOLDER_MIME_TYPE:
                folder_name = str(item.get("name", ""))
                folder_name_norm = _normalize_match_text(folder_name)
                if folder_name_norm in {"estadistica", "estadisticas"}:
                    continue
                if not is_carousel_folder_name(folder_name):
                    continue
                sub_children = [
                    c for c in _list_folder_children_media(service, str(item["id"])) if _is_supported_media(c)
                ]
                if 2 <= len(sub_children) <= 10:
                    carousel_entry = dict(item)
                    carousel_entry["is_carousel"] = True
                    carousel_entry["carousel_items"] = sub_children
                    carousel_entry["size"] = sum(int(c.get("size") or 0) for c in sub_children)
                    results.append(carousel_entry)
                elif len(sub_children) > 10:
                    logger.warning(
                        "Carpeta '%s' tiene %d archivos; Instagram admite máximo 10 en un carrusel (tomando primeros 10).",
                        folder_name,
                        len(sub_children),
                    )
                    carousel_entry = dict(item)
                    carousel_entry["is_carousel"] = True
                    carousel_entry["carousel_items"] = sub_children[:10]
                    carousel_entry["size"] = sum(int(c.get("size") or 0) for c in sub_children[:10])
                    results.append(carousel_entry)
                else:
                    logger.warning(
                        "Carpeta '%s' marcada con (CARRUSEL) tiene %d archivo(s) válido(s); necesita al menos 2.",
                        folder_name,
                        len(sub_children),
                    )

        return results
    except HttpError as exc:
        raise DriveServiceError(
            f"Error de Google Drive al listar carpeta mensual {month_folder_id}: {exc}"
        ) from exc


def list_pending_files(
    month_folder_id: str,
    state_path: Path | str | None = None,
    credentials_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """List MP4/JPG/JPEG/PNG files in `month_folder_id` not yet recorded as finalized in `state.json`."""
    state = load_state(state_path)
    recorded_files: dict[str, Any] = state["files"]
    media_files = list_month_media_files(month_folder_id, credentials_path=credentials_path)

    pending: list[dict[str, Any]] = []
    for file_meta in media_files:
        file_id = file_meta["id"]
        local_status = str(recorded_files.get(file_id, {}).get("status", "")).upper()
        if local_status in FINAL_SKIP_STATUSES:
            continue

        drive_status = str((file_meta.get("appProperties") or {}).get("status", "")).lower()
        if drive_status == "published":
            continue

        pending.append(file_meta)

    logger.info(
        "Escaneo de carpeta mensual (%s): %d multimedia, %d pendientes por procesar.",
        month_folder_id,
        len(media_files),
        len(pending),
    )
    return pending


def set_temp_public_link(
    file_id: str,
    credentials_path: Path | str | None = None,
) -> TempPublicLink:
    """Grant temporary public read permission and return the direct download URL."""
    try:
        service = _get_drive_client(credentials_path)
        permission = _execute_with_retry(
            service.permissions().create(
                fileId=file_id,
                body={"role": "reader", "type": "anyone"},
                fields="id",
                supportsAllDrives=True,
            ),
            f"set_temp_public_link({file_id})",
        )
        permission_id = permission["id"]
        direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        logger.info(
            "Permiso público temporal creado para file_id=%s (permission_id=%s).",
            file_id,
            permission_id,
        )
        return TempPublicLink(url=direct_url, permission_id=permission_id)
    except HttpError as exc:
        raise DriveServiceError(
            f"No se pudo crear el permiso público temporal para {file_id}: {exc}"
        ) from exc


def revoke_public_permission(
    file_id: str,
    permission_id: str,
    credentials_path: Path | str | None = None,
) -> None:
    """Revoke temporary public permission without moving the file."""
    if not permission_id:
        return
    try:
        service = _get_drive_client(credentials_path)
        _execute_with_retry(
            service.permissions().delete(
                fileId=file_id,
                permissionId=permission_id,
                supportsAllDrives=True,
            ),
            f"revoke_public_permission({file_id})",
        )
        logger.info(
            "Permiso público %s revocado para file_id=%s.",
            permission_id,
            file_id,
        )
    except HttpError as exc:
        if exc.resp.status == 404:
            logger.warning(
                "El permiso %s de file_id=%s ya no existía al intentar revocarlo.",
                permission_id,
                file_id,
            )
        else:
            raise DriveServiceError(
                f"Error al revocar permiso {permission_id} de file_id={file_id}: {exc}"
            ) from exc


def mark_drive_file_published(
    file_id: str,
    credentials_path: Path | str | None = None,
) -> None:
    """Set `appProperties={'status': 'published'}` on the Drive file as secondary backup."""
    try:
        service = _get_drive_client(credentials_path)
        _execute_with_retry(
            service.files().update(
                fileId=file_id,
                body={"appProperties": {"status": "published"}},
                fields="id, appProperties",
                supportsAllDrives=True,
            ),
            f"mark_drive_file_published({file_id})",
        )
        logger.info(
            "Propiedad appProperties.status='published' guardada en Drive para file_id=%s.",
            file_id,
        )
    except HttpError as exc:
        logger.warning(
            "No se pudo escribir appProperties en Drive para file_id=%s (se mantiene respaldo en state.json): %s",
            file_id,
            exc,
        )


def get_monthly_doc_metadata(
    month_folder_id: str,
    credentials_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Return metadata (`id`, `name`, `modifiedTime`) of the newest Google Doc in `month_folder_id`, or None."""
    try:
        service = _get_drive_client(credentials_path)
        query = (
            f"'{month_folder_id}' in parents "
            f"and mimeType = '{GOOGLE_DOC_MIME_TYPE}' "
            "and trashed = false"
        )
        response = _execute_with_retry(
            service.files().list(
                q=query,
                spaces="drive",
                corpora="allDrives",
                fields="files(id, name, modifiedTime)",
                orderBy="modifiedTime desc",
                pageSize=10,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ),
            f"get_monthly_doc_metadata({month_folder_id})",
        )
        docs = response.get("files", [])
        return dict(docs[0]) if docs else None
    except HttpError as exc:
        raise DriveServiceError(
            f"Error de Google Drive buscando Google Doc en {month_folder_id}: {exc}"
        ) from exc


def find_previous_month_guidelines_text(
    root_folder_id: str,
    year_str: str,
    month_str: str,
    credentials_path: Path | str | None = None,
    lookback_months: int = 12,
) -> str | None:
    """Walk backwards up to `lookback_months` to locate and export the latest previous `Pautas` doc."""
    year = int(year_str)
    month = int(month_str)

    try:
        service = _get_drive_client(credentials_path)
        for _ in range(lookback_months):
            month -= 1
            if month < 1:
                month = 12
                year -= 1

            prev_year_id = _find_subfolder_id(service, root_folder_id, str(year))
            if not prev_year_id:
                continue
            prev_month_id = _find_subfolder_id(service, prev_year_id, f"{month:02d}")
            if not prev_month_id:
                continue

            doc_meta = get_monthly_doc_metadata(prev_month_id, credentials_path=credentials_path)
            if doc_meta:
                logger.info(
                    "Usando pautas de referencia del mes anterior %04d/%02d ('%s').",
                    year,
                    month,
                    doc_meta.get("name", doc_meta["id"]),
                )
                return get_guidelines_text(
                    str(doc_meta["id"]),
                    credentials_path=credentials_path,
                )
    except Exception as exc:
        logger.warning("No se pudo obtener pautas del mes anterior: %s", exc)

    return None


def _lines_to_html_paragraphs(lines: list[str], bold_prefixes: tuple[str, ...] = ()) -> str:
    paragraphs: list[str] = []
    for idx, raw_line in enumerate(lines):
        line = raw_line.rstrip()
        if not line:
            paragraphs.append(
                '<p style="margin:0;padding:0;font-family:Arial;font-size:11pt;line-height:1.15;height:11pt;">'
                '<span style="font-family:Arial;font-size:11pt;"></span></p>'
            )
            continue

        escaped = html_lib.escape(line)
        is_header = idx == 0 or line.startswith("===")
        matched_prefix = next((p for p in bold_prefixes if line.startswith(p)), None)

        if is_header:
            inner = f'<span style="font-family:Arial;font-size:11pt;font-weight:700;">{escaped}</span>'
        elif matched_prefix:
            rest = html_lib.escape(line[len(matched_prefix) :])
            inner = (
                f'<span style="font-family:Arial;font-size:11pt;font-weight:700;">'
                f"{html_lib.escape(matched_prefix)}</span>"
                f'<span style="font-family:Arial;font-size:11pt;">{rest}</span>'
            )
        else:
            inner = f'<span style="font-family:Arial;font-size:11pt;">{escaped}</span>'

        paragraphs.append(
            f'<p style="margin:0;padding:0;font-family:Arial;font-size:11pt;line-height:1.15;">{inner}</p>'
        )
    return "".join(paragraphs)


def create_monthly_guidelines_doc(
    month_folder_id: str,
    plain_text_content: str,
    doc_title: str = "Pautas",
    credentials_path: Path | str | None = None,
) -> str:
    """Create a new Google Doc named `Pautas` inside `month_folder_id` from formatted text."""
    lines = plain_text_content.splitlines()
    body_html = _lines_to_html_paragraphs(
        lines,
        bold_prefixes=(
            "Fecha:",
            "Horario:",
            "Pauta en Meta Ads:",
            "Título / Gancho:",
            "Caption:",
            "Hashtags:",
        ),
    )
    full_html = f"<html><body>{body_html}</body></html>"

    try:
        service = _get_drive_client(credentials_path)
        media = MediaInMemoryUpload(
            full_html.encode("utf-8"),
            mimetype="text/html",
            resumable=False,
        )
        created = _execute_with_retry(
            service.files().create(
                body={
                    "name": doc_title,
                    "mimeType": GOOGLE_DOC_MIME_TYPE,
                    "parents": [month_folder_id],
                },
                media_body=media,
                fields="id, name",
                supportsAllDrives=True,
            ),
            f"create_monthly_guidelines_doc({month_folder_id})",
        )
        doc_id = str(created["id"])
        logger.info(
            "Documento '%s' creado automáticamente en Drive (id=%s, folder=%s).",
            doc_title,
            doc_id,
            month_folder_id,
        )
        return doc_id
    except HttpError as exc:
        raise DriveServiceError(
            f"Error creando el documento '{doc_title}' en carpeta {month_folder_id}: {exc}"
        ) from exc


def compute_next_numbering_and_filename(
    month_folder_id: str,
    guidelines_text: str,
    post_category: str,
    short_title: str,
    original_filename: str,
    credentials_path: Path | str | None = None,
    is_carousel: bool = False,
) -> tuple[str, str]:
    """Compute the next sequential doc header and Drive filename (`NN. Title.ext`)."""
    service = _get_drive_client(credentials_path)
    query = f"'{month_folder_id}' in parents and trashed = false"
    all_files: list[dict[str, Any]] = []
    page_token: str | None = None

    while True:
        response = _execute_with_retry(
            service.files().list(
                q=query,
                spaces="drive",
                corpora="allDrives",
                fields="nextPageToken, files(id, name, mimeType)",
                pageSize=100,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ),
            f"scan_numbering({month_folder_id})",
        )
        all_files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    max_drive_num = 0
    for f in all_files:
        is_carousel_folder = (
            f.get("mimeType") == GOOGLE_FOLDER_MIME_TYPE
            and is_carousel_folder_name(str(f.get("name", "")))
        )
        if not (_is_supported_media(f) or is_carousel_folder):
            continue
        name = str(f.get("name", ""))
        m = re.match(r"^\s*(\d+)\b", name)
        if m:
            max_drive_num = max(max_drive_num, int(m.group(1)))
    next_drive_num = max_drive_num + 1

    max_main_num = 0
    max_protip_num = 0
    for line in guidelines_text.splitlines():
        stripped = line.strip()
        m_main = re.match(r"^(\d+)\.\s+\S+", stripped)
        if m_main:
            max_main_num = max(max_main_num, int(m_main.group(1)))
        m_protip = re.match(r"^Pro\s*Tip\s+(\d+)\s*:", stripped, flags=re.IGNORECASE)
        if m_protip:
            max_protip_num = max(max_protip_num, int(m_protip.group(1)))

    raw_title = re.sub(r"^\(CARRUSEL\)\s*", "", short_title, flags=re.IGNORECASE)
    clean_title = re.sub(r'[\\/:*?"<>|]+', "", raw_title).strip()[:60] or "Publicación"
    ext = "" if is_carousel else (Path(original_filename).suffix or ".mp4")
    carousel_tag = "(CARRUSEL) " if is_carousel else ""

    if post_category.upper() == "PRO_TIP":
        doc_header = f"Pro Tip {max_protip_num + 1}: {clean_title}"
        new_filename = f"{next_drive_num:02d}. Pro tip {carousel_tag}{clean_title}{ext}"
    else:
        doc_header = f"{max_main_num + 1}. {clean_title}"
        new_filename = f"{next_drive_num:02d}. {carousel_tag}{clean_title}{ext}"

    return doc_header, new_filename


def format_guideline_entry_lines(
    doc_header: str,
    scheduled_dt: datetime,
    promote_in_ads: bool,
    ads_recommendation: str,
    hook_title: str,
    caption: str,
    hashtags: str,
    has_keyword_cta: bool = False,
    keyword: str = "",
    target_link: str = "",
) -> list[str]:
    """Format an approved proposal into the exact multi-line structure used in `Pautas`."""
    weekday_name = SPANISH_WEEKDAYS[scheduled_dt.weekday()]
    fecha_str = f"{weekday_name} {scheduled_dt.strftime('%d/%m/%Y')}"
    horario_str = f"{scheduled_dt.strftime('%H:%M')} hrs"

    clean_ads = ads_recommendation.strip()
    if promote_in_ads:
        ads_line = f"SÍ ({clean_ads})" if clean_ads else "SÍ"
    else:
        ads_line = f"Orgánico ({clean_ads})" if clean_ads else "Orgánico."

    lines = [
        doc_header,
        f"Fecha: {fecha_str}",
        f"Horario: {horario_str}",
        f"Pauta en Meta Ads: {ads_line}",
        f"Título / Gancho: {hook_title.strip()}",
        "Caption:",
        *[line for line in caption.strip().splitlines()],
        f"Hashtags: {hashtags.strip()}",
    ]
    if has_keyword_cta and keyword.strip():
        dm_detail = f"Palabra clave '{keyword.strip().upper()}'"
        if target_link.strip():
            dm_detail += f" -> {target_link.strip()}"
        lines.append(f"DM automático: {dm_detail}")
    return lines


def append_entry_to_monthly_doc(
    doc_id: str,
    entry_lines: list[str],
    credentials_path: Path | str | None = None,
) -> str:
    """Append an approved post entry to the end of `Pautas` (Docs API first, Drive HTML fallback)."""
    plain_block = "\n" + "\n".join(entry_lines) + "\n"

    try:
        docs_service = _get_docs_client(credentials_path)
        doc = _execute_with_retry(
            docs_service.documents().get(documentId=doc_id),
            f"docs_get({doc_id})",
        )
        content = doc.get("body", {}).get("content", [])
        end_index = content[-1].get("endIndex", 1) if content else 1
        insert_index = max(1, end_index - 1)

        _execute_with_retry(
            docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={
                    "requests": [
                        {
                            "insertText": {
                                "location": {"index": insert_index},
                                "text": plain_block,
                            }
                        }
                    ]
                },
            ),
            f"docs_append({doc_id})",
        )
        logger.info("Pauta anexada al Google Doc %s mediante Google Docs API.", doc_id)
        return "docs_api"
    except HttpError as docs_exc:
        if docs_exc.resp.status not in {403, 404}:
            raise DriveServiceError(
                f"Error de Google Docs API al actualizar {doc_id}: {docs_exc}"
            ) from docs_exc
        logger.warning(
            "Google Docs API no disponible (%s); usando respaldo por Drive HTML para anexar al Doc %s.",
            docs_exc.resp.status,
            doc_id,
        )

    try:
        drive_service = _get_drive_client(credentials_path)
        html_bytes: bytes = _execute_with_retry(
            drive_service.files().export(fileId=doc_id, mimeType="text/html"),
            f"export_html_for_append({doc_id})",
        )
        current_html = html_bytes.decode("utf-8")

        new_paragraphs = _lines_to_html_paragraphs(
            [""] + entry_lines,
            bold_prefixes=(
                "Fecha:",
                "Horario:",
                "Pauta en Meta Ads:",
                "Título / Gancho:",
                "Caption:",
                "Hashtags:",
                "DM automático:",
            ),
        )
        if "</body>" in current_html:
            updated_html = current_html.replace("</body>", f"{new_paragraphs}</body>", 1)
        else:
            updated_html = current_html + new_paragraphs

        media = MediaInMemoryUpload(
            updated_html.encode("utf-8"),
            mimetype="text/html",
            resumable=False,
        )
        _execute_with_retry(
            drive_service.files().update(
                fileId=doc_id,
                media_body=media,
                fields="id, name",
                supportsAllDrives=True,
            ),
            f"drive_html_append({doc_id})",
        )
        logger.info("Pauta anexada al Google Doc %s mediante respaldo Drive HTML.", doc_id)
        return "drive_html_fallback"
    except HttpError as drive_exc:
        raise DriveServiceError(
            f"Error al anexar pauta en Google Doc {doc_id}: {drive_exc}"
        ) from drive_exc


def rename_drive_file(
    file_id: str,
    new_name: str,
    credentials_path: Path | str | None = None,
) -> str:
    """Rename a file in Google Drive while preserving its immutable `file_id`."""
    try:
        service = _get_drive_client(credentials_path)
        updated = _execute_with_retry(
            service.files().update(
                fileId=file_id,
                body={"name": new_name},
                fields="id, name",
                supportsAllDrives=True,
            ),
            f"rename_drive_file({file_id})",
        )
        final_name = str(updated.get("name", new_name))
        logger.info("Archivo en Drive renombrado: file_id=%s -> '%s'.", file_id, final_name)
        return final_name
    except HttpError as exc:
        raise DriveServiceError(
            f"Error al renombrar archivo {file_id} a '{new_name}': {exc}"
        ) from exc


def download_drive_file_to_temp(
    file_id: str,
    filename: str,
    credentials_path: Path | str | None = None,
) -> Path:
    """Download a Drive media file to a temporary local path for multimodal Gemini analysis."""
    suffix = Path(filename).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(prefix="nhc_media_", suffix=suffix, delete=False)
    tmp_path = Path(tmp.name)

    try:
        service = _get_drive_client(credentials_path)
        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
        downloader = MediaIoBaseDownload(tmp, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk(num_retries=2)
        tmp.close()
        logger.info(
            "Archivo '%s' descargado temporalmente para análisis multimodal (%d bytes).",
            filename,
            tmp_path.stat().st_size,
        )
        return tmp_path
    except Exception as exc:
        try:
            tmp.close()
        except Exception:
            pass
        tmp_path.unlink(missing_ok=True)
        raise DriveServiceError(
            f"Error descargando '{filename}' ({file_id}) para análisis IA: {exc}"
        ) from exc


@dataclass(slots=True)
class GuidelineEntry:
    """Structured representation of one entry inside the monthly `Pautas` Google Doc."""

    header: str
    post_category: Literal["MAIN", "PRO_TIP"]
    short_title: str
    fecha_raw: str
    horario_raw: str
    scheduled_dt: datetime | None
    ads_raw: str
    promote_in_ads: bool
    ads_daily_budget_usd: float
    ads_duration_days: int
    hook_title: str
    caption: str
    hashtags: str
    has_keyword_cta: bool
    keyword: str
    target_link: str
    raw_block: str
    is_marked_published: bool = False


@dataclass(slots=True)
class UnpublishedGuidelineItem:
    """An entry from `Pautas` that has not yet been published on Instagram."""

    entry: GuidelineEntry
    drive_file: dict[str, Any] | None
    local_status: str
    is_overdue: bool
    has_active_job: bool


_ENTRY_BOLD_PREFIXES = (
    "Fecha:",
    "Horario:",
    "Pauta en Meta Ads:",
    "Título / Gancho:",
    "Caption:",
    "Hashtags:",
    "DM automático:",
)


_PUBLISHED_HEADER_TAG_RE = re.compile(r"\s*\[(?:PUBLICADO|PUBLISHED)\]\s*", re.IGNORECASE)


def _strip_published_marker(text: str) -> str:
    return _PUBLISHED_HEADER_TAG_RE.sub(" ", text).strip()


def _match_entry_header(line: str) -> tuple[Literal["MAIN", "PRO_TIP"], str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    m_protip = re.match(r"^Pro\s*Tip\s+\d+\s*:\s*(.+)$", stripped, flags=re.IGNORECASE)
    if m_protip:
        return "PRO_TIP", _strip_published_marker(m_protip.group(1).strip())
    m_main = re.match(r"^\d+\.\s+(.+)$", stripped)
    if m_main:
        title = _strip_published_marker(m_main.group(1).strip())
        if title.lower().startswith("título"):
            return None
        return "MAIN", title
    return None


def _parse_fecha_horario(
    fecha_raw: str,
    horario_raw: str,
    default_tz: tzinfo | None = None,
) -> datetime | None:
    tz = default_tz or datetime.now().astimezone().tzinfo
    date_match = re.search(r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?", fecha_raw)
    if not date_match:
        iso_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", fecha_raw)
        if not iso_match:
            return None
        year, month, day = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
    else:
        day = int(date_match.group(1))
        month = int(date_match.group(2))
        year_raw = date_match.group(3)
        if year_raw:
            year = int(year_raw)
            if year < 100:
                year += 2000
        else:
            year = datetime.now().year

    hour, minute = 18, 0
    time_match = re.search(r"(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.)?", horario_raw, re.IGNORECASE)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        ampm = (time_match.group(3) or "").lower().replace(".", "")
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0

    try:
        return datetime(year, month, day, hour, minute, tzinfo=tz)
    except ValueError:
        return None


def _parse_ads_field(ads_raw: str) -> tuple[bool, float, int]:
    norm = _normalize_match_text(ads_raw)
    if norm.startswith("si"):
        budget_m = re.search(r"\$\s*(\d+(?:[.,]\d+)?)", ads_raw)
        days_m = re.search(r"(\d+)\s*d[ií]as?", ads_raw, re.IGNORECASE)
        budget = float(budget_m.group(1).replace(",", ".")) if budget_m else 4.0
        days = int(days_m.group(1)) if days_m else 5
        return True, budget, days
    return False, 0.0, 0


def _parse_dm_field(dm_raw: str) -> tuple[bool, str, str]:
    if not dm_raw.strip():
        return False, "", ""
    kw_m = re.search(r"['\"]([A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9_-]+)['\"]", dm_raw)
    link_m = re.search(r"(https?://\S+)", dm_raw)
    keyword = kw_m.group(1).upper() if kw_m else ""
    link = link_m.group(1).strip() if link_m else ""
    return bool(keyword), keyword, link


def parse_guidelines_entries(
    guidelines_text: str,
    default_tz: tzinfo | None = None,
) -> list[GuidelineEntry]:
    """Parse structured post entries (`N. Título` / `Pro Tip N: Título`) from `Pautas`."""
    lines = guidelines_text.splitlines()
    start_idx = 0
    for idx, line in enumerate(lines):
        if re.match(r"^===\s*PAUTAS DEL MES\b", line.strip(), re.IGNORECASE):
            start_idx = idx + 1
            break

    blocks: list[tuple[str, Literal["MAIN", "PRO_TIP"], str, list[str]]] = []
    current_header = ""
    current_cat: Literal["MAIN", "PRO_TIP"] = "MAIN"
    current_title = ""
    current_lines: list[str] = []

    for line in lines[start_idx:]:
        matched = _match_entry_header(line)
        if matched:
            if current_header:
                blocks.append((current_header, current_cat, current_title, current_lines))
            current_cat, current_title = matched
            current_header = line.strip()
            current_lines = [line.rstrip()]
        elif current_header:
            current_lines.append(line.rstrip())

    if current_header:
        blocks.append((current_header, current_cat, current_title, current_lines))

    entries: list[GuidelineEntry] = []
    for header, cat, title, block_lines in blocks:
        # Trim trailing empty lines
        while block_lines and not block_lines[-1].strip():
            block_lines.pop()

        fecha_raw = ""
        horario_raw = ""
        ads_raw = ""
        hook_title = ""
        hashtags = ""
        dm_raw = ""
        caption_lines: list[str] = []
        in_caption = False

        for raw in block_lines[1:]:
            s = raw.strip()
            lower_s = _normalize_match_text(s)
            if lower_s.startswith("fecha:"):
                in_caption = False
                fecha_raw = s.split(":", 1)[1].strip()
            elif lower_s.startswith("horario:"):
                in_caption = False
                horario_raw = s.split(":", 1)[1].strip()
            elif lower_s.startswith("pauta en meta ads:"):
                in_caption = False
                ads_raw = s.split(":", 1)[1].strip()
            elif lower_s.startswith("titulo / gancho:") or lower_s.startswith("titulo/gancho:"):
                in_caption = False
                hook_title = s.split(":", 1)[1].strip()
            elif lower_s.startswith("caption:"):
                in_caption = True
                rest = s.split(":", 1)[1].strip()
                if rest:
                    caption_lines.append(rest)
            elif lower_s.startswith("hashtags:"):
                in_caption = False
                hashtags = s.split(":", 1)[1].strip()
            elif lower_s.startswith("dm automatico:") or lower_s.startswith("dm automático:"):
                in_caption = False
                dm_raw = s.split(":", 1)[1].strip()
            elif in_caption:
                caption_lines.append(raw.rstrip())

        scheduled_dt = _parse_fecha_horario(fecha_raw, horario_raw, default_tz=default_tz)
        promote, budget, days = _parse_ads_field(ads_raw)
        has_cta, keyword, target_link = _parse_dm_field(dm_raw)

        entries.append(
            GuidelineEntry(
                header=header,
                post_category=cat,
                short_title=title,
                fecha_raw=fecha_raw,
                horario_raw=horario_raw,
                scheduled_dt=scheduled_dt,
                ads_raw=ads_raw,
                promote_in_ads=promote,
                ads_daily_budget_usd=budget,
                ads_duration_days=days,
                hook_title=hook_title,
                caption="\n".join(caption_lines).strip(),
                hashtags=hashtags,
                has_keyword_cta=has_cta,
                keyword=keyword,
                target_link=target_link,
                raw_block="\n".join(block_lines).strip(),
                is_marked_published=bool(_PUBLISHED_HEADER_TAG_RE.search(header)),
            )
        )

    return entries


def _stem_tokens(text: str) -> set[str]:
    norm = _normalize_match_text(text)
    norm = re.sub(r"^\d+[\.\-\s:]*", "", norm)
    norm = re.sub(r"\b(pro\s*tip|protip|video|mp4|jpg|jpeg|png)\b", " ", norm)
    tokens = {tok for tok in re.findall(r"[a-z0-9]{3,}", norm)}
    return tokens


def _match_entry_to_drive_file(
    entry: GuidelineEntry,
    month_files: list[dict[str, Any]],
    recorded_files: dict[str, Any],
) -> dict[str, Any] | None:
    norm_header = _normalize_match_text(entry.header)
    for file_meta in month_files:
        fid = file_meta["id"]
        rec = recorded_files.get(fid, {}) if isinstance(recorded_files.get(fid), dict) else {}
        matched = _normalize_match_text(str(rec.get("matched_entry", "")))
        doc_header = _normalize_match_text(str(rec.get("doc_header", "")))
        if norm_header and (matched == norm_header or doc_header == norm_header):
            return file_meta

    entry_tokens = _stem_tokens(entry.short_title)
    if not entry_tokens:
        return None

    best_file: dict[str, Any] | None = None
    best_score = 0.0
    for file_meta in month_files:
        stem = Path(str(file_meta.get("name", ""))).stem
        file_tokens = _stem_tokens(stem)
        if not file_tokens:
            continue
        overlap = len(entry_tokens & file_tokens)
        if overlap == 0:
            continue
        score = overlap / max(len(entry_tokens), len(file_tokens))
        if score > best_score and (overlap >= 1 and score >= 0.45):
            best_score = score
            best_file = file_meta
    return best_file


def _entry_matches_ig_post(
    entry: GuidelineEntry,
    ig_media_list: list[dict[str, Any]],
) -> bool:
    """Cross-check whether `entry` was already published using ONLY caption + hashtags (never hook_title)."""
    if not ig_media_list:
        return False

    caption_norm = _normalize_match_text(entry.caption)
    caption_probe = caption_norm[:80].strip() if len(caption_norm) >= 20 else ""
    cap_words = set(re.findall(r"[a-z0-9]{4,}", caption_norm))
    entry_tags = {
        tag.lower()
        for tag in re.findall(r"#[\wáéíóúüñÁÉÍÓÚÜÑ]+", f"{entry.hashtags} {entry.caption}")
    }

    if not caption_norm and not entry_tags:
        return False

    for media in ig_media_list:
        ig_raw = str(media.get("caption", ""))
        ig_cap = _normalize_match_text(ig_raw)
        if not ig_cap:
            continue

        # 1. Exact or prefix normalized caption match
        if caption_probe and caption_probe in ig_cap:
            return True
        if 10 <= len(caption_norm) < 20 and caption_norm in ig_cap:
            return True

        ig_words = set(re.findall(r"[a-z0-9]{4,}", ig_cap))
        cap_overlap = (
            (len(cap_words & ig_words) / len(cap_words))
            if cap_words
            else 0.0
        )

        # 2. High caption word overlap
        if len(cap_words) >= 4 and cap_overlap >= 0.60:
            return True

        # 3. Hashtags + caption cross-check
        if entry_tags:
            ig_tags = {
                tag.lower()
                for tag in re.findall(r"#[\wáéíóúüñÁÉÍÓÚÜÑ]+", ig_raw)
            }
            shared_tags = len(entry_tags & ig_tags)
            if shared_tags >= min(len(entry_tags), 3) and shared_tags >= 2:
                if not cap_words or cap_overlap >= 0.30:
                    return True
            if len(entry_tags) >= 5 and entry_tags.issubset(ig_tags):
                if not cap_words or bool(cap_words & ig_words):
                    return True
    return False


def find_unpublished_guidelines(
    guidelines_text: str,
    month_files: list[dict[str, Any]],
    state_path: Path | str | None = None,
    recent_ig_media: list[dict[str, Any]] | None = None,
    active_scheduler_file_ids: set[str] | None = None,
    reference_now: datetime | None = None,
) -> list[UnpublishedGuidelineItem]:
    """Return `Pautas` entries that do not appear published yet in state, Drive, or Instagram."""
    now = reference_now or datetime.now().astimezone()
    state = load_state(state_path)
    recorded_files: dict[str, Any] = state["files"]
    published_headers: dict[str, Any] = state.get("published_guideline_headers", {})
    published_headers_clean = {
        _normalize_match_text(_strip_published_marker(k)) for k in published_headers
    }
    active_ids = active_scheduler_file_ids or set()
    ig_posts = recent_ig_media or []

    entries = parse_guidelines_entries(guidelines_text, default_tz=now.tzinfo)
    unpublished: list[UnpublishedGuidelineItem] = []

    for entry in entries:
        if entry.is_marked_published:
            continue

        header_key = _normalize_match_text(_strip_published_marker(entry.header))
        if header_key in published_headers_clean or _normalize_match_text(entry.header) in published_headers:
            continue

        drive_file = _match_entry_to_drive_file(entry, month_files, recorded_files)
        local_status = "UNLINKED"
        has_active_job = False

        if drive_file is not None:
            fid = drive_file["id"]
            has_active_job = fid in active_ids
            rec = recorded_files.get(fid, {}) if isinstance(recorded_files.get(fid), dict) else {}
            local_status = str(rec.get("status", "IN_DRIVE")).upper()
            drive_app_status = str((drive_file.get("appProperties") or {}).get("status", "")).lower()

            if (
                local_status in {"PUBLISHED", "DISCARDED"}
                or drive_app_status == "published"
                or bool(_PUBLISHED_HEADER_TAG_RE.search(str(drive_file.get("name", ""))))
            ):
                continue

        if _entry_matches_ig_post(entry, ig_posts):
            continue

        is_overdue = bool(entry.scheduled_dt is not None and entry.scheduled_dt < now)
        unpublished.append(
            UnpublishedGuidelineItem(
                entry=entry,
                drive_file=drive_file,
                local_status=local_status,
                is_overdue=is_overdue,
                has_active_job=has_active_job,
            )
        )

    return unpublished


def update_entry_in_monthly_doc(
    doc_id: str,
    target_header: str,
    new_entry_lines: list[str],
    credentials_path: Path | str | None = None,
) -> str:
    """Replace an existing entry block inside `Pautas` in-place (Docs API first, Drive HTML fallback)."""
    norm_target = _normalize_match_text(target_header)
    new_block = "\n".join(new_entry_lines) + "\n\n"

    try:
        docs_service = _get_docs_client(credentials_path)
        doc = _execute_with_retry(
            docs_service.documents().get(documentId=doc_id),
            f"docs_get_for_update({doc_id})",
        )
        content = doc.get("body", {}).get("content", [])
        doc_end_index = content[-1].get("endIndex", 1) if content else 1

        paragraphs: list[tuple[int, int, str]] = []
        for elem in content:
            para = elem.get("paragraph")
            if not para:
                continue
            start_i = int(elem.get("startIndex", 0))
            end_i = int(elem.get("endIndex", 0))
            text_parts = [
                el.get("textRun", {}).get("content", "")
                for el in para.get("elements", [])
            ]
            para_text = "".join(text_parts).strip()
            paragraphs.append((start_i, end_i, para_text))

        # Respect `=== PAUTAS DEL MES` divider if present
        search_from = 0
        for idx, (_, _, ptext) in enumerate(paragraphs):
            if re.match(r"^===\s*PAUTAS DEL MES\b", ptext, re.IGNORECASE):
                search_from = idx + 1
                break

        found_idx: int | None = None
        for idx in range(search_from, len(paragraphs)):
            if _normalize_match_text(paragraphs[idx][2]) == norm_target:
                found_idx = idx
                break

        if found_idx is None:
            raise DriveServiceError(
                f"No se encontró la pauta '{target_header}' en el documento {doc_id}."
            )

        start_index = paragraphs[found_idx][0]
        end_index = doc_end_index - 1
        for idx in range(found_idx + 1, len(paragraphs)):
            if _match_entry_header(paragraphs[idx][2]) is not None:
                end_index = paragraphs[idx][0]
                break

        safe_end_index = min(end_index, max(start_index + 1, doc_end_index - 1))
        requests: list[dict[str, Any]] = []
        if safe_end_index > start_index:
            requests.append(
                {
                    "deleteContentRange": {
                        "range": {
                            "startIndex": start_index,
                            "endIndex": safe_end_index,
                        }
                    }
                }
            )
        requests.append(
            {
                "insertText": {
                    "location": {"index": start_index},
                    "text": new_block,
                }
            }
        )
        _execute_with_retry(
            docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={"requests": requests},
            ),
            f"docs_update_entry({doc_id})",
        )
        logger.info(
            "Pauta '%s' actualizada en Google Doc %s mediante Google Docs API.",
            target_header,
            doc_id,
        )
        return "docs_api"
    except HttpError as docs_exc:
        if docs_exc.resp.status not in {403, 404}:
            raise DriveServiceError(
                f"Error de Google Docs API al actualizar '{target_header}' en {doc_id}: {docs_exc}"
            ) from docs_exc
        logger.warning(
            "Google Docs API no disponible (%s); usando respaldo Drive HTML para actualizar '%s'.",
            docs_exc.resp.status,
            target_header,
        )

    try:
        drive_service = _get_drive_client(credentials_path)
        html_bytes: bytes = _execute_with_retry(
            drive_service.files().export(fileId=doc_id, mimeType="text/html"),
            f"export_html_for_update({doc_id})",
        )
        current_html = html_bytes.decode("utf-8")

        p_matches = list(re.finditer(r"<p\b[^>]*>.*?</p>", current_html, flags=re.DOTALL | re.IGNORECASE))
        p_texts = [
            html_lib.unescape(re.sub(r"<[^>]+>", "", m.group(0))).strip()
            for m in p_matches
        ]

        search_from = 0
        for idx, ptext in enumerate(p_texts):
            if re.match(r"^===\s*PAUTAS DEL MES\b", ptext, re.IGNORECASE):
                search_from = idx + 1
                break

        found_idx = None
        for idx in range(search_from, len(p_texts)):
            if _normalize_match_text(p_texts[idx]) == norm_target:
                found_idx = idx
                break

        if found_idx is None:
            raise DriveServiceError(
                f"No se encontró la pauta '{target_header}' en el HTML del documento {doc_id}."
            )

        end_idx = len(p_matches)
        for idx in range(found_idx + 1, len(p_texts)):
            if _match_entry_header(p_texts[idx]) is not None:
                end_idx = idx
                break

        replace_start = p_matches[found_idx].start()
        replace_end = p_matches[end_idx].start() if end_idx < len(p_matches) else p_matches[-1].end()
        replacement_html = _lines_to_html_paragraphs(
            new_entry_lines + [""],
            bold_prefixes=_ENTRY_BOLD_PREFIXES,
        )
        updated_html = current_html[:replace_start] + replacement_html + current_html[replace_end:]

        media = MediaInMemoryUpload(
            updated_html.encode("utf-8"),
            mimetype="text/html",
            resumable=False,
        )
        _execute_with_retry(
            drive_service.files().update(
                fileId=doc_id,
                media_body=media,
                fields="id, name",
                supportsAllDrives=True,
            ),
            f"drive_html_update_entry({doc_id})",
        )
        logger.info(
            "Pauta '%s' actualizada en Google Doc %s mediante respaldo Drive HTML.",
            target_header,
            doc_id,
        )
        return "drive_html_fallback"
    except HttpError as drive_exc:
        raise DriveServiceError(
            f"Error al actualizar '{target_header}' en Google Doc {doc_id}: {drive_exc}"
        ) from drive_exc


def mark_entry_published_in_monthly_doc(
    month_folder_id: str,
    target_header: str,
    credentials_path: Path | str | None = None,
) -> str:
    """Append ` [PUBLICADO]` to the target entry header inside the monthly `Pautas` Google Doc."""
    clean_target = _strip_published_marker(target_header)
    if not clean_target or not month_folder_id:
        return "skipped"

    doc_meta = get_monthly_doc_metadata(month_folder_id, credentials_path=credentials_path)
    if not doc_meta:
        return "no_doc"

    doc_id = str(doc_meta["id"])
    norm_target = _normalize_match_text(clean_target)

    try:
        docs_service = _get_docs_client(credentials_path)
        doc = _execute_with_retry(
            docs_service.documents().get(documentId=doc_id),
            f"docs_get_for_mark_pub({doc_id})",
        )
        content = doc.get("body", {}).get("content", [])

        paragraphs: list[tuple[int, int, str]] = []
        for elem in content:
            para = elem.get("paragraph")
            if not para:
                continue
            start_i = int(elem.get("startIndex", 0))
            end_i = int(elem.get("endIndex", 0))
            raw_text = "".join(
                el.get("textRun", {}).get("content", "")
                for el in para.get("elements", [])
            )
            paragraphs.append((start_i, end_i, raw_text))

        search_from = 0
        for idx, (_, _, raw_t) in enumerate(paragraphs):
            if re.match(r"^===\s*PAUTAS DEL MES\b", raw_t.strip(), re.IGNORECASE):
                search_from = idx + 1
                break

        found_para: tuple[int, int, str] | None = None
        for para_info in paragraphs[search_from:]:
            stripped = para_info[2].strip()
            if _normalize_match_text(_strip_published_marker(stripped)) == norm_target:
                found_para = para_info
                break

        if found_para is None:
            logger.debug("No se encontró '%s' en Doc %s para marcar [PUBLICADO].", clean_target, doc_id)
            return "not_found"

        start_i, end_i, raw_t = found_para
        if _PUBLISHED_HEADER_TAG_RE.search(raw_t):
            return "already_marked"

        insert_index = max(start_i, end_i - 1)
        _execute_with_retry(
            docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={
                    "requests": [
                        {
                            "insertText": {
                                "location": {"index": insert_index},
                                "text": " [PUBLICADO]",
                            }
                        }
                    ]
                },
            ),
            f"docs_mark_published({doc_id})",
        )
        logger.info("Encabezado '%s [PUBLICADO]' marcado en Google Doc %s.", clean_target, doc_id)
        return "docs_api"
    except HttpError as docs_exc:
        if docs_exc.resp.status not in {403, 404}:
            raise DriveServiceError(
                f"Error marcando [PUBLICADO] en Doc {doc_id} para '{clean_target}': {docs_exc}"
            ) from docs_exc

    try:
        drive_service = _get_drive_client(credentials_path)
        html_bytes: bytes = _execute_with_retry(
            drive_service.files().export(fileId=doc_id, mimeType="text/html"),
            f"export_html_mark_pub({doc_id})",
        )
        current_html = html_bytes.decode("utf-8")
        p_matches = list(re.finditer(r"(<p\b[^>]*>)(.*?)(</p>)", current_html, flags=re.DOTALL | re.IGNORECASE))
        search_from = 0
        for idx, m in enumerate(p_matches):
            ptext = html_lib.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip()
            if re.match(r"^===\s*PAUTAS DEL MES\b", ptext, re.IGNORECASE):
                search_from = idx + 1
                break

        target_match = None
        for m in p_matches[search_from:]:
            ptext = html_lib.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip()
            if _normalize_match_text(_strip_published_marker(ptext)) == norm_target:
                target_match = m
                break

        if target_match is None:
            return "not_found"
        plain_hit = html_lib.unescape(re.sub(r"<[^>]+>", "", target_match.group(2))).strip()
        if _PUBLISHED_HEADER_TAG_RE.search(plain_hit):
            return "already_marked"

        insert_pos = target_match.end(2)
        updated_html = current_html[:insert_pos] + " [PUBLICADO]" + current_html[insert_pos:]
        media = MediaInMemoryUpload(updated_html.encode("utf-8"), mimetype="text/html", resumable=False)
        _execute_with_retry(
            drive_service.files().update(
                fileId=doc_id,
                media_body=media,
                fields="id, name",
                supportsAllDrives=True,
            ),
            f"drive_html_mark_pub({doc_id})",
        )
        logger.info("Encabezado '%s [PUBLICADO]' marcado vía respaldo HTML en Doc %s.", clean_target, doc_id)
        return "drive_html_fallback"
    except HttpError as drive_exc:
        raise DriveServiceError(
            f"Error marcando [PUBLICADO] en Doc {doc_id} vía Drive HTML: {drive_exc}"
        ) from drive_exc


def compute_month_week_info(reference_dt: datetime) -> tuple[int, datetime, datetime]:
    """Compute `(week_number, week_start_dt, week_end_dt)` within the month of `reference_dt`.

    Weeks close on Sunday (`weekday() == 6`). If the 1st of the month is Sunday, it is
    grouped into Week 1 along with the following Mon-Sun so Week 1 is never a single day.
    """
    year = reference_dt.year
    month = reference_dt.month
    tz = reference_dt.tzinfo
    days_in_month = calendar.monthrange(year, month)[1]

    weeks: list[tuple[int, int]] = []
    start_day = 1
    for day in range(1, days_in_month + 1):
        wd = datetime(year, month, day).weekday()
        if wd == 6 and not (day == 1):
            weeks.append((start_day, day))
            start_day = day + 1
    if start_day <= days_in_month:
        weeks.append((start_day, days_in_month))

    target_day = reference_dt.day
    for idx, (s_day, e_day) in enumerate(weeks, start=1):
        if s_day <= target_day <= e_day:
            start_dt = datetime(year, month, s_day, 0, 0, 0, tzinfo=tz)
            end_dt = datetime(year, month, e_day, 23, 59, 59, tzinfo=tz)
            return idx, start_dt, end_dt

    return 1, reference_dt, reference_dt


def is_last_weekly_report_of_month(reference_dt: datetime) -> bool:
    """Return True when `reference_dt` falls in the final Sunday-closing week of its month."""
    _, _, week_end = compute_month_week_info(reference_dt)
    days_in_month = calendar.monthrange(reference_dt.year, reference_dt.month)[1]
    return (days_in_month - week_end.day) < 7


def _find_or_create_subfolder_ci(
    service: Resource,
    parent_id: str,
    candidate_names: set[str],
    default_name: str,
) -> tuple[str, bool]:
    query = (
        f"'{parent_id}' in parents "
        f"and mimeType = '{GOOGLE_FOLDER_MIME_TYPE}' "
        "and trashed = false"
    )
    response = _execute_with_retry(
        service.files().list(
            q=query,
            spaces="drive",
            corpora="allDrives",
            fields="files(id, name)",
            pageSize=100,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ),
        f"list_subfolders_ci({parent_id})",
    )
    norm_candidates = {_normalize_match_text(c) for c in candidate_names}
    for folder in response.get("files", []):
        if _normalize_match_text(str(folder.get("name", ""))) in norm_candidates:
            return str(folder["id"]), False

    new_id = _create_subfolder(service, parent_id, default_name)
    return new_id, True


def ensure_statistics_folder(
    month_folder_id: str,
    credentials_path: Path | str | None = None,
) -> tuple[str, bool]:
    """Find or create the `Estadística(s)` subfolder inside `month_folder_id`."""
    try:
        service = _get_drive_client(credentials_path)
        return _find_or_create_subfolder_ci(
            service=service,
            parent_id=month_folder_id,
            candidate_names={"estadistica", "estadisticas"},
            default_name="Estadísticas",
        )
    except HttpError as exc:
        raise DriveServiceError(
            f"Error resolviendo carpeta de Estadísticas en {month_folder_id}: {exc}"
        ) from exc


def ensure_statistics_week_folder(
    month_folder_id: str,
    week_num: int,
    credentials_path: Path | str | None = None,
) -> tuple[str, str, bool]:
    """Ensure `Estadísticas/Semana {week_num}` exists inside `month_folder_id`."""
    try:
        service = _get_drive_client(credentials_path)
        stats_id, stats_created = _find_or_create_subfolder_ci(
            service=service,
            parent_id=month_folder_id,
            candidate_names={"estadistica", "estadisticas"},
            default_name="Estadísticas",
        )
        week_label = f"Semana {week_num}"
        week_id, week_created = _find_or_create_subfolder_ci(
            service=service,
            parent_id=stats_id,
            candidate_names={week_label, f"semana{week_num}"},
            default_name=week_label,
        )
        return stats_id, week_id, (stats_created or week_created)
    except HttpError as exc:
        raise DriveServiceError(
            f"Error resolviendo carpeta Semana {week_num} de Estadísticas en {month_folder_id}: {exc}"
        ) from exc


def _find_doc_by_title_ci(
    service: Resource,
    folder_id: str,
    doc_title: str,
) -> str | None:
    query = (
        f"'{folder_id}' in parents "
        f"and mimeType = '{GOOGLE_DOC_MIME_TYPE}' "
        "and trashed = false"
    )
    response = _execute_with_retry(
        service.files().list(
            q=query,
            spaces="drive",
            corpora="allDrives",
            fields="files(id, name)",
            pageSize=50,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ),
        f"find_doc_ci({folder_id})",
    )
    target_norm = _normalize_match_text(doc_title)
    for item in response.get("files", []):
        if _normalize_match_text(str(item.get("name", ""))) == target_norm:
            return str(item["id"])
    return None


def create_or_update_google_doc(
    folder_id: str,
    doc_title: str,
    plain_text_content: str,
    bold_prefixes: tuple[str, ...] = (),
    credentials_path: Path | str | None = None,
) -> tuple[str, bool]:
    """Create a Google Doc in `folder_id` or overwrite an existing one with the same title."""
    lines = plain_text_content.splitlines()
    body_html = _lines_to_html_paragraphs(lines, bold_prefixes=bold_prefixes)
    full_html = f"<html><body>{body_html}</body></html>"

    try:
        service = _get_drive_client(credentials_path)
        existing_id = _find_doc_by_title_ci(service, folder_id, doc_title)
        media = MediaInMemoryUpload(
            full_html.encode("utf-8"),
            mimetype="text/html",
            resumable=False,
        )
        if existing_id:
            _execute_with_retry(
                service.files().update(
                    fileId=existing_id,
                    media_body=media,
                    fields="id, name",
                    supportsAllDrives=True,
                ),
                f"update_google_doc({existing_id})",
            )
            logger.info("Documento '%s' actualizado en Drive (id=%s).", doc_title, existing_id)
            return existing_id, False

        created = _execute_with_retry(
            service.files().create(
                body={
                    "name": doc_title,
                    "mimeType": GOOGLE_DOC_MIME_TYPE,
                    "parents": [folder_id],
                },
                media_body=media,
                fields="id, name",
                supportsAllDrives=True,
            ),
            f"create_google_doc({doc_title})",
        )
        new_id = str(created["id"])
        logger.info("Documento '%s' creado en Drive (id=%s, folder=%s).", doc_title, new_id, folder_id)
        return new_id, True
    except HttpError as exc:
        raise DriveServiceError(
            f"Error creando/actualizando documento '{doc_title}' en carpeta {folder_id}: {exc}"
        ) from exc


def append_or_create_scripts_doc(
    month_folder_id: str,
    script_lines: list[str],
    doc_title: str = "Guiones",
    credentials_path: Path | str | None = None,
) -> tuple[str, bool]:
    """Append a generated script block to `Guiones` in `month_folder_id` (creating it if needed)."""
    try:
        service = _get_drive_client(credentials_path)
        existing_id = _find_doc_by_title_ci(service, month_folder_id, doc_title)
        if existing_id:
            append_entry_to_monthly_doc(
                doc_id=existing_id,
                entry_lines=script_lines,
                credentials_path=credentials_path,
            )
            return existing_id, False

        header_text = (
            "BANCO DE GUIONES DE GRABACIÓN - @NELHEALTHCOACH\n\n"
            + "\n".join(script_lines)
            + "\n"
        )
        return create_or_update_google_doc(
            folder_id=month_folder_id,
            doc_title=doc_title,
            plain_text_content=header_text,
            bold_prefixes=(
                "Fecha:",
                "Formato:",
                "Gancho (0-3s):",
                "Desarrollo (3-45s):",
                "Cierre y CTA (45-60s):",
                "Notas de grabación:",
                "Caption:",
                "Hashtags:",
            ),
            credentials_path=credentials_path,
        )
    except HttpError as exc:
        raise DriveServiceError(
            f"Error guardando guion en '{doc_title}' (folder={month_folder_id}): {exc}"
        ) from exc


def _list_week_subfolders(service: Resource, stats_folder_id: str) -> list[tuple[int, str, str]]:
    """Return sorted `[(week_num, folder_id, folder_name), ...]` inside an `Estadísticas` folder."""
    query = (
        f"'{stats_folder_id}' in parents "
        f"and mimeType = '{GOOGLE_FOLDER_MIME_TYPE}' "
        "and trashed = false"
    )
    response = _execute_with_retry(
        service.files().list(
            q=query,
            spaces="drive",
            corpora="allDrives",
            fields="files(id, name)",
            pageSize=50,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ),
        f"list_week_subfolders({stats_folder_id})",
    )
    weeks: list[tuple[int, str, str]] = []
    for folder in response.get("files", []):
        fname = str(folder.get("name", "")).strip()
        m = re.search(r"semana\s*(\d+)", fname, flags=re.IGNORECASE)
        if m:
            weeks.append((int(m.group(1)), str(folder["id"]), fname))
    weeks.sort(key=lambda item: item[0])
    return weeks


def _read_reports_in_week_folder(
    service: Resource,
    week_folder_id: str,
    period_label: str,
    credentials_path: Path | str | None = None,
) -> list[dict[str, str]]:
    """Export text of Google Docs / text files inside one `Semana N` folder, skipping images."""
    query = f"'{week_folder_id}' in parents and trashed = false"
    response = _execute_with_retry(
        service.files().list(
            q=query,
            spaces="drive",
            corpora="allDrives",
            fields="files(id, name, mimeType, modifiedTime)",
            orderBy="modifiedTime desc",
            pageSize=30,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ),
        f"list_week_reports({week_folder_id})",
    )
    reports: list[dict[str, str]] = []
    for item in response.get("files", []):
        mime_type = str(item.get("mimeType", "")).lower()
        name = str(item.get("name", ""))
        is_doc = mime_type == GOOGLE_DOC_MIME_TYPE
        is_text = mime_type.startswith("text/") or name.lower().endswith((".txt", ".md"))
        if not (is_doc or is_text):
            continue
        try:
            text = get_guidelines_text(str(item["id"]), credentials_path=credentials_path)
            if text.strip():
                reports.append(
                    {
                        "label": period_label,
                        "title": name,
                        "text": text.strip(),
                    }
                )
        except Exception as exc:
            logger.debug("No se pudo leer informe '%s' (%s): %s", name, item.get("id"), exc)
    return reports


def collect_previous_weekly_reports(
    root_folder_id: str,
    month_folder_id: str,
    year_str: str,
    month_str: str,
    current_week_num: int,
    credentials_path: Path | str | None = None,
) -> list[dict[str, str]]:
    """Collect prior weekly reports for structure reference, 7-day comparison, and monthly rollover.

    - For week >= 2: reads earlier `Semana N` reports from the current month (`1..current_week_num-1`).
    - For week == 1 (or if no prior weeks exist in current month): looks up the previous month's
      `Estadísticas` folder and reads its final week report (e.g., Semana 4 with monthly summary).
    """
    collected: list[dict[str, str]] = []
    try:
        service = _get_drive_client(credentials_path)
        stats_query = (
            f"'{month_folder_id}' in parents "
            f"and mimeType = '{GOOGLE_FOLDER_MIME_TYPE}' "
            "and trashed = false"
        )
        stats_resp = _execute_with_retry(
            service.files().list(
                q=stats_query,
                spaces="drive",
                corpora="allDrives",
                fields="files(id, name)",
                pageSize=50,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ),
            f"find_stats_folder({month_folder_id})",
        )
        stats_folder = next(
            (
                f
                for f in stats_resp.get("files", [])
                if _normalize_match_text(str(f.get("name", ""))) in {"estadistica", "estadisticas"}
            ),
            None,
        )
        if stats_folder:
            week_folders = _list_week_subfolders(service, str(stats_folder["id"]))
            for w_num, w_id, w_name in week_folders:
                if w_num < current_week_num:
                    collected.extend(
                        _read_reports_in_week_folder(
                            service,
                            w_id,
                            f"{month_str}/{year_str} - {w_name}",
                            credentials_path=credentials_path,
                        )
                    )

        if current_week_num <= 1 or not collected:
            year = int(year_str)
            month = int(month_str)
            for _ in range(3):
                month -= 1
                if month < 1:
                    month = 12
                    year -= 1
                prev_year_id = _find_subfolder_id(service, root_folder_id, str(year))
                if not prev_year_id:
                    continue
                prev_month_id = _find_subfolder_id(service, prev_year_id, f"{month:02d}")
                if not prev_month_id:
                    continue
                prev_stats_resp = _execute_with_retry(
                    service.files().list(
                        q=(
                            f"'{prev_month_id}' in parents "
                            f"and mimeType = '{GOOGLE_FOLDER_MIME_TYPE}' "
                            "and trashed = false"
                        ),
                        spaces="drive",
                        corpora="allDrives",
                        fields="files(id, name)",
                        pageSize=50,
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    ),
                    f"find_prev_stats({year}/{month:02d})",
                )
                prev_stats = next(
                    (
                        f
                        for f in prev_stats_resp.get("files", [])
                        if _normalize_match_text(str(f.get("name", ""))) in {"estadistica", "estadisticas"}
                    ),
                    None,
                )
                if not prev_stats:
                    continue
                prev_weeks = _list_week_subfolders(service, str(prev_stats["id"]))
                if not prev_weeks:
                    continue
                # Walk backwards from the final week (e.g. Semana 4 with monthly summary) until a report doc is found
                for _, last_w_id, last_w_name in reversed(prev_weeks):
                    prev_reports = _read_reports_in_week_folder(
                        service,
                        last_w_id,
                        f"{month:02d}/{year} - {last_w_name} (cierre mes anterior)",
                        credentials_path=credentials_path,
                    )
                    if prev_reports:
                        collected.extend(prev_reports)
                        break
                if collected:
                    break
    except Exception as exc:
        logger.warning("No se pudieron leer informes semanales previos: %s", exc)

    return collected
