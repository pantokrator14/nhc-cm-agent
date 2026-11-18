"""Google Drive integration for hierarchical YYYY/MM folders and local state tracking."""

from __future__ import annotations

import html as html_lib
import json
import logging
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

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


def _normalize_state_shape(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"files": {}, "media_posts": {}, "processed_comments": {}}
    return {
        "files": data["files"] if isinstance(data.get("files"), dict) else {},
        "media_posts": data["media_posts"] if isinstance(data.get("media_posts"), dict) else {},
        "processed_comments": (
            data["processed_comments"]
            if isinstance(data.get("processed_comments"), dict)
            else {}
        ),
    }


def _read_state_unlocked(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"files": {}, "media_posts": {}, "processed_comments": {}}
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
            state = {"files": {}, "media_posts": {}, "processed_comments": {}}

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
            state = {"files": {}, "media_posts": {}, "processed_comments": {}}

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
            state = {"files": {}, "media_posts": {}, "processed_comments": {}}

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


def list_pending_files(
    month_folder_id: str,
    state_path: Path | str | None = None,
    credentials_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """List MP4/JPG/JPEG/PNG files in `month_folder_id` not yet recorded as finalized in `state.json`."""
    state = load_state(state_path)
    recorded_files: dict[str, Any] = state["files"]

    try:
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
                f"list_pending_files({month_folder_id})",
            )
            all_files.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break

        pending: list[dict[str, Any]] = []
        for file_meta in all_files:
            if not _is_supported_media(file_meta):
                continue

            file_id = file_meta["id"]
            local_status = str(recorded_files.get(file_id, {}).get("status", "")).upper()
            if local_status in FINAL_SKIP_STATUSES:
                continue

            drive_status = str((file_meta.get("appProperties") or {}).get("status", "")).lower()
            if drive_status == "published":
                continue

            pending.append(file_meta)

        logger.info(
            "Escaneo de carpeta mensual (%s): %d elementos, %d pendientes por procesar.",
            month_folder_id,
            len(all_files),
            len(pending),
        )
        return pending
    except HttpError as exc:
        raise DriveServiceError(
            f"Error de Google Drive al listar carpeta mensual {month_folder_id}: {exc}"
        ) from exc


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
        if not _is_supported_media(f):
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

    clean_title = re.sub(r'[\\/:*?"<>|]+', "", short_title).strip()[:60] or "Publicación"
    ext = Path(original_filename).suffix or ".mp4"

    if post_category.upper() == "PRO_TIP":
        doc_header = f"Pro Tip {max_protip_num + 1}: {clean_title}"
        new_filename = f"{next_drive_num:02d}. Pro tip {clean_title}{ext}"
    else:
        doc_header = f"{max_main_num + 1}. {clean_title}"
        new_filename = f"{next_drive_num:02d}. {clean_title}{ext}"

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
