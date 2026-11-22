"""TikTok Content Posting API v2 service for organic-only video publishing."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from drive_service import get_tiktok_auth_state, save_tiktok_auth_state

logger = logging.getLogger(__name__)

TIKTOK_API_BASE_URL = "https://open.tiktokapis.com/v2"
TIKTOK_SINGLE_CHUNK_MAX_BYTES = 64 * 1024 * 1024  # 64 MB
TIKTOK_MULTI_CHUNK_BYTES = 20 * 1024 * 1024  # 20 MB
DEFAULT_POLL_INTERVAL_SECONDS = 5
DEFAULT_POLL_TIMEOUT_SECONDS = 180


class TikTokPublisherError(RuntimeError):
    """Raised when TikTok OAuth token refresh, upload, or publishing fails."""


@dataclass(frozen=True, slots=True)
class TikTokPublishResult:
    """Result metadata of an organic TikTok video publish/draft upload."""

    publish_id: str
    mode: str  # DIRECT | INBOX
    privacy_level: str
    status: str


def _extract_tiktok_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"

    err = payload.get("error")
    if isinstance(err, dict):
        code = err.get("code", "")
        msg = err.get("message", "")
        log_id = err.get("log_id", "")
        if code or msg:
            return f"{code}: {msg} (log_id={log_id})"
    if payload.get("error_description"):
        return f"{payload.get('error')}: {payload.get('error_description')}"
    return str(payload)


def _compute_tiktok_chunks(video_size: int) -> tuple[int, list[tuple[int, int]]]:
    """Compute TikTok-compliant `(chunk_size, [(start, end_inclusive), ...])`."""
    if video_size <= 0:
        raise TikTokPublisherError("El archivo de video está vacío (0 bytes).")
    if video_size <= TIKTOK_SINGLE_CHUNK_MAX_BYTES:
        return video_size, [(0, video_size - 1)]

    chunk_size = TIKTOK_MULTI_CHUNK_BYTES
    total_chunks = max(1, video_size // chunk_size)
    ranges: list[tuple[int, int]] = []
    for idx in range(total_chunks):
        start = idx * chunk_size
        end = (video_size - 1) if idx == total_chunks - 1 else (start + chunk_size - 1)
        ranges.append((start, end))
    return chunk_size, ranges


async def _get_valid_access_token(
    client: httpx.AsyncClient,
    client_key: str,
    client_secret: str,
    env_refresh_token: str,
    env_access_token: str,
    state_path: Path | str | None = None,
) -> str:
    persisted = await asyncio.to_thread(get_tiktok_auth_state, state_path)
    now_ts = time.time()

    access_token = str(persisted.get("access_token") or env_access_token or "").strip()
    expires_at = float(persisted.get("expires_at") or 0.0)
    refresh_token = str(persisted.get("refresh_token") or env_refresh_token or "").strip()

    if access_token and (expires_at == 0.0 or expires_at > now_ts + 300):
        # If token came purely from .env without expires_at and refresh exists, we can still try it,
        # or if refresh is available and expires_at is set and stale, refresh below.
        if persisted.get("access_token") or not refresh_token:
            return access_token

    if not refresh_token or not client_key or not client_secret:
        if access_token:
            return access_token
        raise TikTokPublisherError(
            "Faltan credenciales de TikTok (TIKTOK_CLIENT_KEY, TIKTOK_CLIENT_SECRET, TIKTOK_REFRESH_TOKEN)."
        )

    response = await client.post(
        f"{TIKTOK_API_BASE_URL}/oauth/token/",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "client_key": client_key,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
    )
    if response.is_error:
        raise TikTokPublisherError(
            f"Error renovando token OAuth de TikTok: {_extract_tiktok_error(response)}"
        )

    payload = response.json()
    new_access = str(payload.get("access_token", "")).strip()
    new_refresh = str(payload.get("refresh_token", refresh_token)).strip() or refresh_token
    expires_in = int(payload.get("expires_in") or 86400)

    if not new_access:
        raise TikTokPublisherError(
            f"TikTok no devolvió access_token al renovar: {payload}"
        )

    await asyncio.to_thread(
        save_tiktok_auth_state,
        {
            "access_token": new_access,
            "refresh_token": new_refresh,
            "expires_at": time.time() + expires_in,
            "open_id": str(payload.get("open_id", persisted.get("open_id", ""))),
            "scope": str(payload.get("scope", persisted.get("scope", ""))),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        state_path,
    )
    logger.info("Token OAuth de TikTok renovado y guardado en state.json.")
    return new_access


async def _poll_publish_status(
    client: httpx.AsyncClient,
    access_token: str,
    publish_id: str,
    poll_interval: int = DEFAULT_POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_POLL_TIMEOUT_SECONDS,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    while True:
        response = await client.post(
            f"{TIKTOK_API_BASE_URL}/post/publish/status/fetch/",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=UTF-8",
            },
            json={"publish_id": publish_id},
        )
        if response.is_error:
            raise TikTokPublisherError(
                f"Error consultando estado de TikTok ({publish_id}): {_extract_tiktok_error(response)}"
            )

        data = response.json().get("data") or {}
        status = str(data.get("status", "PROCESSING")).upper()
        logger.info("Estado TikTok publish_id=%s: %s", publish_id, status)

        if status in {"PUBLISH_COMPLETE", "SEND_TO_USER_INBOX"}:
            return status
        if status == "FAILED":
            reason = data.get("fail_reason") or "Sin detalle"
            raise TikTokPublisherError(
                f"Publicación en TikTok falló (publish_id={publish_id}): {reason}"
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TikTokPublisherError(
                f"Timeout ({timeout_seconds}s) esperando publicación en TikTok (publish_id={publish_id}, último estado={status})."
            )
        await asyncio.sleep(min(poll_interval, remaining))


async def publish_video_to_tiktok(
    *,
    video_path: Path,
    caption: str,
    client_key: str,
    client_secret: str,
    refresh_token: str,
    access_token: str = "",
    publish_mode: str = "DIRECT",
    privacy_level: str = "PUBLIC_TO_EVERYONE",
    state_path: Path | str | None = None,
) -> TikTokPublishResult:
    """Upload and publish a local MP4 video organically to TikTok via Content Posting API v2."""
    if not video_path.is_file():
        raise TikTokPublisherError(f"Archivo de video no encontrado para TikTok: {video_path}")

    video_size = video_path.stat().st_size
    chunk_size, chunk_ranges = _compute_tiktok_chunks(video_size)
    mode = publish_mode.strip().upper()
    if mode not in {"DIRECT", "INBOX"}:
        mode = "DIRECT"

    safe_title = caption.strip()[:2200]

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        token = await _get_valid_access_token(
            client=client,
            client_key=client_key,
            client_secret=client_secret,
            env_refresh_token=refresh_token,
            env_access_token=access_token,
            state_path=state_path,
        )

        if mode == "INBOX":
            init_url = f"{TIKTOK_API_BASE_URL}/post/publish/inbox/video/init/"
            init_body: dict[str, Any] = {
                "source_info": {
                    "source": "FILE_UPLOAD",
                    "video_size": video_size,
                    "chunk_size": chunk_size,
                    "total_chunk_count": len(chunk_ranges),
                }
            }
        else:
            init_url = f"{TIKTOK_API_BASE_URL}/post/publish/video/init/"
            init_body = {
                "post_info": {
                    "title": safe_title,
                    "privacy_level": privacy_level,
                    "disable_duet": False,
                    "disable_comment": False,
                    "disable_stitch": False,
                },
                "source_info": {
                    "source": "FILE_UPLOAD",
                    "video_size": video_size,
                    "chunk_size": chunk_size,
                    "total_chunk_count": len(chunk_ranges),
                },
            }

        init_resp = await client.post(
            init_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=UTF-8",
            },
            json=init_body,
        )
        if init_resp.is_error:
            raise TikTokPublisherError(
                f"Error inicializando publicación en TikTok ({mode}): {_extract_tiktok_error(init_resp)}"
            )

        init_payload = init_resp.json()
        err = init_payload.get("error") or {}
        if err.get("code") not in {None, "", "ok"}:
            raise TikTokPublisherError(
                f"TikTok rechazó init ({mode}): {err.get('code')} - {err.get('message')}"
            )

        data = init_payload.get("data") or {}
        publish_id = str(data.get("publish_id", "")).strip()
        upload_url = str(data.get("upload_url", "")).strip()
        if not publish_id or not upload_url:
            raise TikTokPublisherError(
                f"Respuesta incompleta de TikTok init: {init_payload}"
            )

        with video_path.open("rb") as fh:
            for start, end in chunk_ranges:
                fh.seek(start)
                chunk_bytes = fh.read(end - start + 1)
                put_resp = await client.put(
                    upload_url,
                    headers={
                        "Content-Type": "video/mp4",
                        "Content-Length": str(len(chunk_bytes)),
                        "Content-Range": f"bytes {start}-{end}/{video_size}",
                    },
                    content=chunk_bytes,
                )
                if put_resp.is_error:
                    raise TikTokPublisherError(
                        f"Error subiendo chunk {start}-{end} a TikTok: HTTP {put_resp.status_code} {put_resp.text}"
                    )

        final_status = await _poll_publish_status(
            client=client,
            access_token=token,
            publish_id=publish_id,
        )

    logger.info(
        "TikTok orgánico completado (mode=%s, publish_id=%s, status=%s).",
        mode,
        publish_id,
        final_status,
    )
    return TikTokPublishResult(
        publish_id=publish_id,
        mode=mode,
        privacy_level=privacy_level if mode == "DIRECT" else "INBOX",
        status=final_status,
    )


async def fetch_tiktok_recent_videos(
    *,
    client_key: str,
    client_secret: str,
    refresh_token: str,
    access_token: str = "",
    state_path: Path | str | None = None,
    max_count: int = 20,
) -> list[dict[str, Any]]:
    """Best-effort fetch of recent organic TikTok video stats (requires `video.list` scope)."""
    if not ((refresh_token and client_key and client_secret) or access_token):
        return []

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
            token = await _get_valid_access_token(
                client=client,
                client_key=client_key,
                client_secret=client_secret,
                env_refresh_token=refresh_token,
                env_access_token=access_token,
                state_path=state_path,
            )
            response = await client.post(
                f"{TIKTOK_API_BASE_URL}/video/list/",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json; charset=UTF-8",
                },
                params={
                    "fields": "id,title,create_time,like_count,comment_count,share_count,view_count"
                },
                json={"max_count": min(max(int(max_count), 1), 20)},
            )
            if response.is_error:
                logger.info(
                    "Estadísticas de TikTok no disponibles (%s); se omiten del informe.",
                    _extract_tiktok_error(response),
                )
                return []
            data = response.json().get("data") or {}
            videos = data.get("videos", [])
            return list(videos) if isinstance(videos, list) else []
    except Exception as exc:
        logger.info("No se pudieron consultar videos recientes de TikTok: %s", exc)
        return []
