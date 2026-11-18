"""Instagram Graph API v21.0 publishing service."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Literal

import httpx

from drive_service import save_published_media

logger = logging.getLogger(__name__)

GRAPH_API_BASE_URL = "https://graph.facebook.com/v21.0"
DEFAULT_POLL_INTERVAL_SECONDS = 10
DEFAULT_POLL_TIMEOUT_SECONDS = 180


class MetaPublisherError(RuntimeError):
    """Raised when creating, polling, or publishing an Instagram container fails."""


def _extract_meta_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message", "Error desconocido")
        code = error.get("code")
        subcode = error.get("error_subcode")
        return f"{message} (code={code}, subcode={subcode})"
    return str(payload)


async def _create_media_container(
    client: httpx.AsyncClient,
    ig_user_id: str,
    access_token: str,
    media_url: str,
    media_type: Literal["REELS", "IMAGE", "STORIES"],
    caption: str,
    is_video: bool = False,
) -> str:
    endpoint = f"{GRAPH_API_BASE_URL}/{ig_user_id}/media"
    data: dict[str, Any] = {
        "access_token": access_token,
    }
    if media_type == "REELS":
        data["media_type"] = "REELS"
        data["video_url"] = media_url
        data["caption"] = caption
    elif media_type == "IMAGE":
        data["image_url"] = media_url
        data["caption"] = caption
    elif media_type == "STORIES":
        data["media_type"] = "STORIES"
        if is_video:
            data["video_url"] = media_url
        else:
            data["image_url"] = media_url
    else:
        raise MetaPublisherError(f"media_type no soportado: {media_type}")

    response = await client.post(endpoint, data=data)
    if response.is_error:
        raise MetaPublisherError(
            f"Error al crear contenedor en Instagram: {_extract_meta_error(response)}"
        )

    payload = response.json()
    creation_id = payload.get("id")
    if not creation_id:
        raise MetaPublisherError(
            f"Meta no devolvió un ID de contenedor válido: {payload}"
        )

    logger.info(
        "Contenedor IG creado (creation_id=%s, media_type=%s).",
        creation_id,
        media_type,
    )
    return str(creation_id)


async def _wait_for_container_ready(
    client: httpx.AsyncClient,
    creation_id: str,
    access_token: str,
    poll_interval: int = DEFAULT_POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_POLL_TIMEOUT_SECONDS,
) -> None:
    endpoint = f"{GRAPH_API_BASE_URL}/{creation_id}"
    deadline = time.monotonic() + timeout_seconds

    while True:
        response = await client.get(
            endpoint,
            params={
                "fields": "status_code,status",
                "access_token": access_token,
            },
        )
        if response.is_error:
            raise MetaPublisherError(
                f"Error al consultar estado del contenedor {creation_id}: "
                f"{_extract_meta_error(response)}"
            )

        payload = response.json()
        status_code = str(payload.get("status_code", "")).upper()
        logger.info(
            "Estado del contenedor %s: status_code=%s",
            creation_id,
            status_code or "UNKNOWN",
        )

        if status_code == "FINISHED":
            return
        if status_code in {"ERROR", "EXPIRED"}:
            detail = payload.get("status", "Sin detalle adicional")
            raise MetaPublisherError(
                f"El contenedor {creation_id} falló con status_code={status_code}: {detail}"
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MetaPublisherError(
                f"Timeout ({timeout_seconds}s) esperando que el contenedor {creation_id} "
                f"llegue a FINISHED (último estado: {status_code or 'UNKNOWN'})."
            )

        await asyncio.sleep(min(poll_interval, remaining))


async def _publish_container(
    client: httpx.AsyncClient,
    ig_user_id: str,
    creation_id: str,
    access_token: str,
) -> str:
    endpoint = f"{GRAPH_API_BASE_URL}/{ig_user_id}/media_publish"
    response = await client.post(
        endpoint,
        data={
            "creation_id": creation_id,
            "access_token": access_token,
        },
    )
    if response.is_error:
        raise MetaPublisherError(
            f"Error al publicar contenedor {creation_id}: {_extract_meta_error(response)}"
        )

    payload = response.json()
    media_id = payload.get("id")
    if not media_id:
        raise MetaPublisherError(
            f"Meta no devolvió el ID de publicación tras media_publish: {payload}"
        )

    logger.info("Publicación exitosa en Instagram. media_id=%s", media_id)
    return str(media_id)


async def publish_to_instagram(
    ig_user_id: str,
    access_token: str,
    media_url: str,
    media_type: Literal["REELS", "IMAGE", "STORIES"],
    caption: str,
    *,
    is_video: bool = False,
    file_id: str | None = None,
    has_keyword_cta: bool = False,
    keyword: str = "",
    target_link: str = "",
    public_reply_text: str = "",
    state_path: Path | str | None = None,
    poll_interval: int = DEFAULT_POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_POLL_TIMEOUT_SECONDS,
) -> str:
    """Create, poll until FINISHED, publish an Instagram container, and record `media_id`.

    Returns:
        The published Instagram `media_id`.
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        creation_id = await _create_media_container(
            client=client,
            ig_user_id=ig_user_id,
            access_token=access_token,
            media_url=media_url,
            media_type=media_type,
            caption=caption,
            is_video=is_video,
        )
        await _wait_for_container_ready(
            client=client,
            creation_id=creation_id,
            access_token=access_token,
            poll_interval=poll_interval,
            timeout_seconds=timeout_seconds,
        )
        media_id = await _publish_container(
            client=client,
            ig_user_id=ig_user_id,
            creation_id=creation_id,
            access_token=access_token,
        )

    await asyncio.to_thread(
        save_published_media,
        media_id=media_id,
        has_keyword_cta=has_keyword_cta,
        keyword=keyword,
        target_link=target_link,
        public_reply_text=public_reply_text,
        file_id=file_id,
        state_path=state_path,
    )
    logger.info(
        "Metadatos de media_id=%s guardados en state.json (has_keyword_cta=%s, keyword=%s).",
        media_id,
        has_keyword_cta,
        keyword or "N/A",
    )
    return media_id
