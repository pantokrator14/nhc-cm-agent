"""Instagram Graph API v21.0 publishing service."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import httpx

from drive_service import (
    _token_fingerprint,
    clear_meta_auth_state,
    get_cached_location_id,
    save_cached_location_id,
    save_meta_auth_state,
    save_published_media,
)

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
        details: list[str] = []
        user_title = str(error.get("error_user_title", "") or "").strip()
        user_msg = str(error.get("error_user_msg", "") or "").strip()
        if user_title or user_msg:
            details.append(" — ".join(p for p in (user_title, user_msg) if p))
        err_data = error.get("error_data")
        if isinstance(err_data, dict) and err_data.get("blame_field_specs"):
            details.append(f"campos={err_data['blame_field_specs']}")
        extra = f" | {'; '.join(details)}" if details else ""
        return f"{message} (code={code}, subcode={subcode}){extra}"
    return str(payload)


def _is_location_container_error(error_msg: str) -> bool:
    lowered = error_msg.lower()
    return "location" in lowered or "invalid_location" in lowered


async def _create_media_container(
    client: httpx.AsyncClient,
    ig_user_id: str,
    access_token: str,
    media_url: str,
    media_type: Literal["REELS", "IMAGE", "STORIES"],
    caption: str,
    is_video: bool = False,
    location_id: str = "",
) -> str:
    endpoint = f"{GRAPH_API_BASE_URL}/{ig_user_id}/media"
    data: dict[str, Any] = {
        "access_token": access_token,
    }
    if media_type == "REELS":
        data["media_type"] = "REELS"
        data["video_url"] = media_url
        data["caption"] = caption
        if location_id:
            data["location_id"] = location_id
    elif media_type == "IMAGE":
        data["image_url"] = media_url
        data["caption"] = caption
        if location_id:
            data["location_id"] = location_id
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
        err_msg = _extract_meta_error(response)
        if location_id and "location_id" in data and _is_location_container_error(err_msg):
            logger.warning(
                "Meta rechazó location_id=%s (%s); reintentando publicación sin ubicación.",
                location_id,
                err_msg,
            )
            data_no_loc = {k: v for k, v in data.items() if k != "location_id"}
            response = await client.post(endpoint, data=data_no_loc)
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
    location_id: str = "",
    file_id: str | None = None,
    has_keyword_cta: bool = False,
    keyword: str = "",
    target_link: str = "",
    public_reply_text: str = "",
    dm_message: str = "",
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
            location_id=location_id,
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
        dm_message=dm_message,
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


async def fetch_recent_instagram_media(
    ig_user_id: str,
    access_token: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Fetch recent published media posts on Instagram to cross-check unpublished guidelines."""
    endpoint = f"{GRAPH_API_BASE_URL}/{ig_user_id}/media"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            response = await client.get(
                endpoint,
                params={
                    "fields": "id,caption,media_type,timestamp,permalink",
                    "limit": min(max(int(limit), 1), 100),
                    "access_token": access_token,
                },
            )
            if response.is_error:
                logger.warning(
                    "No se pudieron consultar publicaciones recientes de IG (%s): %s",
                    ig_user_id,
                    _extract_meta_error(response),
                )
                return []
            payload = response.json()
            data = payload.get("data", [])
            return list(data) if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("Error de red consultando IG /media para %s: %s", ig_user_id, exc)
        return []


async def fetch_instagram_weekly_metrics(
    ig_user_id: str,
    access_token: str,
    since_dt: datetime,
    until_dt: datetime,
    limit: int = 50,
) -> dict[str, Any]:
    """Fetch Instagram profile summary and post-level insights within `[since_dt, until_dt]`."""
    profile: dict[str, Any] = {}
    posts_in_range: list[dict[str, Any]] = []

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
            prof_resp = await client.get(
                f"{GRAPH_API_BASE_URL}/{ig_user_id}",
                params={
                    "fields": "username,followers_count,media_count",
                    "access_token": access_token,
                },
            )
            if not prof_resp.is_error:
                profile = prof_resp.json()

            media_resp = await client.get(
                f"{GRAPH_API_BASE_URL}/{ig_user_id}/media",
                params={
                    "fields": (
                        "id,caption,media_type,media_product_type,"
                        "timestamp,like_count,comments_count,permalink"
                    ),
                    "limit": min(max(int(limit), 1), 100),
                    "access_token": access_token,
                },
            )
            if media_resp.is_error:
                logger.warning(
                    "No se pudo listar media de IG para métricas: %s",
                    _extract_meta_error(media_resp),
                )
                return {"profile": profile, "posts": []}

            raw_media = list(media_resp.json().get("data", []))
            for item in raw_media:
                ts_raw = str(item.get("timestamp", "")).strip()
                if not ts_raw:
                    continue
                try:
                    post_dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                except ValueError:
                    continue

                if not (since_dt <= post_dt <= until_dt):
                    continue

                media_id = str(item["id"])
                insights_map: dict[str, Any] = {}
                for metric_set in (
                    "reach,saved,shares,views,total_interactions",
                    "reach,saved,shares",
                ):
                    ins_resp = await client.get(
                        f"{GRAPH_API_BASE_URL}/{media_id}/insights",
                        params={"metric": metric_set, "access_token": access_token},
                    )
                    if not ins_resp.is_error:
                        for m in ins_resp.json().get("data", []):
                            name = str(m.get("name", ""))
                            vals = m.get("values") or []
                            if name and vals:
                                insights_map[name] = vals[0].get("value", 0)
                        break

                posts_in_range.append(
                    {
                        "id": media_id,
                        "timestamp": ts_raw,
                        "media_type": item.get("media_product_type") or item.get("media_type"),
                        "caption_preview": str(item.get("caption", "")).splitlines()[0][:120]
                        if item.get("caption")
                        else "",
                        "likes": item.get("like_count", 0),
                        "comments": item.get("comments_count", 0),
                        "permalink": item.get("permalink", ""),
                        "insights": insights_map,
                    }
                )
    except Exception as exc:
        logger.warning("Error consultando métricas semanales de IG: %s", exc)

    return {"profile": profile, "posts": posts_in_range}


async def publish_carousel_to_instagram(
    ig_user_id: str,
    access_token: str,
    media_items: list[tuple[str, bool]],
    caption: str,
    *,
    location_id: str = "",
    file_id: str | None = None,
    has_keyword_cta: bool = False,
    keyword: str = "",
    target_link: str = "",
    public_reply_text: str = "",
    dm_message: str = "",
    state_path: Path | str | None = None,
    poll_interval: int = DEFAULT_POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_POLL_TIMEOUT_SECONDS,
) -> str:
    """Publish an Instagram multi-slide CAROUSEL (2-10 image/video items) and record `media_id`."""
    if not (2 <= len(media_items) <= 10):
        raise MetaPublisherError(
            f"Un carrusel de Instagram requiere entre 2 y 10 elementos (recibidos: {len(media_items)})."
        )

    endpoint = f"{GRAPH_API_BASE_URL}/{ig_user_id}/media"
    child_ids: list[str] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        for idx, (media_url, is_video) in enumerate(media_items, start=1):
            child_data: dict[str, Any] = {
                "is_carousel_item": "true",
                "access_token": access_token,
            }
            if is_video:
                child_data["media_type"] = "VIDEO"
                child_data["video_url"] = media_url
            else:
                child_data["image_url"] = media_url

            child_resp = await client.post(endpoint, data=child_data)
            if child_resp.is_error:
                raise MetaPublisherError(
                    f"Error creando item {idx} del carrusel en Instagram: {_extract_meta_error(child_resp)}"
                )
            child_id = str(child_resp.json().get("id", "")).strip()
            if not child_id:
                raise MetaPublisherError(
                    f"Meta no devolvió ID para el item {idx} del carrusel: {child_resp.text}"
                )
            await _wait_for_container_ready(
                client=client,
                creation_id=child_id,
                access_token=access_token,
                poll_interval=poll_interval,
                timeout_seconds=timeout_seconds,
            )
            child_ids.append(child_id)

        parent_data: dict[str, Any] = {
            "media_type": "CAROUSEL",
            "children": ",".join(child_ids),
            "caption": caption,
            "access_token": access_token,
        }
        if location_id:
            parent_data["location_id"] = location_id

        parent_resp = await client.post(endpoint, data=parent_data)
        if parent_resp.is_error:
            err_msg = _extract_meta_error(parent_resp)
            if location_id and "location_id" in parent_data and _is_location_container_error(err_msg):
                logger.warning(
                    "Meta rechazó location_id=%s en carrusel (%s); reintentando sin ubicación.",
                    location_id,
                    err_msg,
                )
                parent_no_loc = {k: v for k, v in parent_data.items() if k != "location_id"}
                parent_resp = await client.post(endpoint, data=parent_no_loc)
            if parent_resp.is_error:
                raise MetaPublisherError(
                    f"Error creando contenedor padre CAROUSEL: {_extract_meta_error(parent_resp)}"
                )
        parent_id = str(parent_resp.json().get("id", "")).strip()
        if not parent_id:
            raise MetaPublisherError(f"Meta no devolvió ID padre CAROUSEL: {parent_resp.text}")

        await _wait_for_container_ready(
            client=client,
            creation_id=parent_id,
            access_token=access_token,
            poll_interval=poll_interval,
            timeout_seconds=timeout_seconds,
        )
        media_id = await _publish_container(
            client=client,
            ig_user_id=ig_user_id,
            creation_id=parent_id,
            access_token=access_token,
        )

    await asyncio.to_thread(
        save_published_media,
        media_id=media_id,
        has_keyword_cta=has_keyword_cta,
        keyword=keyword,
        target_link=target_link,
        public_reply_text=public_reply_text,
        dm_message=dm_message,
        file_id=file_id,
        state_path=state_path,
    )
    return media_id


async def send_instagram_comment_reply(
    access_token: str,
    comment_id: str,
    reply_text: str,
) -> None:
    """Post a public reply to an Instagram comment."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
        resp = await client.post(
            f"{GRAPH_API_BASE_URL}/{comment_id}/replies",
            data={"message": reply_text, "access_token": access_token},
        )
        if resp.is_error:
            raise MetaPublisherError(
                f"Error respondiendo comentario {comment_id}: {_extract_meta_error(resp)}"
            )


async def send_instagram_dm(
    ig_user_id: str,
    access_token: str,
    recipient_id: str,
    message_text: str,
) -> None:
    """Send an Instagram Direct Message reply to `recipient_id` (within 24h messaging window)."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
        resp = await client.post(
            f"{GRAPH_API_BASE_URL}/{ig_user_id}/messages",
            json={
                "recipient": {"id": recipient_id},
                "message": {"text": message_text},
                "access_token": access_token,
            },
        )
        if resp.is_error and resp.status_code in {400, 404}:
            page_creds = await _resolve_linked_page_credentials(client, ig_user_id, access_token)
            if page_creds is not None:
                page_id, page_token = page_creds
                resp = await client.post(
                    f"{GRAPH_API_BASE_URL}/{page_id}/messages",
                    json={
                        "recipient": {"id": recipient_id},
                        "message": {"text": message_text},
                        "access_token": page_token,
                    },
                )
        if resp.is_error:
            raise MetaPublisherError(
                f"Error enviando DM a {recipient_id}: {_extract_meta_error(resp)}"
            )


async def fetch_instagram_user_profile(
    access_token: str,
    user_id: str,
) -> dict[str, str]:
    """Best-effort fetch of `@username` and `name` for an IG user/sender ID."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.get(
                f"{GRAPH_API_BASE_URL}/{user_id}",
                params={"fields": "username,name", "access_token": access_token},
            )
            if resp.is_error:
                return {}
            data = resp.json()
            return {
                "username": str(data.get("username", "")).strip(),
                "name": str(data.get("name", "")).strip(),
            }
    except Exception:
        return {}


async def fetch_instagram_media_context(
    access_token: str,
    media_id: str,
) -> dict[str, str]:
    """Best-effort fetch of caption/permalink for an IG post referenced by a comment."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.get(
                f"{GRAPH_API_BASE_URL}/{media_id}",
                params={"fields": "caption,permalink,media_type", "access_token": access_token},
            )
            if resp.is_error:
                return {}
            data = resp.json()
            return {
                "caption": str(data.get("caption", "")).strip(),
                "permalink": str(data.get("permalink", "")).strip(),
                "media_type": str(data.get("media_type", "")).strip(),
            }
    except Exception:
        return {}


@dataclass(frozen=True, slots=True)
class MetaTokenStatus:
    """Inspection result for Meta access token validity and expiration."""

    is_valid: bool
    is_permanent: bool
    expires_at: int
    days_remaining: int | None
    refreshed: bool
    active_token: str
    message: str


async def check_and_refresh_meta_token(
    access_token: str,
    *,
    app_id: str = "",
    app_secret: str = "",
    state_path: Path | str | None = None,
    warn_days_threshold: int = 7,
    source_env_token: str = "",
) -> MetaTokenStatus:
    """Inspect `META_ACCESS_TOKEN` via `/debug_token`, extend if possible, and report status."""
    debug_auth = f"{app_id}|{app_secret}" if (app_id and app_secret) else access_token
    current_token = access_token.strip()
    env_fp = _token_fingerprint(source_env_token or current_token)

    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        resp = await client.get(
            f"{GRAPH_API_BASE_URL}/debug_token",
            params={"input_token": current_token, "access_token": debug_auth},
        )
        if resp.is_error:
            err_msg = _extract_meta_error(resp)
            if state_path is not None:
                await asyncio.to_thread(clear_meta_auth_state, state_path)
            return MetaTokenStatus(
                is_valid=False,
                is_permanent=False,
                expires_at=0,
                days_remaining=None,
                refreshed=False,
                active_token=current_token,
                message=f"No se pudo validar META_ACCESS_TOKEN ({err_msg}).",
            )

        data = resp.json().get("data") or {}
        is_valid = bool(data.get("is_valid"))
        expires_at = int(data.get("expires_at") or data.get("data_access_expires_at") or 0)

        if not is_valid:
            err_detail = (data.get("error") or {}).get("message") or "Token inválido o expirado"
            if state_path is not None:
                await asyncio.to_thread(clear_meta_auth_state, state_path)
            return MetaTokenStatus(
                is_valid=False,
                is_permanent=False,
                expires_at=expires_at,
                days_remaining=0,
                refreshed=False,
                active_token=current_token,
                message=f"META_ACCESS_TOKEN inválido o expirado: {err_detail}",
            )

        # expires_at == 0 means non-expiring (System User or Page token)
        if expires_at == 0:
            await asyncio.to_thread(
                save_meta_auth_state,
                {
                    "access_token": current_token,
                    "expires_at": 0,
                    "source_env_fp": env_fp,
                    "checked_at": datetime.utcnow().isoformat() + "Z",
                },
                state_path,
            )
            return MetaTokenStatus(
                is_valid=True,
                is_permanent=True,
                expires_at=0,
                days_remaining=None,
                refreshed=False,
                active_token=current_token,
                message="Token permanente activo (sin fecha de expiración).",
            )

        now_ts = int(time.time())
        days_remaining = max(0, (expires_at - now_ts) // 86400)
        refreshed = False

        # Attempt automatic exchange if <= 14 days remaining and app credentials are configured
        if days_remaining <= 14 and app_id and app_secret:
            ex_resp = await client.get(
                f"{GRAPH_API_BASE_URL}/oauth/access_token",
                params={
                    "grant_type": "fb_exchange_token",
                    "client_id": app_id,
                    "client_secret": app_secret,
                    "fb_exchange_token": current_token,
                },
            )
            if not ex_resp.is_error:
                ex_data = ex_resp.json()
                new_tok = str(ex_data.get("access_token", "")).strip()
                expires_in = int(ex_data.get("expires_in") or 0)
                if new_tok:
                    current_token = new_tok
                    if expires_in > 0:
                        expires_at = now_ts + expires_in
                        days_remaining = max(0, expires_in // 86400)
                    refreshed = True

        await asyncio.to_thread(
            save_meta_auth_state,
            {
                "access_token": current_token,
                "expires_at": expires_at,
                "source_env_fp": env_fp,
                "checked_at": datetime.utcnow().isoformat() + "Z",
            },
            state_path,
        )

        if refreshed:
            msg = f"META_ACCESS_TOKEN extendido automáticamente (~{days_remaining} días restantes)."
        elif days_remaining <= warn_days_threshold:
            msg = (
                f"META_ACCESS_TOKEN vence en {days_remaining} día(s). "
                "Renuévalo pronto o genera un Token de Usuario del Sistema permanente en Meta Business Suite."
            )
        else:
            msg = f"META_ACCESS_TOKEN vigente ({days_remaining} días restantes)."

        return MetaTokenStatus(
            is_valid=True,
            is_permanent=False,
            expires_at=expires_at,
            days_remaining=days_remaining,
            refreshed=refreshed,
            active_token=current_token,
            message=msg,
        )


_LOCATION_SEARCH_TERMS: dict[str, tuple[str, ...]] = {
    "Estados Unidos": ("United States", "Estados Unidos"),
    "California": ("California",),
}


async def resolve_instagram_location_id(
    access_token: str,
    location_name: str,
    *,
    override_id: str = "",
    state_path: Path | str | None = None,
) -> str:
    """Resolve and cache a valid Facebook Place Page ID for Instagram `location_id`."""
    if override_id.strip():
        return override_id.strip()

    label = location_name.strip()
    if not label:
        return ""

    cached = await asyncio.to_thread(get_cached_location_id, label, state_path)
    if cached:
        return cached

    queries = _LOCATION_SEARCH_TERMS.get(label, (label,))
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(12.0)) as client:
            for query in queries:
                resp = await client.get(
                    f"{GRAPH_API_BASE_URL}/pages/search",
                    params={
                        "q": query,
                        "fields": "id,name,location",
                        "limit": 10,
                        "access_token": access_token,
                    },
                )
                if resp.is_error:
                    logger.warning(
                        "Meta /pages/search falló para '%s': %s",
                        query,
                        _extract_meta_error(resp),
                    )
                    continue
                candidates = resp.json().get("data") or []
                best_id = ""
                for page in candidates:
                    if not isinstance(page, dict) or not page.get("id"):
                        continue
                    loc = page.get("location")
                    if not isinstance(loc, dict):
                        continue
                    has_coords = (
                        loc.get("latitude") is not None and loc.get("longitude") is not None
                    )
                    if not has_coords:
                        continue
                    page_name = str(page.get("name", "")).strip().lower()
                    if page_name == query.lower() or page_name == label.lower():
                        best_id = str(page["id"]).strip()
                        break
                    if not best_id:
                        best_id = str(page["id"]).strip()
                if best_id:
                    await asyncio.to_thread(
                        save_cached_location_id,
                        label,
                        best_id,
                        state_path,
                    )
                    logger.info(
                        "Ubicación IG '%s' resuelta y guardada (location_id=%s).",
                        label,
                        best_id,
                    )
                    return best_id
    except Exception as exc:
        logger.warning("No se pudo resolver location_id para '%s': %s", label, exc)

    return ""


def _parse_meta_timestamp(raw_ts: str) -> datetime | None:
    """Parse an ISO-8601 Meta Graph timestamp into an aware UTC datetime."""
    cleaned = raw_ts.strip()
    if not cleaned:
        return None
    normalized = cleaned.replace("Z", "+00:00")
    if len(normalized) >= 5 and normalized[-5] in {"+", "-"} and ":" not in normalized[-5:]:
        normalized = f"{normalized[:-2]}:{normalized[-2:]}"
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def _resolve_linked_page_credentials(
    client: httpx.AsyncClient,
    ig_user_id: str,
    access_token: str,
) -> tuple[str, str] | None:
    """Resolve `(page_id, page_access_token)` linked to `ig_user_id` via `/me/accounts`."""
    try:
        resp = await client.get(
            f"{GRAPH_API_BASE_URL}/me/accounts",
            params={
                "fields": "id,name,access_token,instagram_business_account{id}",
                "limit": 25,
                "access_token": access_token,
            },
        )
        if resp.is_error:
            return None
        pages = resp.json().get("data") or []
        fallback_pair: tuple[str, str] | None = None
        for page in pages:
            if not isinstance(page, dict):
                continue
            page_id = str(page.get("id", "")).strip()
            page_token = str(page.get("access_token", "")).strip() or access_token
            if not page_id:
                continue
            ig_acct = page.get("instagram_business_account")
            linked_ig_id = str(ig_acct.get("id", "")).strip() if isinstance(ig_acct, dict) else ""
            if linked_ig_id == ig_user_id:
                return page_id, page_token
            if linked_ig_id and fallback_pair is None:
                fallback_pair = (page_id, page_token)
        return fallback_pair
    except Exception:
        return None


async def fetch_instagram_dm_messages_detailed(
    ig_user_id: str,
    access_token: str,
    *,
    limit_conversations: int = 20,
    since_minutes: int = 4320,
) -> tuple[list[dict[str, str]], int, str | None]:
    """Poll Instagram Conversations API (`platform=instagram`) and return `(messages, threads_count, error_msg)`."""
    cutoff_utc = datetime.now(timezone.utc) - timedelta(minutes=max(1, int(since_minutes)))
    base_params = {
        "platform": "instagram",
        "fields": "id,updated_time,participants,messages.limit(10){id,created_time,from,message}",
        "limit": min(max(int(limit_conversations), 1), 25),
    }
    threads_by_id: dict[str, dict[str, Any]] = {}
    own_sender_ids: set[str] = {ig_user_id}
    any_success = False
    last_error: str | None = None

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            candidates: list[tuple[str, dict[str, Any]]] = [
                (
                    f"{GRAPH_API_BASE_URL}/{ig_user_id}/conversations",
                    {**base_params, "access_token": access_token},
                ),
                (
                    f"{GRAPH_API_BASE_URL}/me/conversations",
                    {**base_params, "access_token": access_token},
                ),
            ]
            page_creds = await _resolve_linked_page_credentials(client, ig_user_id, access_token)
            if page_creds is not None:
                page_id, page_token = page_creds
                own_sender_ids.add(page_id)
                candidates.append(
                    (
                        f"{GRAPH_API_BASE_URL}/{page_id}/conversations",
                        {**base_params, "access_token": page_token},
                    )
                )
                candidates.append(
                    (
                        f"{GRAPH_API_BASE_URL}/{page_id}/conversations",
                        {**base_params, "folder": "other", "access_token": page_token},
                    )
                )

            for endpoint, req_params in candidates:
                resp = await client.get(endpoint, params=req_params)
                if resp.is_error:
                    last_error = _extract_meta_error(resp)
                    continue
                any_success = True
                raw_threads = resp.json().get("data") or []
                if isinstance(raw_threads, list):
                    for t in raw_threads:
                        if isinstance(t, dict) and t.get("id"):
                            threads_by_id.setdefault(str(t["id"]), t)
                # Keep checking page endpoints if primary endpoint returned 0 threads
                if threads_by_id and "folder" not in req_params:
                    break

            if not any_success:
                logger.warning(
                    "Polling Conversations IG no disponible: %s",
                    last_error or "Error desconocido",
                )
                return [], 0, last_error or "Error consultando Conversations API de Meta"
    except Exception as exc:
        logger.warning("Error consultando Conversations API de IG: %s", exc)
        return [], 0, str(exc)

    threads = list(threads_by_id.values())
    collected: list[dict[str, str]] = []
    for thread in threads:
        participants_map: dict[str, str] = {}
        raw_parts = (thread.get("participants") or {}).get("data") or []
        for p in raw_parts:
            if isinstance(p, dict) and p.get("id"):
                participants_map[str(p["id"])] = str(p.get("username") or p.get("name") or "").strip()

        raw_msgs = [
            m for m in ((thread.get("messages") or {}).get("data") or []) if isinstance(m, dict)
        ]
        # Sort newest-first when timestamps are present so we stop at our latest outgoing reply.
        if any(m.get("created_time") for m in raw_msgs):
            raw_msgs.sort(
                key=lambda m: _parse_meta_timestamp(str(m.get("created_time", "")))
                or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )

        for msg in raw_msgs:
            from_obj = msg.get("from") if isinstance(msg.get("from"), dict) else {}
            from_id = str(from_obj.get("id", "")).strip()
            if from_id and from_id in own_sender_ids:
                # Thread already answered by us; older messages before this reply are already handled.
                break

            mid = str(msg.get("id", "")).strip()
            text = str(msg.get("message", "") or "").strip()
            from_username = str(from_obj.get("username") or participants_map.get(from_id, "")).strip()
            created_raw = str(msg.get("created_time", "")).strip()
            if not mid or not from_id:
                continue
            if not text:
                text = "[Mensaje multimedia / sticker / respuesta a historia sin texto]"

            created_dt = _parse_meta_timestamp(created_raw) if created_raw else None
            if created_dt is not None and created_dt < cutoff_utc:
                continue

            collected.append(
                {
                    "event_id": mid,
                    "sender_id": from_id,
                    "sender_username": from_username,
                    "message_text": text,
                    "created_time": created_raw,
                }
            )

    collected.sort(key=lambda item: item.get("created_time", ""))
    return collected, len(threads), None


async def fetch_instagram_dm_messages_with_status(
    ig_user_id: str,
    access_token: str,
    *,
    limit_conversations: int = 20,
    since_minutes: int = 4320,
) -> tuple[list[dict[str, str]], str | None]:
    """Poll Instagram Conversations API (`platform=instagram`) and return `(messages, error_msg)`."""
    messages, _, err = await fetch_instagram_dm_messages_detailed(
        ig_user_id=ig_user_id,
        access_token=access_token,
        limit_conversations=limit_conversations,
        since_minutes=since_minutes,
    )
    return messages, err


async def fetch_recent_instagram_dm_messages(
    ig_user_id: str,
    access_token: str,
    *,
    limit_conversations: int = 20,
    since_minutes: int = 4320,
) -> list[dict[str, str]]:
    """Poll Instagram Conversations API (`platform=instagram`) to catch inbox and Message Requests."""
    messages, _ = await fetch_instagram_dm_messages_with_status(
        ig_user_id=ig_user_id,
        access_token=access_token,
        limit_conversations=limit_conversations,
        since_minutes=since_minutes,
    )
    return messages
