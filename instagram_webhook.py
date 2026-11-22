"""Native FastAPI Instagram Webhook server for automated keyword DM & public replies."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from config import Settings
from drive_service import (
    get_effective_meta_token,
    get_published_media,
    is_comment_processed,
    mark_comment_processed,
)
from meta_publisher import GRAPH_API_BASE_URL, _extract_meta_error

logger = logging.getLogger(__name__)


def _keyword_matches(keyword: str, comment_text: str) -> bool:
    normalized_keyword = keyword.strip().upper()
    if not normalized_keyword:
        return False

    normalized_text = comment_text.upper()
    if re.fullmatch(r"[\wáéíóúüñÁÉÍÓÚÜÑ]+", normalized_keyword):
        pattern = rf"(?<!\w){re.escape(normalized_keyword)}(?!\w)"
        return re.search(pattern, normalized_text) is not None
    return normalized_keyword in normalized_text


async def _send_private_dm(
    client: httpx.AsyncClient,
    ig_user_id: str,
    access_token: str,
    comment_id: str,
    message_text: str,
) -> bool:
    endpoint = f"{GRAPH_API_BASE_URL}/{ig_user_id}/messages"
    payload = {
        "recipient": {"comment_id": comment_id},
        "message": {"text": message_text},
        "access_token": access_token,
    }
    response = await client.post(endpoint, json=payload)
    if response.is_error:
        logger.error(
            "Error enviando DM automático para comment_id=%s: %s",
            comment_id,
            _extract_meta_error(response),
        )
        return False

    logger.info("DM automático enviado con éxito para comment_id=%s.", comment_id)
    return True


async def _send_public_reply(
    client: httpx.AsyncClient,
    access_token: str,
    comment_id: str,
    reply_text: str,
) -> bool:
    endpoint = f"{GRAPH_API_BASE_URL}/{comment_id}/replies"
    response = await client.post(
        endpoint,
        data={
            "message": reply_text,
            "access_token": access_token,
        },
    )
    if response.is_error:
        logger.error(
            "Error respondiendo públicamente comment_id=%s: %s",
            comment_id,
            _extract_meta_error(response),
        )
        return False

    logger.info("Respuesta pública publicada para comment_id=%s.", comment_id)
    return True


async def _process_comment_event(
    settings: Settings,
    comment_id: str,
    media_id: str,
    comment_text: str,
    username: str,
    from_id: str = "",
    interaction_callback: Callable[..., Awaitable[None]] | None = None,
) -> None:
    try:
        already_processed = await asyncio.to_thread(
            is_comment_processed,
            comment_id,
            settings.state_file_path,
        )
        if already_processed:
            logger.debug("comment_id=%s ya fue procesado; se omite duplicado.", comment_id)
            return

        active_token = await asyncio.to_thread(
            get_effective_meta_token,
            settings.meta_access_token,
            settings.state_file_path,
        )

        media_record = await asyncio.to_thread(
            get_published_media,
            media_id,
            settings.state_file_path,
        )
        keyword_matched = False
        if media_record and media_record.get("has_keyword_cta"):
            keyword = str(media_record.get("keyword", "")).strip().upper()
            keyword_matched = _keyword_matches(keyword, comment_text)

        if not keyword_matched:
            if interaction_callback is not None and comment_text.strip():
                await interaction_callback(
                    interaction_type="COMMENT",
                    event_id=comment_id,
                    sender_id=from_id,
                    sender_username=username,
                    message_text=comment_text,
                    media_id=media_id,
                )
            return

        keyword = str(media_record.get("keyword", "")).strip().upper()
        target_link = str(media_record.get("target_link", "")).strip()
        public_reply_text = str(media_record.get("public_reply_text", "")).strip()

        logger.info(
            "Palabra clave '%s' detectada en comment_id=%s (media_id=%s, user=@%s).",
            keyword,
            comment_id,
            media_id,
            username or "desconocido",
        )

        dm_sent = False
        reply_sent = False

        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
            if target_link:
                greeting = f"¡Hola @{username}! 💚" if username else "¡Hola! 💚"
                dm_text = (
                    f"{greeting} Gracias por comentar {keyword}.\n\n"
                    f"Aquí tienes el enlace que solicitaste:\n{target_link}"
                )
                dm_sent = await _send_private_dm(
                    client=client,
                    ig_user_id=settings.ig_user_id,
                    access_token=active_token,
                    comment_id=comment_id,
                    message_text=dm_text,
                )
            else:
                logger.warning(
                    "media_id=%s tiene keyword='%s' pero target_link vacío; se omite DM.",
                    media_id,
                    keyword,
                )

            if public_reply_text:
                reply_sent = await _send_public_reply(
                    client=client,
                    access_token=active_token,
                    comment_id=comment_id,
                    reply_text=public_reply_text,
                )

        if dm_sent or reply_sent:
            await asyncio.to_thread(
                mark_comment_processed,
                comment_id=comment_id,
                media_id=media_id,
                username=username,
                dm_sent=dm_sent,
                reply_sent=reply_sent,
                state_path=settings.state_file_path,
            )
    except Exception as exc:
        logger.exception(
            "Error inesperado procesando comentario comment_id=%s (media_id=%s): %s",
            comment_id,
            media_id,
            exc,
        )


async def _process_dm_event(
    settings: Settings,
    message_id: str,
    sender_id: str,
    message_text: str,
    interaction_callback: Callable[..., Awaitable[None]] | None = None,
) -> None:
    try:
        if interaction_callback is None or not message_text.strip():
            return
        already_processed = await asyncio.to_thread(
            is_comment_processed,
            message_id,
            settings.state_file_path,
        )
        if already_processed:
            logger.debug("DM mid=%s ya fue procesado; se omite duplicado.", message_id)
            return

        await interaction_callback(
            interaction_type="DM",
            event_id=message_id,
            sender_id=sender_id,
            sender_username="",
            message_text=message_text,
            media_id="",
        )
    except Exception as exc:
        logger.exception("Error inesperado procesando DM mid=%s: %s", message_id, exc)


def create_webhook_app(
    settings: Settings,
    interaction_callback: Callable[..., Awaitable[None]] | None = None,
) -> FastAPI:
    """Create the FastAPI application for Meta Instagram Webhook verification and events."""
    app = FastAPI(
        title="Nel Health Coach Instagram Webhook",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
    )

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/webhook/instagram")
    async def verify_instagram_webhook(request: Request) -> PlainTextResponse:
        params = request.query_params
        mode = params.get("hub.mode")
        token = params.get("hub.verify_token")
        challenge = params.get("hub.challenge")

        if mode == "subscribe" and token == settings.webhook_verify_token and challenge is not None:
            logger.info("Webhook de Instagram verificado correctamente por Meta.")
            return PlainTextResponse(content=challenge, status_code=200)

        logger.warning(
            "Intento fallido de verificación de webhook (mode=%s).",
            mode,
        )
        raise HTTPException(status_code=403, detail="Webhook verification failed")

    @app.post("/webhook/instagram")
    async def handle_instagram_webhook(
        request: Request,
        background_tasks: BackgroundTasks,
    ) -> dict[str, str]:
        try:
            payload: dict[str, Any] = await request.json()
        except Exception as exc:
            logger.warning("Payload JSON inválido recibido en webhook: %s", exc)
            raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

        entries = payload.get("entry")
        if not isinstance(entries, list):
            return {"status": "ignored"}

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            # 1. Comment events
            changes = entry.get("changes")
            if isinstance(changes, list):
                for change in changes:
                    if not isinstance(change, dict) or change.get("field") != "comments":
                        continue

                    value = change.get("value")
                    if not isinstance(value, dict):
                        continue

                    comment_id = str(value.get("id", "")).strip()
                    comment_text = str(value.get("text", ""))
                    media_obj = value.get("media") if isinstance(value.get("media"), dict) else {}
                    media_id = str(media_obj.get("id", "")).strip()
                    from_obj = value.get("from") if isinstance(value.get("from"), dict) else {}
                    from_id = str(from_obj.get("id", "")).strip()
                    username = str(from_obj.get("username", "")).strip()

                    if not comment_id or not media_id:
                        continue

                    if from_id and from_id == settings.ig_user_id:
                        logger.debug("Comentario propio (%s) ignorado.", comment_id)
                        continue

                    background_tasks.add_task(
                        _process_comment_event,
                        settings,
                        comment_id,
                        media_id,
                        comment_text,
                        username,
                        from_id,
                        interaction_callback,
                    )

            # 2. Direct Message events (`entry[].messaging[]` and `entry[].standby[]`)
            for dm_bucket_key in ("messaging", "standby"):
                dm_events = entry.get(dm_bucket_key)
                if not isinstance(dm_events, list):
                    continue
                for event in dm_events:
                    if not isinstance(event, dict):
                        continue
                    msg_obj = event.get("message")
                    if not isinstance(msg_obj, dict) or msg_obj.get("is_echo"):
                        continue
                    sender_obj = event.get("sender") if isinstance(event.get("sender"), dict) else {}
                    sender_id = str(sender_obj.get("id", "")).strip()
                    mid = str(msg_obj.get("mid", "")).strip()
                    msg_text = str(msg_obj.get("text", ""))
                    if not sender_id or not mid or not msg_text.strip():
                        continue
                    if sender_id == settings.ig_user_id:
                        continue

                    background_tasks.add_task(
                        _process_dm_event,
                        settings,
                        mid,
                        sender_id,
                        msg_text,
                        interaction_callback,
                    )

        return {"status": "ok"}

    return app
