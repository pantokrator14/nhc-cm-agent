"""Telegram Human-in-the-Loop approval bot, multimodal analysis, and publication orchestrator."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import shutil
import subprocess
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import Settings
from drive_service import (
    SPANISH_WEEKDAYS,
    DriveOAuthExpiredError,
    DriveServiceError,
    DriveStorageQuotaError,
    GoogleOAuthStatus,
    TempPublicLink,
    UnpublishedGuidelineItem,
    append_entry_to_monthly_doc,
    append_or_create_scripts_doc,
    check_google_oauth_token_health,
    clear_meta_auth_state,
    collect_previous_weekly_reports,
    compute_month_week_info,
    compute_next_numbering_and_filename,
    create_or_update_google_doc,
    download_drive_file_to_temp,
    ensure_current_month_folder,
    ensure_statistics_week_folder,
    find_unpublished_guidelines,
    format_guideline_entry_lines,
    get_effective_meta_token,
    get_guidelines_text,
    get_monthly_doc_metadata,
    get_monthly_guidelines,
    get_next_location_label,
    get_performance_memory,
    get_scheduled_times_from_state,
    is_comment_processed,
    is_last_weekly_report_of_month,
    list_month_media_files,
    load_scheduled_sessions,
    mark_comment_processed,
    mark_drive_file_published,
    mark_entry_published_in_monthly_doc,
    mark_file_state,
    mark_guideline_header_published,
    parse_guidelines_entries,
    remove_scheduled_session,
    rename_drive_file,
    resolve_metrics_reference_dt,
    revoke_public_permission,
    save_performance_memory,
    save_scheduled_session,
    set_temp_public_link,
    update_entry_in_monthly_doc,
)
from gemini_agent import (
    MAX_INSTAGRAM_HASHTAGS,
    IgReplySuggestions,
    PostProposal,
    RevisionResult,
    TrendPostIdea,
    TrendReport,
    VideoScriptResult,
    fetch_social_trends_brief,
    generate_ig_reply_suggestions,
    generate_post_proposal,
    generate_trend_ideas,
    generate_video_script,
    generate_weekly_metrics_report,
    is_story_filename,
    normalize_hashtags,
    revise_post_proposal,
)
from meta_ads_service import (
    MetaAdsError,
    fetch_meta_ads_weekly_insights,
    list_accessible_ad_accounts,
    promote_instagram_post,
)
from meta_publisher import (
    MetaTokenStatus,
    check_and_refresh_meta_token,
    fetch_instagram_dm_messages_with_status,
    fetch_instagram_media_context,
    fetch_instagram_user_profile,
    fetch_instagram_weekly_metrics,
    fetch_recent_instagram_media,
    publish_carousel_to_instagram,
    publish_to_instagram,
    resolve_instagram_location_id,
    send_instagram_comment_reply,
    send_instagram_dm,
)
from tiktok_publisher import (
    TikTokPublisherError,
    fetch_tiktok_recent_videos,
    publish_video_to_tiktok,
)

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
MAX_MULTIMODAL_DOWNLOAD_BYTES = 500 * 1024 * 1024  # 500 MB


@dataclass(slots=True)
class ProposalSession:
    """In-memory state for a post proposal awaiting human approval or scheduled execution."""

    proposal_id: str
    month_folder_id: str
    file_metadata: dict[str, Any]
    guidelines_text: str
    proposal: PostProposal
    status: str = "pending"  # pending | scheduled | publishing | published | discarded | failed
    message_id: int | None = None
    last_feedback: str = ""
    revision_history: list[str] = field(default_factory=list)
    is_existing_guideline: bool = False
    existing_doc_header: str = ""
    doc_header: str = ""
    technical_note: str = ""
    location_label: str = ""
    admin_messages: dict[int, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class IgInteractionSession:
    """In-memory state for an incoming IG comment or DM awaiting admin approval."""

    interaction_id: str
    event_id: str
    interaction_type: str  # COMMENT | DM
    sender_id: str
    sender_label: str
    message_text: str
    media_id: str = ""
    post_context: str = ""
    suggested_replies: list[str] = field(default_factory=list)
    reasoning: str = ""
    urgency: str = "media"
    status: str = "pending"  # pending | replied | ignored
    admin_messages: dict[int, int] = field(default_factory=dict)
    handled_by: str = ""
    sent_reply_text: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _inspect_video_technical_specs(
    media_path: Path | None,
    file_metadata: dict[str, Any],
) -> str:
    """Check resolution, aspect ratio (9:16), duration, and file size via `ffprobe`."""
    if media_path is None or not media_path.is_file() or not _is_video_metadata(file_metadata):
        return ""
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return ""

    try:
        proc = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-show_entries",
                "format=duration,size",
                "-of",
                "json",
                str(media_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return ""
        data = json.loads(proc.stdout)
        streams = data.get("streams") or []
        fmt = data.get("format") or {}
        width = int(streams[0].get("width", 0)) if streams else 0
        height = int(streams[0].get("height", 0)) if streams else 0
        duration = float(fmt.get("duration") or 0.0)
        size_bytes = int(fmt.get("size") or media_path.stat().st_size)
        size_mb = size_bytes / (1024 * 1024)

        parts: list[str] = []
        warnings: list[str] = []
        if width and height:
            orientation = "vertical" if height >= width else "horizontal"
            parts.append(f"{width}×{height} ({orientation})")
            if width > height:
                warnings.append("conviene 9:16 vertical para Reels/TikTok")
        if duration > 0:
            parts.append(f"{duration:.0f}s")
            if duration > 90:
                warnings.append("supera 90s (límite recomendado Reels)")
            elif duration < 3:
                warnings.append("dura menos de 3s")
        parts.append(f"{size_mb:.1f} MB")
        if size_mb > 250:
            warnings.append("archivo muy pesado (>250 MB)")

        summary = " • ".join(parts)
        if warnings:
            return f"⚠️ {summary} ({'; '.join(warnings)})"
        return f"✅ {summary}"
    except Exception:
        return ""


def _is_carousel_metadata(file_metadata: dict[str, Any]) -> bool:
    return bool(file_metadata.get("is_carousel")) or bool(file_metadata.get("carousel_items"))


def _is_video_metadata(file_metadata: dict[str, Any]) -> bool:
    if _is_carousel_metadata(file_metadata):
        return False
    mime_type = str(file_metadata.get("mimeType", "")).lower()
    name = str(file_metadata.get("name", "")).lower()
    return mime_type.startswith("video/") or name.endswith(".mp4")


def _build_approval_keyboard(
    proposal_id: str,
    show_force_apply: bool = False,
    is_existing_guideline: bool = False,
    show_tiktok_toggle: bool = False,
    publish_to_tiktok: bool = True,
) -> InlineKeyboardMarkup:
    approve_label = (
        "✅ Guardar en Pautas y Programar"
        if is_existing_guideline
        else "✅ Aprobar y Programar"
    )
    rows = [
        [
            InlineKeyboardButton(
                approve_label,
                callback_data=f"approve:{proposal_id}",
            )
        ]
    ]
    if is_existing_guideline:
        rows.append(
            [
                InlineKeyboardButton(
                    "💾 Solo actualizar Doc Pautas",
                    callback_data=f"save_doc_only:{proposal_id}",
                )
            ]
        )
    if show_tiktok_toggle:
        tt_label = (
            "🎵 TikTok (Orgánico): SÍ"
            if publish_to_tiktok
            else "🎵 TikTok (Orgánico): NO"
        )
        rows.append(
            [
                InlineKeyboardButton(
                    tt_label,
                    callback_data=f"toggle_tiktok:{proposal_id}",
                )
            ]
        )
    if show_force_apply:
        rows.append(
            [
                InlineKeyboardButton(
                    "⚡ Aplicar exactamente mi cambio",
                    callback_data=f"force_edit:{proposal_id}",
                )
            ]
        )
    rows.append(
        [
            InlineKeyboardButton(
                "✏️ Pedir Cambio",
                callback_data=f"edit:{proposal_id}",
            ),
            InlineKeyboardButton(
                "❌ Cancelar" if is_existing_guideline else "❌ Descartar",
                callback_data=f"discard:{proposal_id}",
            ),
        ]
    )
    return InlineKeyboardMarkup(rows)


def _format_proposal_html(
    session: ProposalSession,
    header: str = "📸 Nueva propuesta de pauta (archivo nuevo)",
    ai_note: str = "",
    saved_audience_name: str = "mujeres",
    tiktok_configured: bool = False,
) -> str:
    p = session.proposal
    filename = html.escape(str(session.file_metadata.get("name", "Sin nombre")))
    is_carousel = _is_carousel_metadata(session.file_metadata)
    carousel_items = session.file_metadata.get("carousel_items") or []
    is_video = _is_video_metadata(session.file_metadata)
    media_type = html.escape(p.media_type)
    if p.media_type == "STORIES":
        category_label = "Historia IG (24h)"
    else:
        category_label = "Pro Tip (Mar/Jue/Sáb)" if p.post_category == "PRO_TIP" else "Principal (Lun/Mié/Vie)"
    short_title = html.escape(p.short_title) if p.short_title else "N/A"
    content_summary = html.escape(p.content_summary) if p.content_summary else "N/A"
    scheduled_time = html.escape(p.scheduled_time)
    schedule_reasoning = html.escape(p.schedule_reasoning) if p.schedule_reasoning else ""
    trend_alignment = html.escape(p.trend_alignment) if p.trend_alignment else ""

    if p.promote_in_ads:
        total_usd = round(p.ads_daily_budget_usd * p.ads_duration_days, 2)
        ads_badge = (
            f"✅ SÍ promocionar (${p.ads_daily_budget_usd:.2f}/día × "
            f"{p.ads_duration_days} días = ${total_usd:.2f} USD | "
            f"Público: '{saved_audience_name}' + tarjeta de la cuenta)"
        )
    else:
        ads_badge = "⚪ Orgánico (no promocionar)"
    ads_detail = html.escape(p.ads_recommendation) if p.ads_recommendation else ""

    if p.media_type == "STORIES":
        tiktok_badge = "⚪ No aplica (las Historias de Instagram no se publican en TikTok)"
    elif is_carousel:
        tiktok_badge = "⚪ No aplica (TikTok vía API solo admite videos MP4, no carruseles)"
    elif not is_video:
        tiktok_badge = "⚪ No aplica (TikTok vía API solo admite videos MP4)"
    elif not p.publish_to_tiktok:
        tiktok_badge = "⚪ Omitido (solo Instagram)"
    elif tiktok_configured:
        tiktok_badge = "✅ SÍ — Orgánico (sin pauta pagada)"
    else:
        tiktok_badge = "🟡 Listo en orgánico (falta configurar TIKTOK_* en .env)"

    hook_title = html.escape(p.hook_title) if p.hook_title else "N/A"
    keyword_display = html.escape(p.keyword) if p.has_keyword_cta and p.keyword else "Ninguna"
    link_display = html.escape(p.target_link) if p.has_keyword_cta and p.target_link else "N/A"
    reply_display = (
        html.escape(p.public_reply_text)
        if p.has_keyword_cta and p.public_reply_text
        else "N/A"
    )
    caption = html.escape(p.caption)
    hashtag_list = p.hashtags.split()
    hashtags = html.escape(p.hashtags)

    note_block = f"💡 <b>Razonamiento IA:</b>\n<i>{html.escape(ai_note)}</i>\n\n" if ai_note else ""
    doc_line = (
        f"📌 <b>Pauta en Doc:</b> <code>{html.escape(session.existing_doc_header)}</code>\n"
        if session.is_existing_guideline and session.existing_doc_header
        else ""
    )
    carousel_line = (
        f"🖼️ <b>Carrusel:</b> {len(carousel_items)} diapositivas detectadas en carpeta\n"
        if is_carousel and carousel_items
        else ""
    )
    tech_line = (
        f"🎞️ <b>Video:</b> {html.escape(session.technical_note)}\n"
        if session.technical_note
        else ""
    )

    location_line = (
        "📍 <b>Ubicación (IG):</b> California\n"
        if p.media_type != "STORIES"
        else ""
    )

    meta_block = (
        f"<b>{html.escape(header)}</b>\n\n"
        + note_block
        + doc_line
        + f"📁 <b>Archivo:</b> <code>{filename}</code>\n"
        + carousel_line
        + tech_line
        + f"🎥 <b>Contenido analizado:</b> <i>{content_summary}</i>\n"
        f"🏷️ <b>Cadencia / Título:</b> {category_label} — {short_title}\n"
        f"🎬 <b>Formato:</b> {media_type}\n"
        + location_line
        + f"🕒 <b>Fecha y hora sugerida:</b> <code>{scheduled_time}</code>\n"
        + (f"   <i>{schedule_reasoning}</i>\n" if schedule_reasoning else "")
        + (f"📈 <b>Tendencia aplicada:</b> <i>{trend_alignment}</i>\n" if trend_alignment else "")
        + f"🎯 <b>Gancho / Título:</b> {hook_title}\n"
        f"📣 <b>Meta Ads (IG):</b> {html.escape(ads_badge)}\n"
        + (f"   <i>{ads_detail}</i>\n" if ads_detail else "")
        + f"🎵 <b>TikTok (Orgánico):</b> {html.escape(tiktok_badge)}\n"
        f"🔑 <b>Palabra clave (IG):</b> <code>{keyword_display}</code>\n"
        f"🔗 <b>Enlace DM:</b> {link_display}\n"
        f"💬 <b>Respuesta pública:</b> {reply_display}\n\n"
        f"📝 <b>Caption sugerido:</b>\n"
    )
    hashtags_block = (
        f"\n\n🏷️ <b>Hashtags ({len(hashtag_list)}/{MAX_INSTAGRAM_HASHTAGS}):</b>\n{hashtags}"
        if hashtags
        else ""
    )
    text = meta_block + caption + hashtags_block

    if len(text) > TELEGRAM_MAX_MESSAGE_LENGTH:
        max_caption_len = max(
            0,
            TELEGRAM_MAX_MESSAGE_LENGTH - len(meta_block) - len(hashtags_block) - 10,
        )
        text = meta_block + caption[:max_caption_len] + "..." + hashtags_block
    return text


class NelHealthCoachBot:
    """Coordinates multimodal Gemini drafting, Telegram approval, and publication."""

    def __init__(
        self,
        settings: Settings,
        scheduler: AsyncIOScheduler,
        poll_interval_minutes: int = 5,
        scan_callback: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.settings = settings
        self.scheduler = scheduler
        self.poll_interval_minutes = poll_interval_minutes
        self.scan_callback = scan_callback
        self.application: Application | None = None
        self.pending_proposals: dict[str, ProposalSession] = {}
        self.awaiting_feedback_proposal_id: str | None = None
        self.awaiting_feedback_by_chat: dict[int, str] = {}
        self.cached_unpublished_pautas: dict[str, UnpublishedGuidelineItem] = {}
        self.cached_pautas_month_folder_id: str = ""
        self.cached_pautas_guidelines_text: str = ""
        self.notified_overdue_by_day: dict[str, set[str]] = {}
        self.cached_trend_ideas: list[TrendPostIdea] = []
        self.pending_ig_interactions: dict[str, IgInteractionSession] = {}
        self.awaiting_ig_reply_id: str | None = None
        self.awaiting_ig_reply_by_chat: dict[int, str] = {}
        self.awaiting_ig_refine_by_chat: dict[int, str] = {}
        self.notified_meta_token_day: str = ""
        self.last_meta_token_status: MetaTokenStatus | None = None
        self.notified_google_oauth_day: str = ""
        self.last_google_oauth_status: GoogleOAuthStatus | None = None

    def _serialize_session(self, session: ProposalSession) -> dict[str, Any]:
        return {
            "proposal_id": session.proposal_id,
            "month_folder_id": session.month_folder_id,
            "file_metadata": session.file_metadata,
            "guidelines_text": session.guidelines_text,
            "proposal": session.proposal.model_dump(),
            "status": session.status,
            "is_existing_guideline": session.is_existing_guideline,
            "existing_doc_header": session.existing_doc_header,
            "doc_header": session.doc_header,
            "technical_note": session.technical_note,
            "location_label": session.location_label,
        }

    async def _persist_scheduled_session(self, session: ProposalSession) -> None:
        await asyncio.to_thread(
            save_scheduled_session,
            session.proposal_id,
            self._serialize_session(session),
            self.settings.state_file_path,
        )

    async def rehydrate_scheduled_jobs(self) -> None:
        """Restore persisted `APPROVED` jobs from `state.json` after a restart."""
        saved = await asyncio.to_thread(load_scheduled_sessions, self.settings.state_file_path)
        if not saved:
            return

        now_utc = datetime.now(timezone.utc)
        restored_count = 0
        missed_sessions: list[ProposalSession] = []

        for proposal_id, payload in saved.items():
            try:
                proposal = PostProposal.model_validate(payload["proposal"])
                scheduled_dt = datetime.fromisoformat(proposal.scheduled_time)
                if scheduled_dt.tzinfo is None:
                    scheduled_dt = scheduled_dt.replace(tzinfo=self._get_reference_now().tzinfo)

                session = ProposalSession(
                    proposal_id=proposal_id,
                    month_folder_id=str(payload.get("month_folder_id", "")),
                    file_metadata=dict(payload.get("file_metadata", {})),
                    guidelines_text=str(payload.get("guidelines_text", "")),
                    proposal=proposal,
                    status="scheduled",
                    is_existing_guideline=bool(payload.get("is_existing_guideline", False)),
                    existing_doc_header=str(payload.get("existing_doc_header", "")),
                    doc_header=str(payload.get("doc_header", "")),
                    technical_note=str(payload.get("technical_note", "")),
                    location_label="" if proposal.media_type == "STORIES" else "California",
                )
            except Exception as exc:
                logger.warning("No se pudo restaurar sesión programada %s: %s", proposal_id, exc)
                continue

            if scheduled_dt.astimezone(timezone.utc) > now_utc:
                self.pending_proposals[proposal_id] = session
                self.scheduler.add_job(
                    self.execute_publication,
                    trigger=DateTrigger(run_date=scheduled_dt),
                    args=[proposal_id],
                    id=f"publish_{proposal_id}",
                    replace_existing=True,
                    misfire_grace_time=3600,
                )
                restored_count += 1
            else:
                session.status = "pending"
                self.pending_proposals[proposal_id] = session
                missed_sessions.append(session)

        if restored_count:
            logger.info("Se rehidrataron %d publicaciones programadas desde state.json.", restored_count)

        for missed in missed_sessions:
            file_name = str(missed.file_metadata.get("name", missed.proposal.short_title))
            await self._send_admin_message(
                text=self._format_session_html(
                    missed,
                    header=f"⏰ Publicación pendiente tras reinicio ({file_name})",
                    ai_note=(
                        "Su horario programado pasó mientras el agente estaba apagado. "
                        "Elige si deseas publicarla ahora, reprogramar su fecha/hora o cancelarla."
                    ),
                ),
                reply_markup=self._build_session_keyboard(missed),
            )

    def _format_session_html(
        self,
        session: ProposalSession,
        header: str = "📸 Nueva propuesta de pauta (archivo nuevo)",
        ai_note: str = "",
    ) -> str:
        return _format_proposal_html(
            session,
            header=header,
            ai_note=ai_note,
            saved_audience_name=self.settings.meta_saved_audience_name,
            tiktok_configured=self.settings.tiktok_enabled,
        )

    def _build_session_keyboard(
        self,
        session: ProposalSession,
        show_force_apply: bool = False,
    ) -> InlineKeyboardMarkup:
        return _build_approval_keyboard(
            session.proposal_id,
            show_force_apply=show_force_apply,
            is_existing_guideline=session.is_existing_guideline,
            show_tiktok_toggle=(
                _is_video_metadata(session.file_metadata)
                and not _is_carousel_metadata(session.file_metadata)
                and session.proposal.media_type != "STORIES"
            ),
            publish_to_tiktok=session.proposal.publish_to_tiktok,
        )

    def _get_reference_now(self) -> datetime:
        return datetime.now(self.settings.tzinfo)

    def _get_effective_meta_token(self) -> str:
        return get_effective_meta_token(
            self.settings.meta_access_token,
            state_path=self.settings.state_file_path,
        )

    def _is_authorized_admin(self, update: Update) -> bool:
        chat = update.effective_chat
        if chat is None or chat.id not in self.settings.telegram_admin_chat_ids:
            logger.warning(
                "Acceso ignorado de chat_id no autorizado: %s",
                chat.id if chat else "None",
            )
            return False
        return True

    def _resolve_admin_actor_name(self, update: Update) -> str:
        user = update.effective_user
        if user is None:
            return "Admin"
        if user.username:
            return f"@{user.username}"
        return user.full_name or str(user.id)

    async def _broadcast_admin_message(
        self,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
        parse_mode: str | None = ParseMode.HTML,
    ) -> dict[int, int]:
        if self.application is None:
            raise RuntimeError("Telegram Application no inicializada.")
        sent: dict[int, int] = {}
        last_exc: Exception | None = None
        for chat_id in self.settings.telegram_admin_chat_ids:
            try:
                msg = await self.application.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_markup=reply_markup,
                    parse_mode=parse_mode,
                    disable_web_page_preview=True,
                )
                sent[chat_id] = msg.message_id
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "No se pudo enviar mensaje a chat_id=%s: %s",
                    chat_id,
                    exc,
                )
        if not sent and last_exc is not None:
            raise last_exc
        return sent

    async def _broadcast_admin_document(
        self,
        document_bytes: bytes,
        filename: str,
        caption: str = "",
    ) -> bool:
        if self.application is None or not hasattr(self.application.bot, "send_document"):
            return False
        sent_any = False
        for chat_id in self.settings.telegram_admin_chat_ids:
            try:
                bio = BytesIO(document_bytes)
                bio.name = filename
                await self.application.bot.send_document(
                    chat_id=chat_id,
                    document=bio,
                    filename=filename,
                    caption=caption[:1024] if caption else None,
                    parse_mode=ParseMode.HTML if caption else None,
                )
                sent_any = True
            except Exception as exc:
                logger.warning(
                    "No se pudo enviar documento '%s' a chat_id=%s: %s",
                    filename,
                    chat_id,
                    exc,
                )
        return sent_any

    async def _send_admin_message(
        self,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
        parse_mode: str | None = ParseMode.HTML,
    ) -> int:
        sent = await self._broadcast_admin_message(
            text=text,
            reply_markup=reply_markup,
            parse_mode=parse_mode,
        )
        primary_id = self.settings.telegram_admin_chat_id
        return sent.get(primary_id) or next(iter(sent.values()))

    async def _send_session_card(
        self,
        session: ProposalSession,
        *,
        header: str = "📸 Nueva propuesta de pauta (archivo nuevo)",
        ai_note: str = "",
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> int:
        sent_map = await self._broadcast_admin_message(
            text=self._format_session_html(session, header=header, ai_note=ai_note),
            reply_markup=reply_markup,
        )
        session.admin_messages = sent_map
        primary_id = self.settings.telegram_admin_chat_id
        msg_id = sent_map.get(primary_id) or next(iter(sent_map.values()))
        session.message_id = msg_id
        return msg_id

    async def _sync_proposal_cards(
        self,
        session: ProposalSession,
        *,
        header: str,
        ai_note: str = "",
        reply_markup: InlineKeyboardMarkup | None = None,
        fallback_query: Any | None = None,
    ) -> None:
        text = self._format_session_html(session, header=header, ai_note=ai_note)
        edited_any = False
        if self.application is not None and session.admin_messages:
            for chat_id, msg_id in session.admin_messages.items():
                try:
                    await self.application.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=msg_id,
                        text=text,
                        parse_mode=ParseMode.HTML,
                        reply_markup=reply_markup,
                        disable_web_page_preview=True,
                    )
                    edited_any = True
                except Exception as exc:
                    logger.debug(
                        "No se pudo sincronizar propuesta %s en chat_id=%s: %s",
                        session.proposal_id,
                        chat_id,
                        exc,
                    )
        if not edited_any and fallback_query is not None:
            await fallback_query.edit_message_text(
                text=text,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup,
            )

    async def notify_error(self, message: str) -> None:
        """Send an error alert to the Telegram admin chat."""
        try:
            await self._send_admin_message(
                text=f"⚠️ <b>Alerta del Agente</b>\n\n<pre>{html.escape(message)}</pre>"
            )
        except Exception as notify_exc:
            logger.error("No se pudo enviar alerta de error por Telegram: %s", notify_exc)

    async def notify_info(self, html_message: str) -> None:
        """Send an informational notification to the Telegram admin chat."""
        try:
            await self._send_admin_message(text=html_message)
        except Exception as notify_exc:
            logger.error("No se pudo enviar notificación por Telegram: %s", notify_exc)

    def has_active_session_for_file(self, file_id: str) -> bool:
        """Return True if a file already has an active in-flight proposal session."""
        return any(
            s.file_metadata.get("id") == file_id
            and s.status in {"pending", "scheduled", "publishing"}
            for s in self.pending_proposals.values()
        )

    async def _send_long_html_message(
        self,
        html_text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> int:
        """Send HTML text safely splitting across paragraphs if longer than Telegram's 4096-char limit."""
        if len(html_text) <= TELEGRAM_MAX_MESSAGE_LENGTH:
            return await self._send_admin_message(text=html_text, reply_markup=reply_markup)

        chunks: list[str] = []
        current = ""
        for paragraph in html_text.split("\n\n"):
            candidate = f"{current}\n\n{paragraph}" if current else paragraph
            if len(candidate) <= TELEGRAM_MAX_MESSAGE_LENGTH - 20:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                    current = paragraph
                else:
                    # Single paragraph > limit: hard split safely
                    for i in range(0, len(paragraph), TELEGRAM_MAX_MESSAGE_LENGTH - 50):
                        chunks.append(paragraph[i : i + TELEGRAM_MAX_MESSAGE_LENGTH - 50])
                    current = ""
        if current:
            chunks.append(current)

        last_msg_id = 0
        for idx, chunk in enumerate(chunks):
            markup = reply_markup if idx == len(chunks) - 1 else None
            last_msg_id = await self._send_admin_message(text=chunk, reply_markup=markup)
        return last_msg_id

    def _active_scheduler_file_ids(self) -> set[str]:
        active_job_ids = {job.id for job in self.scheduler.get_jobs() if job.id.startswith("publish_")}
        file_ids: set[str] = set()
        for pid, sess in self.pending_proposals.items():
            if f"publish_{pid}" in active_job_ids or sess.status in {"scheduled", "publishing"}:
                fid = str(sess.file_metadata.get("id", ""))
                if fid:
                    file_ids.add(fid)
        return file_ids

    def _build_session_from_guideline(
        self,
        item: UnpublishedGuidelineItem,
        month_folder_id: str,
        guidelines_text: str,
    ) -> ProposalSession:
        entry = item.entry
        drive_file = item.drive_file or {}
        now = self._get_reference_now()
        scheduled_dt = entry.scheduled_dt or (now + timedelta(hours=2))
        if scheduled_dt.tzinfo is None:
            scheduled_dt = scheduled_dt.replace(tzinfo=now.tzinfo)

        file_name = str(drive_file.get("name", entry.header))
        mime_type = str(drive_file.get("mimeType", "")).lower()
        is_video = mime_type.startswith("video/") or file_name.lower().endswith(".mp4")
        is_story = is_story_filename(file_name) or bool(
            re.match(r"^historia\b", entry.header.strip(), re.IGNORECASE)
        )
        media_type = "STORIES" if is_story else ("REELS" if is_video else "IMAGE")
        clean_hashtags = (
            ""
            if is_story
            else normalize_hashtags(entry.hashtags, max_count=MAX_INSTAGRAM_HASHTAGS)
        )

        proposal = PostProposal(
            content_summary=f"Pauta existente del documento: {entry.header}",
            has_presenter_on_camera=(entry.post_category == "MAIN" and is_video),
            post_category=entry.post_category,
            short_title=entry.short_title or entry.header,
            hook_title=entry.hook_title or entry.short_title,
            caption=entry.caption or entry.hook_title or entry.short_title,
            hashtags=clean_hashtags,
            trend_alignment="",
            media_type=media_type,
            scheduled_time=scheduled_dt.isoformat(),
            schedule_reasoning="Fecha/hora tomada del documento de Pautas.",
            promote_in_ads=entry.promote_in_ads,
            ads_daily_budget_usd=entry.ads_daily_budget_usd if entry.promote_in_ads else 0.0,
            ads_duration_days=entry.ads_duration_days if entry.promote_in_ads else 0,
            ads_recommendation=entry.ads_raw or ("Orgánico." if not entry.promote_in_ads else "Meta Ads."),
            has_keyword_cta=entry.has_keyword_cta,
            keyword=entry.keyword,
            target_link=entry.target_link,
            public_reply_text="",
            publish_to_tiktok=is_video and not is_story,
            tiktok_caption=entry.caption or entry.hook_title or entry.short_title,
        )
        proposal_id = uuid.uuid4().hex[:10]
        return ProposalSession(
            proposal_id=proposal_id,
            month_folder_id=month_folder_id,
            file_metadata=dict(drive_file),
            guidelines_text=guidelines_text,
            proposal=proposal,
            is_existing_guideline=True,
            existing_doc_header=entry.header,
        )

    async def propose_pending_files(
        self,
        month_folder_id: str,
        pending_files: list[dict[str, Any]],
        guidelines_text: str,
    ) -> None:
        """Download each unlisted file, analyze it multimodally with Gemini, and send to Telegram."""
        occupied_slots = await asyncio.to_thread(
            get_scheduled_times_from_state,
            self.settings.state_file_path,
        )
        for existing in self.pending_proposals.values():
            if existing.status in {"pending", "scheduled", "publishing"}:
                occupied_slots.append(existing.proposal.scheduled_time)

        trends_brief = ""
        try:
            trends_brief = await asyncio.to_thread(
                fetch_social_trends_brief,
                self._get_reference_now(),
                self.settings.gemini_api_key,
                self.settings.gemini_model,
                False,
            )
        except Exception as trend_exc:
            logger.warning("No se pudo obtener resumen de tendencias para el lote: %s", trend_exc)

        perf_memory = ""
        try:
            mem_data = await asyncio.to_thread(get_performance_memory, self.settings.state_file_path)
            perf_memory = str(mem_data.get("winning_insights", "")).strip()
        except Exception:
            perf_memory = ""

        for file_meta in pending_files:
            file_id = file_meta["id"]
            file_name = str(file_meta.get("name", file_id))

            if self.has_active_session_for_file(file_id):
                logger.debug(
                    "Archivo '%s' (%s) ya tiene una propuesta activa; se omite.",
                    file_name,
                    file_id,
                )
                continue

            temp_media_paths: list[Path] = []
            try:
                is_carousel = _is_carousel_metadata(file_meta)
                if is_carousel:
                    carousel_items = list(file_meta.get("carousel_items") or [])[:5]
                    for item in carousel_items:
                        item_id = str(item.get("id", ""))
                        item_name = str(item.get("name", item_id))
                        item_size = int(item.get("size", 0) or 0)
                        if not item_id or item_size > MAX_MULTIMODAL_DOWNLOAD_BYTES:
                            continue
                        try:
                            slide_path = await asyncio.to_thread(
                                download_drive_file_to_temp,
                                item_id,
                                item_name,
                                self.settings.google_application_credentials,
                            )
                            temp_media_paths.append(slide_path)
                        except Exception as dl_exc:
                            logger.warning(
                                "No se pudo descargar slide '%s' del carrusel '%s' (%s).",
                                item_name,
                                file_name,
                                dl_exc,
                            )
                    technical_note = (
                        f"Carrusel de {len(file_meta.get('carousel_items') or [])} diapositivas"
                    )
                else:
                    file_size = int(file_meta.get("size", 0) or 0)
                    if file_size <= MAX_MULTIMODAL_DOWNLOAD_BYTES:
                        try:
                            temp_path = await asyncio.to_thread(
                                download_drive_file_to_temp,
                                file_id,
                                file_name,
                                self.settings.google_application_credentials,
                            )
                            temp_media_paths.append(temp_path)
                        except Exception as dl_exc:
                            logger.warning(
                                "No se pudo descargar '%s' para análisis multimodal (%s); usando metadatos.",
                                file_name,
                                dl_exc,
                            )
                    else:
                        logger.warning(
                            "Archivo '%s' supera 500 MB (%d bytes); se analizará por metadatos.",
                            file_name,
                            file_size,
                        )

                    technical_note = await asyncio.to_thread(
                        _inspect_video_technical_specs,
                        temp_media_paths[0] if temp_media_paths else None,
                        file_meta,
                    )

                proposal = await asyncio.to_thread(
                    generate_post_proposal,
                    guidelines_text,
                    file_meta,
                    self._get_reference_now(),
                    self.settings.gemini_api_key,
                    self.settings.gemini_model,
                    occupied_slots,
                    temp_media_paths[0] if temp_media_paths else None,
                    trends_brief,
                    perf_memory,
                    temp_media_paths if is_carousel else None,
                )
                occupied_slots.append(proposal.scheduled_time)

                proposal_id = uuid.uuid4().hex[:10]
                next_loc_label = (
                    ""
                    if proposal.media_type == "STORIES"
                    else await asyncio.to_thread(
                        get_next_location_label,
                        advance=False,
                        state_path=self.settings.state_file_path,
                    )
                )
                session = ProposalSession(
                    proposal_id=proposal_id,
                    month_folder_id=month_folder_id,
                    file_metadata=file_meta,
                    guidelines_text=guidelines_text,
                    proposal=proposal,
                    technical_note=technical_note,
                    location_label=next_loc_label,
                )

                await self._send_session_card(
                    session,
                    reply_markup=self._build_session_keyboard(session),
                )
                self.pending_proposals[proposal_id] = session

                await asyncio.to_thread(
                    mark_file_state,
                    file_id,
                    "PENDING_APPROVAL",
                    {
                        "name": file_name,
                        "month_folder_id": month_folder_id,
                        "proposal_id": proposal_id,
                    },
                    self.settings.state_file_path,
                )
                logger.info(
                    "Propuesta multimodal enviada a Telegram para '%s' (proposal_id=%s).",
                    file_name,
                    proposal_id,
                )
            except Exception as exc:
                logger.exception(
                    "Error generando/enviando propuesta para '%s': %s",
                    file_name,
                    exc,
                )
                await self.notify_error(
                    f"Error procesando el archivo '{file_name}': {exc}"
                )
            finally:
                for p_tmp in temp_media_paths:
                    p_tmp.unlink(missing_ok=True)

    async def handle_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle inline buttons for proposals, unpublished guidelines, and trend reports."""
        if not self._is_authorized_admin(update):
            return

        query = update.callback_query
        if query is None or not query.data:
            return
        await query.answer()

        if ":" not in query.data:
            return
        action, target_id = query.data.split(":", 1)

        if action == "trends" and target_id == "refresh":
            await self._run_trends_analysis(reply_target=query.message, force_refresh=True)
            return

        if action == "script_idea":
            await self._handle_script_from_idea(query.message, target_id)
            return

        if action in {
            "ig_reply_send",
            "ig_reply_refine",
            "ig_reply_regen",
            "ig_reply_custom",
            "ig_reply_ignore",
        }:
            await self._handle_ig_reply_callback(update, query, action, target_id)
            return

        if action in {"pauta_open", "pauta_edit", "pauta_schedule", "pauta_mark_pub", "pauta_list", "pauta_refresh"}:
            await self._handle_pauta_callback(query, action, target_id)
            return

        proposal_id = target_id
        session = self.pending_proposals.get(proposal_id)
        if session is None:
            await query.edit_message_reply_markup(reply_markup=None)
            await self._send_admin_message(
                "⚠️ Esta propuesta ya expiró o fue procesada anteriormente."
            )
            return

        if session.status != "pending":
            await query.edit_message_reply_markup(reply_markup=None)
            await self._send_admin_message(
                f"ℹ️ Esta propuesta ya se encuentra en estado: <code>{html.escape(session.status)}</code>."
            )
            return

        file_name = str(session.file_metadata.get("name", session.file_metadata.get("id")))
        chat_id = update.effective_chat.id if update.effective_chat else self.settings.telegram_admin_chat_id
        actor = self._resolve_admin_actor_name(update)

        if action == "toggle_tiktok":
            if _is_video_metadata(session.file_metadata):
                session.proposal.publish_to_tiktok = not session.proposal.publish_to_tiktok
            else:
                session.proposal.publish_to_tiktok = False
            await self._sync_proposal_cards(
                session,
                header="📸 Nueva propuesta de pauta (archivo nuevo)",
                reply_markup=self._build_session_keyboard(session),
                fallback_query=query,
            )
            return

        if action == "discard":
            session.status = "discarded"
            if self.awaiting_feedback_proposal_id == proposal_id:
                self.awaiting_feedback_proposal_id = None
            for k_cid in [k for k, v in self.awaiting_feedback_by_chat.items() if v == proposal_id]:
                self.awaiting_feedback_by_chat.pop(k_cid, None)

            if not session.is_existing_guideline and session.file_metadata.get("id"):
                await asyncio.to_thread(
                    mark_file_state,
                    session.file_metadata["id"],
                    "DISCARDED",
                    {"name": file_name},
                    self.settings.state_file_path,
                )
            await asyncio.to_thread(
                remove_scheduled_session,
                proposal_id,
                self.settings.state_file_path,
            )
            cancel_header = (
                f"❌ Edición cancelada por {actor} ({session.existing_doc_header})"
                if session.is_existing_guideline
                else f"❌ Propuesta descartada por {actor}"
            )
            await self._sync_proposal_cards(
                session,
                header=cancel_header,
                reply_markup=None,
                fallback_query=query,
            )
            logger.info("Propuesta %s (%s) descartada/cancelada por %s.", proposal_id, file_name, actor)
            return

        if action == "edit":
            self.awaiting_feedback_proposal_id = proposal_id
            self.awaiting_feedback_by_chat[chat_id] = proposal_id
            await query.message.reply_text(
                f"✏️ Escribe los cambios que deseas para <b>{html.escape(file_name)}</b>.\n"
                "Analizaré tu solicitud con criterio estratégico (y si insistes, aplicaré exactamente lo que indiques).",
                parse_mode=ParseMode.HTML,
            )
            return

        if action == "force_edit":
            if not session.last_feedback:
                await query.message.reply_text(
                    "ℹ️ No hay un cambio previo pendiente para forzar. Pulsa ✏️ Pedir Cambio."
                )
                return

            await self._sync_proposal_cards(
                session,
                header=f"⏳ {actor} solicitó aplicar exactamente su instrucción...",
                reply_markup=None,
                fallback_query=query,
            )
            try:
                revision: RevisionResult = await asyncio.to_thread(
                    revise_post_proposal,
                    session.guidelines_text,
                    session.file_metadata,
                    session.proposal,
                    session.last_feedback,
                    self._get_reference_now(),
                    self.settings.gemini_api_key,
                    self.settings.gemini_model,
                    True,
                    session.revision_history,
                )
                session.proposal = revision.proposal
                await self._send_session_card(
                    session,
                    header="✅ Cambio aplicado exactamente según tu indicación",
                    ai_note=revision.reasoning_message,
                    reply_markup=self._build_session_keyboard(session, show_force_apply=False),
                )
            except Exception as exc:
                logger.exception("Error forzando cambio en %s: %s", proposal_id, exc)
                await self.notify_error(f"Error aplicando tu cambio en '{file_name}': {exc}")
            return

        if action == "save_doc_only":
            if self.awaiting_feedback_proposal_id == proposal_id:
                self.awaiting_feedback_proposal_id = None
            try:
                scheduled_dt = datetime.fromisoformat(session.proposal.scheduled_time)
            except ValueError as exc:
                await self.notify_error(f"Fecha inválida en pauta {proposal_id}: {exc}")
                return
            if scheduled_dt.tzinfo is None:
                scheduled_dt = scheduled_dt.replace(tzinfo=self._get_reference_now().tzinfo)

            session.status = "saved"
            try:
                doc_header, final_name = await asyncio.to_thread(
                    self._record_approval_in_drive_sync,
                    session,
                    scheduled_dt,
                )
                await query.edit_message_text(
                    text=self._format_session_html(
                        session,
                        header=f"💾 Documento Pautas actualizado ({doc_header or final_name}) — Sin programar envío aún",
                    ),
                    parse_mode=ParseMode.HTML,
                    reply_markup=None,
                )
            except Exception as exc:
                logger.exception("Error actualizando solo Doc Pautas para %s: %s", proposal_id, exc)
                await self.notify_error(f"No se pudo actualizar el documento Pautas: {exc}")
            return

        if action == "approve":
            if self.awaiting_feedback_proposal_id == proposal_id:
                self.awaiting_feedback_proposal_id = None

            try:
                scheduled_dt = datetime.fromisoformat(session.proposal.scheduled_time)
            except ValueError as exc:
                await self.notify_error(
                    f"Fecha programada inválida en propuesta {proposal_id}: {exc}"
                )
                return

            if scheduled_dt.tzinfo is None:
                scheduled_dt = scheduled_dt.replace(tzinfo=self._get_reference_now().tzinfo)

            session.status = "scheduled"
            await query.edit_message_text(
                text=self._format_session_html(
                    session,
                    header="⏳ Aprobando: actualizando Pautas y preparando publicación...",
                ),
                parse_mode=ParseMode.HTML,
                reply_markup=None,
            )

            file_id = session.file_metadata["id"]
            original_name = file_name
            doc_header = ""
            final_name = original_name
            drive_sync_warnings: list[str] = []

            try:
                doc_header, final_name = await asyncio.to_thread(
                    self._record_approval_in_drive_sync,
                    session,
                    scheduled_dt,
                )
                session.file_metadata["name"] = final_name
                if doc_header:
                    session.doc_header = doc_header
            except Exception as sync_exc:
                logger.exception(
                    "Error al actualizar Pautas / Drive para '%s': %s",
                    original_name,
                    sync_exc,
                )
                drive_sync_warnings.append(str(sync_exc))

            now = datetime.now(timezone.utc)
            run_immediately = scheduled_dt.astimezone(timezone.utc) <= now
            run_date = now + timedelta(seconds=1) if run_immediately else scheduled_dt

            await asyncio.to_thread(
                mark_file_state,
                file_id,
                "APPROVED",
                {
                    "name": final_name,
                    "original_name": original_name,
                    "doc_header": doc_header,
                    "scheduled_time": scheduled_dt.isoformat(),
                },
                self.settings.state_file_path,
            )

            await self._persist_scheduled_session(session)

            self.scheduler.add_job(
                self.execute_publication,
                trigger=DateTrigger(run_date=run_date),
                args=[proposal_id],
                id=f"publish_{proposal_id}",
                replace_existing=True,
                misfire_grace_time=3600,
            )

            status_header = (
                f"🚀 Aprobado por {actor} ({doc_header or final_name}) — Publicando inmediatamente"
                if run_immediately
                else f"✅ Aprobado por {actor} ({doc_header or final_name}) — Programado para {scheduled_dt.isoformat()}"
            )
            await self._sync_proposal_cards(
                session,
                header=status_header,
                reply_markup=None,
                fallback_query=query,
            )
            if drive_sync_warnings:
                await self.notify_error(
                    f"La publicación de '{original_name}' fue aprobada y programada, "
                    "pero hubo un problema actualizando Drive/Pautas:\n"
                    + "\n".join(drive_sync_warnings)
                )
            logger.info(
                "Propuesta %s (%s -> %s) aprobada. Ejecución programada para %s (inmediata=%s).",
                proposal_id,
                original_name,
                final_name,
                run_date.isoformat(),
                run_immediately,
            )

    def _record_approval_in_drive_sync(
        self,
        session: ProposalSession,
        scheduled_dt: datetime,
    ) -> tuple[str, str]:
        """Append or update approved proposal in the monthly `Pautas` doc and rename Drive file if new."""
        creds = self.settings.google_application_credentials
        month_folder_id = session.month_folder_id
        file_id = str(session.file_metadata.get("id", ""))
        old_name = str(session.file_metadata.get("name", file_id))
        p = session.proposal

        doc_meta = get_monthly_doc_metadata(month_folder_id, credentials_path=creds)
        latest_guidelines = (
            get_guidelines_text(str(doc_meta["id"]), credentials_path=creds)
            if doc_meta
            else session.guidelines_text
        )
        session.guidelines_text = latest_guidelines

        if session.is_existing_guideline and session.existing_doc_header:
            doc_header = session.existing_doc_header
            new_filename = old_name
        else:
            doc_header, new_filename = compute_next_numbering_and_filename(
                month_folder_id=month_folder_id,
                guidelines_text=latest_guidelines,
                post_category=p.post_category,
                short_title=p.short_title,
                original_filename=old_name,
                credentials_path=creds,
                is_carousel=_is_carousel_metadata(session.file_metadata),
                is_story=(p.media_type == "STORIES"),
            )

        if doc_meta:
            ads_detail = p.ads_recommendation
            if p.promote_in_ads and p.ads_daily_budget_usd > 0:
                ads_detail = (
                    f"${p.ads_daily_budget_usd:.2f}/día × {p.ads_duration_days} días — "
                    f"{p.ads_recommendation}"
                )
            entry_lines = format_guideline_entry_lines(
                doc_header=doc_header,
                scheduled_dt=scheduled_dt,
                promote_in_ads=p.promote_in_ads,
                ads_recommendation=ads_detail,
                hook_title=p.hook_title,
                caption=p.caption,
                hashtags=p.hashtags,
                has_keyword_cta=p.has_keyword_cta,
                keyword=p.keyword,
                target_link=p.target_link,
            )
            if session.is_existing_guideline and session.existing_doc_header:
                update_entry_in_monthly_doc(
                    doc_id=str(doc_meta["id"]),
                    target_header=session.existing_doc_header,
                    new_entry_lines=entry_lines,
                    credentials_path=creds,
                )
            else:
                append_entry_to_monthly_doc(
                    doc_id=str(doc_meta["id"]),
                    entry_lines=entry_lines,
                    credentials_path=creds,
                )

        if session.is_existing_guideline or not file_id or new_filename == old_name:
            final_name = old_name
        else:
            final_name = rename_drive_file(
                file_id=file_id,
                new_name=new_filename,
                credentials_path=creds,
            )
        return doc_header, final_name

    async def handle_admin_text(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Evaluate admin change requests critically, explain alternatives, and obey insistence."""
        if not self._is_authorized_admin(update):
            return

        message = update.effective_message
        if message is None or not message.text:
            return
        chat_id = update.effective_chat.id if update.effective_chat else self.settings.telegram_admin_chat_id

        pending_refine_id = self.awaiting_ig_refine_by_chat.pop(chat_id, None)
        if pending_refine_id:
            ig_session = self.pending_ig_interactions.get(pending_refine_id)
            if ig_session is None or ig_session.status != "pending":
                await message.reply_text(
                    "ℹ️ Esa interacción ya fue respondida o descartada por otro administrador."
                )
                return
            instruction = message.text.strip()
            if not instruction:
                await message.reply_text("⚠️ La instrucción está vacía. Intenta de nuevo.")
                return
            await message.reply_text("🤖 Refinando las opciones con tu indicación...")
            refined: IgReplySuggestions = await asyncio.to_thread(
                generate_ig_reply_suggestions,
                interaction_type="DM" if ig_session.interaction_type == "DM" else "COMMENT",
                sender_label=ig_session.sender_label,
                message_text=ig_session.message_text,
                post_context=ig_session.post_context,
                previous_options=ig_session.suggested_replies,
                admin_instruction=instruction,
                api_key=self.settings.gemini_api_key,
                model=self.settings.gemini_model,
            )
            ig_session.suggested_replies = list(refined.options)
            ig_session.reasoning = refined.intent_summary
            await self._sync_ig_interaction_cards(ig_session)
            await message.reply_text(
                "✅ Opciones actualizadas arriba según tu indicación. Pulsa el botón de la opción que prefieras o vuelve a refinar."
            )
            return

        pending_ig_id = self.awaiting_ig_reply_by_chat.pop(chat_id, None)
        if pending_ig_id is None and self.awaiting_ig_reply_id:
            pending_ig_id = self.awaiting_ig_reply_id
        if pending_ig_id:
            if self.awaiting_ig_reply_id == pending_ig_id:
                self.awaiting_ig_reply_id = None
            ig_session = self.pending_ig_interactions.get(pending_ig_id)
            if ig_session is None or ig_session.status != "pending":
                await message.reply_text(
                    "ℹ️ Esa interacción ya fue respondida o descartada por otro administrador."
                )
                return
            custom_reply = message.text.strip()
            if not custom_reply:
                await message.reply_text("⚠️ El texto de respuesta está vacío. Intenta de nuevo.")
                return
            actor = self._resolve_admin_actor_name(update)
            await self._send_ig_interaction_reply(
                ig_session,
                custom_reply,
                actor_name=actor,
                reply_message=message,
            )
            return

        proposal_id = self.awaiting_feedback_by_chat.pop(chat_id, None) or self.awaiting_feedback_proposal_id
        if not proposal_id or proposal_id not in self.pending_proposals:
            await message.reply_text(
                "ℹ️ No hay ninguna edición pendiente. Usa /check para buscar archivos nuevos, /pautas para ver pautas sin publicar o /tendencias para ideas virales."
            )
            return

        session = self.pending_proposals[proposal_id]
        if session.status != "pending":
            if self.awaiting_feedback_proposal_id == proposal_id:
                self.awaiting_feedback_proposal_id = None
            await message.reply_text(
                f"ℹ️ La propuesta ya cambió de estado ({session.status})."
            )
            return

        feedback = message.text.strip()
        if self.awaiting_feedback_proposal_id == proposal_id:
            self.awaiting_feedback_proposal_id = None
        session.last_feedback = feedback
        session.revision_history.append(feedback)
        await message.reply_text("🧠 Analizando tu solicitud con criterio estratégico...")

        try:
            try:
                latest_guidelines = await asyncio.to_thread(
                    get_monthly_guidelines,
                    session.month_folder_id,
                    self.settings.google_application_credentials,
                )
                session.guidelines_text = latest_guidelines
            except Exception as exc:
                logger.warning(
                    "No se pudieron refrescar pautas del mes en edición; usando caché: %s",
                    exc,
                )

            revision: RevisionResult = await asyncio.to_thread(
                revise_post_proposal,
                session.guidelines_text,
                session.file_metadata,
                session.proposal,
                feedback,
                self._get_reference_now(),
                self.settings.gemini_api_key,
                self.settings.gemini_model,
                False,
                session.revision_history,
            )
            session.proposal = revision.proposal

            await self._sync_proposal_cards(
                session,
                header="🔄 Propuesta anterior reemplazada por revisión",
                reply_markup=None,
            )

            is_alternative = revision.ai_assessment == "ALTERNATIVE_SUGGESTED"
            header = (
                "💡 Alternativa estratégica sugerida"
                if is_alternative
                else "✏️ Propuesta actualizada"
            )
            await self._send_session_card(
                session,
                header=header,
                ai_note=revision.reasoning_message,
                reply_markup=self._build_session_keyboard(
                    session,
                    show_force_apply=is_alternative,
                ),
            )
        except Exception as exc:
            logger.exception("Error reescribiendo propuesta %s: %s", proposal_id, exc)
            self.awaiting_feedback_proposal_id = proposal_id
            self.awaiting_feedback_by_chat[chat_id] = proposal_id
            await self.notify_error(
                f"No se pudo evaluar/reescribir la propuesta: {exc}\n"
                "Puedes enviar nuevamente tus instrucciones de cambio."
            )

    async def execute_publication(self, proposal_id: str) -> None:
        """Run Drive temp link -> IG publish -> Meta Ads (if approved) -> Drive revoke & mark PUBLISHED."""
        session = self.pending_proposals.get(proposal_id)
        if session is None:
            logger.error("No se encontró ProposalSession para proposal_id=%s", proposal_id)
            return

        file_id = session.file_metadata["id"]
        file_name = str(session.file_metadata.get("name", file_id))
        is_carousel = _is_carousel_metadata(session.file_metadata)
        is_video = _is_video_metadata(session.file_metadata)
        proposal = session.proposal
        session.status = "publishing"

        meta_token = await asyncio.to_thread(self._get_effective_meta_token)
        temp_links: list[tuple[str, TempPublicLink]] = []
        media_id: str | None = None

        try:
            logger.info("Iniciando flujo de publicación para '%s' (%s)...", file_name, file_id)
            location_label = ""
            location_id = ""
            if proposal.media_type != "STORIES":
                location_label = await asyncio.to_thread(
                    get_next_location_label,
                    advance=False,
                    state_path=self.settings.state_file_path,
                )
                location_id = await resolve_instagram_location_id(
                    access_token=meta_token,
                    location_name=location_label,
                    override_id=self.settings.meta_location_id_ca,
                    state_path=self.settings.state_file_path,
                )

            if is_carousel:
                carousel_items = list(session.file_metadata.get("carousel_items") or [])
                if len(carousel_items) < 2:
                    raise RuntimeError(
                        f"El carrusel '{file_name}' necesita al menos 2 archivos válidos."
                    )
                carousel_payload: list[tuple[str, bool]] = []
                for slide_meta in carousel_items:
                    slide_id = str(slide_meta.get("id", ""))
                    if not slide_id:
                        continue
                    slide_link = await asyncio.to_thread(
                        set_temp_public_link,
                        slide_id,
                        self.settings.google_application_credentials,
                    )
                    temp_links.append((slide_id, slide_link))
                    carousel_payload.append((slide_link.url, _is_video_metadata(slide_meta)))

                media_id = await publish_carousel_to_instagram(
                    ig_user_id=self.settings.ig_user_id,
                    access_token=meta_token,
                    media_items=carousel_payload,
                    caption=proposal.full_instagram_caption,
                    location_id=location_id,
                    file_id=file_id,
                    has_keyword_cta=proposal.has_keyword_cta,
                    keyword=proposal.keyword,
                    target_link=proposal.target_link,
                    public_reply_text=proposal.public_reply_text,
                    state_path=self.settings.state_file_path,
                )
            else:
                link_info = await asyncio.to_thread(
                    set_temp_public_link,
                    file_id,
                    self.settings.google_application_credentials,
                )
                temp_links.append((file_id, link_info))

                media_id = await publish_to_instagram(
                    ig_user_id=self.settings.ig_user_id,
                    access_token=meta_token,
                    media_url=link_info.url,
                    media_type=proposal.media_type,
                    caption=proposal.full_instagram_caption,
                    is_video=is_video,
                    location_id=location_id,
                    file_id=file_id,
                    has_keyword_cta=proposal.has_keyword_cta,
                    keyword=proposal.keyword,
                    target_link=proposal.target_link,
                    public_reply_text=proposal.public_reply_text,
                    state_path=self.settings.state_file_path,
                )

            if proposal.media_type != "STORIES" and location_label:
                await asyncio.to_thread(
                    get_next_location_label,
                    advance=True,
                    state_path=self.settings.state_file_path,
                )

            ads_summary = "No requerida (orgánico)"
            if proposal.promote_in_ads:
                if self.settings.meta_ad_account_id:
                    try:
                        ad_res = await promote_instagram_post(
                            access_token=meta_token,
                            ig_user_id=self.settings.ig_user_id,
                            ad_account_id=self.settings.meta_ad_account_id,
                            ig_media_id=media_id,
                            post_name=proposal.short_title or file_name,
                            daily_budget_usd=proposal.ads_daily_budget_usd,
                            duration_days=proposal.ads_duration_days,
                            target_link=proposal.target_link,
                            saved_audience_name=self.settings.meta_saved_audience_name,
                        )
                        ads_summary = (
                            f"Activa (Campaign={ad_res.campaign_id}, Ad={ad_res.ad_id}, "
                            f"${ad_res.daily_budget_usd:.2f}/día × {ad_res.duration_days} días | "
                            f"{ad_res.audience_used} | Pago: {ad_res.payment_method_summary})"
                        )
                    except MetaAdsError as ads_exc:
                        logger.error(
                            "Publicación IG exitosa (%s), pero falló promoción Meta Ads: %s",
                            media_id,
                            ads_exc,
                        )
                        ads_summary = f"⚠️ Error creando campaña: {ads_exc}"
                else:
                    try:
                        accounts = await list_accessible_ad_accounts(meta_token)
                        acc_str = ", ".join(
                            f"{a.get('id')} ({a.get('name')})" for a in accounts
                        ) or "ninguna"
                    except Exception:
                        acc_str = "no detectadas"
                    ads_summary = (
                        "⚠️ Recomendada, pero falta META_AD_ACCOUNT_ID en .env "
                        f"(cuentas disponibles: {acc_str})"
                    )

            tiktok_summary = "Omitido (no aplica o desactivado)"
            if is_video and not is_carousel and proposal.publish_to_tiktok:
                if self.settings.tiktok_enabled:
                    temp_tt_path: Path | None = None
                    try:
                        temp_tt_path = await asyncio.to_thread(
                            download_drive_file_to_temp,
                            file_id,
                            file_name,
                            self.settings.google_application_credentials,
                        )
                        tt_res = await publish_video_to_tiktok(
                            video_path=temp_tt_path,
                            caption=proposal.full_tiktok_caption,
                            client_key=self.settings.tiktok_client_key,
                            client_secret=self.settings.tiktok_client_secret,
                            refresh_token=self.settings.tiktok_refresh_token,
                            access_token=self.settings.tiktok_access_token,
                            publish_mode=self.settings.tiktok_publish_mode,
                            privacy_level=self.settings.tiktok_privacy_level,
                            state_path=self.settings.state_file_path,
                        )
                        tiktok_summary = (
                            f"Publicado orgánico (mode={tt_res.mode}, status={tt_res.status}, "
                            f"publish_id={tt_res.publish_id})"
                        )
                    except (TikTokPublisherError, Exception) as tt_exc:
                        logger.error(
                            "Publicación IG exitosa (%s), pero falló subida orgánica a TikTok: %s",
                            media_id,
                            tt_exc,
                        )
                        tiktok_summary = f"⚠️ Error en TikTok (IG sí se publicó): {tt_exc}"
                    finally:
                        if temp_tt_path is not None:
                            temp_tt_path.unlink(missing_ok=True)
                else:
                    tiktok_summary = "Pendiente de configurar TIKTOK_* en .env (solo orgánico)"

            for target_file_id, temp_link in temp_links:
                await asyncio.to_thread(
                    revoke_public_permission,
                    target_file_id,
                    temp_link.permission_id,
                    self.settings.google_application_credentials,
                )
            await asyncio.to_thread(
                mark_drive_file_published,
                file_id,
                self.settings.google_application_credentials,
            )

            session.status = "published"
            await asyncio.to_thread(
                mark_file_state,
                file_id,
                "PUBLISHED",
                {
                    "name": file_name,
                    "ig_media_id": media_id,
                    "published_at": datetime.now(timezone.utc).isoformat(),
                    "ads_summary": ads_summary,
                    "tiktok_summary": tiktok_summary,
                },
                self.settings.state_file_path,
            )
            header_to_mark = session.doc_header or session.existing_doc_header
            if header_to_mark:
                await asyncio.to_thread(
                    mark_guideline_header_published,
                    header_to_mark,
                    session.month_folder_id,
                    file_id,
                    self.settings.state_file_path,
                )
                if session.month_folder_id:
                    try:
                        await asyncio.to_thread(
                            mark_entry_published_in_monthly_doc,
                            session.month_folder_id,
                            header_to_mark,
                            self.settings.google_application_credentials,
                        )
                    except Exception as doc_mark_exc:
                        logger.warning(
                            "Publicación exitosa en IG, pero no se pudo marcar [PUBLICADO] en Pautas (%s): %s",
                            header_to_mark,
                            doc_mark_exc,
                        )
            await asyncio.to_thread(
                remove_scheduled_session,
                proposal_id,
                self.settings.state_file_path,
            )

            dm_summary = (
                f"Activa en webhook (keyword={proposal.keyword})"
                if proposal.has_keyword_cta
                else "No requerida (sin palabra clave)"
            )
            loc_line = (
                f"📍 <b>Ubicación (IG):</b> {html.escape(location_label)}"
                + (" (aplicada)\n" if location_id else " (pendiente de location_id)\n")
                if location_label
                else ""
            )
            await self._send_admin_message(
                "🎉 <b>¡Publicación completada con éxito!</b>\n\n"
                f"📁 <b>Archivo:</b> <code>{html.escape(file_name)}</code>\n"
                f"📸 <b>IG Media ID:</b> <code>{html.escape(media_id)}</code>\n"
                + loc_line
                + f"📣 <b>Meta Ads (IG):</b> {html.escape(ads_summary)}\n"
                f"🎵 <b>TikTok (Orgánico):</b> {html.escape(tiktok_summary)}\n"
                f"🤖 <b>Auto-DM / Replies (IG):</b> {html.escape(dm_summary)}\n"
                "🔒 <b>Drive:</b> Permiso público revocado y marcado como <code>PUBLISHED</code>."
            )
            logger.info("Flujo completo finalizado para '%s' (media_id=%s).", file_name, media_id)

        except Exception as exc:
            logger.exception("Fallo en execute_publication para '%s': %s", file_name, exc)
            session.status = "failed"

            for target_file_id, temp_link in temp_links:
                try:
                    await asyncio.to_thread(
                        revoke_public_permission,
                        target_file_id,
                        temp_link.permission_id,
                        self.settings.google_application_credentials,
                    )
                except Exception as revoke_exc:
                    logger.error(
                        "No se pudo revocar permiso %s tras fallo en '%s': %s",
                        temp_link.permission_id,
                        file_name,
                        revoke_exc,
                    )

            await asyncio.to_thread(
                mark_file_state,
                file_id,
                "FAILED",
                {"name": file_name, "error": str(exc)},
                self.settings.state_file_path,
            )
            await asyncio.to_thread(
                remove_scheduled_session,
                proposal_id,
                self.settings.state_file_path,
            )
            await self.notify_error(
                f"Falló la publicación de '{file_name}' ({file_id}).\n"
                f"Permiso público revocado (si aplicaba). Error:\n{exc}"
            )

    async def _render_pautas_list_and_send(
        self,
        *,
        status_note: str = "",
    ) -> None:
        now = self._get_reference_now()
        year_str = now.strftime("%Y")
        month_str = now.strftime("%m")
        creds = self.settings.google_application_credentials

        resolution = await asyncio.to_thread(
            ensure_current_month_folder,
            self.settings.drive_root_folder_id,
            year_str,
            month_str,
            creds,
            False,
        )
        if not resolution.folder_id:
            await self._send_admin_message(
                f"ℹ️ No existe todavía la carpeta <code>{year_str}/{month_str}</code> en Google Drive."
            )
            return

        month_folder_id = resolution.folder_id
        guidelines_text = await asyncio.to_thread(
            get_monthly_guidelines,
            month_folder_id,
            creds,
        )
        month_files = await asyncio.to_thread(
            list_month_media_files,
            month_folder_id,
            creds,
        )
        recent_ig = await fetch_recent_instagram_media(
            ig_user_id=self.settings.ig_user_id,
            access_token=self._get_effective_meta_token(),
            limit=50,
        )
        unpublished = await asyncio.to_thread(
            find_unpublished_guidelines,
            guidelines_text,
            month_files,
            self.settings.state_file_path,
            recent_ig,
            self._active_scheduler_file_ids(),
            now,
        )

        self.cached_pautas_month_folder_id = month_folder_id
        self.cached_pautas_guidelines_text = guidelines_text
        self.cached_unpublished_pautas = {}

        if not unpublished:
            await self._send_admin_message(
                f"✅ Todas las pautas registradas en el documento de <code>{year_str}/{month_str}</code> "
                "ya aparecen publicadas (o marcadas como publicadas)."
            )
            return

        lines = [
            f"📋 <b>Pautas sin publicar en {year_str}/{month_str} ({len(unpublished)})</b>",
        ]
        if status_note:
            lines.append(f"<i>{html.escape(status_note)}</i>")
        lines.append(
            "Selecciona una pauta para verla, mover su fecha/hora, actualizar su caption o programarla:\n"
        )

        keyboard_rows: list[list[InlineKeyboardButton]] = []
        for idx, item in enumerate(unpublished):
            key = f"p{idx}"
            self.cached_unpublished_pautas[key] = item
            entry = item.entry
            file_label = (
                str(item.drive_file.get("name", ""))
                if item.drive_file
                else "Sin archivo en Drive"
            )
            if item.has_active_job:
                badge = "🕒 En cola"
            elif item.is_overdue:
                badge = "⏰ Atrasada"
            elif item.drive_file is None:
                badge = "⚠️ Sin archivo"
            else:
                badge = "📅 Pendiente"

            when_parts = [p for p in (entry.fecha_raw, entry.horario_raw) if p]
            when_str = " ".join(when_parts) if when_parts else "Sin fecha"
            lines.append(
                f"• <b>{html.escape(entry.header)}</b> [{badge}]\n"
                f"  📁 <code>{html.escape(file_label)}</code> | 🕒 {html.escape(when_str)}"
            )

            btn_icon = "⏰" if item.is_overdue else ("🕒" if item.has_active_job else "📌")
            short_btn = f"{btn_icon} {entry.header}"
            if item.drive_file:
                short_btn += f" ({file_label})"
            keyboard_rows.append(
                [
                    InlineKeyboardButton(
                        short_btn[:60],
                        callback_data=f"pauta_open:{key}",
                    )
                ]
            )

        keyboard_rows.append(
            [InlineKeyboardButton("🔄 Actualizar lista", callback_data="pauta_refresh:list")]
        )
        await self._send_long_html_message(
            "\n".join(lines),
            reply_markup=InlineKeyboardMarkup(keyboard_rows),
        )

    def _format_guideline_detail_html(self, key: str, item: UnpublishedGuidelineItem) -> str:
        entry = item.entry
        file_name = (
            str(item.drive_file.get("name", ""))
            if item.drive_file
            else "⚠️ No encontrado en Drive"
        )
        status_label = (
            "⏰ Atrasada (fecha en Pautas ya pasó)"
            if item.is_overdue
            else ("🕒 En cola del agente" if item.has_active_job else "📅 Pendiente")
        )
        raw_tags = entry.hashtags.split()
        norm_tags = normalize_hashtags(entry.hashtags, max_count=MAX_INSTAGRAM_HASHTAGS)
        tag_warn = (
            f" <i>(⚠️ En el doc hay {len(raw_tags)}; se recortará a {MAX_INSTAGRAM_HASHTAGS})</i>"
            if len(raw_tags) > MAX_INSTAGRAM_HASHTAGS
            else ""
        )
        when_str = " ".join(p for p in (entry.fecha_raw, entry.horario_raw) if p) or "Sin fecha"

        text = (
            f"📌 <b>{html.escape(entry.header)}</b>\n\n"
            f"📊 <b>Estado:</b> {status_label}\n"
            f"📁 <b>Archivo Drive:</b> <code>{html.escape(file_name)}</code>\n"
            f"🕒 <b>Fecha/Hora actual:</b> <code>{html.escape(when_str)}</code>\n"
            f"📣 <b>Meta Ads:</b> {html.escape(entry.ads_raw or 'Orgánico')}\n"
            f"🎯 <b>Gancho / Título:</b> {html.escape(entry.hook_title or 'N/A')}\n\n"
            f"📝 <b>Caption actual:</b>\n{html.escape(entry.caption or 'N/A')}\n\n"
            f"🏷️ <b>Hashtags:</b> {html.escape(norm_tags or entry.hashtags or 'Ninguno')}{tag_warn}"
        )
        if len(text) > TELEGRAM_MAX_MESSAGE_LENGTH:
            text = text[: TELEGRAM_MAX_MESSAGE_LENGTH - 10] + "..."
        return text

    async def _handle_pauta_callback(
        self,
        query: Any,
        action: str,
        target_id: str,
    ) -> None:
        if action in {"pauta_refresh", "pauta_list"}:
            try:
                await self._render_pautas_list_and_send()
            except Exception as exc:
                logger.exception("Error listando pautas sin publicar: %s", exc)
                await self.notify_error(f"Error consultando pautas sin publicar: {exc}")
            return

        item = self.cached_unpublished_pautas.get(target_id)
        if item is None:
            await self._send_admin_message(
                "⚠️ La lista de pautas expiró. Ejecuta /pautas nuevamente para refrescarla."
            )
            return

        entry = item.entry
        if action == "pauta_open":
            rows: list[list[InlineKeyboardButton]] = []
            if item.drive_file is not None:
                rows.append(
                    [
                        InlineKeyboardButton(
                            "✏️ Editar / Mover fecha o Caption",
                            callback_data=f"pauta_edit:{target_id}",
                        )
                    ]
                )
                sched_label = (
                    "🚀 Publicar ahora (con esta pauta)"
                    if item.is_overdue or entry.scheduled_dt is None
                    else "🚀 Programar en su fecha"
                )
                rows.append(
                    [
                        InlineKeyboardButton(
                            sched_label,
                            callback_data=f"pauta_schedule:{target_id}",
                        )
                    ]
                )
            rows.append(
                [
                    InlineKeyboardButton(
                        "☑️ Marcar como ya publicada",
                        callback_data=f"pauta_mark_pub:{target_id}",
                    ),
                    InlineKeyboardButton(
                        "⬅️ Volver a lista",
                        callback_data="pauta_list:back",
                    ),
                ]
            )
            await self._send_admin_message(
                text=self._format_guideline_detail_html(target_id, item),
                reply_markup=InlineKeyboardMarkup(rows),
            )
            return

        if action == "pauta_edit":
            if item.drive_file is None:
                await self._send_admin_message(
                    f"⚠️ No hay archivo multimedia vinculado en Drive para <b>{html.escape(entry.header)}</b>."
                )
                return
            session = self._build_session_from_guideline(
                item=item,
                month_folder_id=self.cached_pautas_month_folder_id,
                guidelines_text=self.cached_pautas_guidelines_text,
            )
            self.pending_proposals[session.proposal_id] = session
            self.awaiting_feedback_proposal_id = session.proposal_id
            file_name = str(item.drive_file.get("name", entry.header))
            await self._send_admin_message(
                f"✏️ Editando <b>{html.escape(entry.header)}</b> (<code>{html.escape(file_name)}</code>).\n\n"
                "Escribe qué deseas cambiar (por ejemplo: <i>'Muévelo para hoy a las 19:00, "
                "actualiza el caption con enfoque en tendencias y deja 5 hashtags'</i>)."
            )
            return

        if action == "pauta_schedule":
            if item.drive_file is None:
                await self._send_admin_message(
                    f"⚠️ No se puede programar <b>{html.escape(entry.header)}</b> porque no se encontró su archivo en Drive."
                )
                return
            session = self._build_session_from_guideline(
                item=item,
                month_folder_id=self.cached_pautas_month_folder_id,
                guidelines_text=self.cached_pautas_guidelines_text,
            )
            self.pending_proposals[session.proposal_id] = session
            scheduled_dt = datetime.fromisoformat(session.proposal.scheduled_time)
            if scheduled_dt.tzinfo is None:
                scheduled_dt = scheduled_dt.replace(tzinfo=self._get_reference_now().tzinfo)
            now_utc = datetime.now(timezone.utc)
            run_immediately = scheduled_dt.astimezone(timezone.utc) <= now_utc
            run_date = now_utc + timedelta(seconds=1) if run_immediately else scheduled_dt
            session.status = "scheduled"

            file_id = str(item.drive_file["id"])
            file_name = str(item.drive_file.get("name", entry.header))
            await asyncio.to_thread(
                mark_file_state,
                file_id,
                "APPROVED",
                {
                    "name": file_name,
                    "doc_header": entry.header,
                    "scheduled_time": scheduled_dt.isoformat(),
                },
                self.settings.state_file_path,
            )
            await self._persist_scheduled_session(session)
            self.scheduler.add_job(
                self.execute_publication,
                trigger=DateTrigger(run_date=run_date),
                args=[session.proposal_id],
                id=f"publish_{session.proposal_id}",
                replace_existing=True,
                misfire_grace_time=3600,
            )
            await self._send_admin_message(
                text=self._format_session_html(
                    session,
                    header=(
                        f"🚀 Publicando ahora ({entry.header})"
                        if run_immediately
                        else f"✅ Programado ({entry.header}) para {scheduled_dt.isoformat()}"
                    ),
                )
            )
            return

        if action == "pauta_mark_pub":
            await asyncio.to_thread(
                mark_guideline_header_published,
                entry.header,
                self.cached_pautas_month_folder_id,
                str(item.drive_file["id"]) if item.drive_file else "",
                self.settings.state_file_path,
            )
            if self.cached_pautas_month_folder_id:
                try:
                    await asyncio.to_thread(
                        mark_entry_published_in_monthly_doc,
                        self.cached_pautas_month_folder_id,
                        entry.header,
                        self.settings.google_application_credentials,
                    )
                except Exception as doc_mark_exc:
                    logger.warning(
                        "No se pudo escribir [PUBLICADO] en Google Docs para '%s': %s",
                        entry.header,
                        doc_mark_exc,
                    )
            if item.drive_file is not None:
                fid = str(item.drive_file["id"])
                fname = str(item.drive_file.get("name", entry.header))
                await asyncio.to_thread(
                    mark_file_state,
                    fid,
                    "PUBLISHED",
                    {"name": fname, "doc_header": entry.header, "manually_published": True},
                    self.settings.state_file_path,
                )
                await asyncio.to_thread(
                    mark_drive_file_published,
                    fid,
                    self.settings.google_application_credentials,
                )
            self.cached_unpublished_pautas.pop(target_id, None)
            await self._send_admin_message(
                f"☑️ Marcada <b>{html.escape(entry.header)}</b> como ya publicada. "
                "Usa /pautas para ver las restantes."
            )

    def _format_trend_report_html(self, report: TrendReport) -> str:
        lines = [
            "📈 <b>Análisis de Tendencias e Ideas Virales (@nelhealthcoach)</b>",
            "",
            f"🌐 <b>Panorama actual en redes:</b>\n{html.escape(report.market_summary)}",
        ]
        if report.viral_formats_to_replicate:
            lines.append("\n🎬 <b>Formatos virales replicables para ganar seguidores:</b>")
            for idx, fmt in enumerate(report.viral_formats_to_replicate, start=1):
                lines.append(f"{idx}. {html.escape(fmt)}")

        if report.new_post_ideas:
            lines.append("\n💡 <b>Nuevas pautas sugeridas (con fuerza orgánica propia):</b>")
            for idx, idea in enumerate(report.new_post_ideas, start=1):
                cat_label = (
                    "Pro Tip (Mar/Jue/Sáb)"
                    if idea.post_category == "PRO_TIP"
                    else "Principal a cámara (Lun/Mié/Vie)"
                )
                tag_count = len(idea.hashtags.split())
                lines.append(
                    f"\n<b>{idx}. {html.escape(idea.title)}</b> — <i>{cat_label}</i>\n"
                    f"🎯 <b>Gancho (3 seg):</b> {html.escape(idea.viral_hook)}\n"
                    f"🔥 <b>Por qué tiene fuerza:</b> {html.escape(idea.why_it_trends)}\n"
                    f"🎥 <b>Guía para grabarlo/replicarlo:</b> {html.escape(idea.recording_guide)}\n"
                    f"📝 <b>Caption borrador:</b> {html.escape(idea.suggested_caption)}\n"
                    f"🏷️ <b>Hashtags ({tag_count}/{MAX_INSTAGRAM_HASHTAGS}):</b> {html.escape(idea.hashtags)}"
                )
        return "\n".join(lines)

    async def _collect_unpublished_guidelines(
        self,
        month_folder_id: str,
        guidelines_text: str,
        now: datetime | None = None,
    ) -> list[UnpublishedGuidelineItem]:
        ref_now = now or self._get_reference_now()
        month_files = await asyncio.to_thread(
            list_month_media_files,
            month_folder_id,
            self.settings.google_application_credentials,
        )
        recent_ig = await fetch_recent_instagram_media(
            ig_user_id=self.settings.ig_user_id,
            access_token=self._get_effective_meta_token(),
            limit=50,
        )
        return await asyncio.to_thread(
            find_unpublished_guidelines,
            guidelines_text,
            month_files,
            self.settings.state_file_path,
            recent_ig,
            self._active_scheduler_file_ids(),
            ref_now,
        )

    async def notify_overdue_pautas_if_needed(
        self,
        month_folder_id: str,
        guidelines_text: str,
    ) -> None:
        """Proactively notify Manuel once per day when overdue guidelines exist in `Pautas`."""
        now = self._get_reference_now()
        day_key = now.strftime("%Y-%m-%d")
        already_notified = self.notified_overdue_by_day.setdefault(day_key, set())

        try:
            unpublished = await self._collect_unpublished_guidelines(
                month_folder_id=month_folder_id,
                guidelines_text=guidelines_text,
                now=now,
            )
        except Exception as exc:
            logger.debug("No se pudo verificar pautas atrasadas en ciclo automático: %s", exc)
            return

        overdue_new = [
            item
            for item in unpublished
            if item.is_overdue
            and not item.has_active_job
            and item.entry.header not in already_notified
        ]
        if not overdue_new:
            return

        for item in overdue_new:
            already_notified.add(item.entry.header)

        lines = [
            f"⏰ <b>Detección automática: {len(overdue_new)} pauta(s) atrasada(s) sin publicar</b>",
            "",
        ]
        for item in overdue_new:
            when_str = " ".join(
                p for p in (item.entry.fecha_raw, item.entry.horario_raw) if p
            ) or "Sin fecha"
            file_label = (
                str(item.drive_file.get("name", ""))
                if item.drive_file
                else "Sin archivo en Drive"
            )
            lines.append(
                f"• <b>{html.escape(item.entry.header)}</b> — <code>{html.escape(file_label)}</code> "
                f"(<i>{html.escape(when_str)}</i>)"
            )
        lines.append("\nPulsa el botón para mover su fecha/hora, editar el caption o publicarla:")

        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "📋 Ver y reprogramar pautas atrasadas",
                        callback_data="pauta_refresh:list",
                    )
                ]
            ]
        )
        await self._send_long_html_message("\n".join(lines), reply_markup=keyboard)

    async def _run_trends_analysis(
        self,
        *,
        reply_target: Any = None,
        force_refresh: bool = False,
    ) -> None:
        if reply_target is not None:
            await reply_target.reply_text(
                "🌐 Analizando tendencias actuales en Instagram Reels, TikTok y Shorts para sugerir nuevas ideas..."
            )
        now = self._get_reference_now()
        year_str = now.strftime("%Y")
        month_str = now.strftime("%m")
        creds = self.settings.google_application_credentials

        guidelines_text = ""
        try:
            resolution = await asyncio.to_thread(
                ensure_current_month_folder,
                self.settings.drive_root_folder_id,
                year_str,
                month_str,
                creds,
                False,
            )
            if resolution.folder_id:
                guidelines_text = await asyncio.to_thread(
                    get_monthly_guidelines,
                    resolution.folder_id,
                    creds,
                )
        except Exception as exc:
            logger.debug("Sin pautas previas para contexto de tendencias (%s/%s): %s", year_str, month_str, exc)

        perf_memory = ""
        try:
            mem_data = await asyncio.to_thread(get_performance_memory, self.settings.state_file_path)
            perf_memory = str(mem_data.get("winning_insights", "")).strip()
        except Exception:
            perf_memory = ""

        try:
            report = await asyncio.to_thread(
                generate_trend_ideas,
                guidelines_text,
                now,
                self.settings.gemini_api_key,
                self.settings.gemini_model,
                force_refresh,
                perf_memory,
            )
            self.cached_trend_ideas = list(report.new_post_ideas)
            report_html = self._format_trend_report_html(report)

            kb_rows: list[list[InlineKeyboardButton]] = []
            script_row: list[InlineKeyboardButton] = []
            for idx, idea in enumerate(self.cached_trend_ideas):
                script_row.append(
                    InlineKeyboardButton(
                        f"📝 Guion {idx + 1}",
                        callback_data=f"script_idea:{idx}",
                    )
                )
                if len(script_row) == 3:
                    kb_rows.append(script_row)
                    script_row = []
            if script_row:
                kb_rows.append(script_row)

            kb_rows.append(
                [
                    InlineKeyboardButton(
                        "🔄 Actualizar sugerencias y tendencias",
                        callback_data="trends:refresh",
                    )
                ]
            )
            await self._send_long_html_message(
                report_html,
                reply_markup=InlineKeyboardMarkup(kb_rows),
            )
        except Exception as exc:
            logger.exception("Error generando reporte de tendencias: %s", exc)
            await self.notify_error(f"Error analizando tendencias de redes sociales: {exc}")

    async def _generate_and_send_script(self, topic_or_idea: str) -> None:
        now = self._get_reference_now()
        year_str = now.strftime("%Y")
        month_str = now.strftime("%m")
        creds = self.settings.google_application_credentials

        resolution = await asyncio.to_thread(
            ensure_current_month_folder,
            self.settings.drive_root_folder_id,
            year_str,
            month_str,
            creds,
            True,
        )
        month_folder_id = resolution.folder_id
        guidelines_text = ""
        if month_folder_id:
            try:
                guidelines_text = await asyncio.to_thread(
                    get_monthly_guidelines,
                    month_folder_id,
                    creds,
                )
            except Exception:
                guidelines_text = ""

        mem_data = await asyncio.to_thread(get_performance_memory, self.settings.state_file_path)
        perf_memory = str(mem_data.get("winning_insights", "")).strip()

        script: VideoScriptResult = await asyncio.to_thread(
            generate_video_script,
            topic_or_idea,
            guidelines_text,
            self.settings.gemini_api_key,
            self.settings.gemini_model,
            perf_memory,
        )

        doc_url = ""
        drive_warning = ""
        if month_folder_id:
            cat_str = "Pro Tip (Mar/Jue/Sáb)" if script.format_type == "PRO_TIP" else "Principal a cámara (Lun/Mié/Vie)"
            script_lines = [
                f"GUION: {script.title}",
                f"Fecha: {now.strftime('%d/%m/%Y')}",
                f"Formato: {cat_str}",
                f"Gancho (0-3s): {script.hook_0_3s}",
                f"Desarrollo (3-45s): {script.body_3_45s}",
                f"Cierre y CTA (45-60s): {script.cta_45_60s}",
                f"Notas de grabación: {script.recording_notes}",
                f"Caption: {script.caption}",
                f"Hashtags: {script.hashtags}",
            ]
            try:
                doc_id, _ = await asyncio.to_thread(
                    append_or_create_scripts_doc,
                    month_folder_id,
                    script_lines,
                    "Guiones",
                    creds,
                )
                doc_url = f"https://docs.google.com/document/d/{doc_id}/edit"
            except DriveServiceError as drive_exc:
                logger.warning("No se pudo guardar guion en Google Docs: %s", drive_exc)
                drive_warning = str(drive_exc)

        tag_count = len(script.hashtags.split())
        msg_lines = [
            f"🎬 <b>Guion listo para grabar: {html.escape(script.title)}</b>",
            f"🏷️ <b>Formato:</b> {'Pro Tip' if script.format_type == 'PRO_TIP' else 'Principal a cámara'}",
            "",
            f"🎯 <b>Gancho (0-3s):</b>\n{html.escape(script.hook_0_3s)}",
            f"🗣️ <b>Desarrollo (3-45s):</b>\n{html.escape(script.body_3_45s)}",
            f"📣 <b>Cierre y CTA (45-60s):</b>\n{html.escape(script.cta_45_60s)}",
            f"🎥 <b>Notas de grabación:</b>\n<i>{html.escape(script.recording_notes)}</i>",
            "",
            f"📝 <b>Caption sugerido:</b>\n{html.escape(script.caption)}",
            f"🏷️ <b>Hashtags ({tag_count}/{MAX_INSTAGRAM_HASHTAGS}):</b>\n{html.escape(script.hashtags)}",
        ]
        if doc_url:
            msg_lines.append(f"\n📄 <a href=\"{doc_url}\">Abrir documento Guiones en Drive ({year_str}/{month_str})</a>")
        elif drive_warning:
            msg_lines.append(
                f"\n⚠️ <b>No se pudo guardar automáticamente en Google Docs 'Guiones':</b>\n"
                f"<i>{html.escape(drive_warning)}</i>"
            )
        await self._send_long_html_message("\n".join(msg_lines))

    async def _handle_script_from_idea(self, reply_target: Any, idx_str: str) -> None:
        try:
            idx = int(idx_str)
            idea = self.cached_trend_ideas[idx]
        except (ValueError, IndexError):
            await self._send_admin_message(
                "⚠️ Esa sugerencia ya expiró. Usa /sugerencias para generar nuevas o /guion <tema>."
            )
            return

        if reply_target is not None:
            await reply_target.reply_text(
                f"📝 Redactando guion listo para grabar sobre: {idea.title}..."
            )
        topic = (
            f"Título: {idea.title}\n"
            f"Categoría: {idea.post_category}\n"
            f"Gancho: {idea.viral_hook}\n"
            f"Guía: {idea.recording_guide}\n"
            f"Enfoque: {idea.why_it_trends}"
        )
        try:
            await self._generate_and_send_script(topic)
        except Exception as exc:
            logger.exception("Error generando guion desde idea %s: %s", idx_str, exc)
            await self.notify_error(f"Error generando guion: {exc}")

    async def generate_and_save_weekly_report(
        self,
        *,
        reference_dt: datetime | None = None,
    ) -> None:
        """Collect IG + Meta Ads + TikTok weekly metrics, save Google Doc in Estadísticas/Semana X, and notify."""
        now = reference_dt or self._get_reference_now()
        year_str = now.strftime("%Y")
        month_str = now.strftime("%m")
        creds = self.settings.google_application_credentials

        week_num, week_start, week_end = compute_month_week_info(now)
        is_month_end = is_last_weekly_report_of_month(now)
        spanish_months = (
            "",
            "enero",
            "febrero",
            "marzo",
            "abril",
            "mayo",
            "junio",
            "julio",
            "agosto",
            "septiembre",
            "octubre",
            "noviembre",
            "diciembre",
        )
        month_label = f"{spanish_months[now.month]} {now.year}"
        period_short = f"del {week_start.strftime('%d/%m')} al {week_end.strftime('%d/%m')}"
        report_title = f"Informe Semanal de métricas semana {week_num} ({period_short})"

        # Ensure at least the previous 7 days are covered in metrics (helpful on Week 1 of a new month)
        rolling_7d_start = (now - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
        metrics_since_dt = min(week_start, rolling_7d_start)
        metrics_until_dt = max(week_end, now)

        resolution = await asyncio.to_thread(
            ensure_current_month_folder,
            self.settings.drive_root_folder_id,
            year_str,
            month_str,
            creds,
            True,
        )
        if not resolution.folder_id:
            raise RuntimeError(f"No se pudo resolver la carpeta {year_str}/{month_str} en Drive.")
        month_folder_id = resolution.folder_id

        _, week_folder_id, _ = await asyncio.to_thread(
            ensure_statistics_week_folder,
            month_folder_id,
            week_num,
            creds,
        )

        previous_reports = await asyncio.to_thread(
            collect_previous_weekly_reports,
            self.settings.drive_root_folder_id,
            month_folder_id,
            year_str,
            month_str,
            week_num,
            creds,
        )

        guidelines_text = ""
        try:
            guidelines_text = await asyncio.to_thread(
                get_monthly_guidelines,
                month_folder_id,
                creds,
            )
        except Exception:
            guidelines_text = ""

        meta_token = self._get_effective_meta_token()
        ig_metrics = await fetch_instagram_weekly_metrics(
            ig_user_id=self.settings.ig_user_id,
            access_token=meta_token,
            since_dt=metrics_since_dt.astimezone(timezone.utc),
            until_dt=metrics_until_dt.astimezone(timezone.utc),
            limit=50,
        )
        meta_ads_metrics = await fetch_meta_ads_weekly_insights(
            access_token=meta_token,
            ad_account_id=self.settings.meta_ad_account_id,
            since_date=metrics_since_dt.strftime("%Y-%m-%d"),
            until_date=metrics_until_dt.strftime("%Y-%m-%d"),
        )
        tiktok_videos = await fetch_tiktok_recent_videos(
            client_key=self.settings.tiktok_client_key,
            client_secret=self.settings.tiktok_client_secret,
            refresh_token=self.settings.tiktok_refresh_token,
            access_token=self.settings.tiktok_access_token,
            state_path=self.settings.state_file_path,
        )

        report = await asyncio.to_thread(
            generate_weekly_metrics_report,
            report_title,
            f"{metrics_since_dt.strftime('%d/%m/%Y')} - {metrics_until_dt.strftime('%d/%m/%Y')} (semana {week_num}: {period_short})",
            ig_metrics,
            meta_ads_metrics,
            tiktok_videos,
            guidelines_text,
            self.settings.gemini_api_key,
            self.settings.gemini_model,
            previous_reports,
            is_month_end,
            month_label,
            week_num == 1,
        )

        await asyncio.to_thread(
            save_performance_memory,
            {
                "last_updated": now.isoformat(),
                "last_week_label": f"{year_str}/{month_str} - Semana {week_num}",
                "winning_insights": report.winning_insights,
            },
            self.settings.state_file_path,
        )

        folder_url = f"https://drive.google.com/drive/folders/{week_folder_id}"
        extra_note = (
            f"\n📅 <i>Incluye Resumen Mensual Consolidado de {month_label}.</i>"
            if is_month_end
            else ""
        )

        try:
            doc_id, created = await asyncio.to_thread(
                create_or_update_google_doc,
                week_folder_id,
                report_title,
                report.report_document_text,
                (
                    "Resumen",
                    "RESUMEN MENSUAL",
                    "Análisis",
                    "Cuenta:",
                    "Publicaciones:",
                    "Meta Ads:",
                    "TikTok:",
                    "Ganadores:",
                    "Recomendaciones:",
                ),
                creds,
            )
        except DriveServiceError as drive_exc:
            logger.warning(
                "Informe semanal generado pero no pudo crearse en Google Docs (%s): %s",
                week_folder_id,
                drive_exc,
            )
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", report_title).strip("_") or "informe_semanal"
            doc_sent = await self._broadcast_admin_document(
                document_bytes=report.report_document_text.encode("utf-8"),
                filename=f"{safe_name}.txt",
                caption=f"📊 <b>{html.escape(report_title)}</b> (respaldo local por error de Drive)",
            )
            quota_note = (
                "La Service Account de Google Drive no tiene cuota para crear archivos nuevos en carpetas personales."
                if isinstance(drive_exc, DriveStorageQuotaError)
                else str(drive_exc)
            )
            msg_parts = [
                f"📊 <b>Informe Semanal de Métricas generado (sin guardar en Google Docs)</b>",
                f"📄 <b>Título:</b> <code>{html.escape(report_title)}</code>",
                f"📁 <b>Carpeta destino:</b> <code>{year_str}/{month_str}/Estadísticas/Semana {week_num}</code>{extra_note}",
                f"⚠️ <b>Motivo en Drive:</b> {html.escape(quota_note)}",
                f"📝 <b>Resumen ejecutivo:</b>\n{html.escape(report.executive_summary)}",
                f"📂 <a href=\"{folder_url}\">Abrir carpeta Semana {week_num}</a>",
                f"ℹ️ <i>{html.escape(str(drive_exc))}</i>",
            ]
            if not doc_sent:
                msg_parts.append("📄 <b>Informe completo:</b>")
                raw_text = report.report_document_text.strip()
                step = 3200
                for start in range(0, len(raw_text), step):
                    chunk_plain = raw_text[start : start + step]
                    msg_parts.append(f"<pre>{html.escape(chunk_plain)}</pre>")
            await self._send_long_html_message("\n\n".join(msg_parts))
            return

        doc_url = f"https://docs.google.com/document/d/{doc_id}/edit"
        verb = "creado" if created else "actualizado"
        await self._send_admin_message(
            f"📊 <b>Informe Semanal de Métricas {verb}</b>\n\n"
            f"📄 <b>Documento:</b> <code>{html.escape(report_title)}</code>\n"
            f"📁 <b>Ubicación:</b> <code>{year_str}/{month_str}/Estadísticas/Semana {week_num}</code>"
            f"{extra_note}\n\n"
            f"🔗 <a href=\"{doc_url}\">Abrir informe en Google Docs</a>\n"
            f"📂 <a href=\"{folder_url}\">Abrir carpeta Semana {week_num}</a>"
        )

    async def cmd_start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not self._is_authorized_admin(update):
            return
        await update.effective_message.reply_text(
            "🤖 <b>Agente @nelhealthcoach activo</b>\n\n"
            f"Reviso la carpeta mensual actual en Google Drive cada <b>{self.poll_interval_minutes} minutos</b> "
            "(Reels, Imágenes y Carruseles) y reenvío comentarios/DMs de Instagram con respuestas asistidas "
            f"desde el webhook (puerto <b>{self.settings.webhook_port}</b>).\n"
            "Comandos disponibles:\n"
            "• /check — Escanear la carpeta del mes, detectar pautas atrasadas y proponer archivos nuevos\n"
            "• /mensajes (o /dms) — Revisar DMs y solicitudes de mensajes pendientes en Instagram\n"
            "• /pautas — Listar pautas del documento aún sin publicar (editar fecha, caption o publicar)\n"
            "• /calendario — Ver agenda y huecos libres de los próximos 14 días\n"
            "• /sugerencias (o /suggest, /tendencias) — Analizar tendencias actuales y sugerir nuevas ideas virales\n"
            "• /guion [tema] — Generar un guion hablado listo para grabar y guardarlo en Drive\n"
            "• /metricas [anterior | semana N | fecha] — Generar informe semanal en Google Docs\n"
            "• /token — Verificar vigencia y renovar META_ACCESS_TOKEN\n"
            "• /id — Ver tu Chat ID personal con este bot (para los 3 administradores)\n"
            "• /status — Ver propuestas pendientes y publicaciones programadas",
            parse_mode=ParseMode.HTML,
        )

    async def cmd_check(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not self._is_authorized_admin(update):
            return
        await update.effective_message.reply_text("🔍 Escaneando carpeta del mes actual en Drive...")
        if self.scan_callback is not None:
            await self.scan_callback()
        else:
            logger.warning("cmd_check invocado sin scan_callback configurado.")

    async def cmd_pautas(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not self._is_authorized_admin(update):
            return
        await update.effective_message.reply_text(
            "📋 Revisando el documento de Pautas, Drive e Instagram para listar pautas sin publicar..."
        )
        try:
            await self._render_pautas_list_and_send()
        except Exception as exc:
            logger.exception("Error ejecutando /pautas: %s", exc)
            await self.notify_error(f"Error al listar pautas sin publicar: {exc}")

    async def cmd_tendencias(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not self._is_authorized_admin(update):
            return
        await self._run_trends_analysis(
            reply_target=update.effective_message,
            force_refresh=True,
        )

    async def cmd_calendario(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not self._is_authorized_admin(update):
            return

        now = self._get_reference_now()
        year_str = now.strftime("%Y")
        month_str = now.strftime("%m")
        creds = self.settings.google_application_credentials

        try:
            resolution = await asyncio.to_thread(
                ensure_current_month_folder,
                self.settings.drive_root_folder_id,
                year_str,
                month_str,
                creds,
                False,
            )
            guidelines_text = ""
            if resolution.folder_id:
                guidelines_text = await asyncio.to_thread(
                    get_monthly_guidelines,
                    resolution.folder_id,
                    creds,
                )
        except Exception as exc:
            await self.notify_error(f"No se pudo leer Pautas para el calendario: {exc}")
            return

        entries = parse_guidelines_entries(guidelines_text, default_tz=now.tzinfo)
        by_date: dict[str, list[str]] = {}
        for entry in entries:
            if entry.scheduled_dt is not None:
                dkey = entry.scheduled_dt.strftime("%Y-%m-%d")
                hora = entry.scheduled_dt.strftime("%H:%M")
                by_date.setdefault(dkey, []).append(f"{entry.header} ({hora})")

        for sess in self.pending_proposals.values():
            if sess.status in {"scheduled", "publishing"}:
                try:
                    sdt = datetime.fromisoformat(sess.proposal.scheduled_time)
                    dkey = sdt.strftime("%Y-%m-%d")
                    label = sess.existing_doc_header or sess.proposal.short_title
                    entry_txt = f"{label} ({sdt.strftime('%H:%M')} - en cola)"
                    if entry_txt not in by_date.get(dkey, []):
                        by_date.setdefault(dkey, []).append(entry_txt)
                except ValueError:
                    pass

        lines = [f"📅 <b>Calendario Editorial (próximos 14 días)</b>", ""]
        for offset in range(14):
            day_dt = now + timedelta(days=offset)
            wd = day_dt.weekday()
            if wd == 6:  # Domingo: día de informe semanal
                lines.append(
                    f"📊 <b>{SPANISH_WEEKDAYS[wd]} {day_dt.strftime('%d/%m')}</b> — <i>Informe semanal de métricas</i>"
                )
                continue

            expected_slot = "MAIN (Reels)" if wd in {0, 2, 4} else "PRO_TIP"
            dkey = day_dt.strftime("%Y-%m-%d")
            items = by_date.get(dkey, [])
            if items:
                joined = ", ".join(html.escape(i) for i in items)
                lines.append(
                    f"✅ <b>{SPANISH_WEEKDAYS[wd]} {day_dt.strftime('%d/%m')}</b> [{expected_slot}]: {joined}"
                )
            else:
                lines.append(
                    f"⬜ <b>{SPANISH_WEEKDAYS[wd]} {day_dt.strftime('%d/%m')}</b> [{expected_slot}]: <i>Hueco libre por pautar</i>"
                )

        await self._send_long_html_message("\n".join(lines))

    async def cmd_guion(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not self._is_authorized_admin(update):
            return

        topic = " ".join(context.args).strip() if context.args else ""
        if not topic:
            await update.effective_message.reply_text(
                "✍️ Indica el tema después del comando, por ejemplo:\n"
                "<code>/guion Beneficios de caminar después de comer</code>\n\n"
                "También puedes usar /sugerencias y pulsar el botón <b>📝 Guion</b> de cualquier idea.",
                parse_mode=ParseMode.HTML,
            )
            return

        await update.effective_message.reply_text(
            f"📝 Generando guion para: <b>{html.escape(topic)}</b>...",
            parse_mode=ParseMode.HTML,
        )
        try:
            await self._generate_and_send_script(topic)
        except Exception as exc:
            logger.exception("Error ejecutando /guion: %s", exc)
            await self.notify_error(f"Error generando guion: {exc}")

    async def cmd_metricas(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not self._is_authorized_admin(update):
            return

        now = self._get_reference_now()
        try:
            target_dt, period_label = resolve_metrics_reference_dt(context.args or [], now)
        except ValueError as exc:
            await update.effective_message.reply_text(
                f"⚠️ {html.escape(str(exc))}\n\n"
                "<b>Ejemplos de uso:</b>\n"
                "• <code>/metricas</code> — semana actual\n"
                "• <code>/metricas anterior</code> — semana pasada\n"
                "• <code>/metricas semana 3</code> — semana 3 del mes actual\n"
                "• <code>/metricas 2026-11-22</code> — semana que contiene esa fecha",
                parse_mode=ParseMode.HTML,
            )
            return

        await update.effective_message.reply_text(
            f"📊 Recopilando métricas de Instagram, Meta Ads y TikTok para <b>{html.escape(period_label)}</b>...",
            parse_mode=ParseMode.HTML,
        )
        try:
            await self.generate_and_save_weekly_report(reference_dt=target_dt)
        except Exception as exc:
            logger.exception("Error ejecutando /metricas: %s", exc)
            await self.notify_error(f"Error generando informe semanal de métricas: {exc}")

    async def cmd_mensajes(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not self._is_authorized_admin(update):
            return

        await update.effective_message.reply_text(
            "📩 Revisando mensajes directos y solicitudes de mensajes en Instagram (últimos 30 días)..."
        )
        try:
            forwarded, total_found, error_msg = await self._poll_instagram_dms_detailed(
                limit_conversations=25,
                since_minutes=43200,
            )
        except Exception as exc:
            logger.exception("Error ejecutando /mensajes: %s", exc)
            await self.notify_error(f"Error consultando mensajes de Instagram: {exc}")
            return

        if error_msg:
            await update.effective_message.reply_text(
                "⚠️ <b>No se pudieron consultar los DMs/Solicitudes desde la API de Meta:</b>\n"
                f"<code>{html.escape(error_msg)}</code>\n\n"
                "<b>Qué revisar en Meta / Instagram:</b>\n"
                "1. En la app de Instagram: <i>Configuración → Mensajes y respuestas a historias → Controles de mensajes → Herramientas conectadas → Permitir acceso a los mensajes</i> (activado).\n"
                "2. En Meta App Dashboard: permiso <code>instagram_manage_messages</code> (o <code>pages_messaging</code>) con <b>Acceso avanzado</b> y app en modo <b>Live</b>.\n"
                "3. Webhook suscrito a <code>comments</code>, <code>messages</code> y <code>standby</code>.",
                parse_mode=ParseMode.HTML,
            )
            return

        pending_in_memory = len(
            [s for s in self.pending_ig_interactions.values() if s.status == "pending"]
        )
        if forwarded > 0:
            await update.effective_message.reply_text(
                f"✅ Se detectaron y enviaron <b>{forwarded}</b> mensaje(s)/solicitud(es) pendiente(s) con sugerencias de respuesta.",
                parse_mode=ParseMode.HTML,
            )
            return

        extra = (
            f" Ya tienes <b>{pending_in_memory}</b> tarjeta(s) pendiente(s) arriba sin responder/ignorar."
            if pending_in_memory > 0
            else ""
        )
        await update.effective_message.reply_text(
            f"✅ Revisión completada: no hay mensajes o solicitudes nuevas sin procesar (hilos sin responder detectados por API: {total_found}).{extra}\n\n"
            "<i>Nota: Meta solo expone solicitudes activas de los últimos 30 días y requiere tener activado 'Permitir acceso a los mensajes' en Instagram.</i>",
            parse_mode=ParseMode.HTML,
        )

    def _format_ig_interaction_html(self, session: IgInteractionSession) -> str:
        kind_badge = (
            "💬 <b>Nuevo Comentario en Instagram</b>"
            if session.interaction_type == "COMMENT"
            else "📩 <b>Nuevo Mensaje Directo (DM) en Instagram</b>"
        )
        lines = [
            kind_badge,
            f"👤 <b>De:</b> <code>{html.escape(session.sender_label)}</code>",
        ]
        if session.reasoning:
            lines.append(f"🎯 <b>Intención:</b> <i>{html.escape(session.reasoning)}</i>")
        if session.post_context:
            lines.append(f"🖼️ <b>Publicación:</b> <i>{html.escape(session.post_context)}</i>")
        lines.extend(
            [
                "",
                "📝 <b>Mensaje exacto:</b>",
                f"<blockquote>{html.escape(session.message_text)}</blockquote>",
            ]
        )
        if session.status == "replied":
            lines.extend(
                [
                    "",
                    f"✅ <b>Respondido por {html.escape(session.handled_by or 'Admin')}:</b>",
                    f"<i>{html.escape(session.sent_reply_text)}</i>",
                ]
            )
            return "\n".join(lines)
        if session.status == "ignored":
            lines.extend(
                [
                    "",
                    f"🚫 <i>Marcado como ignorado por {html.escape(session.handled_by or 'Admin')}.</i>",
                ]
            )
            return "\n".join(lines)

        lines.append("\n🤖 <b>Opciones sugeridas:</b>")
        for idx, opt in enumerate(session.suggested_replies, start=1):
            lines.append(f"<b>{idx}.</b> {html.escape(opt)}")
        return "\n".join(lines)

    def _build_ig_interaction_keyboard(
        self,
        session: IgInteractionSession,
    ) -> InlineKeyboardMarkup:
        rows: list[list[InlineKeyboardButton]] = []
        for idx in range(len(session.suggested_replies)):
            rows.append(
                [
                    InlineKeyboardButton(
                        f"💬 Enviar Opción {idx + 1}",
                        callback_data=f"ig_reply_send:{session.interaction_id}:{idx}",
                    )
                ]
            )
        rows.append(
            [
                InlineKeyboardButton(
                    "✏️ Refinar con IA",
                    callback_data=f"ig_reply_refine:{session.interaction_id}",
                ),
                InlineKeyboardButton(
                    "🔄 Regenerar desde cero",
                    callback_data=f"ig_reply_regen:{session.interaction_id}",
                ),
            ]
        )
        rows.append(
            [
                InlineKeyboardButton(
                    "✍️ Escribir respuesta manual",
                    callback_data=f"ig_reply_custom:{session.interaction_id}",
                ),
                InlineKeyboardButton(
                    "🚫 Ignorar",
                    callback_data=f"ig_reply_ignore:{session.interaction_id}",
                ),
            ]
        )
        return InlineKeyboardMarkup(rows)

    async def _sync_ig_interaction_cards(self, session: IgInteractionSession) -> None:
        if self.application is None:
            return
        text = self._format_ig_interaction_html(session)
        markup = (
            self._build_ig_interaction_keyboard(session)
            if session.status == "pending"
            else None
        )
        for chat_id, msg_id in session.admin_messages.items():
            try:
                await self.application.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg_id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=markup,
                    disable_web_page_preview=True,
                )
            except Exception as exc:
                logger.warning(
                    "No se pudo actualizar tarjeta IG (%s) en chat_id=%s: %s",
                    session.interaction_id,
                    chat_id,
                    exc,
                )

    async def handle_incoming_ig_interaction(
        self,
        *,
        interaction_type: str,
        event_id: str,
        sender_id: str,
        sender_username: str,
        message_text: str,
        media_id: str = "",
    ) -> None:
        """Notify admins about an incoming non-keyword IG comment or DM with AI reply drafts."""
        if not event_id or not message_text.strip():
            return
        already_done = await asyncio.to_thread(
            is_comment_processed,
            event_id,
            self.settings.state_file_path,
        )
        if already_done:
            return

        for existing in self.pending_ig_interactions.values():
            if existing.event_id == event_id:
                return

        meta_token = await asyncio.to_thread(self._get_effective_meta_token)
        sender_label = (
            f"@{sender_username.lstrip('@')}" if sender_username.strip() else ""
        )
        if not sender_label and sender_id:
            profile = await fetch_instagram_user_profile(meta_token, sender_id)
            if profile.get("username"):
                sender_label = f"@{profile['username'].lstrip('@')}"
            elif profile.get("name"):
                sender_label = f"{profile['name']} (ID {sender_id})"
        if not sender_label:
            sender_label = f"Usuario {sender_id or 'desconocido'}"

        post_context = ""
        normalized_type = "DM" if interaction_type.upper() == "DM" else "COMMENT"
        if normalized_type == "COMMENT" and media_id:
            media_ctx = await fetch_instagram_media_context(meta_token, media_id)
            parts = []
            if media_ctx.get("media_type"):
                parts.append(media_ctx["media_type"])
            if media_ctx.get("caption"):
                caption_snippet = media_ctx["caption"].replace("\n", " ")[:140]
                parts.append(f'"{caption_snippet}"')
            if media_ctx.get("permalink"):
                parts.append(media_ctx["permalink"])
            post_context = " — ".join(parts)

        suggestions: IgReplySuggestions = await asyncio.to_thread(
            generate_ig_reply_suggestions,
            interaction_type=normalized_type,
            sender_label=sender_label,
            message_text=message_text.strip(),
            post_context=post_context,
            api_key=self.settings.gemini_api_key,
            model=self.settings.gemini_model,
        )

        interaction_id = uuid.uuid4().hex[:10]
        session = IgInteractionSession(
            interaction_id=interaction_id,
            event_id=event_id,
            interaction_type=normalized_type,
            sender_id=sender_id,
            sender_label=sender_label,
            message_text=message_text.strip(),
            media_id=media_id,
            post_context=post_context,
            suggested_replies=list(suggestions.options),
            reasoning=suggestions.intent_summary,
        )
        self.pending_ig_interactions[interaction_id] = session
        sent_map = await self._broadcast_admin_message(
            self._format_ig_interaction_html(session),
            reply_markup=self._build_ig_interaction_keyboard(session),
        )
        session.admin_messages = sent_map
        logger.info(
            "Interacción IG (%s) de %s enviada a %d chat(s) admin (interaction_id=%s).",
            normalized_type,
            sender_label,
            len(sent_map),
            interaction_id,
        )

    async def _send_ig_interaction_reply(
        self,
        session: IgInteractionSession,
        reply_text: str,
        *,
        actor_name: str,
        reply_message: Any | None = None,
    ) -> None:
        meta_token = await asyncio.to_thread(self._get_effective_meta_token)
        try:
            if session.interaction_type == "COMMENT":
                await send_instagram_comment_reply(
                    access_token=meta_token,
                    comment_id=session.event_id,
                    reply_text=reply_text,
                )
            else:
                await send_instagram_dm(
                    ig_user_id=self.settings.ig_user_id,
                    access_token=meta_token,
                    recipient_id=session.sender_id,
                    message_text=reply_text,
                )
        except Exception as exc:
            logger.error(
                "Error enviando respuesta IG (%s, %s): %s",
                session.interaction_type,
                session.event_id,
                exc,
            )
            err_msg = f"❌ No se pudo enviar la respuesta a Instagram: {exc}"
            if reply_message is not None:
                await reply_message.reply_text(err_msg)
            else:
                await self.notify_error(err_msg)
            return

        await asyncio.to_thread(
            mark_comment_processed,
            session.event_id,
            session.media_id or session.interaction_type,
            session.sender_label,
            session.interaction_type == "DM",
            session.interaction_type == "COMMENT",
            self.settings.state_file_path,
        )
        session.status = "replied"
        session.handled_by = actor_name
        session.sent_reply_text = reply_text
        await self._sync_ig_interaction_cards(session)
        if reply_message is not None:
            await reply_message.reply_text(
                f"✅ Respuesta enviada en Instagram por {actor_name}."
            )

    async def _handle_ig_reply_callback(
        self,
        update: Update,
        query: Any,
        action: str,
        target_id: str,
    ) -> None:
        actor = self._resolve_admin_actor_name(update)
        if action == "ig_reply_send":
            if ":" not in target_id:
                return
            interaction_id, idx_raw = target_id.split(":", 1)
            session = self.pending_ig_interactions.get(interaction_id)
            if session is None:
                await query.message.reply_text("ℹ️ Esta interacción ya expiró de memoria.")
                return
            if session.status != "pending":
                await query.message.reply_text(
                    f"ℹ️ Ya fue gestionada por {session.handled_by or 'otro administrador'}."
                )
                return
            try:
                opt_idx = int(idx_raw)
                chosen = session.suggested_replies[opt_idx]
            except (ValueError, IndexError):
                await query.message.reply_text("⚠️ Opción de respuesta inválida.")
                return
            await self._send_ig_interaction_reply(session, chosen, actor_name=actor)
            return

        session = self.pending_ig_interactions.get(target_id)
        if session is None:
            await query.message.reply_text("ℹ️ Esta interacción ya expiró de memoria.")
            return
        if session.status != "pending":
            await query.message.reply_text(
                f"ℹ️ Ya fue gestionada por {session.handled_by or 'otro administrador'}."
            )
            return

        if action == "ig_reply_refine":
            chat_id = (
                update.effective_chat.id
                if update.effective_chat
                else self.settings.telegram_admin_chat_id
            )
            self.awaiting_ig_reply_by_chat.pop(chat_id, None)
            self.awaiting_ig_refine_by_chat[chat_id] = target_id
            await query.message.reply_text(
                f"✏️ Escribe cómo quieres ajustar las opciones para <b>{html.escape(session.sender_label)}</b> "
                "(por ejemplo: <i>'toma la opción 2, hazla más corta y menciona que atendemos de lunes a viernes'</i>):",
                parse_mode=ParseMode.HTML,
            )
            return

        if action == "ig_reply_regen":
            regenerated: IgReplySuggestions = await asyncio.to_thread(
                generate_ig_reply_suggestions,
                interaction_type="DM" if session.interaction_type == "DM" else "COMMENT",
                sender_label=session.sender_label,
                message_text=session.message_text,
                post_context=session.post_context,
                previous_options=session.suggested_replies,
                regenerate=True,
                api_key=self.settings.gemini_api_key,
                model=self.settings.gemini_model,
            )
            session.suggested_replies = list(regenerated.options)
            session.reasoning = regenerated.intent_summary
            await self._sync_ig_interaction_cards(session)
            return

        if action == "ig_reply_custom":
            chat_id = (
                update.effective_chat.id
                if update.effective_chat
                else self.settings.telegram_admin_chat_id
            )
            self.awaiting_ig_refine_by_chat.pop(chat_id, None)
            self.awaiting_ig_reply_id = target_id
            self.awaiting_ig_reply_by_chat[chat_id] = target_id
            await query.message.reply_text(
                f"✍️ Escribe ahora tu respuesta manual exacta para <b>{html.escape(session.sender_label)}</b> "
                "(el próximo mensaje que envíes se publicará directamente en Instagram):",
                parse_mode=ParseMode.HTML,
            )
            return

        if action == "ig_reply_ignore":
            await asyncio.to_thread(
                mark_comment_processed,
                session.event_id,
                session.media_id or session.interaction_type,
                session.sender_label,
                False,
                False,
                self.settings.state_file_path,
            )
            session.status = "ignored"
            session.handled_by = actor
            if self.awaiting_ig_reply_id == target_id:
                self.awaiting_ig_reply_id = None
            for k_cid in [k for k, v in self.awaiting_ig_reply_by_chat.items() if v == target_id]:
                self.awaiting_ig_reply_by_chat.pop(k_cid, None)
            for k_cid in [k for k, v in self.awaiting_ig_refine_by_chat.items() if v == target_id]:
                self.awaiting_ig_refine_by_chat.pop(k_cid, None)
            await self._sync_ig_interaction_cards(session)

    async def _poll_instagram_dms_detailed(
        self,
        *,
        limit_conversations: int = 20,
        since_minutes: int = 4320,
    ) -> tuple[int, int, str | None]:
        """Poll Instagram Conversations API and return `(forwarded, total_found, error_msg)`."""
        meta_token = await asyncio.to_thread(self._get_effective_meta_token)
        recent_dms, error_msg = await fetch_instagram_dm_messages_with_status(
            ig_user_id=self.settings.ig_user_id,
            access_token=meta_token,
            limit_conversations=limit_conversations,
            since_minutes=since_minutes,
        )
        if error_msg:
            return 0, 0, error_msg

        forwarded = 0
        for dm in recent_dms:
            event_id = dm.get("event_id", "")
            if not event_id:
                continue
            already_done = await asyncio.to_thread(
                is_comment_processed,
                event_id,
                self.settings.state_file_path,
            )
            if already_done or any(
                s.event_id == event_id for s in self.pending_ig_interactions.values()
            ):
                continue
            await self.handle_incoming_ig_interaction(
                interaction_type="DM",
                event_id=event_id,
                sender_id=dm.get("sender_id", ""),
                sender_username=dm.get("sender_username", ""),
                message_text=dm.get("message_text", ""),
                media_id="",
            )
            forwarded += 1
        return forwarded, len(recent_dms), None

    async def poll_instagram_dms(self) -> int:
        """Poll Instagram Conversations API to catch DMs and Message Requests not delivered by webhook."""
        forwarded, _, _ = await self._poll_instagram_dms_detailed()
        return forwarded

    async def check_meta_token_health(self, *, notify_always: bool = False) -> MetaTokenStatus:
        """Inspect META_ACCESS_TOKEN expiry, auto-extend when possible, and alert admins if needed."""
        current_token = await asyncio.to_thread(self._get_effective_meta_token)
        env_token = self.settings.meta_access_token.strip()
        status = await check_and_refresh_meta_token(
            access_token=current_token,
            app_id=self.settings.meta_app_id,
            app_secret=self.settings.meta_app_secret,
            state_path=self.settings.state_file_path,
            warn_days_threshold=7,
            source_env_token=env_token,
        )
        if not status.is_valid and env_token and current_token.strip() != env_token:
            await asyncio.to_thread(clear_meta_auth_state, self.settings.state_file_path)
            status = await check_and_refresh_meta_token(
                access_token=env_token,
                app_id=self.settings.meta_app_id,
                app_secret=self.settings.meta_app_secret,
                state_path=self.settings.state_file_path,
                warn_days_threshold=7,
                source_env_token=env_token,
            )
        self.last_meta_token_status = status
        today_key = self._get_reference_now().strftime("%Y-%m-%d")

        expires_iso = (
            datetime.fromtimestamp(status.expires_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            if status.expires_at > 0
            else "N/A"
        )
        needs_warning = (
            not status.is_permanent
            and status.days_remaining is not None
            and status.days_remaining <= 7
        )

        if status.refreshed:
            await self._send_admin_message(
                "🔄 <b>META_ACCESS_TOKEN renovado automáticamente</b>\n\n"
                f"• Nuevo vencimiento estimado: <b>{status.days_remaining} días</b> "
                f"(<code>{html.escape(expires_iso)}</code>)\n"
                "• Guardado automáticamente en <code>state.json</code>."
            )
            self.notified_meta_token_day = today_key
            return status

        should_alert = (
            not status.is_valid or needs_warning or notify_always
        )
        if should_alert and (notify_always or self.notified_meta_token_day != today_key):
            if not status.is_valid:
                header = "🚨 <b>ALERTA: META_ACCESS_TOKEN inválido o vencido</b>"
            elif needs_warning:
                header = (
                    f"⚠️ <b>ALERTA PREVENTIVA: META_ACCESS_TOKEN vence en {status.days_remaining} día(s)</b>"
                )
            else:
                header = "🔐 <b>Estado de META_ACCESS_TOKEN</b>"

            days_text = (
                "No expira (System User Token permanente)"
                if status.is_permanent
                else (
                    f"{status.days_remaining} días restantes ({expires_iso})"
                    if status.days_remaining is not None
                    else "Sin fecha reportada por Meta"
                )
            )
            hint_lines: list[str] = []
            if not status.is_permanent and not (
                self.settings.meta_app_id and self.settings.meta_app_secret
            ):
                hint_lines.append(
                    "💡 Para que pueda auto-extenderlo hasta ~60 días por código, agrega "
                    "<code>META_APP_ID</code> y <code>META_APP_SECRET</code> en <code>.env</code> "
                    "(o usa un System User Token permanente de Meta Business Manager)."
                )
            msg = (
                f"{header}\n\n"
                f"• <b>Validez:</b> {'Válido ✅' if status.is_valid else 'Inválido/Vencido ❌'}\n"
                f"• <b>Vigencia:</b> {html.escape(days_text)}\n"
                + (
                    f"• <b>Detalle:</b> {html.escape(status.message)}\n"
                    if status.message
                    else ""
                )
                + ("\n" + "\n".join(hint_lines) if hint_lines else "")
            )
            await self._send_admin_message(msg)
            self.notified_meta_token_day = today_key
        return status

    async def check_google_oauth_health(self, *, notify_always: bool = False) -> GoogleOAuthStatus:
        """Check Google OAuth user token validity/age and remind admins before 7-day Testing expiry."""
        status = await asyncio.to_thread(
            check_google_oauth_token_health,
            self.settings.google_application_credentials,
            self.settings.google_oauth_credentials,
            5.0,
        )
        self.last_google_oauth_status = status
        today_key = self._get_reference_now().strftime("%Y-%m-%d")

        if not status.enabled:
            if notify_always:
                await self._send_admin_message(
                    "📁 <b>Google Drive Auth:</b> usando <code>Service Account</code> "
                    "(sin <code>GOOGLE_OAUTH_CREDENTIALS</code> configurado)."
                )
            return status

        should_alert = (not status.is_valid) or status.needs_renewal or notify_always
        if should_alert and (notify_always or self.notified_google_oauth_day != today_key):
            if not status.is_valid:
                header = "🚨 <b>ALERTA: Token OAuth de Google Drive expirado/inválido</b>"
            elif status.needs_renewal:
                rem = status.days_until_7d_expiry if status.days_until_7d_expiry is not None else "?"
                header = f"⚠️ <b>RECORDATORIO: Renovar OAuth de Google Drive (quedan ~{rem} día(s) del límite de 7 días en modo Prueba)</b>"
            else:
                header = "📁 <b>Estado de Google Drive OAuth</b>"

            age_str = f"{status.age_days:.1f} días" if status.age_days is not None else "N/D"
            rem_str = (
                f"~{status.days_until_7d_expiry} día(s) (si la app sigue en modo Prueba)"
                if status.days_until_7d_expiry is not None
                else "N/D"
            )
            steps = (
                "\n\n<b>Pasos para renovar (2 minutos):</b>\n"
                "1. En tu PC ejecuta:\n"
                "   <code>.venv/bin/python scripts/google_oauth_setup.py</code>\n"
                "2. Abre la URL y autoriza con <code>ceo@nelhealthcoach.com</code>.\n"
                "3. Sube el archivo nuevo al servidor DigitalOcean:\n"
                "   <code>scp credentials/oauth_token.json root@TU_IP:/root/nhc-cm-agent/credentials/oauth_token.json</code>\n"
                "4. Reinicia el servicio en el servidor:\n"
                "   <code>ssh root@TU_IP \"systemctl restart nel-agent\"</code>"
            )
            msg = (
                f"{header}\n\n"
                f"• <b>Estado:</b> {'Válido ✅' if status.is_valid else 'Expirado/Inválido ❌'}\n"
                f"• <b>Antigüedad:</b> {html.escape(age_str)}\n"
                f"• <b>Vigencia estimada (Testing):</b> {html.escape(rem_str)}\n"
                f"• <b>Detalle:</b> {html.escape(status.message)}"
                f"{steps if (not status.is_valid or status.needs_renewal or notify_always) else ''}"
            )
            await self._send_admin_message(msg)
            self.notified_google_oauth_day = today_key
        return status

    async def cmd_id(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        chat = update.effective_chat
        if chat is None or update.effective_message is None:
            return
        authorized = chat.id in self.settings.telegram_admin_chat_ids
        status_note = (
            "✅ Ya estás autorizado en este bot."
            if authorized
            else "ℹ️ Envía este número al administrador para agregarlo a <code>TELEGRAM_EXTRA_ADMIN_CHAT_IDS</code> en <code>.env</code>."
        )
        await update.effective_message.reply_text(
            f"🆔 Tu <b>Chat ID</b> directo con el bot es: <code>{chat.id}</code>\n{status_note}",
            parse_mode=ParseMode.HTML,
        )

    async def cmd_token(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not self._is_authorized_admin(update):
            return

        await update.effective_message.reply_text("🔐 Verificando estado de META_ACCESS_TOKEN y Google Drive OAuth...")
        try:
            await self.check_meta_token_health(notify_always=True)
            await self.check_google_oauth_health(notify_always=True)
        except Exception as exc:
            logger.exception("Error ejecutando /token: %s", exc)
            await self.notify_error(f"Error consultando estado de credenciales: {exc}")

    async def cmd_status(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not self._is_authorized_admin(update):
            return

        pending = [s for s in self.pending_proposals.values() if s.status == "pending"]
        scheduled_jobs = [
            job for job in self.scheduler.get_jobs() if job.id.startswith("publish_")
        ]

        token_line = "• Meta Token: usa /token para verificar vencimiento"
        if self.last_meta_token_status is not None:
            st = self.last_meta_token_status
            if not st.is_valid:
                token_line = "• Meta Token: <b>❌ Inválido / Vencido</b> (usa /token)"
            elif st.is_permanent:
                token_line = "• Meta Token: <b>✅ Permanente (System User)</b>"
            elif st.days_remaining is not None:
                token_line = f"• Meta Token: <b>✅ Válido ({st.days_remaining} días restantes)</b>"

        lines = [
            "<b>📊 Estado del Agente</b>",
            f"• Propuestas esperando aprobación: <b>{len(pending)}</b>",
            f"• Comentarios/DMs pendientes: <b>{len([s for s in self.pending_ig_interactions.values() if s.status == 'pending'])}</b>",
            f"• Publicaciones programadas: <b>{len(scheduled_jobs)}</b>",
            token_line,
            f"• Administradores configurados: <b>{len(self.settings.telegram_admin_chat_ids)} chat(s)</b>",
            f"• Meta Ads público guardado: <code>{html.escape(self.settings.meta_saved_audience_name)}</code>",
            (
                f"• TikTok (@nelhealtcoachoficial, Orgánico): <b>Configurado ({html.escape(self.settings.tiktok_publish_mode)})</b>"
                if self.settings.tiktok_enabled
                else "• TikTok (@nelhealtcoachoficial, Orgánico): <i>Sin credenciales TIKTOK_* en .env</i>"
            ),
        ]
        for job in scheduled_jobs:
            run_time = (
                job.next_run_time.astimezone(self.settings.tzinfo).strftime("%Y-%m-%d %H:%M (%Z)")
                if job.next_run_time
                else "N/A"
            )
            lines.append(
                f"  - <code>{html.escape(job.id)}</code> → <code>{html.escape(run_time)}</code>"
            )

        await update.effective_message.reply_text(
            "\n".join(lines),
            parse_mode=ParseMode.HTML,
        )


def register_handlers(application: Application, bot_controller: NelHealthCoachBot) -> None:
    """Attach command, callback, and text handlers to the Telegram Application."""
    bot_controller.application = application
    application.add_handler(CommandHandler(["start", "help"], bot_controller.cmd_start))
    application.add_handler(CommandHandler("check", bot_controller.cmd_check))
    application.add_handler(CommandHandler(["mensajes", "dms", "inbox"], bot_controller.cmd_mensajes))
    application.add_handler(CommandHandler(["pautas", "pendientes"], bot_controller.cmd_pautas))
    application.add_handler(CommandHandler("calendario", bot_controller.cmd_calendario))
    application.add_handler(
        CommandHandler(["tendencias", "sugerencias", "suggest"], bot_controller.cmd_tendencias)
    )
    application.add_handler(CommandHandler("guion", bot_controller.cmd_guion))
    application.add_handler(CommandHandler(["metricas", "informe"], bot_controller.cmd_metricas))
    application.add_handler(CommandHandler("token", bot_controller.cmd_token))
    application.add_handler(CommandHandler("id", bot_controller.cmd_id))
    application.add_handler(CommandHandler("status", bot_controller.cmd_status))
    application.add_handler(CallbackQueryHandler(bot_controller.handle_callback))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, bot_controller.handle_admin_text)
    )
