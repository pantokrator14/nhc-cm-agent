"""Telegram Human-in-the-Loop approval bot, multimodal analysis, and publication orchestrator."""

from __future__ import annotations

import asyncio
import html
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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
    TempPublicLink,
    append_entry_to_monthly_doc,
    compute_next_numbering_and_filename,
    download_drive_file_to_temp,
    format_guideline_entry_lines,
    get_guidelines_text,
    get_monthly_doc_metadata,
    get_monthly_guidelines,
    get_scheduled_times_from_state,
    mark_drive_file_published,
    mark_file_state,
    rename_drive_file,
    revoke_public_permission,
    set_temp_public_link,
)
from gemini_agent import (
    PostProposal,
    RevisionResult,
    generate_post_proposal,
    revise_post_proposal,
)
from meta_ads_service import MetaAdsError, list_accessible_ad_accounts, promote_instagram_post
from meta_publisher import publish_to_instagram

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
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _build_approval_keyboard(
    proposal_id: str,
    show_force_apply: bool = False,
) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                "✅ Aprobar y Programar",
                callback_data=f"approve:{proposal_id}",
            )
        ]
    ]
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
                "❌ Descartar",
                callback_data=f"discard:{proposal_id}",
            ),
        ]
    )
    return InlineKeyboardMarkup(rows)


def _format_proposal_html(
    session: ProposalSession,
    header: str = "📸 Nueva propuesta de pauta (archivo nuevo)",
    ai_note: str = "",
) -> str:
    p = session.proposal
    filename = html.escape(str(session.file_metadata.get("name", "Sin nombre")))
    media_type = html.escape(p.media_type)
    category_label = "Pro Tip (Mar/Jue/Sáb)" if p.post_category == "PRO_TIP" else "Principal (Lun/Mié/Vie)"
    short_title = html.escape(p.short_title) if p.short_title else "N/A"
    content_summary = html.escape(p.content_summary) if p.content_summary else "N/A"
    scheduled_time = html.escape(p.scheduled_time)
    schedule_reasoning = html.escape(p.schedule_reasoning) if p.schedule_reasoning else ""

    if p.promote_in_ads:
        total_usd = round(p.ads_daily_budget_usd * p.ads_duration_days, 2)
        ads_badge = (
            f"✅ SÍ promocionar (${p.ads_daily_budget_usd:.2f}/día × "
            f"{p.ads_duration_days} días = ${total_usd:.2f} USD)"
        )
    else:
        ads_badge = "⚪ Orgánico (no promocionar)"
    ads_detail = html.escape(p.ads_recommendation) if p.ads_recommendation else ""

    hook_title = html.escape(p.hook_title) if p.hook_title else "N/A"
    keyword_display = html.escape(p.keyword) if p.has_keyword_cta and p.keyword else "Ninguna"
    link_display = html.escape(p.target_link) if p.has_keyword_cta and p.target_link else "N/A"
    reply_display = (
        html.escape(p.public_reply_text)
        if p.has_keyword_cta and p.public_reply_text
        else "N/A"
    )
    caption = html.escape(p.caption)
    hashtags = html.escape(p.hashtags)

    note_block = f"💡 <b>Razonamiento IA:</b>\n<i>{html.escape(ai_note)}</i>\n\n" if ai_note else ""

    meta_block = (
        f"<b>{html.escape(header)}</b>\n\n"
        + note_block
        + f"📁 <b>Archivo:</b> <code>{filename}</code>\n"
        f"🎥 <b>Contenido analizado:</b> <i>{content_summary}</i>\n"
        f"🏷️ <b>Cadencia / Título:</b> {category_label} — {short_title}\n"
        f"🎬 <b>Formato:</b> {media_type}\n"
        f"🕒 <b>Fecha y hora sugerida:</b> <code>{scheduled_time}</code>\n"
        + (f"   <i>{schedule_reasoning}</i>\n" if schedule_reasoning else "")
        + f"🎯 <b>Gancho / Título:</b> {hook_title}\n"
        f"📣 <b>Meta Ads:</b> {html.escape(ads_badge)}\n"
        + (f"   <i>{ads_detail}</i>\n" if ads_detail else "")
        + f"🔑 <b>Palabra clave:</b> <code>{keyword_display}</code>\n"
        f"🔗 <b>Enlace DM:</b> {link_display}\n"
        f"💬 <b>Respuesta pública:</b> {reply_display}\n\n"
        f"📝 <b>Caption sugerido:</b>\n"
    )
    hashtags_block = f"\n\n🏷️ <b>Hashtags:</b>\n{hashtags}" if hashtags else ""
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

    def _get_reference_now(self) -> datetime:
        return datetime.now().astimezone()

    def _is_authorized_admin(self, update: Update) -> bool:
        chat = update.effective_chat
        if chat is None or chat.id != self.settings.telegram_admin_chat_id:
            logger.warning(
                "Acceso ignorado de chat_id no autorizado: %s",
                chat.id if chat else "None",
            )
            return False
        return True

    async def _send_admin_message(
        self,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
        parse_mode: str | None = ParseMode.HTML,
    ) -> int:
        if self.application is None:
            raise RuntimeError("Telegram Application no inicializada.")
        msg = await self.application.bot.send_message(
            chat_id=self.settings.telegram_admin_chat_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode=parse_mode,
            disable_web_page_preview=True,
        )
        return msg.message_id

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

            temp_media_path: Path | None = None
            try:
                file_size = int(file_meta.get("size", 0) or 0)
                if file_size <= MAX_MULTIMODAL_DOWNLOAD_BYTES:
                    try:
                        temp_media_path = await asyncio.to_thread(
                            download_drive_file_to_temp,
                            file_id,
                            file_name,
                            self.settings.google_application_credentials,
                        )
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

                proposal = await asyncio.to_thread(
                    generate_post_proposal,
                    guidelines_text,
                    file_meta,
                    self._get_reference_now(),
                    self.settings.gemini_api_key,
                    self.settings.gemini_model,
                    occupied_slots,
                    temp_media_path,
                )
                occupied_slots.append(proposal.scheduled_time)

                proposal_id = uuid.uuid4().hex[:10]
                session = ProposalSession(
                    proposal_id=proposal_id,
                    month_folder_id=month_folder_id,
                    file_metadata=file_meta,
                    guidelines_text=guidelines_text,
                    proposal=proposal,
                )

                message_id = await self._send_admin_message(
                    text=_format_proposal_html(session),
                    reply_markup=_build_approval_keyboard(proposal_id),
                )
                session.message_id = message_id
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
                if temp_media_path is not None:
                    temp_media_path.unlink(missing_ok=True)

    async def handle_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle inline buttons: Approve & Schedule, Request Edit, Force Edit, or Discard."""
        if not self._is_authorized_admin(update):
            return

        query = update.callback_query
        if query is None or not query.data:
            return
        await query.answer()

        if ":" not in query.data:
            return
        action, proposal_id = query.data.split(":", 1)

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

        if action == "discard":
            session.status = "discarded"
            if self.awaiting_feedback_proposal_id == proposal_id:
                self.awaiting_feedback_proposal_id = None

            await asyncio.to_thread(
                mark_file_state,
                session.file_metadata["id"],
                "DISCARDED",
                {"name": file_name},
                self.settings.state_file_path,
            )
            await query.edit_message_text(
                text=_format_proposal_html(session, header="❌ Propuesta descartada"),
                parse_mode=ParseMode.HTML,
                reply_markup=None,
            )
            logger.info("Propuesta %s (%s) descartada por el admin.", proposal_id, file_name)
            return

        if action == "edit":
            self.awaiting_feedback_proposal_id = proposal_id
            await self._send_admin_message(
                f"✏️ Escribe los cambios que deseas para <b>{html.escape(file_name)}</b>.\n"
                "Analizaré tu solicitud con criterio estratégico (y si insistes, aplicaré exactamente lo que indiques)."
            )
            return

        if action == "force_edit":
            if not session.last_feedback:
                await self._send_admin_message(
                    "ℹ️ No hay un cambio previo pendiente para forzar. Pulsa ✏️ Pedir Cambio."
                )
                return

            await query.edit_message_text(
                text=_format_proposal_html(
                    session,
                    header="⏳ Aplicando exactamente tu instrucción...",
                ),
                parse_mode=ParseMode.HTML,
                reply_markup=None,
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
                new_msg_id = await self._send_admin_message(
                    text=_format_proposal_html(
                        session,
                        header="✅ Cambio aplicado exactamente según tu indicación",
                        ai_note=revision.reasoning_message,
                    ),
                    reply_markup=_build_approval_keyboard(proposal_id, show_force_apply=False),
                )
                session.message_id = new_msg_id
            except Exception as exc:
                logger.exception("Error forzando cambio en %s: %s", proposal_id, exc)
                await self.notify_error(f"Error aplicando tu cambio en '{file_name}': {exc}")
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
                text=_format_proposal_html(
                    session,
                    header="⏳ Aprobando: anexando a Pautas y organizando Drive...",
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
            except Exception as sync_exc:
                logger.exception(
                    "Error al anexar a Pautas / renombrar '%s' en Drive: %s",
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

            self.scheduler.add_job(
                self.execute_publication,
                trigger=DateTrigger(run_date=run_date),
                args=[proposal_id],
                id=f"publish_{proposal_id}",
                replace_existing=True,
                misfire_grace_time=3600,
            )

            status_header = (
                f"🚀 Aprobado ({doc_header or final_name}) — Publicando inmediatamente"
                if run_immediately
                else f"✅ Aprobado ({doc_header or final_name}) — Programado para {scheduled_dt.isoformat()}"
            )
            await query.edit_message_text(
                text=_format_proposal_html(session, header=status_header),
                parse_mode=ParseMode.HTML,
                reply_markup=None,
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
        """Append approved proposal to the monthly `Pautas` doc and rename the Drive file."""
        creds = self.settings.google_application_credentials
        month_folder_id = session.month_folder_id
        file_id = str(session.file_metadata["id"])
        old_name = str(session.file_metadata.get("name", file_id))
        p = session.proposal

        doc_meta = get_monthly_doc_metadata(month_folder_id, credentials_path=creds)
        latest_guidelines = (
            get_guidelines_text(str(doc_meta["id"]), credentials_path=creds)
            if doc_meta
            else session.guidelines_text
        )
        session.guidelines_text = latest_guidelines

        doc_header, new_filename = compute_next_numbering_and_filename(
            month_folder_id=month_folder_id,
            guidelines_text=latest_guidelines,
            post_category=p.post_category,
            short_title=p.short_title,
            original_filename=old_name,
            credentials_path=creds,
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
            append_entry_to_monthly_doc(
                doc_id=str(doc_meta["id"]),
                entry_lines=entry_lines,
                credentials_path=creds,
            )

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

        proposal_id = self.awaiting_feedback_proposal_id
        if not proposal_id or proposal_id not in self.pending_proposals:
            await message.reply_text(
                "ℹ️ No hay ninguna edición pendiente. Usa /check para buscar archivos nuevos o /status para ver el estado."
            )
            return

        session = self.pending_proposals[proposal_id]
        if session.status != "pending":
            self.awaiting_feedback_proposal_id = None
            await message.reply_text(
                f"ℹ️ La propuesta ya cambió de estado ({session.status})."
            )
            return

        feedback = message.text.strip()
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

            if session.message_id is not None and self.application is not None:
                try:
                    await self.application.bot.edit_message_text(
                        chat_id=self.settings.telegram_admin_chat_id,
                        message_id=session.message_id,
                        text=_format_proposal_html(
                            session,
                            header="🔄 Propuesta anterior reemplazada por revisión",
                        ),
                        parse_mode=ParseMode.HTML,
                        reply_markup=None,
                    )
                except TelegramError as edit_exc:
                    logger.debug("No se pudo editar mensaje anterior: %s", edit_exc)

            is_alternative = revision.ai_assessment == "ALTERNATIVE_SUGGESTED"
            header = (
                "💡 Alternativa estratégica sugerida"
                if is_alternative
                else "✏️ Propuesta actualizada"
            )
            new_message_id = await self._send_admin_message(
                text=_format_proposal_html(
                    session,
                    header=header,
                    ai_note=revision.reasoning_message,
                ),
                reply_markup=_build_approval_keyboard(
                    proposal_id,
                    show_force_apply=is_alternative,
                ),
            )
            session.message_id = new_message_id
        except Exception as exc:
            logger.exception("Error reescribiendo propuesta %s: %s", proposal_id, exc)
            self.awaiting_feedback_proposal_id = proposal_id
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
        mime_type = str(session.file_metadata.get("mimeType", "")).lower()
        is_video = mime_type.startswith("video/") or file_name.lower().endswith(".mp4")
        proposal = session.proposal
        session.status = "publishing"

        link_info: TempPublicLink | None = None
        media_id: str | None = None

        try:
            logger.info("Iniciando flujo de publicación para '%s' (%s)...", file_name, file_id)
            link_info = await asyncio.to_thread(
                set_temp_public_link,
                file_id,
                self.settings.google_application_credentials,
            )

            media_id = await publish_to_instagram(
                ig_user_id=self.settings.ig_user_id,
                access_token=self.settings.meta_access_token,
                media_url=link_info.url,
                media_type=proposal.media_type,
                caption=proposal.full_instagram_caption,
                is_video=is_video,
                file_id=file_id,
                has_keyword_cta=proposal.has_keyword_cta,
                keyword=proposal.keyword,
                target_link=proposal.target_link,
                public_reply_text=proposal.public_reply_text,
                state_path=self.settings.state_file_path,
            )

            ads_summary = "No requerida (orgánico)"
            if proposal.promote_in_ads:
                if self.settings.meta_ad_account_id:
                    try:
                        ad_res = await promote_instagram_post(
                            access_token=self.settings.meta_access_token,
                            ig_user_id=self.settings.ig_user_id,
                            ad_account_id=self.settings.meta_ad_account_id,
                            ig_media_id=media_id,
                            post_name=proposal.short_title or file_name,
                            daily_budget_usd=proposal.ads_daily_budget_usd,
                            duration_days=proposal.ads_duration_days,
                            target_link=proposal.target_link,
                        )
                        ads_summary = (
                            f"Activa (Campaign={ad_res.campaign_id}, Ad={ad_res.ad_id}, "
                            f"${ad_res.daily_budget_usd:.2f}/día × {ad_res.duration_days} días)"
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
                        accounts = await list_accessible_ad_accounts(self.settings.meta_access_token)
                        acc_str = ", ".join(
                            f"{a.get('id')} ({a.get('name')})" for a in accounts
                        ) or "ninguna"
                    except Exception:
                        acc_str = "no detectadas"
                    ads_summary = (
                        "⚠️ Recomendada, pero falta META_AD_ACCOUNT_ID en .env "
                        f"(cuentas disponibles: {acc_str})"
                    )

            await asyncio.to_thread(
                revoke_public_permission,
                file_id,
                link_info.permission_id,
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
                },
                self.settings.state_file_path,
            )

            dm_summary = (
                f"Activa en webhook (keyword={proposal.keyword})"
                if proposal.has_keyword_cta
                else "No requerida (sin palabra clave)"
            )
            await self._send_admin_message(
                "🎉 <b>¡Publicación completada con éxito!</b>\n\n"
                f"📁 <b>Archivo:</b> <code>{html.escape(file_name)}</code>\n"
                f"📸 <b>IG Media ID:</b> <code>{html.escape(media_id)}</code>\n"
                f"📣 <b>Meta Ads:</b> {html.escape(ads_summary)}\n"
                f"🤖 <b>Auto-DM / Replies:</b> {html.escape(dm_summary)}\n"
                "🔒 <b>Drive:</b> Permiso público revocado y marcado como <code>PUBLISHED</code>."
            )
            logger.info("Flujo completo finalizado para '%s' (media_id=%s).", file_name, media_id)

        except Exception as exc:
            logger.exception("Fallo en execute_publication para '%s': %s", file_name, exc)
            session.status = "failed"

            if link_info is not None:
                try:
                    await asyncio.to_thread(
                        revoke_public_permission,
                        file_id,
                        link_info.permission_id,
                        self.settings.google_application_credentials,
                    )
                except Exception as revoke_exc:
                    logger.error(
                        "No se pudo revocar permiso %s tras fallo en '%s': %s",
                        link_info.permission_id,
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
            await self.notify_error(
                f"Falló la publicación de '{file_name}' ({file_id}).\n"
                f"Permiso público revocado (si aplicaba). Error:\n{exc}"
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
            f"y escucho comentarios de Instagram en el puerto <b>{self.settings.webhook_port}</b>.\n"
            "Comandos disponibles:\n"
            "• /check — Escanear la carpeta del mes ahora mismo\n"
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

        lines = [
            "<b>📊 Estado del Agente</b>",
            f"• Propuestas esperando aprobación: <b>{len(pending)}</b>",
            f"• Publicaciones programadas: <b>{len(scheduled_jobs)}</b>",
        ]
        for job in scheduled_jobs:
            run_time = job.next_run_time.isoformat() if job.next_run_time else "N/A"
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
    application.add_handler(CommandHandler("status", bot_controller.cmd_status))
    application.add_handler(CallbackQueryHandler(bot_controller.handle_callback))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, bot_controller.handle_admin_text)
    )
