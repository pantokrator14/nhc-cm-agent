"""Unified asyncio orchestrator: FastAPI Webhook + Telegram Polling + APScheduler."""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes

from config import ConfigurationError, Settings, load_settings
from drive_service import (
    create_monthly_guidelines_doc,
    ensure_current_month_folder,
    find_previous_month_guidelines_text,
    get_monthly_doc_metadata,
    get_monthly_guidelines,
    list_pending_files,
    mark_file_state,
)
from gemini_agent import (
    classify_files_against_guidelines,
    generate_monthly_template_from_previous,
)
from instagram_webhook import create_webhook_app
from telegram_bot import NelHealthCoachBot, register_handlers

logger = logging.getLogger(__name__)

DRIVE_POLL_INTERVAL_MINUTES = 5


def _configure_logging() -> None:
    """Configure clean stdout logging suitable for systemd / journalctl."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)
    logging.getLogger("telegram").setLevel(logging.INFO)
    logging.getLogger("apscheduler").setLevel(logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)


async def _telegram_error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    logger.exception("Excepción no controlada en Telegram handler: %s", context.error)


def _polling_error_callback(error: Exception) -> None:
    logger.warning("Error de red en polling de Telegram (el agente reintentará): %s", error)


async def run_monthly_drive_review(
    settings: Settings,
    bot_controller: NelHealthCoachBot,
    scan_lock: asyncio.Lock,
) -> None:
    """Resolve current YYYY/MM Drive folder, read monthly guidelines, and propose new media."""
    if scan_lock.locked():
        logger.info("Escaneo de Drive ya en ejecución; se omite ciclo superpuesto.")
        return

    async with scan_lock:
        local_now = datetime.now().astimezone()
        year_str = local_now.strftime("%Y")
        month_str = local_now.strftime("%m")
        logger.info(
            "Iniciando revisión de Drive para el periodo %s/%s...",
            year_str,
            month_str,
        )

        try:
            resolution = await asyncio.to_thread(
                ensure_current_month_folder,
                settings.drive_root_folder_id,
                year_str,
                month_str,
                settings.google_application_credentials,
                True,
            )
        except Exception as exc:
            logger.exception(
                "Error consultando o creando carpeta mensual %s/%s en Drive: %s",
                year_str,
                month_str,
                exc,
            )
            await bot_controller.notify_error(
                f"Error al buscar/crear la carpeta {year_str}/{month_str} en Google Drive: {exc}"
            )
            return

        month_folder_id = resolution.folder_id
        if not month_folder_id:
            logger.warning(
                "Carpeta mensual '%s/%s' no disponible en Google Drive (root_folder_id=%s).",
                year_str,
                month_str,
                settings.drive_root_folder_id,
            )
            return

        created_doc_id: str | None = None
        try:
            existing_doc = await asyncio.to_thread(
                get_monthly_doc_metadata,
                month_folder_id,
                settings.google_application_credentials,
            )
            if existing_doc is None:
                prev_text = await asyncio.to_thread(
                    find_previous_month_guidelines_text,
                    settings.drive_root_folder_id,
                    year_str,
                    month_str,
                    settings.google_application_credentials,
                )
                template_text = await asyncio.to_thread(
                    generate_monthly_template_from_previous,
                    prev_text or "",
                    year_str,
                    month_str,
                    settings.gemini_api_key,
                    settings.gemini_model,
                )
                created_doc_id = await asyncio.to_thread(
                    create_monthly_guidelines_doc,
                    month_folder_id,
                    template_text,
                    "Pautas",
                    settings.google_application_credentials,
                )
        except Exception as exc:
            logger.exception(
                "Error verificando/creando documento 'Pautas' para %s/%s: %s",
                year_str,
                month_str,
                exc,
            )
            await bot_controller.notify_error(
                f"No se pudo crear automáticamente el documento 'Pautas' en {year_str}/{month_str}: {exc}"
            )

        if resolution.created or created_doc_id:
            drive_url = f"https://drive.google.com/drive/folders/{month_folder_id}"
            await bot_controller.notify_info(
                f"👋 <b>¡Hola! Ha iniciado otro mes (<code>{year_str}/{month_str}</code>)</b> "
                "y ya he creado la carpeta correspondiente junto con el documento <b>Pautas</b> "
                "(tomando el mes anterior como referencia de formato y marca).\n\n"
                "📤 Sube los archivos multimedia que tengas a dicha carpeta para analizarlos y hacer sus propuestas de pautas.\n\n"
                f"🔗 <a href=\"{drive_url}\">Abrir carpeta {year_str}/{month_str} en Google Drive</a>"
            )

        try:
            pending_files = await asyncio.to_thread(
                list_pending_files,
                month_folder_id,
                settings.state_file_path,
                settings.google_application_credentials,
            )
        except Exception as exc:
            logger.exception(
                "Error listando archivos en carpeta mensual %s/%s: %s",
                year_str,
                month_str,
                exc,
            )
            await bot_controller.notify_error(
                f"Error al listar archivos en Drive ({year_str}/{month_str}): {exc}"
            )
            return

        if not pending_files:
            logger.info("No hay archivos nuevos por procesar en %s/%s.", year_str, month_str)
            return

        candidate_files = [
            f for f in pending_files if not bot_controller.has_active_session_for_file(f["id"])
        ]
        if not candidate_files:
            logger.info("No hay archivos nuevos por evaluar en %s/%s.", year_str, month_str)
            return

        try:
            guidelines_text = await asyncio.to_thread(
                get_monthly_guidelines,
                month_folder_id,
                settings.google_application_credentials,
            )
        except Exception as exc:
            logger.exception(
                "Error leyendo las pautas del mes %s/%s: %s",
                year_str,
                month_str,
                exc,
            )
            await bot_controller.notify_error(
                f"Se detectaron {len(candidate_files)} archivo(s) nuevo(s) en {year_str}/{month_str}, "
                f"pero falló la lectura del Google Doc de pautas: {exc}"
            )
            return

        try:
            classification = await asyncio.to_thread(
                classify_files_against_guidelines,
                guidelines_text,
                candidate_files,
                settings.gemini_api_key,
                settings.gemini_model,
            )
        except Exception as exc:
            logger.exception(
                "Error clasificando archivos contra el documento de pautas: %s",
                exc,
            )
            await bot_controller.notify_error(
                f"Error comparando archivos de {year_str}/{month_str} con las pautas: {exc}"
            )
            return

        unlisted_files: list[dict] = []
        for file_meta in candidate_files:
            file_id = file_meta["id"]
            file_name = file_meta.get("name", file_id)
            match = classification.get(file_id)

            if match and match.already_in_guidelines:
                logger.info(
                    "Omitiendo '%s': ya aparece en el documento de pautas (%s).",
                    file_name,
                    match.matched_entry or "coincidencia detectada",
                )
                await asyncio.to_thread(
                    mark_file_state,
                    file_id,
                    "ALREADY_IN_GUIDELINES",
                    {
                        "name": file_name,
                        "matched_entry": match.matched_entry,
                    },
                    settings.state_file_path,
                )
            else:
                unlisted_files.append(file_meta)

        if not unlisted_files:
            logger.info(
                "Todos los archivos pendientes en %s/%s ya están en la lista de pautas.",
                year_str,
                month_str,
            )
            return

        await bot_controller.propose_pending_files(
            month_folder_id=month_folder_id,
            pending_files=unlisted_files,
            guidelines_text=guidelines_text,
        )


async def run_agent(settings: Settings) -> None:
    """Run FastAPI/Uvicorn, Telegram Bot polling, and APScheduler in a single asyncio loop."""
    scheduler = AsyncIOScheduler()
    scan_lock = asyncio.Lock()

    bot_controller = NelHealthCoachBot(
        settings=settings,
        scheduler=scheduler,
        poll_interval_minutes=DRIVE_POLL_INTERVAL_MINUTES,
    )

    async def scheduled_review() -> None:
        try:
            await run_monthly_drive_review(settings, bot_controller, scan_lock)
        except Exception as exc:
            logger.exception("Error inesperado en ciclo programado de Drive: %s", exc)

    bot_controller.scan_callback = scheduled_review

    application = ApplicationBuilder().token(settings.telegram_bot_token).build()
    register_handlers(application, bot_controller)
    application.add_error_handler(_telegram_error_handler)

    fastapi_app = create_webhook_app(settings)
    uvicorn_config = uvicorn.Config(
        app=fastapi_app,
        host="0.0.0.0",
        port=settings.webhook_port,
        log_config=None,
        access_log=True,
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)

    async with application:
        await application.start()
        if application.updater is None:
            raise RuntimeError("Telegram Updater no está disponible.")

        await application.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            error_callback=_polling_error_callback,
        )
        logger.info("Bot de Telegram iniciado en modo polling.")

        scheduler.start()
        scheduler.add_job(
            scheduled_review,
            trigger=IntervalTrigger(minutes=DRIVE_POLL_INTERVAL_MINUTES),
            id="drive_poll_job",
            replace_existing=True,
            next_run_time=datetime.now().astimezone(),
            max_instances=1,
        )
        logger.info(
            "APScheduler iniciado: escaneo de Drive cada %d minutos.",
            DRIVE_POLL_INTERVAL_MINUTES,
        )

        try:
            await application.bot.send_message(
                chat_id=settings.telegram_admin_chat_id,
                text=(
                    "🟢 <b>Agente @nelhealthcoach iniciado</b>\n"
                    f"• Drive YYYY/MM cada {DRIVE_POLL_INTERVAL_MINUTES} min\n"
                    f"• Webhook IG en puerto {settings.webhook_port}\n"
                    "Usa /check para forzar revisión o /status para ver la cola."
                ),
                parse_mode=ParseMode.HTML,
            )
        except Exception as exc:
            logger.warning("No se pudo enviar mensaje de inicio por Telegram: %s", exc)

        logger.info(
            "Iniciando servidor FastAPI/Uvicorn en 0.0.0.0:%d...",
            settings.webhook_port,
        )
        try:
            await uvicorn_server.serve()
        except asyncio.CancelledError:
            logger.info("Bucle principal cancelado; iniciando apagado limpio...")
        finally:
            logger.info("Deteniendo APScheduler y Telegram polling...")
            if scheduler.running:
                scheduler.shutdown(wait=False)
            if application.updater.running:
                await application.updater.stop()
            await application.stop()
            logger.info("Apagado completo del agente.")


def main() -> None:
    """Entrypoint for CLI and systemd service execution."""
    _configure_logging()

    try:
        settings = load_settings()
    except ConfigurationError as exc:
        logger.critical("Error de configuración: %s", exc)
        sys.exit(1)

    try:
        asyncio.run(run_agent(settings))
    except KeyboardInterrupt:
        logger.info("Agente detenido manualmente (KeyboardInterrupt).")


if __name__ == "__main__":
    main()
