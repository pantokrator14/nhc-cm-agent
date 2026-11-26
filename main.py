"""Unified asyncio orchestrator: FastAPI Webhook + Telegram Polling + APScheduler."""

from __future__ import annotations

import asyncio
import html
import logging
import sys
from datetime import datetime, timedelta

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes

from config import ConfigurationError, Settings, load_settings
from drive_service import (
    create_monthly_guidelines_doc,
    ensure_current_month_folder,
    ensure_statistics_folder,
    find_previous_month_guidelines_text,
    get_monthly_doc_metadata,
    get_monthly_guidelines,
    list_pending_files,
    mark_file_state,
    move_unpublished_media_from_previous_month,
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
    *,
    notify_always: bool = False,
) -> None:
    """Resolve current YYYY/MM Drive folder, read monthly guidelines, and propose new media."""
    if scan_lock.locked():
        logger.info("Escaneo de Drive ya en ejecución; se omite ciclo superpuesto.")
        if notify_always:
            await bot_controller.notify_info(
                "⏳ Ya hay un escaneo de Drive en ejecución; espera unos segundos y vuelve a intentar."
            )
        return

    async with scan_lock:
        local_now = datetime.now(settings.tzinfo)
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

        rollover_summary = ""
        if resolution.created or created_doc_id:
            try:
                rollover = await asyncio.to_thread(
                    move_unpublished_media_from_previous_month,
                    root_folder_id=settings.drive_root_folder_id,
                    new_month_folder_id=month_folder_id,
                    year_str=year_str,
                    month_str=month_str,
                    state_path=settings.state_file_path,
                    credentials_path=settings.google_application_credentials,
                )
                if rollover.moved_names:
                    preview = ", ".join(html.escape(n) for n in rollover.moved_names[:10])
                    extra = (
                        f" (+{len(rollover.moved_names) - 10} más)"
                        if len(rollover.moved_names) > 10
                        else ""
                    )
                    rollover_summary = (
                        f"\n\n📦 <b>Contenido no publicado trasladado desde {html.escape(rollover.previous_period)}:</b> "
                        f"<b>{len(rollover.moved_names)}</b> elemento(s) ({preview}{extra})."
                    )
            except Exception as rollover_exc:
                logger.warning(
                    "No se pudo trasladar contenido no publicado del mes anterior a %s/%s: %s",
                    year_str,
                    month_str,
                    rollover_exc,
                )

            drive_url = f"https://drive.google.com/drive/folders/{month_folder_id}"
            await bot_controller.notify_info(
                f"👋 <b>¡Hola! Ha iniciado otro mes (<code>{year_str}/{month_str}</code>)</b> "
                "y ya he creado la carpeta correspondiente junto con el documento <b>Pautas</b> "
                "(tomando el mes anterior como referencia de formato y marca)."
                f"{rollover_summary}\n\n"
                "📤 Sube los archivos multimedia que tengas a dicha carpeta para analizarlos y hacer sus propuestas de pautas.\n\n"
                f"🔗 <a href=\"{drive_url}\">Abrir carpeta {year_str}/{month_str} en Google Drive</a>"
            )

        try:
            await asyncio.to_thread(
                ensure_statistics_folder,
                month_folder_id,
                settings.google_application_credentials,
            )
        except Exception as stats_exc:
            logger.debug("No se pudo verificar subcarpeta de Estadísticas en %s/%s: %s", year_str, month_str, stats_exc)

        guidelines_text: str | None = None
        try:
            guidelines_text = await asyncio.to_thread(
                get_monthly_guidelines,
                month_folder_id,
                settings.google_application_credentials,
            )
            await bot_controller.notify_overdue_pautas_if_needed(
                month_folder_id=month_folder_id,
                guidelines_text=guidelines_text,
            )
        except Exception as exc:
            logger.debug("No se pudieron revisar pautas atrasadas al inicio del ciclo: %s", exc)

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
            if notify_always:
                await bot_controller.notify_info(
                    f"✅ Escaneo completado (<code>{year_str}/{month_str}</code>): no hay archivos nuevos pendientes en Drive "
                    "(todos los multimedia ya están publicados, programados o registrados en Pautas).\n\n"
                    "Usa /pautas para ver las pautas pendientes de publicar o /status para ver la cola."
                )
            return

        candidate_files = [
            f for f in pending_files if not bot_controller.has_active_session_for_file(f["id"])
        ]
        if not candidate_files:
            logger.info("No hay archivos nuevos por evaluar en %s/%s.", year_str, month_str)
            if notify_always:
                preview_names = ", ".join(
                    html.escape(str(f.get("name", f.get("id", "")))) for f in pending_files[:6]
                )
                more = f" (+{len(pending_files) - 6} más)" if len(pending_files) > 6 else ""
                await bot_controller.notify_info(
                    f"✅ Escaneo completado (<code>{year_str}/{month_str}</code>): hay <b>{len(pending_files)}</b> archivo(s) "
                    f"que ya tienen propuesta enviada esperando tu aprobación en Telegram ({preview_names}{more}).\n\n"
                    "Usa /status para ver el resumen o /pautas para revisar pautas existentes."
                )
            return

        if guidelines_text is None:
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
        matched_count = 0
        for file_meta in candidate_files:
            file_id = file_meta["id"]
            file_name = file_meta.get("name", file_id)
            match = classification.get(file_id)

            if match and match.already_in_guidelines:
                matched_count += 1
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
            if notify_always:
                await bot_controller.notify_info(
                    f"✅ Escaneo completado (<code>{year_str}/{month_str}</code>): "
                    f"los {matched_count} archivo(s) detectado(s) ya figuraban en el documento <b>Pautas</b>.\n\n"
                    "Usa /pautas para listarlos, editarlos o programarlos."
                )
            return

        await bot_controller.propose_pending_files(
            month_folder_id=month_folder_id,
            pending_files=unlisted_files,
            guidelines_text=guidelines_text,
        )
        if notify_always:
            await bot_controller.notify_info(
                f"✅ Escaneo completado (<code>{year_str}/{month_str}</code>): "
                f"se enviaron <b>{len(unlisted_files)}</b> propuesta(s) nueva(s)."
            )


async def run_agent(settings: Settings) -> None:
    """Run FastAPI/Uvicorn, Telegram Bot polling, and APScheduler in a single asyncio loop."""
    scheduler = AsyncIOScheduler(timezone=settings.tzinfo)
    scan_lock = asyncio.Lock()

    bot_controller = NelHealthCoachBot(
        settings=settings,
        scheduler=scheduler,
        poll_interval_minutes=DRIVE_POLL_INTERVAL_MINUTES,
    )

    async def scheduled_review(*, manual: bool = False) -> None:
        try:
            await run_monthly_drive_review(
                settings,
                bot_controller,
                scan_lock,
                notify_always=manual,
            )
        except Exception as exc:
            logger.exception("Error inesperado en ciclo programado de Drive: %s", exc)
            if manual:
                await bot_controller.notify_error(f"Error al escanear Drive: {exc}")

    bot_controller.scan_callback = scheduled_review

    application = ApplicationBuilder().token(settings.telegram_bot_token).build()
    register_handlers(application, bot_controller)
    application.add_error_handler(_telegram_error_handler)

    fastapi_app = create_webhook_app(
        settings,
        interaction_callback=bot_controller.handle_incoming_ig_interaction,
    )
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
        try:
            await bot_controller.rehydrate_scheduled_jobs()
        except Exception as rehydrate_exc:
            logger.warning("Error restaurando publicaciones programadas tras reinicio: %s", rehydrate_exc)

        async def scheduled_weekly_metrics() -> None:
            try:
                await bot_controller.generate_and_save_weekly_report()
            except Exception as metrics_exc:
                logger.exception("Error generando informe semanal de métricas: %s", metrics_exc)
                await bot_controller.notify_error(
                    f"Error en la generación automática del informe semanal de métricas: {metrics_exc}"
                )

        async def scheduled_meta_token_check() -> None:
            try:
                await bot_controller.check_meta_token_health()
            except Exception as tok_exc:
                logger.warning("No se pudo verificar estado de META_ACCESS_TOKEN: %s", tok_exc)
            try:
                await bot_controller.check_google_oauth_health()
            except Exception as goauth_exc:
                logger.warning("No se pudo verificar estado de Google OAuth: %s", goauth_exc)

        async def scheduled_ig_dm_poll() -> None:
            try:
                await bot_controller.poll_instagram_dms()
            except Exception as dm_exc:
                logger.warning("Error en polling de respaldo para DMs/Solicitudes IG: %s", dm_exc)

        scheduler.add_job(
            scheduled_review,
            trigger=IntervalTrigger(minutes=DRIVE_POLL_INTERVAL_MINUTES),
            id="drive_poll_job",
            replace_existing=True,
            next_run_time=datetime.now(settings.tzinfo),
            max_instances=1,
        )
        scheduler.add_job(
            scheduled_weekly_metrics,
            trigger=CronTrigger(day_of_week="sun", hour=20, minute=0, timezone=settings.tzinfo),
            id="weekly_metrics_report_job",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=3600,
        )
        scheduler.add_job(
            scheduled_meta_token_check,
            trigger=CronTrigger(hour=9, minute=0, timezone=settings.tzinfo),
            id="meta_token_health_job",
            replace_existing=True,
            next_run_time=datetime.now(settings.tzinfo),
            max_instances=1,
            misfire_grace_time=3600,
        )
        scheduler.add_job(
            scheduled_ig_dm_poll,
            trigger=IntervalTrigger(minutes=3),
            id="ig_dm_poll_job",
            replace_existing=True,
            next_run_time=datetime.now(settings.tzinfo) + timedelta(seconds=20),
            max_instances=1,
        )
        logger.info(
            "APScheduler iniciado (zona horaria %s): escaneo de Drive cada %d min + respaldo DMs cada 3 min + informe semanal domingos 20:00 + chequeo diario de META_ACCESS_TOKEN.",
            settings.timezone_name,
            DRIVE_POLL_INTERVAL_MINUTES,
        )

        try:
            await application.bot.set_my_commands(
                [
                    BotCommand("check", "Escanear Drive ahora y proponer archivos nuevos"),
                    BotCommand("mensajes", "Revisar DMs y solicitudes de mensajes en Instagram"),
                    BotCommand("pautas", "Listar pautas del documento aún sin publicar"),
                    BotCommand("calendario", "Ver agenda editorial de los próximos 14 días"),
                    BotCommand("sugerencias", "Analizar tendencias y sugerir nuevas ideas virales"),
                    BotCommand("guion", "Generar guion listo para grabar y guardarlo en Drive"),
                    BotCommand("metricas", "Generar informe semanal (opcional: anterior o semana N)"),
                    BotCommand("token", "Verificar vigencia y renovar META_ACCESS_TOKEN"),
                    BotCommand("id", "Ver tu Chat ID personal con este bot"),
                    BotCommand("status", "Ver propuestas pendientes y publicaciones programadas"),
                    BotCommand("help", "Mostrar ayuda y comandos disponibles"),
                ]
            )
            await bot_controller._send_admin_message(
                text=(
                    "🟢 <b>Agente @nelhealthcoach iniciado</b>\n"
                    f"• Zona horaria: <code>{settings.timezone_name}</code> (Hora Venezuela UTC-4)\n"
                    f"• Drive YYYY/MM cada {DRIVE_POLL_INTERVAL_MINUTES} min (Reels, Imágenes y carpetas <code>(CARRUSEL)</code>)\n"
                    "• Ubicación IG fija: <code>California</code>\n"
                    "• Comentarios, DMs y Solicitudes de mensaje IG con respuestas asistidas activados\n"
                    "• Monitoreo diario preventivo de META_ACCESS_TOKEN activado\n"
                    "• Informe semanal de métricas: Domingos 20:00 hrs\n"
                    f"• Webhook IG en puerto {settings.webhook_port}\n"
                    "Comandos: /check | /mensajes | /pautas | /calendario | /sugerencias | /guion | /metricas | /token | /id | /status"
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
