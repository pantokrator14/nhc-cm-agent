"""Strict workflow tests for config loading, weekly metrics fallback delivery, and /guion resilience."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from apscheduler.schedulers.asyncio import AsyncIOScheduler

import config
from drive_service import DriveStorageQuotaError, MonthFolderResolution, get_performance_memory
from gemini_agent import VideoScriptResult, WeeklyMetricsReport
from telegram_bot import TELEGRAM_MAX_MESSAGE_LENGTH, NelHealthCoachBot


def _make_settings(tmp: Path, sa_path: Path) -> config.Settings:
    return config.Settings(
        gemini_api_key="gem-key",
        gemini_model="gemini-2.5-flash",
        google_application_credentials=sa_path,
        drive_root_folder_id="root_folder_1",
        telegram_bot_token="123:abc",
        telegram_admin_chat_id=1001,
        telegram_extra_admin_chat_ids=(1002,),
        meta_access_token="meta-tok",
        ig_user_id="ig-123",
        base_dir=tmp,
        state_file_path=tmp / "state.json",
    )


def _make_bot(tmp: Path) -> tuple[NelHealthCoachBot, MagicMock]:
    sa_path = tmp / "sa.json"
    sa_path.write_text(json.dumps({"type": "service_account", "client_email": "sa@test.iam"}), encoding="utf-8")
    settings = _make_settings(tmp, sa_path)
    scheduler = MagicMock(spec=AsyncIOScheduler)
    scheduler.get_jobs.return_value = []
    bot_ctrl = NelHealthCoachBot(settings=settings, scheduler=scheduler)

    mock_tg_bot = MagicMock()
    msg_counter = {"n": 0}

    async def _send_msg(**kwargs):
        msg_counter["n"] += 1
        return SimpleNamespace(message_id=msg_counter["n"])

    mock_tg_bot.send_message = AsyncMock(side_effect=_send_msg)
    mock_tg_bot.send_document = AsyncMock(return_value=SimpleNamespace(message_id=999))
    bot_ctrl.application = SimpleNamespace(bot=mock_tg_bot)
    return bot_ctrl, mock_tg_bot


class TestConfigValidation(unittest.TestCase):
    def test_load_settings_validates_oauth_path_and_impersonation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sa = tmp / "sa.json"
            sa.write_text("{}", encoding="utf-8")
            oauth = tmp / "oauth.json"
            oauth.write_text("{}", encoding="utf-8")
            env_file = tmp / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "GEMINI_API_KEY=k1",
                        f"GOOGLE_APPLICATION_CREDENTIALS={sa}",
                        f"GOOGLE_OAUTH_CREDENTIALS={oauth}",
                        "GOOGLE_DRIVE_IMPERSONATE_EMAIL=ceo@nelhealthcoach.com",
                        "DRIVE_ROOT_FOLDER_ID=root123",
                        "TELEGRAM_BOT_TOKEN=tg123",
                        "TELEGRAM_ADMIN_CHAT_ID=12345",
                        "META_ACCESS_TOKEN=meta123",
                        "IG_USER_ID=ig123",
                    ]
                ),
                encoding="utf-8",
            )
            clean_keys = {
                "GEMINI_API_KEY": "",
                "GOOGLE_APPLICATION_CREDENTIALS": "",
                "GOOGLE_OAUTH_CREDENTIALS": "",
                "GOOGLE_DRIVE_IMPERSONATE_EMAIL": "",
                "DRIVE_ROOT_FOLDER_ID": "",
                "TELEGRAM_BOT_TOKEN": "",
                "TELEGRAM_ADMIN_CHAT_ID": "",
                "META_ACCESS_TOKEN": "",
                "IG_USER_ID": "",
            }
            with patch.dict(os.environ, clean_keys, clear=False):
                for k in clean_keys:
                    os.environ.pop(k, None)
                settings = config.load_settings(env_file)
                self.assertEqual(settings.google_drive_impersonate_email, "ceo@nelhealthcoach.com")
                self.assertEqual(settings.google_oauth_credentials, oauth.resolve())

    def test_load_settings_rejects_missing_oauth_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sa = tmp / "sa.json"
            sa.write_text("{}", encoding="utf-8")
            env_file = tmp / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "GEMINI_API_KEY=k1",
                        f"GOOGLE_APPLICATION_CREDENTIALS={sa}",
                        f"GOOGLE_OAUTH_CREDENTIALS={tmp / 'does_not_exist.json'}",
                        "DRIVE_ROOT_FOLDER_ID=root123",
                        "TELEGRAM_BOT_TOKEN=tg123",
                        "TELEGRAM_ADMIN_CHAT_ID=12345",
                        "META_ACCESS_TOKEN=meta123",
                        "IG_USER_ID=ig123",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(config.ConfigurationError):
                    config.load_settings(env_file)


class TestWeeklyReportAndScriptWorkflows(unittest.IsolatedAsyncioTestCase):
    async def test_weekly_report_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bot_ctrl, mock_tg_bot = _make_bot(tmp)
            ref_dt = datetime(2026, 11, 22, 20, 0, tzinfo=timezone.utc)
            fake_report = WeeklyMetricsReport(
                executive_summary="Crecimiento sólido del 15%.",
                report_document_text="Resumen Ejecutivo\nSubió el alcance.\nRecomendaciones:\nSeguir igual.",
                winning_insights="Reels matutinos con gancho directo tuvieron mejor retención.",
            )

            with patch("telegram_bot.ensure_current_month_folder", return_value=MonthFolderResolution("m1", False)), patch(
                "telegram_bot.ensure_statistics_week_folder", return_value=("stats1", "week3_id", True)
            ), patch("telegram_bot.collect_previous_weekly_reports", return_value=[]), patch(
                "telegram_bot.get_monthly_guidelines", return_value="PAUTAS"
            ), patch(
                "telegram_bot.fetch_instagram_weekly_metrics", AsyncMock(return_value={"profile": {}, "posts": []})
            ), patch(
                "telegram_bot.fetch_meta_ads_weekly_insights", AsyncMock(return_value=[])
            ), patch(
                "telegram_bot.fetch_tiktok_recent_videos", AsyncMock(return_value=[])
            ), patch(
                "telegram_bot.generate_weekly_metrics_report", return_value=fake_report
            ), patch(
                "telegram_bot.create_or_update_google_doc", return_value=("doc_999", True)
            ):
                await bot_ctrl.generate_and_save_weekly_report(reference_dt=ref_dt)

            mem = get_performance_memory(tmp / "state.json")
            self.assertEqual(mem["winning_insights"], fake_report.winning_insights)
            self.assertEqual(mock_tg_bot.send_message.call_count, 2)
            sent_text = mock_tg_bot.send_message.call_args_list[0].kwargs["text"]
            self.assertIn("https://docs.google.com/document/d/doc_999/edit", sent_text)
            self.assertIn("https://drive.google.com/drive/folders/week3_id", sent_text)

    async def test_weekly_report_survives_storage_quota_error_and_sends_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bot_ctrl, mock_tg_bot = _make_bot(tmp)
            ref_dt = datetime(2026, 11, 22, 20, 0, tzinfo=timezone.utc)
            fake_report = WeeklyMetricsReport(
                executive_summary="Alcance semanal +20%.",
                report_document_text="INFORME SEMANA 3\nDetalle completo de métricas IG, Meta Ads y TikTok.",
                winning_insights="Los carruseles educativos lograron más guardados.",
            )

            with patch("telegram_bot.ensure_current_month_folder", return_value=MonthFolderResolution("m1", False)), patch(
                "telegram_bot.ensure_statistics_week_folder", return_value=("stats1", "18__8pcfrbQDTSldqYTAX6MSjoosRy4aT", False)
            ), patch("telegram_bot.collect_previous_weekly_reports", return_value=[]), patch(
                "telegram_bot.get_monthly_guidelines", return_value="PAUTAS"
            ), patch(
                "telegram_bot.fetch_instagram_weekly_metrics", AsyncMock(return_value={"profile": {}, "posts": []})
            ), patch(
                "telegram_bot.fetch_meta_ads_weekly_insights", AsyncMock(return_value=[])
            ), patch(
                "telegram_bot.fetch_tiktok_recent_videos", AsyncMock(return_value=[])
            ), patch(
                "telegram_bot.generate_weekly_metrics_report", return_value=fake_report
            ), patch(
                "telegram_bot.create_or_update_google_doc",
                side_effect=DriveStorageQuotaError("Service Account sin cuota (403 storageQuotaExceeded)."),
            ):
                await bot_ctrl.generate_and_save_weekly_report(reference_dt=ref_dt)

            # Insights memory MUST still be saved
            mem = get_performance_memory(tmp / "state.json")
            self.assertEqual(mem["winning_insights"], fake_report.winning_insights)

            # Document attachment sent to both admins with full report bytes
            self.assertEqual(mock_tg_bot.send_document.call_count, 2)
            doc_bytes = mock_tg_bot.send_document.call_args_list[0].kwargs["document"].getvalue()
            self.assertEqual(doc_bytes.decode("utf-8"), fake_report.report_document_text)

            # Summary + folder link + quota explanation sent
            sent_text = mock_tg_bot.send_message.call_args_list[0].kwargs["text"]
            self.assertIn("sin guardar en Google Docs", sent_text)
            self.assertIn("18__8pcfrbQDTSldqYTAX6MSjoosRy4aT", sent_text)
            self.assertIn("Alcance semanal +20%.", sent_text)

    async def test_weekly_report_chunked_html_fallback_when_send_document_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bot_ctrl, mock_tg_bot = _make_bot(tmp)
            mock_tg_bot.send_document = AsyncMock(side_effect=RuntimeError("upload failed"))
            ref_dt = datetime(2026, 11, 22, 20, 0, tzinfo=timezone.utc)
            long_body = ("Línea de métrica importante <test & verify>\n" * 180).strip()
            fake_report = WeeklyMetricsReport(
                executive_summary="Resumen breve.",
                report_document_text=long_body,
                winning_insights="Insight clave.",
            )

            with patch("telegram_bot.ensure_current_month_folder", return_value=MonthFolderResolution("m1", False)), patch(
                "telegram_bot.ensure_statistics_week_folder", return_value=("stats1", "week3_id", False)
            ), patch("telegram_bot.collect_previous_weekly_reports", return_value=[]), patch(
                "telegram_bot.get_monthly_guidelines", return_value="PAUTAS"
            ), patch(
                "telegram_bot.fetch_instagram_weekly_metrics", AsyncMock(return_value={"profile": {}, "posts": []})
            ), patch(
                "telegram_bot.fetch_meta_ads_weekly_insights", AsyncMock(return_value=[])
            ), patch(
                "telegram_bot.fetch_tiktok_recent_videos", AsyncMock(return_value=[])
            ), patch(
                "telegram_bot.generate_weekly_metrics_report", return_value=fake_report
            ), patch(
                "telegram_bot.create_or_update_google_doc",
                side_effect=DriveStorageQuotaError("storageQuotaExceeded"),
            ):
                await bot_ctrl.generate_and_save_weekly_report(reference_dt=ref_dt)

            admin_1001_msgs = [
                call.kwargs["text"]
                for call in mock_tg_bot.send_message.call_args_list
                if call.kwargs["chat_id"] == 1001
            ]
            self.assertGreaterEqual(len(admin_1001_msgs), 2)
            for msg in admin_1001_msgs:
                self.assertLessEqual(len(msg), TELEGRAM_MAX_MESSAGE_LENGTH)
                # Every <pre> tag must be properly balanced within each chunk
                self.assertEqual(len(re.findall(r"<pre>", msg)), len(re.findall(r"</pre>", msg)))

            combined = "\n".join(admin_1001_msgs)
            self.assertIn("Línea de métrica importante &lt;test &amp; verify&gt;", combined)

    async def test_generate_and_send_script_survives_drive_quota_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bot_ctrl, mock_tg_bot = _make_bot(tmp)
            fake_script = VideoScriptResult(
                title="Caminar después de comer",
                format_type="PRO_TIP",
                hook_0_3s="¿Sabías esto de caminar 10 minutos?",
                body_3_45s="Ayuda a regular la glucosa tras cada comida.",
                cta_45_60s="Guarda este video y compártelo.",
                recording_notes="Luz natural y plano medio.",
                caption="Camina 10 minutos después de comer.",
                hashtags="#salud #habitos #nelhealthcoach",
            )

            with patch("telegram_bot.ensure_current_month_folder", return_value=MonthFolderResolution("m1", False)), patch(
                "telegram_bot.get_monthly_guidelines", return_value="PAUTAS"
            ), patch("telegram_bot.generate_video_script", return_value=fake_script), patch(
                "telegram_bot.append_or_create_scripts_doc",
                side_effect=DriveStorageQuotaError("Service Account sin cuota"),
            ):
                await bot_ctrl._generate_and_send_script("Caminar después de comer")

            sent_text = mock_tg_bot.send_message.call_args_list[0].kwargs["text"]
            self.assertIn("Caminar después de comer", sent_text)
            self.assertIn("¿Sabías esto de caminar 10 minutos?", sent_text)
            self.assertIn("No se pudo guardar automáticamente en Google Docs", sent_text)


if __name__ == "__main__":
    unittest.main()
