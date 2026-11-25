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


class TestMetricsArgsAndDmPolling(unittest.IsolatedAsyncioTestCase):
    def test_resolve_metrics_reference_dt_supports_previous_and_specific_week(self) -> None:
        from drive_service import compute_month_week_info, resolve_metrics_reference_dt

        tz = timezone.utc
        tue_week4 = datetime(2026, 11, 24, 15, 0, tzinfo=tz)

        # `anterior` from week 4 (23-29 Nov) resolves to week 3 (16-22 Nov)
        prev_dt, prev_label = resolve_metrics_reference_dt(["anterior"], tue_week4)
        w_num, w_start, w_end = compute_month_week_info(prev_dt)
        self.assertEqual(w_num, 3)
        self.assertEqual((w_start.day, w_end.day), (16, 22))
        self.assertIn("Semana 3", prev_label)

        # `semana 3` or `3` resolves to week 3
        s3_dt, _ = resolve_metrics_reference_dt(["semana", "3"], tue_week4)
        self.assertEqual(compute_month_week_info(s3_dt)[0], 3)
        n3_dt, _ = resolve_metrics_reference_dt(["3"], tue_week4)
        self.assertEqual(compute_month_week_info(n3_dt)[0], 3)

        # Explicit date resolves to that week
        d_dt, _ = resolve_metrics_reference_dt(["2026-11-22"], tue_week4)
        self.assertEqual(compute_month_week_info(d_dt)[0], 3)

        # Invalid week raises ValueError
        with self.assertRaises(ValueError):
            resolve_metrics_reference_dt(["semana", "9"], tue_week4)

    async def test_cmd_metricas_accepts_previous_week_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bot_ctrl, _ = _make_bot(tmp)
            bot_ctrl._get_reference_now = MagicMock(
                return_value=datetime(2026, 11, 24, 15, 0, tzinfo=timezone.utc)
            )
            bot_ctrl.generate_and_save_weekly_report = AsyncMock()

            reply_msg = MagicMock()
            reply_msg.reply_text = AsyncMock()
            update = SimpleNamespace(
                effective_chat=SimpleNamespace(id=1001),
                effective_user=SimpleNamespace(first_name="Admin"),
                effective_message=reply_msg,
            )
            context = SimpleNamespace(args=["anterior"])

            await bot_ctrl.cmd_metricas(update, context)

            bot_ctrl.generate_and_save_weekly_report.assert_awaited_once()
            passed_dt = bot_ctrl.generate_and_save_weekly_report.call_args.kwargs["reference_dt"]
            self.assertEqual(passed_dt.day, 22)
            self.assertEqual(passed_dt.month, 11)
            self.assertIn("Semana 3", reply_msg.reply_text.call_args.args[0])

    async def test_cmd_mensajes_reports_meta_api_error_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bot_ctrl, _ = _make_bot(tmp)
            reply_msg = MagicMock()
            reply_msg.reply_text = AsyncMock()
            update = SimpleNamespace(
                effective_chat=SimpleNamespace(id=1001),
                effective_user=SimpleNamespace(first_name="Admin"),
                effective_message=reply_msg,
            )
            context = SimpleNamespace(args=[])

            with patch(
                "telegram_bot.fetch_instagram_dm_messages_with_status",
                AsyncMock(return_value=([], "Requires instagram_manage_messages (code=200)")),
            ):
                await bot_ctrl.cmd_mensajes(update, context)

            final_reply = reply_msg.reply_text.call_args_list[-1].args[0]
            self.assertIn("instagram_manage_messages", final_reply)

    async def test_meta_token_health_clears_stale_state_and_uses_new_env_token(self) -> None:
        from drive_service import get_effective_meta_token, load_state, save_meta_auth_state
        from meta_publisher import MetaTokenStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bot_ctrl, mock_tg_bot = _make_bot(tmp)
            state_path = tmp / "state.json"

            # Legacy stale state.json entry (old invalidated token without source_env_fp)
            save_meta_auth_state({"access_token": "old-revoked-tok", "expires_at": 0}, state_path)
            self.assertEqual(get_effective_meta_token("meta-tok", state_path), "old-revoked-tok")

            # First check fails on old token -> must clear state and retry with env token ("meta-tok")
            invalid_st = MetaTokenStatus(
                is_valid=False,
                is_permanent=False,
                expires_at=0,
                days_remaining=0,
                refreshed=False,
                active_token="old-revoked-tok",
                message="Invalid OAuth access token",
            )
            valid_st = MetaTokenStatus(
                is_valid=True,
                is_permanent=True,
                expires_at=0,
                days_remaining=None,
                refreshed=False,
                active_token="meta-tok",
                message="Token permanente activo",
            )

            with patch(
                "telegram_bot.check_and_refresh_meta_token",
                AsyncMock(side_effect=[invalid_st, valid_st]),
            ) as mock_check:
                res = await bot_ctrl.check_meta_token_health(notify_always=True)

            self.assertTrue(res.is_valid)
            self.assertEqual(mock_check.await_count, 2)
            self.assertEqual(mock_check.await_args_list[0].kwargs["access_token"], "old-revoked-tok")
            self.assertEqual(mock_check.await_args_list[1].kwargs["access_token"], "meta-tok")
            self.assertEqual(load_state(state_path)["meta_auth"], {})

            # When source_env_fp mismatches new .env token, get_effective_meta_token picks new env token immediately
            save_meta_auth_state(
                {
                    "access_token": "extended-old-tok",
                    "expires_at": 9999999999,
                    "source_env_fp": "different_fp_000",
                },
                state_path,
            )
            self.assertEqual(get_effective_meta_token("brand-new-env-tok", state_path), "brand-new-env-tok")

    def test_settings_defaults_to_venezuela_timezone_america_caracas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sa = tmp / "sa.json"
            sa.write_text("{}", encoding="utf-8")
            s = _make_settings(tmp, sa)
            self.assertEqual(s.timezone_name, "America/Caracas")
            now_ve = datetime.now(s.tzinfo)
            self.assertEqual(now_ve.utcoffset().total_seconds(), -4 * 3600)

            from drive_service import parse_guidelines_entries

            entries = parse_guidelines_entries(
                "=== PAUTAS DEL MES ===\n"
                "8. Orinar\n"
                "Fecha: Miércoles 25/11/2026\n"
                "Horario: 19:00 hrs\n"
                "Pauta en Meta Ads: Orgánico.\n"
                "Título / Gancho: Test\n"
                "Caption:\nHola\n"
                "Hashtags: #salud\n",
                default_tz=s.tzinfo,
            )
            self.assertEqual(len(entries), 1)
            self.assertIsNotNone(entries[0].scheduled_dt)
            self.assertEqual(entries[0].scheduled_dt.hour, 19)
            self.assertEqual(entries[0].scheduled_dt.utcoffset().total_seconds(), -4 * 3600)

    async def test_fetch_dm_messages_resolves_linked_page_and_skips_answered_threads(self) -> None:
        from meta_publisher import fetch_instagram_dm_messages_with_status

        def _fake_resp(status: int, payload: dict) -> MagicMock:
            resp = MagicMock()
            resp.status_code = status
            resp.is_error = status >= 400
            resp.json.return_value = payload
            resp.text = json.dumps(payload)
            return resp

        async def _mock_get(url: str, params: dict | None = None):
            if "/ig-123/conversations" in url or "/me/conversations" in url:
                return _fake_resp(400, {"error": {"message": "Unsupported get request", "code": 100}})
            if "/me/accounts" in url:
                return _fake_resp(
                    200,
                    {
                        "data": [
                            {
                                "id": "page-999",
                                "name": "Nel Health Coach",
                                "access_token": "page-tok",
                                "instagram_business_account": {"id": "ig-123"},
                            }
                        ]
                    },
                )
            if "/page-999/conversations" in url:
                return _fake_resp(
                    200,
                    {
                        "data": [
                            {
                                "id": "conv_unanswered",
                                "participants": {"data": [{"id": "user-1", "username": "maria"}]},
                                "messages": {
                                    "data": [
                                        {
                                            "id": "mid_new",
                                            "created_time": "2026-11-24T14:00:00+0000",
                                            "from": {"id": "user-1", "username": "maria"},
                                            "message": "Hola, ¿info del programa?",
                                        }
                                    ]
                                },
                            },
                            {
                                "id": "conv_answered",
                                "participants": {"data": [{"id": "user-2", "username": "ana"}]},
                                "messages": {
                                    "data": [
                                        {
                                            "id": "mid_own",
                                            "created_time": "2026-11-24T13:00:00+0000",
                                            "from": {"id": "ig-123", "username": "nelhealthcoach"},
                                            "message": "¡Claro! Te escribo.",
                                        },
                                        {
                                            "id": "mid_old_user",
                                            "created_time": "2026-11-24T12:00:00+0000",
                                            "from": {"id": "user-2", "username": "ana"},
                                            "message": "Pregunta ya respondida",
                                        },
                                    ]
                                },
                            },
                        ]
                    },
                )
            raise AssertionError(f"Unexpected URL: {url}")

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=_mock_get)
        mock_client_cm = MagicMock()
        mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("meta_publisher.httpx.AsyncClient", return_value=mock_client_cm):
            msgs, err = await fetch_instagram_dm_messages_with_status(
                ig_user_id="ig-123",
                access_token="user-tok",
                since_minutes=4320,
            )

        self.assertIsNone(err)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["event_id"], "mid_new")
        self.assertEqual(msgs[0]["sender_username"], "maria")


if __name__ == "__main__":
    unittest.main()
