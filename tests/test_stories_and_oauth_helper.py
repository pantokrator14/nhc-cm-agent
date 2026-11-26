"""Strict tests for Instagram Story detection/normalization and OAuth setup helper."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from google.auth.exceptions import RefreshError

import drive_service
import gemini_agent
from scripts import google_oauth_setup
from tests.test_bot_workflows import _make_bot


class TestStoryDetectionAndNormalization(unittest.TestCase):
    def test_is_story_filename_matches_explicit_story_names_and_avoids_false_positives(self) -> None:
        self.assertTrue(gemini_agent.is_story_filename("historia ORINAR.mp4"))
        self.assertTrue(gemini_agent.is_story_filename("historia SUEN\u0303O.mp4"))
        self.assertTrue(gemini_agent.is_story_filename("historia DISCIPLINA.mp4"))
        self.assertTrue(gemini_agent.is_story_filename("(HISTORIA) Desayuno.jpg"))
        self.assertTrue(gemini_agent.is_story_filename("12. Historia dormir.mp4"))
        self.assertTrue(gemini_agent.is_story_filename("story_ayuno.mp4"))

        self.assertFalse(gemini_agent.is_story_filename("14. orinar.mp4"))
        self.assertFalse(gemini_agent.is_story_filename("La historia del ayuno.mp4"))
        self.assertFalse(gemini_agent.is_story_filename("03. Protip Atención.mp4"))

    def test_normalize_proposal_forces_stories_even_with_presenter_on_camera(self) -> None:
        raw_proposal = gemini_agent.PostProposal(
            content_summary="Manuel hablando en formato historia sobre la hidratación.",
            has_presenter_on_camera=True,
            post_category="PRO_TIP",
            short_title="Historia Orinar",
            hook_title="Dato rápido del día",
            caption="Toma agua al despertar #salud #agua #bienestar",
            hashtags="#salud #agua #bienestar #habitos #nelhealthcoach",
            trend_alignment="Formato cercano tipo historia.",
            media_type="REELS",  # Simula el error reportado: Gemini propuso REELS
            scheduled_time="2026-11-24T10:00:00-04:00",
            schedule_reasoning="Mañana para historia.",
            promote_in_ads=True,
            ads_daily_budget_usd=5.0,
            ads_duration_days=5,
            ads_recommendation="Promocionar.",
            has_keyword_cta=False,
            keyword="",
            target_link="",
            public_reply_text="",
            publish_to_tiktok=True,
            tiktok_caption="Para TikTok",
        )
        file_meta = {
            "id": "file_hist_1",
            "name": "historia ORINAR.mp4",
            "mimeType": "video/mp4",
        }
        normalized = gemini_agent._normalize_proposal(
            raw_proposal,
            file_meta,
            datetime(2026, 11, 23, 19, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(normalized.media_type, "STORIES")
        self.assertEqual(normalized.post_category, "MAIN")
        self.assertEqual(normalized.short_title, "Orinar")
        self.assertFalse(normalized.promote_in_ads)
        self.assertEqual(normalized.ads_daily_budget_usd, 0.0)
        self.assertEqual(normalized.ads_duration_days, 0)
        self.assertFalse(normalized.publish_to_tiktok)
        self.assertEqual(normalized.tiktok_caption, "")
        self.assertEqual(normalized.hashtags, "")
        self.assertEqual(normalized.caption, "Toma agua al despertar")

    def test_compute_next_numbering_and_filename_for_story(self) -> None:
        mock_drive = MagicMock()
        mock_drive.files().list().execute.return_value = {
            "files": [
                {"id": "1", "name": "16. Meditar.mp4", "mimeType": "video/mp4"},
            ]
        }
        guidelines = "10. Meditar\nHistoria 1: Sueño\n"
        with patch.object(drive_service, "_get_drive_client", return_value=mock_drive):
            header, filename = drive_service.compute_next_numbering_and_filename(
                month_folder_id="month_1",
                guidelines_text=guidelines,
                post_category="MAIN",
                short_title="Historia Orinar",
                original_filename="historia ORINAR.mp4",
                is_story=True,
            )
        self.assertEqual(header, "Historia 2: Orinar")
        self.assertEqual(filename, "17. Historia Orinar.mp4")
        self.assertEqual(
            drive_service._match_entry_header("Historia 2: Orinar"),
            ("PRO_TIP", "Orinar"),
        )


class TestStoryTelegramDisplayAndBuild(unittest.TestCase):
    def test_format_session_html_and_build_guideline_for_story(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bot_ctrl, _ = _make_bot(Path(tmpdir))
            entry = drive_service.GuidelineEntry(
                header="Historia 1: Orinar",
                post_category="PRO_TIP",
                short_title="Orinar",
                fecha_raw="24/11/2026",
                horario_raw="10:00 a.m.",
                scheduled_dt=datetime(2026, 11, 24, 10, 0, tzinfo=timezone.utc),
                ads_raw="No.",
                promote_in_ads=False,
                ads_daily_budget_usd=0.0,
                ads_duration_days=0,
                hook_title="Escucha tu cuerpo",
                caption="Historia rápida.",
                hashtags="",
                has_keyword_cta=False,
                keyword="",
                target_link="",
                raw_block="",
            )
            item = drive_service.UnpublishedGuidelineItem(
                entry=entry,
                drive_file={"id": "h1", "name": "historia ORINAR.mp4", "mimeType": "video/mp4"},
                local_status="IN_DRIVE",
                is_overdue=False,
                has_active_job=False,
            )
            session = bot_ctrl._build_session_from_guideline(item, "month_1", "")
            self.assertEqual(session.proposal.media_type, "STORIES")
            self.assertFalse(session.proposal.publish_to_tiktok)

            card_html = bot_ctrl._format_session_html(session)
            self.assertIn("Historia IG (24h)", card_html)
            self.assertIn("STORIES", card_html)
            self.assertIn("las Historias de Instagram no se publican en TikTok", card_html)
            self.assertIn("Nota / Comentario opcional", card_html)
            self.assertNotIn("Caption sugerido", card_html)
            self.assertNotIn("Ubicación (IG)", card_html)

            story_lines = drive_service.format_guideline_entry_lines(
                doc_header="Historia 1: Orinar",
                scheduled_dt=datetime(2026, 11, 24, 10, 0, tzinfo=timezone.utc),
                promote_in_ads=False,
                ads_recommendation="No aplica",
                hook_title="Escucha tu cuerpo",
                caption="Sticker de encuesta en pantalla",
                hashtags="",
                is_story=True,
            )
            self.assertFalse(any(l.startswith("Caption:") for l in story_lines))
            self.assertFalse(any(l.startswith("Hashtags:") for l in story_lines))
            self.assertIn("Nota / Comentario: Sticker de encuesta en pantalla", story_lines)


class TestGoogleOAuthSetupHelper(unittest.TestCase):
    def test_load_client_credentials_and_extract_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "client.json"
            p.write_text(
                json.dumps(
                    {
                        "web": {
                            "client_id": "web-id.apps.googleusercontent.com",
                            "client_secret": "web-sec",
                        }
                    }
                ),
                encoding="utf-8",
            )
            cid, csec, kind = google_oauth_setup.load_client_credentials(p)
            self.assertEqual((cid, csec, kind), ("web-id.apps.googleusercontent.com", "web-sec", "web"))

        url = google_oauth_setup.build_authorization_url("cid123", "http://localhost:8765")
        self.assertIn("access_type=offline", url)
        self.assertIn("prompt=consent", url)
        self.assertIn("drive", url)
        self.assertIn("documents", url)

        code = google_oauth_setup.extract_code_from_input(
            "http://localhost:8765/?code=4/0AcvD&scope=https://www.googleapis.com/auth/drive"
        )
        self.assertEqual(code, "4/0AcvD")


class TestGoogleOAuthHealthReminder(unittest.IsolatedAsyncioTestCase):
    def test_check_google_oauth_token_health_warns_at_5_days_and_on_refresh_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            oauth_file = tmp / "oauth_token.json"
            old_ts = (datetime.now(timezone.utc) - timedelta(days=5, hours=12)).isoformat()
            oauth_file.write_text(
                json.dumps(
                    {
                        "type": "authorized_user",
                        "client_id": "cid",
                        "client_secret": "csec",
                        "refresh_token": "rtok",
                        "created_at": old_ts,
                    }
                ),
                encoding="utf-8",
            )
            mock_creds = MagicMock()
            with patch.dict(os.environ, {"GOOGLE_OAUTH_CREDENTIALS": str(oauth_file)}, clear=False), patch.object(
                drive_service.oauth2_credentials.Credentials,
                "from_authorized_user_file",
                return_value=mock_creds,
            ):
                status = drive_service.check_google_oauth_token_health(warn_age_days=5.0)
            self.assertTrue(status.enabled)
            self.assertTrue(status.is_valid)
            self.assertTrue(status.needs_renewal)
            self.assertEqual(status.days_until_7d_expiry, 1)

            # Simulate expired / revoked token
            mock_creds.refresh.side_effect = RefreshError("invalid_grant: Token has been expired or revoked.")
            with patch.dict(os.environ, {"GOOGLE_OAUTH_CREDENTIALS": str(oauth_file)}, clear=False), patch.object(
                drive_service.oauth2_credentials.Credentials,
                "from_authorized_user_file",
                return_value=mock_creds,
            ):
                bad_status = drive_service.check_google_oauth_token_health(warn_age_days=5.0)
            self.assertFalse(bad_status.is_valid)
            self.assertTrue(bad_status.needs_renewal)
            self.assertIn("invalid_grant", bad_status.message)

    async def test_bot_check_google_oauth_health_sends_developer_reminder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bot_ctrl, mock_tg_bot = _make_bot(Path(tmpdir))
            warn_status = drive_service.GoogleOAuthStatus(
                enabled=True,
                is_valid=True,
                age_days=5.4,
                days_until_7d_expiry=1,
                needs_renewal=True,
                token_path="credentials/oauth_token.json",
                message="Token OAuth de Google activo y válido.",
            )
            with patch("telegram_bot.check_google_oauth_token_health", return_value=warn_status):
                await bot_ctrl.check_google_oauth_health()

            self.assertEqual(mock_tg_bot.send_message.call_count, 2)
            text = mock_tg_bot.send_message.call_args_list[0].kwargs["text"]
            self.assertIn("RECORDATORIO", text)
            self.assertIn("scripts/google_oauth_setup.py", text)
            self.assertIn("ceo@nelhealthcoach.com", text)
            self.assertIn("systemctl restart nel-agent", text)


if __name__ == "__main__":
    unittest.main()
