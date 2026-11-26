"""Tests for editable DM messages, webhook rendering without links, and DM edit learning."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import drive_service
import gemini_agent
import instagram_webhook
from tests.test_bot_workflows import _make_bot, _make_settings


def _sample_proposal(**overrides) -> gemini_agent.PostProposal:
    base = dict(
        content_summary="Manuel explica cómo reducir picos de glucosa.",
        has_presenter_on_camera=True,
        post_category="MAIN",
        short_title="Control de glucosa",
        hook_title="El hábito de 10 minutos tras comer",
        caption=(
            "¿Sientes cansancio después de almorzar?\n\n"
            "Caminar 10 minutos mejora la sensibilidad a la insulina.\n\n"
            "Comenta GLUCOSA para conversar por privado y guarda este video.\n\n"
            "Recuerda: tu comida, tu mejor medicina."
        ),
        hashtags="#SaludMetabolica #Glucosa #HabitosSaludables #Bienestar #NelHealthCoach",
        trend_alignment="Gancho directo con micro-hábito accionable.",
        media_type="REELS",
        scheduled_time="2026-11-28T18:00:00-04:00",
        schedule_reasoning="Horario ideal tras la tarde laboral.",
        promote_in_ads=False,
        ads_daily_budget_usd=0.0,
        ads_duration_days=0,
        ads_recommendation="Orgánico.",
        has_keyword_cta=True,
        keyword="GLUCOSA",
        target_link="",
        dm_message="¡Hola! 💚 Gracias por comentar GLUCOSA. ¿Qué horario te cuesta más cuidar en tus comidas?",
        public_reply_text="¡Te escribí al privado! 💚",
        publish_to_tiktok=True,
        tiktok_caption="Caminar 10 minutos tras comer cambia tu energía.",
    )
    base.update(overrides)
    return gemini_agent.PostProposal(**base)


class TestDmMessageNormalizationAndDrivePersistence(unittest.TestCase):
    def test_normalize_proposal_enforces_website_cta_without_pdf_and_with_pdf(self) -> None:
        raw_no_pdf = _sample_proposal(dm_message="", target_link="")
        file_meta = {"id": "f1", "name": "glucosa.mp4", "mimeType": "video/mp4"}
        norm_no_pdf = gemini_agent._normalize_proposal(
            raw_no_pdf,
            file_meta,
            datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc),
        )
        self.assertTrue(norm_no_pdf.has_keyword_cta)
        self.assertIn("GLUCOSA", norm_no_pdf.dm_message)
        self.assertEqual(norm_no_pdf.target_link, "https://nelhealthcoach.com")
        self.assertTrue(norm_no_pdf.dm_message.endswith("https://nelhealthcoach.com"))

        raw_with_pdf = _sample_proposal(
            dm_message="¡Hola! 💚 Aquí tienes tu guía en PDF.",
            target_link="https://drive.google.com/file/d/pdf123/view",
        )
        norm_with_pdf = gemini_agent._normalize_proposal(
            raw_with_pdf,
            file_meta,
            datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(norm_with_pdf.target_link, "https://drive.google.com/file/d/pdf123/view")
        self.assertIn("https://drive.google.com/file/d/pdf123/view", norm_with_pdf.dm_message)
        self.assertIn("https://nelhealthcoach.com", norm_with_pdf.dm_message)

        raw_disabled = _sample_proposal(has_keyword_cta=False, keyword="GLUCOSA", dm_message="Hola")
        norm_disabled = gemini_agent._normalize_proposal(
            raw_disabled,
            file_meta,
            datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc),
        )
        self.assertFalse(norm_disabled.has_keyword_cta)
        self.assertEqual(norm_disabled.dm_message, "")

    def test_guideline_entry_roundtrips_dm_message_and_public_reply(self) -> None:
        lines = drive_service.format_guideline_entry_lines(
            doc_header="11. Control de glucosa",
            scheduled_dt=datetime(2026, 11, 28, 18, 0, tzinfo=timezone.utc),
            promote_in_ads=False,
            ads_recommendation="Orgánico.",
            hook_title="El hábito de 10 minutos",
            caption="Cuerpo del caption.\n\nRecuerda: tu comida, tu mejor medicina.",
            hashtags="#Salud #Glucosa #Bienestar #Habitos #NelHealthCoach",
            has_keyword_cta=True,
            keyword="GLUCOSA",
            target_link="",
            dm_message="¡Hola! Gracias por comentar GLUCOSA.\n¿Quieres que revisemos tu rutina?",
            public_reply_text="¡Revisa tu DM! 💚",
        )
        doc_text = "=== PAUTAS DEL MES ===\n\n" + "\n".join(lines) + "\n"
        entries = drive_service.parse_guidelines_entries(doc_text)
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0].has_keyword_cta)
        self.assertEqual(entries[0].keyword, "GLUCOSA")
        self.assertEqual(entries[0].public_reply_text, "¡Revisa tu DM! 💚")
        self.assertEqual(
            entries[0].dm_message,
            "¡Hola! Gracias por comentar GLUCOSA.\n¿Quieres que revisemos tu rutina?",
        )

    def test_dm_learning_memory_upserts_by_proposal_and_formats_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            drive_service.record_dm_learning_example(
                proposal_id="prop_1",
                previous_dm="Borrador inicial IA",
                updated_dm="Primera edición de Manuel",
                feedback="Hazlo más cercano",
                keyword="GLUCOSA",
                post_title="Control de glucosa",
                state_path=state_path,
            )
            drive_service.record_dm_learning_example(
                proposal_id="prop_1",
                previous_dm="Primera edición de Manuel",
                updated_dm="Versión final aprobada por Manuel 💚",
                feedback="Sin enlaces, pregunta directa",
                keyword="GLUCOSA",
                post_title="Control de glucosa",
                state_path=state_path,
            )
            mem = drive_service.get_dm_learning_memory(state_path)
            self.assertEqual(len(mem["examples"]), 1)
            ex = mem["examples"][0]
            self.assertEqual(ex["initial_dm"], "Borrador inicial IA")
            self.assertEqual(ex["approved_dm"], "Versión final aprobada por Manuel 💚")
            self.assertEqual(len(ex["feedback_notes"]), 2)

            ctx = drive_service.format_dm_learning_context(mem)
            self.assertIn("Versión final aprobada por Manuel 💚", ctx)
            self.assertIn("GLUCOSA", ctx)


class TestWebhookCustomDmSending(unittest.IsolatedAsyncioTestCase):
    def test_render_dm_message_handles_custom_text_placeholders_and_legacy_fallback(self) -> None:
        rendered = instagram_webhook._render_dm_message(
            dm_message="¡Hola {username}! Gracias por comentar GLUCOSA.",
            keyword="GLUCOSA",
            target_link="",
            username="ana_fit",
            website_url="https://nelhealthcoach.com",
        )
        self.assertIn("¡Hola @ana_fit! Gracias por comentar GLUCOSA.", rendered)
        self.assertTrue(rendered.endswith("https://nelhealthcoach.com"))

        rendered_no_user = instagram_webhook._render_dm_message(
            dm_message="¡Hola {username}! Gracias por comentar GLUCOSA.",
            keyword="GLUCOSA",
            target_link="https://drive.google.com/file/d/guia/view",
            username="",
            website_url="https://nelhealthcoach.com",
        )
        self.assertIn("¡Hola! Gracias por comentar GLUCOSA.", rendered_no_user)
        self.assertIn("https://drive.google.com/file/d/guia/view", rendered_no_user)
        self.assertIn("https://nelhealthcoach.com", rendered_no_user)

        legacy = instagram_webhook._render_dm_message(
            dm_message="",
            keyword="PLAN",
            target_link="https://drive.google.com/file/d/plan/view",
            username="luis",
            website_url="https://nelhealthcoach.com",
        )
        self.assertIn("Gracias por comentar PLAN", legacy)
        self.assertIn("https://drive.google.com/file/d/plan/view", legacy)
        self.assertIn("https://nelhealthcoach.com", legacy)

    async def test_process_comment_event_sends_custom_dm_even_without_target_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sa = tmp / "sa.json"
            sa.write_text("{}", encoding="utf-8")
            settings = _make_settings(tmp, sa)

            drive_service.save_published_media(
                media_id="media_99",
                has_keyword_cta=True,
                keyword="GLUCOSA",
                target_link="",
                public_reply_text="¡Te escribí por DM! 💚",
                dm_message="¡Hola {username}! 💚 Cuéntame cuál es tu duda sobre la glucosa.",
                file_id="f99",
                state_path=settings.state_file_path,
            )

            with patch.object(
                instagram_webhook,
                "_send_private_dm",
                new=AsyncMock(return_value=True),
            ) as mock_dm, patch.object(
                instagram_webhook,
                "_send_public_reply",
                new=AsyncMock(return_value=True),
            ) as mock_reply:
                await instagram_webhook._process_comment_event(
                    settings=settings,
                    comment_id="cmt_123",
                    media_id="media_99",
                    comment_text="Quiero info GLUCOSA por favor",
                    username="maria_salud",
                    from_id="user_55",
                )

            mock_dm.assert_awaited_once()
            self.assertIn(
                "¡Hola @maria_salud! 💚 Cuéntame cuál es tu duda sobre la glucosa.",
                mock_dm.await_args.kwargs["message_text"],
            )
            self.assertIn("https://nelhealthcoach.com", mock_dm.await_args.kwargs["message_text"])
            mock_reply.assert_awaited_once()
            self.assertTrue(drive_service.is_comment_processed("cmt_123", settings.state_file_path))


class TestTelegramDmDisplayEditAndLearningLoop(unittest.IsolatedAsyncioTestCase):
    async def test_edit_dm_button_updates_proposal_records_learning_and_feeds_next_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bot_ctrl, mock_tg_bot = _make_bot(tmp)
            proposal = _sample_proposal()
            from telegram_bot import ProposalSession

            session = ProposalSession(
                proposal_id="p_dm_1",
                month_folder_id="month_1",
                file_metadata={"id": "f1", "name": "glucosa.mp4", "mimeType": "video/mp4"},
                guidelines_text="=== PAUTAS DEL MES ===",
                proposal=proposal,
                initial_dm_message=proposal.dm_message,
            )
            bot_ctrl.pending_proposals["p_dm_1"] = session

            # Card HTML shows the DM message and keyboard includes the edit DM button
            card_html = bot_ctrl._format_session_html(session)
            self.assertIn("Mensaje DM automático:", card_html)
            self.assertIn("¿Qué horario te cuesta más cuidar en tus comidas?", card_html)
            kb = bot_ctrl._build_session_keyboard(session)
            cb_data = [btn.callback_data for row in kb.inline_keyboard for btn in row]
            self.assertIn("edit_dm:p_dm_1", cb_data)

            # Admin taps "✉️ Editar mensaje DM"
            query = MagicMock()
            query.data = "edit_dm:p_dm_1"
            query.answer = AsyncMock()
            query.message = MagicMock()
            query.message.reply_text = AsyncMock()
            update_cb = SimpleNamespace(
                effective_chat=SimpleNamespace(id=1001),
                effective_user=SimpleNamespace(id=1001, username="manuel", full_name="Manuel"),
                callback_query=query,
            )
            await bot_ctrl.handle_callback(update_cb, SimpleNamespace())
            self.assertEqual(bot_ctrl.awaiting_dm_edit_by_chat.get(1001), "p_dm_1")

            # Admin replies with exact DM text
            msg = MagicMock()
            msg.text = "¡Hola! Qué gusto saludarte 💚 ¿En qué comida del día sientes más ansiedad por dulce?"
            msg.reply_text = AsyncMock()
            update_txt = SimpleNamespace(
                effective_chat=SimpleNamespace(id=1001),
                effective_user=SimpleNamespace(id=1001, username="manuel", full_name="Manuel"),
                effective_message=msg,
            )
            await bot_ctrl.handle_admin_text(update_txt, SimpleNamespace())

            self.assertIn(
                "¡Hola! Qué gusto saludarte 💚 ¿En qué comida del día sientes más ansiedad por dulce?",
                session.proposal.dm_message,
            )
            self.assertIn("https://nelhealthcoach.com", session.proposal.dm_message)
            mem = drive_service.get_dm_learning_memory(bot_ctrl.settings.state_file_path)
            self.assertEqual(len(mem["examples"]), 1)
            self.assertIn("ansiedad por dulce", mem["examples"][0]["approved_dm"])

            # Subsequent proposal generation passes learned DM style context to Gemini
            captured_dm_ctx: list[str] = []

            def _fake_generate(*args, **kwargs):
                # 11th positional arg or keyword arg `dm_learning_context`
                dm_ctx = kwargs.get("dm_learning_context")
                if dm_ctx is None and len(args) >= 11:
                    dm_ctx = args[10]
                captured_dm_ctx.append(str(dm_ctx or ""))
                return _sample_proposal(short_title="Nuevo video")

            with patch("telegram_bot.fetch_social_trends_brief", return_value="Tendencia"), patch(
                "telegram_bot.download_drive_file_to_temp",
                side_effect=RuntimeError("skip download"),
            ), patch(
                "telegram_bot.generate_post_proposal",
                side_effect=_fake_generate,
            ):
                await bot_ctrl.propose_pending_files(
                    month_folder_id="month_1",
                    pending_files=[{"id": "f2", "name": "nuevo.mp4", "mimeType": "video/mp4"}],
                    guidelines_text="=== PAUTAS DEL MES ===",
                )

            self.assertEqual(len(captured_dm_ctx), 1)
            self.assertIn("ansiedad por dulce", captured_dm_ctx[0])

    async def test_rejecting_suggested_pdf_removes_pdf_link_and_keeps_website_cta(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bot_ctrl, _ = _make_bot(tmp)
            pdf_url = "https://drive.google.com/file/d/pdf_guia_1/view"
            proposal = _sample_proposal(
                target_link=pdf_url,
                dm_message=(
                    "¡Hola! 💚 Aquí tienes tu guía en PDF:\n"
                    f"{pdf_url}\n\n"
                    "✨ Te invito a visitar nuestra página:\nhttps://nelhealthcoach.com"
                ),
            )
            from telegram_bot import ProposalSession

            session = ProposalSession(
                proposal_id="p_no_pdf",
                month_folder_id="month_1",
                file_metadata={"id": "f1", "name": "glucosa.mp4", "mimeType": "video/mp4"},
                guidelines_text="=== PAUTAS DEL MES ===",
                proposal=proposal,
                initial_dm_message=proposal.dm_message,
            )
            bot_ctrl.pending_proposals["p_no_pdf"] = session
            bot_ctrl.awaiting_dm_edit_by_chat[1001] = "p_no_pdf"

            msg = MagicMock()
            msg.text = "¡Hola! 💚 Gracias por comentar GLUCOSA. Cuéntame qué hábito quieres mejorar."
            msg.reply_text = AsyncMock()
            update_txt = SimpleNamespace(
                effective_chat=SimpleNamespace(id=1001),
                effective_user=SimpleNamespace(id=1001, username="manuel", full_name="Manuel"),
                effective_message=msg,
            )
            await bot_ctrl.handle_admin_text(update_txt, SimpleNamespace())

            self.assertEqual(session.proposal.target_link, "https://nelhealthcoach.com")
            self.assertNotIn(pdf_url, session.proposal.dm_message)
            self.assertIn("https://nelhealthcoach.com", session.proposal.dm_message)

            # Also verify revise_post_proposal deterministic PDF-rejection guard
            fake_rev = gemini_agent.RevisionResult(
                ai_assessment="ALTERNATIVE_SUGGESTED",
                reasoning_message="Prueba",
                proposal=proposal,  # Simula IA olvidando limpiar el PDF
            )
            fake_resp = SimpleNamespace(parsed=fake_rev, text="")
            fake_client = MagicMock()
            fake_client.models.generate_content.return_value = fake_resp
            with patch.object(
                gemini_agent,
                "_resolve_client_and_model",
                return_value=(fake_client, "gemini-2.5-flash"),
            ):
                res = gemini_agent.revise_post_proposal(
                    guidelines_text="=== PAUTAS ===",
                    file_metadata={"id": "f1", "name": "glucosa.mp4", "mimeType": "video/mp4"},
                    previous_proposal=proposal,
                    feedback="No envíes esa guía PDF en este post",
                )
            self.assertEqual(res.ai_assessment, "ACCEPTED")
            self.assertEqual(res.proposal.target_link, "https://nelhealthcoach.com")
            self.assertNotIn(pdf_url, res.proposal.dm_message)
            self.assertIn("https://nelhealthcoach.com", res.proposal.dm_message)


if __name__ == "__main__":
    unittest.main()
