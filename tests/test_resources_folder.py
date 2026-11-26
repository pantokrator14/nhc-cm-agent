"""Tests for the universal `Guías PDF y recursos` Google Drive folder sync & notifications."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import config
import drive_service
from tests.test_bot_workflows import _make_bot


class TestResourcesFolderConfigAndDriveSync(unittest.TestCase):
    def test_load_settings_supports_optional_resources_folder_id_and_name(self) -> None:
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
                        "DRIVE_ROOT_FOLDER_ID=root123",
                        "DRIVE_RESOURCES_FOLDER_ID=res_folder_99",
                        "DRIVE_RESOURCES_FOLDER_NAME=Guías PDF y recursos",
                        "TELEGRAM_BOT_TOKEN=tg123",
                        "TELEGRAM_ADMIN_CHAT_ID=12345",
                        "META_ACCESS_TOKEN=meta123",
                        "IG_USER_ID=ig123",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                settings = config.load_settings(env_file)
            self.assertEqual(settings.drive_resources_folder_id, "res_folder_99")
            self.assertEqual(settings.drive_resources_folder_name, "Guías PDF y recursos")
            self.assertEqual(settings.website_url, "https://nelhealthcoach.com")

    def test_sync_resources_folder_detects_new_pdf_sets_public_read_and_avoids_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sa = tmp / "sa.json"
            sa.write_text('{"type": "service_account", "client_email": "sa@test.iam"}', encoding="utf-8")
            state_path = tmp / "state.json"

            mock_drive = MagicMock()
            # 1) resolve_resources_folder -> find inside root_folder_id (not found -> create)
            # 2) list files inside created folder -> returns 1 PDF
            mock_drive.files().list().execute.side_effect = [
                {"files": []},  # subfolder search in root
                {
                    "files": [
                        {
                            "id": "pdf_res_1",
                            "name": "Guía Glucosa 7 Días.pdf",
                            "mimeType": "application/pdf",
                            "size": "102400",
                            "webViewLink": "https://drive.google.com/file/d/pdf_res_1/view",
                        }
                    ]
                },
                # Second sync call: folder found, same PDF present
                {"files": [{"id": "res_folder_1", "name": "Guías PDF y recursos"}]},
                {
                    "files": [
                        {
                            "id": "pdf_res_1",
                            "name": "Guía Glucosa 7 Días.pdf",
                            "mimeType": "application/pdf",
                            "size": "102400",
                            "webViewLink": "https://drive.google.com/file/d/pdf_res_1/view",
                        }
                    ]
                },
            ]
            mock_drive.files().create().execute.return_value = {"id": "res_folder_1"}
            mock_drive.permissions().list().execute.return_value = {"permissions": []}
            mock_drive.permissions().create().execute.return_value = {"id": "perm_anyone_1"}
            mock_drive.files().get().execute.return_value = {
                "id": "pdf_res_1",
                "webViewLink": "https://drive.google.com/file/d/pdf_res_1/view",
            }
            mock_drive.files().get_media.return_value.execute.return_value = b"%PDF-fake"

            with patch.object(drive_service, "_get_drive_client", return_value=mock_drive), patch.object(
                drive_service,
                "_extract_pdf_text",
                return_value="Guía práctica para controlar picos de glucosa en tus comidas.",
            ):
                res1 = drive_service.sync_resources_folder(
                    root_folder_id="root_1",
                    resources_folder_name="Guías PDF y recursos",
                    state_path=state_path,
                    credentials_path=sa,
                )
                self.assertTrue(res1.folder_created)
                self.assertEqual(res1.folder_id, "res_folder_1")
                self.assertEqual(len(res1.newly_added), 1)
                self.assertEqual(res1.newly_added[0]["name"], "Guía Glucosa 7 Días.pdf")
                self.assertEqual(
                    res1.newly_added[0]["url"],
                    "https://drive.google.com/file/d/pdf_res_1/view",
                )
                self.assertIn("picos de glucosa", res1.newly_added[0]["preview_text"])

                # Second scan: same file is NOT reported as new again
                res2 = drive_service.sync_resources_folder(
                    root_folder_id="root_1",
                    resources_folder_name="Guías PDF y recursos",
                    state_path=state_path,
                    credentials_path=sa,
                )
                self.assertFalse(res2.folder_created)
                self.assertEqual(len(res2.newly_added), 0)
                self.assertEqual(res2.total_resources, 1)

            ctx = drive_service.format_resources_catalog_context(
                drive_service.get_resources_catalog(state_path)
            )
            self.assertIn("Guía Glucosa 7 Días.pdf", ctx)
            self.assertIn("https://drive.google.com/file/d/pdf_res_1/view", ctx)


class TestResourcesTelegramNotificationsAndCommand(unittest.IsolatedAsyncioTestCase):
    async def test_sync_and_notify_resources_notifies_admin_and_cmd_recursos_lists_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bot_ctrl, mock_tg_bot = _make_bot(tmp)

            sync_res = drive_service.ResourceSyncResult(
                folder_id="res_folder_1",
                folder_name="Guías PDF y recursos",
                folder_created=False,
                newly_added=[
                    {
                        "file_id": "pdf_1",
                        "name": "Guía Metabolismo.pdf",
                        "url": "https://drive.google.com/file/d/pdf_1/view",
                        "preview_text": "Guía sobre hábitos diarios para activar el metabolismo.",
                    }
                ],
                total_resources=1,
            )

            with patch("telegram_bot.sync_resources_folder", return_value=sync_res), patch(
                "telegram_bot.summarize_uploaded_resource",
                return_value=(
                    "Guía práctica de hábitos diarios para el metabolismo.",
                    "Ideal para posts de energía; palabra clave: METABOLISMO",
                ),
            ), patch(
                "telegram_bot.get_resources_catalog",
                return_value=[
                    {
                        "file_id": "pdf_1",
                        "name": "Guía Metabolismo.pdf",
                        "url": "https://drive.google.com/file/d/pdf_1/view",
                        "summary": "Guía práctica de hábitos diarios para el metabolismo.",
                    }
                ],
            ):
                await bot_ctrl.sync_and_notify_resources()

                self.assertEqual(mock_tg_bot.send_message.call_count, 2)  # 2 admins
                notif_text = mock_tg_bot.send_message.call_args_list[0].kwargs["text"]
                self.assertIn("Nuevo archivo subido", notif_text)
                self.assertIn("Guías PDF y recursos", notif_text)
                self.assertIn("Guía Metabolismo.pdf", notif_text)
                self.assertIn("https://drive.google.com/file/d/pdf_1/view", notif_text)
                self.assertIn("METABOLISMO", notif_text)

                mock_tg_bot.send_message.reset_mock()
                msg = MagicMock()
                msg.reply_text = AsyncMock()
                update = SimpleNamespace(
                    effective_chat=SimpleNamespace(id=1001),
                    effective_message=msg,
                )
                await bot_ctrl.cmd_recursos(update, SimpleNamespace())
                list_text = mock_tg_bot.send_message.call_args_list[-1].kwargs["text"]
                self.assertIn("Guía Metabolismo.pdf", list_text)
                self.assertIn("1 archivo(s) disponible(s)", list_text)


if __name__ == "__main__":
    unittest.main()
