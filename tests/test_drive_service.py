"""Strict unit and workflow tests for Google Drive auth, Doc creation/fallback, and metrics reporting."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import httplib2
from googleapiclient.errors import HttpError

import drive_service


def _make_http_error(status: int, reason: str, message: str) -> HttpError:
    resp = httplib2.Response({"status": str(status)})
    payload = json.dumps(
        {
            "error": {
                "code": status,
                "message": message,
                "errors": [
                    {
                        "message": message,
                        "domain": "usageLimits" if status == 403 else "global",
                        "reason": reason,
                    }
                ],
            }
        }
    ).encode("utf-8")
    return HttpError(resp=resp, content=payload)


def _write_sa_json(directory: Path, name: str = "sa.json") -> Path:
    path = directory / name
    path.write_text(
        json.dumps(
            {
                "type": "service_account",
                "project_id": "test-proj",
                "private_key_id": "key1",
                "private_key": "-----BEGIN PRIVATE KEY-----\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8wggE7AgEAAkEA\n-----END PRIVATE KEY-----\n",
                "client_email": "nhc-rrss-agent@test-proj.iam.gserviceaccount.com",
                "client_id": "1234567890",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_oauth_json(directory: Path, name: str = "oauth.json") -> Path:
    path = directory / name
    path.write_text(
        json.dumps(
            {
                "type": "authorized_user",
                "client_id": "client-id.apps.googleusercontent.com",
                "client_secret": "client-secret",
                "refresh_token": "1//refresh-token",
            }
        ),
        encoding="utf-8",
    )
    return path


class TestDriveCredentialsAndQuota(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.sa_path = _write_sa_json(self.tmp)
        self.oauth_path = _write_oauth_json(self.tmp)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_load_google_credentials_default_service_account(self) -> None:
        env = {"GOOGLE_OAUTH_CREDENTIALS": "", "GOOGLE_DRIVE_IMPERSONATE_EMAIL": ""}
        with patch.dict(os.environ, env, clear=False), patch.object(
            drive_service.service_account.Credentials,
            "from_service_account_file",
            return_value="sa_creds",
        ) as mock_sa:
            creds = drive_service._load_google_credentials(
                drive_service.DRIVE_SCOPES, self.sa_path
            )
            self.assertEqual(creds, "sa_creds")
            mock_sa.assert_called_once_with(
                str(self.sa_path.resolve()),
                scopes=drive_service.DRIVE_SCOPES,
            )

    def test_load_google_credentials_with_impersonation(self) -> None:
        env = {
            "GOOGLE_OAUTH_CREDENTIALS": "",
            "GOOGLE_DRIVE_IMPERSONATE_EMAIL": "ceo@nelhealthcoach.com",
        }
        with patch.dict(os.environ, env, clear=False), patch.object(
            drive_service.service_account.Credentials,
            "from_service_account_file",
            return_value="dwd_creds",
        ) as mock_sa:
            creds = drive_service._load_google_credentials(
                drive_service.DOCS_SCOPES, self.sa_path
            )
            self.assertEqual(creds, "dwd_creds")
            mock_sa.assert_called_once_with(
                str(self.sa_path.resolve()),
                scopes=drive_service.DOCS_SCOPES,
                subject="ceo@nelhealthcoach.com",
            )

    def test_load_google_credentials_prefers_oauth_env_over_sa_path(self) -> None:
        env = {
            "GOOGLE_OAUTH_CREDENTIALS": str(self.oauth_path),
            "GOOGLE_DRIVE_IMPERSONATE_EMAIL": "ceo@nelhealthcoach.com",
        }
        with patch.dict(os.environ, env, clear=False), patch.object(
            drive_service.oauth2_credentials.Credentials,
            "from_authorized_user_file",
            return_value="oauth_creds",
        ) as mock_oauth:
            creds = drive_service._load_google_credentials(
                drive_service.DRIVE_SCOPES, self.sa_path
            )
            self.assertEqual(creds, "oauth_creds")
            mock_oauth.assert_called_once_with(
                str(self.oauth_path.resolve()),
                scopes=drive_service.DRIVE_SCOPES,
            )

    def test_load_google_credentials_direct_authorized_user_file(self) -> None:
        env = {"GOOGLE_OAUTH_CREDENTIALS": "", "GOOGLE_DRIVE_IMPERSONATE_EMAIL": ""}
        with patch.dict(os.environ, env, clear=False), patch.object(
            drive_service.oauth2_credentials.Credentials,
            "from_authorized_user_file",
            return_value="direct_oauth_creds",
        ) as mock_oauth:
            creds = drive_service._load_google_credentials(
                drive_service.DRIVE_SCOPES, self.oauth_path
            )
            self.assertEqual(creds, "direct_oauth_creds")
            mock_oauth.assert_called_once_with(
                str(self.oauth_path.resolve()),
                scopes=drive_service.DRIVE_SCOPES,
            )

    def test_create_or_update_google_doc_raises_storage_quota_error(self) -> None:
        mock_service = MagicMock()
        mock_service.files().list().execute.return_value = {"files": []}
        quota_exc = _make_http_error(
            403,
            "storageQuotaExceeded",
            "Service Accounts do not have storage quota. Leverage shared drives, or use OAuth delegation instead.",
        )
        mock_service.files().create().execute.side_effect = quota_exc

        with patch.object(drive_service, "_get_drive_client", return_value=mock_service):
            with self.assertRaises(drive_service.DriveStorageQuotaError) as ctx:
                drive_service.create_or_update_google_doc(
                    folder_id="18__8pcfrbQDTSldqYTAX6MSjoosRy4aT",
                    doc_title="Informe Semanal de métricas semana 3 (del 16/11 al 22/11)",
                    plain_text_content="Resumen\nTodo bien",
                    credentials_path=self.sa_path,
                )

        msg = str(ctx.exception)
        self.assertIsInstance(ctx.exception, drive_service.DriveServiceError)
        self.assertIn("storageQuotaExceeded", msg)
        self.assertIn("Unidad Compartida", msg)
        self.assertIn("GOOGLE_DRIVE_IMPERSONATE_EMAIL", msg)
        self.assertIn("GOOGLE_OAUTH_CREDENTIALS", msg)
        self.assertIn("Informe Semanal de métricas semana 3", msg)

    def test_create_monthly_guidelines_doc_raises_storage_quota_error(self) -> None:
        mock_service = MagicMock()
        quota_exc = _make_http_error(
            403,
            "storageQuotaExceeded",
            "Service Accounts do not have storage quota.",
        )
        mock_service.files().create().execute.side_effect = quota_exc

        with patch.object(drive_service, "_get_drive_client", return_value=mock_service):
            with self.assertRaises(drive_service.DriveStorageQuotaError):
                drive_service.create_monthly_guidelines_doc(
                    month_folder_id="month_folder_1",
                    plain_text_content="PAUTAS\n",
                    doc_title="Pautas",
                    credentials_path=self.sa_path,
                )

    def test_create_or_update_google_doc_creates_with_supports_all_drives(self) -> None:
        mock_service = MagicMock()
        mock_service.files().list().execute.return_value = {"files": []}
        mock_service.files().create().execute.return_value = {
            "id": "new_doc_123",
            "name": "Informe Semanal",
        }

        with patch.object(drive_service, "_get_drive_client", return_value=mock_service):
            doc_id, created = drive_service.create_or_update_google_doc(
                folder_id="week_folder_9",
                doc_title="Informe Semanal",
                plain_text_content="Resumen\nContenido",
                bold_prefixes=("Resumen",),
                credentials_path=self.sa_path,
            )

        self.assertEqual(doc_id, "new_doc_123")
        self.assertTrue(created)
        create_kwargs = mock_service.files().create.call_args.kwargs
        self.assertTrue(create_kwargs["supportsAllDrives"])
        self.assertEqual(
            create_kwargs["body"],
            {
                "name": "Informe Semanal",
                "mimeType": drive_service.GOOGLE_DOC_MIME_TYPE,
                "parents": ["week_folder_9"],
            },
        )

    def test_create_or_update_google_doc_falls_back_to_docs_api_on_update_403(self) -> None:
        mock_drive = MagicMock()
        mock_drive.files().list().execute.return_value = {
            "files": [{"id": "existing_doc_77", "name": "Informe Semanal"}]
        }
        mock_drive.files().update().execute.side_effect = _make_http_error(
            403, "storageQuotaExceeded", "Cannot update media"
        )

        mock_docs = MagicMock()
        mock_docs.documents().get().execute.return_value = {
            "body": {"content": [{"endIndex": 1}, {"endIndex": 42}]}
        }
        mock_docs.documents().batchUpdate().execute.return_value = {}

        with patch.object(drive_service, "_get_drive_client", return_value=mock_drive), patch.object(
            drive_service, "_get_docs_client", return_value=mock_docs
        ):
            doc_id, created = drive_service.create_or_update_google_doc(
                folder_id="week_folder_9",
                doc_title="Informe Semanal",
                plain_text_content="Nuevo informe completo",
                credentials_path=self.sa_path,
            )

        self.assertEqual(doc_id, "existing_doc_77")
        self.assertFalse(created)
        batch_kwargs = mock_docs.documents().batchUpdate.call_args.kwargs
        self.assertEqual(batch_kwargs["documentId"], "existing_doc_77")
        self.assertEqual(
            batch_kwargs["body"]["requests"],
            [
                {"deleteContentRange": {"range": {"startIndex": 1, "endIndex": 41}}},
                {"insertText": {"location": {"index": 1}, "text": "Nuevo informe completo"}},
            ],
        )

    def test_get_guidelines_text_uses_supports_all_drives_for_non_doc(self) -> None:
        mock_drive = MagicMock()
        mock_drive.files().get().execute.return_value = {
            "id": "txt_1",
            "name": "pautas.txt",
            "mimeType": "text/plain",
        }
        mock_drive.files().get_media().execute.return_value = b"Contenido de pautas"

        with patch.object(drive_service, "_get_drive_client", return_value=mock_drive):
            text = drive_service.get_guidelines_text("txt_1", credentials_path=self.sa_path)

        self.assertEqual(text, "Contenido de pautas")
        self.assertTrue(mock_drive.files().get_media.call_args.kwargs["supportsAllDrives"])

    def test_non_quota_403_raises_generic_drive_service_error_not_quota_error(self) -> None:
        mock_service = MagicMock()
        mock_service.files().list().execute.return_value = {"files": []}
        perm_exc = _make_http_error(
            403,
            "insufficientFilePermissions",
            "The user does not have sufficient permissions for this file.",
        )
        mock_service.files().create().execute.side_effect = perm_exc

        with patch.object(drive_service, "_get_drive_client", return_value=mock_service):
            with self.assertRaises(drive_service.DriveServiceError) as ctx:
                drive_service.create_or_update_google_doc(
                    folder_id="folder_x",
                    doc_title="Informe",
                    plain_text_content="Texto",
                    credentials_path=self.sa_path,
                )
        self.assertNotIsInstance(ctx.exception, drive_service.DriveStorageQuotaError)


class TestDriveWeeksAndGuidelinesLogic(unittest.TestCase):
    def test_compute_month_week_info_november_2026_sunday_start(self) -> None:
        tz = timezone.utc
        # Nov 1, 2026 is Sunday -> grouped into Week 1 (Nov 1..Nov 8)
        w1, s1, e1 = drive_service.compute_month_week_info(datetime(2026, 11, 1, 12, 0, tzinfo=tz))
        self.assertEqual((w1, s1.day, e1.day), (1, 1, 8))
        self.assertFalse(drive_service.is_last_weekly_report_of_month(datetime(2026, 11, 8, 20, 0, tzinfo=tz)))

        # Nov 22, 2026 is Sunday -> Week 3 (Nov 16..Nov 22), NOT month end (Nov 30 - 22 = 8 >= 7)
        w3, s3, e3 = drive_service.compute_month_week_info(datetime(2026, 11, 22, 20, 0, tzinfo=tz))
        self.assertEqual((w3, s3.day, e3.day), (3, 16, 22))
        self.assertFalse(drive_service.is_last_weekly_report_of_month(datetime(2026, 11, 22, 20, 0, tzinfo=tz)))

        # Nov 29, 2026 is Sunday -> Week 4 (Nov 23..Nov 29), month end (30 - 29 = 1 < 7)
        w4, s4, e4 = drive_service.compute_month_week_info(datetime(2026, 11, 29, 20, 0, tzinfo=tz))
        self.assertEqual((w4, s4.day, e4.day), (4, 23, 29))
        self.assertTrue(drive_service.is_last_weekly_report_of_month(datetime(2026, 11, 29, 20, 0, tzinfo=tz)))

    def test_entry_matches_ig_post_never_matches_hook_title_only(self) -> None:
        entry = drive_service.GuidelineEntry(
            header="1. Ejercitar tu mente",
            post_category="MAIN",
            short_title="Ejercitar tu mente",
            fecha_raw="20/11/2026",
            horario_raw="06:00 p.m.",
            scheduled_dt=None,
            ads_raw="No.",
            promote_in_ads=False,
            ads_daily_budget_usd=0.0,
            ads_duration_days=0,
            hook_title="Tu cerebro necesita entrenamiento diario",
            caption="Hoy hablamos de la neuroplasticidad y cómo cuidar tu memoria cada mañana.",
            hashtags="#saludcerebral #bienestar #nelhealthcoach",
            has_keyword_cta=False,
            keyword="",
            target_link="",
            raw_block="",
        )
        # Same hook_title on IG, completely different caption -> MUST NOT match
        ig_wrong = [{"caption": "Tu cerebro necesita entrenamiento diario en el parque"}]
        self.assertFalse(drive_service._entry_matches_ig_post(entry, ig_wrong))

        # Matching caption -> MUST match
        ig_right = [{"caption": "Hoy hablamos de la neuroplasticidad y cómo cuidar tu memoria cada mañana.\n#saludcerebral #bienestar #nelhealthcoach"}]
        self.assertTrue(drive_service._entry_matches_ig_post(entry, ig_right))

    def test_mark_entry_published_in_monthly_doc_docs_api_and_html_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sa_path = _write_sa_json(Path(tmpdir))
            mock_docs = MagicMock()
            mock_docs.documents().get().execute.return_value = {
                "body": {
                    "content": [
                        {
                            "startIndex": 1,
                            "endIndex": 25,
                            "paragraph": {"elements": [{"textRun": {"content": "=== PAUTAS DEL MES ===\n"}}]},
                        },
                        {
                            "startIndex": 25,
                            "endIndex": 48,
                            "paragraph": {"elements": [{"textRun": {"content": "1. Ejercitar tu mente\n"}}]},
                        },
                    ]
                }
            }
            mock_docs.documents().batchUpdate().execute.return_value = {}

            with patch.object(
                drive_service, "get_monthly_doc_metadata", return_value={"id": "doc_pautas"}
            ), patch.object(drive_service, "_get_docs_client", return_value=mock_docs):
                mode = drive_service.mark_entry_published_in_monthly_doc(
                    "month_1", "1. Ejercitar tu mente", credentials_path=sa_path
                )
            self.assertEqual(mode, "docs_api")
            req = mock_docs.documents().batchUpdate.call_args.kwargs["body"]["requests"][0]
            self.assertEqual(req["insertText"], {"location": {"index": 47}, "text": " [PUBLICADO]"})

            # Fallback to Drive HTML when Docs API returns 403
            mock_docs_403 = MagicMock()
            mock_docs_403.documents().get().execute.side_effect = _make_http_error(403, "forbidden", "Forbidden")
            mock_drive = MagicMock()
            mock_drive.files().export().execute.return_value = (
                b"<html><body><p>=== PAUTAS DEL MES ===</p><p>1. Ejercitar tu mente</p></body></html>"
            )
            mock_drive.files().update().execute.return_value = {"id": "doc_pautas", "name": "Pautas"}

            with patch.object(
                drive_service, "get_monthly_doc_metadata", return_value={"id": "doc_pautas"}
            ), patch.object(drive_service, "_get_docs_client", return_value=mock_docs_403), patch.object(
                drive_service, "_get_drive_client", return_value=mock_drive
            ):
                mode_fb = drive_service.mark_entry_published_in_monthly_doc(
                    "month_1", "1. Ejercitar tu mente", credentials_path=sa_path
                )
            self.assertEqual(mode_fb, "drive_html_fallback")

    def test_find_unpublished_guidelines_respects_published_markers_and_ig_cross_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            guidelines = (
                "=== PAUTAS DEL MES ===\n"
                "1. Apertura [PUBLICADO]\n"
                "Fecha: 01/11/2026\nHorario: 06:00 p.m.\nPauta en Meta Ads: No.\n"
                "Título / Gancho: Bienvenida\nCaption: Primer video del mes.\nHashtags: #salud #bienestar\n\n"
                "2. Presentación\n"
                "Fecha: 05/11/2026\nHorario: 06:00 p.m.\nPauta en Meta Ads: No.\n"
                "Título / Gancho: Quién soy\nCaption: Te cuento mi historia completa como health coach.\nHashtags: #healthcoach #habitos\n\n"
                "3. Flexibilidad\n"
                "Fecha: 20/11/2026\nHorario: 06:00 p.m.\nPauta en Meta Ads: No.\n"
                "Título / Gancho: Estira cada mañana\nCaption: Cuatro ejercicios simples para tus articulaciones.\nHashtags: #movilidad #flexibilidad\n"
            )
            ig_posts = [
                {"caption": "Te cuento mi historia completa como health coach. #healthcoach #habitos"}
            ]
            items = drive_service.find_unpublished_guidelines(
                guidelines_text=guidelines,
                month_files=[],
                state_path=state_path,
                recent_ig_media=ig_posts,
                reference_now=datetime(2026, 11, 22, 20, 0, tzinfo=timezone.utc),
            )
            self.assertEqual([it.entry.header for it in items], ["3. Flexibilidad"])
            self.assertTrue(items[0].is_overdue)

    def test_read_reports_in_week_folder_supports_docx_and_pdf(self) -> None:
        import io
        import zipfile

        docx_buf = io.BytesIO()
        with zipfile.ZipFile(docx_buf, "w") as zf:
            zf.writestr(
                "word/document.xml",
                (
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                    "<w:body><w:p><w:r><w:t>Resumen Semana 1</w:t></w:r></w:p>"
                    "<w:p><w:r><w:t>Alcance: 2360</w:t></w:r></w:p></w:body></w:document>"
                ),
            )
        docx_bytes = docx_buf.getvalue()

        mock_drive = MagicMock()
        mock_drive.files().list().execute.return_value = {
            "files": [
                {
                    "id": "docx_1",
                    "name": "Informe_S1.docx",
                    "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                },
                {
                    "id": "pdf_2",
                    "name": "Informe_S2.pdf",
                    "mimeType": "application/pdf",
                },
                {
                    "id": "img_3",
                    "name": "1.png",
                    "mimeType": "image/png",
                },
            ]
        }
        mock_drive.files().get_media.return_value.execute.side_effect = [docx_bytes, b"%PDF-fake"]

        with patch.object(drive_service, "_extract_pdf_text", return_value="Resumen Semana 2\nAlcance: 655"):
            reports = drive_service._read_reports_in_week_folder(
                mock_drive, "week_folder", "11/2026 - Semana 1"
            )

        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[0]["title"], "Informe_S1.docx")
        self.assertIn("Alcance: 2360", reports[0]["text"])
        self.assertEqual(reports[1]["title"], "Informe_S2.pdf")
        self.assertIn("Alcance: 655", reports[1]["text"])


if __name__ == "__main__":
    unittest.main()
