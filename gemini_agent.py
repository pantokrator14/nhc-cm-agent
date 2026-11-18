"""Gemini AI multimodal filtering, copywriting, scheduling, and reasoning agent."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

CLASSIFIER_SYSTEM_INSTRUCTION = """Eres el asistente editorial de "Nel Health Coach" (@nelhealthcoach).
Tu tarea es comparar una lista de archivos multimedia de Google Drive contra el documento de pautas mensuales ya programadas.

Reglas estrictas de emparejamiento:
1. El documento de pautas contiene los contenidos que YA fueron programados o publicados manualmente (por ejemplo: `1. Apertura`, `2. Presentación`, `Pro Tip 1: Hay que moverse`, `3. A mis 60 años`, etc.).
2. Los nombres de archivo en Drive a veces tienen numeración o títulos abreviados que corresponden a una entrada del documento. Por ejemplo:
   - `01. Apertura.mp4` -> `1. Apertura`
   - `03. Protip Atención.mp4` -> `Pro Tip 2: Donde pones tu atención`
   - `05 Protip Movimiento.mp4` -> `Pro Tip 1: Hay que moverse`
   - `13. PROTIP estoesecono/cost.mp4` -> `Pro Tip 6: Lo económico vs. lo costoso`
   - `14. Orinar.mp4` -> `8. Orinar`
3. Marca `already_in_guidelines = true` para TODO archivo que ya tenga una entrada equivalente en el documento de pautas (por número, título, tema o Pro Tip).
4. Marca `already_in_guidelines = false` ÚNICAMENTE para archivos nuevos cuyo tema/nombre NO aparece en la lista del documento.
"""

PROPOSAL_SYSTEM_INSTRUCTION = """Eres el Director de Contenido, Estratega de Redes y Trafficker de "Nel Health Coach" (@nelhealthcoach).

Tu función:
Analizar DIRECTAMENTE el archivo multimedia (video o imagen) junto con el documento de pautas del mes para proponer una pauta inteligente y lista para aprobar.

Reglas editoriales y estratégicas obligatorias:
1. Análisis real del archivo y regla ESTRICTA de `MAIN` vs `PRO_TIP`:
   - Observa/escucha el contenido real del video o imagen (tema exacto, formato visual, si aparece una persona hablando a cámara o si es solo texto/imágenes estáticas).
   - **REGLA CRÍTICA DE CLASIFICACIÓN**:
     * **Video grabado por Manuel Martínez (persona hablando a cámara / presentador)**: `has_presenter_on_camera = true` -> **SIEMPRE** es `post_category = "MAIN"`, formato `"REELS"`, sin importar que dure pocos segundos. **NUNCA** clasifiques un video grabado por Manuel hablando a cámara como `PRO_TIP`.
     * **Pro Tips (`post_category = "PRO_TIP"`)**: **SOLO** pueden ser (a) videos cortos sin persona hablando a cámara hechos con textos e imágenes estáticas que cambian automáticamente como un carrusel/presentación, o (b) imágenes de texto en formato vertical (para `"STORIES"` o `"IMAGE"`). En ambos casos `has_presenter_on_camera = false`.
   - Resume brevemente lo que trata el archivo en `content_summary`.
2. Cadencia semanal obligatoria (NO copies ciegamente las horas viejas; razona el mejor día y horario según el contenido):
   - **Lunes, Miércoles y Viernes**: Publicaciones principales (`post_category = "MAIN"`), formato `"REELS"` (video) o `"IMAGE"` (imagen).
   - **Martes, Jueves y Sábados**: ÚNICAMENTE `PRO_TIP` (`post_category = "PRO_TIP"`), en formato `"REELS"` / `"STORIES"` (video carrusel de textos/imágenes) o `"STORIES"` / `"IMAGE"` (imagen vertical de texto).
   - Elige la fecha y horario futuro (`scheduled_time`, ISO 8601 con zona horaria) posterior a la fecha actual y a los slots ya ocupados, respetando estrictamente que los `MAIN` caigan en Lunes/Miércoles/Viernes y los `PRO_TIP` en Martes/Jueves/Sábado, y explica por qué en `schedule_reasoning`.
3. Tono y ética de @nelhealthcoach:
   - Cercano, empático, educativo, basado en salud integrativa, metabolismo y hábitos.
   - NUNCA hagas promesas médicas milagrosas ni garantías de cura.
   - Redacta un `caption` con gancho claro y llamado a la acción (CTA), más 4 a 6 `hashtags` relevantes.
4. Pauta en Meta Ads:
   - Evalúa de forma inteligente si el contenido amerita inversión en Meta Ads (`promote_in_ads = true`) o debe ir orgánico (`false`).
   - Si `promote_in_ads = true`, define `ads_daily_budget_usd` (entre 3.0 y 10.0 USD) y `ads_duration_days` (entre 3 y 7 días), y explica en `ads_recommendation` objetivo, público y por qué vale la pena promocionarlo. Si es orgánico, pon `0.0` y `0`.
5. Automatización de DM por comentario:
   - Solo activa `has_keyword_cta = true` si el copy invita explícitamente a comentar una palabra clave para enviar un recurso/enlace por DM (`keyword` en MAYÚSCULAS, `target_link`, `public_reply_text`). Si no aplica, `false` y `""`.
"""

REVISION_SYSTEM_INSTRUCTION = """Eres el Director de Contenido y Estratega de "Nel Health Coach" (@nelhealthcoach) conversando con Manuel (el administrador).

Cuando Manuel pide un cambio sobre una propuesta:
1. Si `force_apply` es True (o si Manuel está insistiendo claramente después de una sugerencia previa), DEBES obedecer exactamente lo que Manuel pide: pon `ai_assessment = "ACCEPTED"`, aplica sus cambios al pie de la letra en `proposal`, y confirma brevemente en `reasoning_message`.
2. Si `force_apply` es False:
   - Razona críticamente el cambio pedido por Manuel evaluando: la regla estricta de contenido (videos grabados por Manuel a cámara = `MAIN` Lun/Mié/Vie; `PRO_TIP` Mar/Jue/Sáb = solo carrusel automático de textos/imágenes o imagen vertical de texto), horario óptimo según el tema, retención del copy/CTA, cumplimiento ético en salud y retorno en Meta Ads.
   - Si el cambio de Manuel es acertado o mejora la propuesta -> `ai_assessment = "ACCEPTED"`, aplícalo en `proposal` y explica en `reasoning_message` por qué es una buena decisión.
   - Si ves que lo que pide Manuel puede perjudicar el rendimiento (ej. mal horario para el tema, romper la cadencia semanal, perder el CTA, promocionar algo que funciona mejor orgánico, etc.) -> `ai_assessment = "ALTERNATIVE_SUGGESTED"`, explica claramente en `reasoning_message` por qué sugieres otra alternativa mejor, y entrega esa alternativa mejorada en `proposal` (para que Manuel decida si acepta tu alternativa o insiste en su cambio).
"""


class GeminiAgentError(RuntimeError):
    """Raised when Gemini fails to generate or validate a structured proposal."""


class FileGuidelineMatch(BaseModel):
    """Classification result indicating whether a Drive file is already in the guidelines doc."""

    file_id: str
    file_name: str
    already_in_guidelines: bool = Field(
        description="True si el archivo ya está programado/listado en el documento de pautas; False si es nuevo y no aparece."
    )
    matched_entry: str = Field(
        description="Título o sección del documento con la que coincide, o cadena vacía si no está en la lista."
    )


class FileClassificationBatch(BaseModel):
    """Batch classification output for Drive candidate files."""

    items: list[FileGuidelineMatch]


class PostProposal(BaseModel):
    """Structured Instagram post proposal returned by Gemini for unlisted media."""

    content_summary: str = Field(
        description="Resumen conciso del contenido real observado/escuchado en el archivo."
    )
    has_presenter_on_camera: bool = Field(
        description="True si es un video grabado con Manuel Martínez / persona hablando a cámara; False si es solo texto/imágenes estáticas tipo carrusel o imagen."
    )
    post_category: Literal["MAIN", "PRO_TIP"] = Field(
        description="OBLIGATORIO 'MAIN' (Lun/Mié/Vie) si has_presenter_on_camera=True. 'PRO_TIP' (Mar/Jue/Sáb) SOLO si es video carrusel de texto/imágenes sin presentador o imagen vertical de texto."
    )
    short_title: str = Field(
        description="Título corto de 2 a 5 palabras sin número inicial ni extensión (ej. 'Adicción al azúcar')."
    )
    hook_title: str = Field(
        description="Título o gancho sugerido para el post."
    )
    caption: str = Field(
        description="Cuerpo del copy optimizado para Instagram con llamada a la acción clara."
    )
    hashtags: str = Field(
        description="Hashtags sugeridos separados por espacios (ej. #NelHealthCoach #SaludMetabolica)."
    )
    media_type: Literal["REELS", "IMAGE", "STORIES"] = Field(
        description="'REELS' para video en feed/reels, 'IMAGE' para imagen en feed, o 'STORIES' para formato historia."
    )
    scheduled_time: str = Field(
        description="Fecha y hora futura sugerida en formato ISO 8601 con zona horaria."
    )
    schedule_reasoning: str = Field(
        description="Explicación breve de por qué conviene publicar este contenido en esa fecha y hora."
    )
    promote_in_ads: bool = Field(
        description="True si es válido/recomendable promocionar este contenido en Meta Ads; False si debe ir orgánico."
    )
    ads_daily_budget_usd: float = Field(
        description="Presupuesto diario sugerido en USD si promote_in_ads=True (ej. 4.0); 0.0 si es orgánico."
    )
    ads_duration_days: int = Field(
        description="Duración sugerida de la campaña en días si promote_in_ads=True (ej. 5); 0 si es orgánico."
    )
    ads_recommendation: str = Field(
        description="Indicación clara de si conviene promocionarlo en Meta Ads o dejarlo orgánico, con justificación."
    )
    has_keyword_cta: bool = Field(
        description="True si el post pide comentar una palabra clave para enviar DM automático."
    )
    keyword: str = Field(
        description="Palabra clave en MAYÚSCULAS si has_keyword_cta es true; cadena vacía si no aplica."
    )
    target_link: str = Field(
        description="Enlace destino para enviar por DM según las pautas, o cadena vacía si no aplica."
    )
    public_reply_text: str = Field(
        description="Respuesta corta pública al comentario cuando aplica keyword CTA, o cadena vacía."
    )

    @property
    def full_instagram_caption(self) -> str:
        """Return caption combined with hashtags cleanly for Instagram publication."""
        clean_caption = self.caption.strip()
        clean_hashtags = self.hashtags.strip()
        if not clean_hashtags:
            return clean_caption
        first_tag = clean_hashtags.split()[0]
        if first_tag in clean_caption:
            return clean_caption
        return f"{clean_caption}\n\n{clean_hashtags}"


class RevisionResult(BaseModel):
    """Structured reasoning and revised proposal when the admin requests changes."""

    ai_assessment: Literal["ACCEPTED", "ALTERNATIVE_SUGGESTED"] = Field(
        description="'ACCEPTED' si se aplica el cambio pedido; 'ALTERNATIVE_SUGGESTED' si el agente propone una opción mejor."
    )
    reasoning_message: str = Field(
        description="Explicación clara y argumentada para Manuel sobre su cambio o la alternativa sugerida."
    )
    proposal: PostProposal


def _resolve_client_and_model(
    api_key: str | None,
    model: str | None,
) -> tuple[genai.Client, str]:
    resolved_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not resolved_key:
        raise GeminiAgentError("GEMINI_API_KEY no está configurado.")

    resolved_model = (
        model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    ).strip() or DEFAULT_GEMINI_MODEL

    return genai.Client(api_key=resolved_key), resolved_model


def _is_video_file(file_metadata: dict[str, Any]) -> bool:
    mime_type = (file_metadata.get("mimeType") or "").lower()
    name = (file_metadata.get("name") or "").lower()
    return mime_type.startswith("video/") or Path(name).suffix == ".mp4"


def _normalize_media_type(
    proposed_type: Literal["REELS", "IMAGE", "STORIES"],
    file_metadata: dict[str, Any],
) -> Literal["REELS", "IMAGE", "STORIES"]:
    is_video = _is_video_file(file_metadata)
    if proposed_type == "STORIES":
        return "STORIES"
    if is_video and proposed_type != "REELS":
        logger.warning(
            "Corrigiendo media_type=%s a REELS para archivo de video '%s'.",
            proposed_type,
            file_metadata.get("name"),
        )
        return "REELS"
    if not is_video and proposed_type != "IMAGE":
        logger.warning(
            "Corrigiendo media_type=%s a IMAGE para archivo de imagen '%s'.",
            proposed_type,
            file_metadata.get("name"),
        )
        return "IMAGE"
    return proposed_type


def _normalize_proposal(
    proposal: PostProposal,
    file_metadata: dict[str, Any],
    reference_now: datetime,
) -> PostProposal:
    media_type = _normalize_media_type(proposal.media_type, file_metadata)

    try:
        parsed_dt = datetime.fromisoformat(proposal.scheduled_time.strip())
    except ValueError as exc:
        raise GeminiAgentError(
            f"Gemini devolvió un scheduled_time inválido ('{proposal.scheduled_time}'): {exc}"
        ) from exc

    if parsed_dt.tzinfo is None:
        parsed_dt = parsed_dt.replace(tzinfo=reference_now.tzinfo)

    keyword = proposal.keyword.strip().upper() if proposal.has_keyword_cta else ""
    has_cta = bool(proposal.has_keyword_cta and keyword)
    target_link = proposal.target_link.strip() if has_cta else ""
    public_reply_text = proposal.public_reply_text.strip() if has_cta else ""

    raw_short = re.sub(r"^\d+[\.\-\s:]+", "", proposal.short_title.strip())
    raw_short = re.sub(r"^pro\s*tip\s*\d*[\.\-\s:]+", "", raw_short, flags=re.IGNORECASE)
    if not raw_short:
        raw_short = Path(str(file_metadata.get("name", "Publicacion"))).stem

    promote = bool(proposal.promote_in_ads)
    daily_budget = round(max(1.0, proposal.ads_daily_budget_usd), 2) if promote else 0.0
    duration_days = max(1, int(proposal.ads_duration_days)) if promote else 0

    post_category = proposal.post_category
    if proposal.has_presenter_on_camera and post_category != "MAIN":
        logger.warning(
            "Corrigiendo post_category=%s a MAIN porque '%s' tiene presentador a cámara (regla: Pro Tips nunca son videos grabados por Manuel).",
            post_category,
            file_metadata.get("name"),
        )
        post_category = "MAIN"
        if media_type == "STORIES":
            media_type = "REELS" if _is_video_file(file_metadata) else "IMAGE"

    return PostProposal(
        content_summary=proposal.content_summary.strip(),
        has_presenter_on_camera=bool(proposal.has_presenter_on_camera),
        post_category=post_category,
        short_title=raw_short.strip(),
        hook_title=proposal.hook_title.strip(),
        caption=proposal.caption.strip(),
        hashtags=proposal.hashtags.strip(),
        media_type=media_type,
        scheduled_time=parsed_dt.isoformat(),
        schedule_reasoning=proposal.schedule_reasoning.strip(),
        promote_in_ads=promote,
        ads_daily_budget_usd=daily_budget,
        ads_duration_days=duration_days,
        ads_recommendation=proposal.ads_recommendation.strip(),
        has_keyword_cta=has_cta,
        keyword=keyword if has_cta else "",
        target_link=target_link,
        public_reply_text=public_reply_text,
    )


def _upload_media_to_gemini(
    client: genai.Client,
    media_path: Path,
    timeout_seconds: int = 180,
) -> Any:
    logger.info("Subiendo '%s' a Gemini Files API para análisis multimodal...", media_path.name)
    uploaded = client.files.upload(file=str(media_path))
    deadline = time.monotonic() + timeout_seconds

    while True:
        state_name = getattr(getattr(uploaded, "state", None), "name", "ACTIVE")
        if state_name == "ACTIVE":
            return uploaded
        if state_name == "FAILED":
            raise GeminiAgentError(
                f"Gemini Files API falló procesando '{media_path.name}'."
            )
        if time.monotonic() >= deadline:
            raise GeminiAgentError(
                f"Timeout esperando procesamiento de '{media_path.name}' en Gemini Files API."
            )
        time.sleep(3)
        uploaded = client.files.get(name=uploaded.name)


def _delete_gemini_file_quietly(client: genai.Client, uploaded_file: Any) -> None:
    if uploaded_file is None or not getattr(uploaded_file, "name", None):
        return
    try:
        client.files.delete(name=uploaded_file.name)
    except Exception as exc:
        logger.debug("No se pudo limpiar archivo temporal en Gemini (%s): %s", uploaded_file.name, exc)


def classify_files_against_guidelines(
    guidelines_text: str,
    files: list[dict[str, Any]],
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, FileGuidelineMatch]:
    """Identify which Drive files already appear in the monthly guidelines document."""
    if not files:
        return {}

    client, resolved_model = _resolve_client_and_model(api_key, model)
    compact_files = [
        {
            "file_id": f["id"],
            "file_name": f.get("name", ""),
            "mimeType": f.get("mimeType", ""),
        }
        for f in files
    ]

    prompt = (
        "=== DOCUMENTO DE PAUTAS DEL MES (YA PROGRAMADAS / EXISTENTES) ===\n"
        f"{guidelines_text}\n\n"
        "=== ARCHIVOS MULTIMEDIA EN LA CARPETA DE DRIVE ===\n"
        f"{json.dumps(compact_files, ensure_ascii=False, indent=2)}\n\n"
        "Clasifica cada archivo indicando si ya existe en la lista de pautas (`already_in_guidelines=true`) "
        "o si es un archivo nuevo sin pauta asignada (`already_in_guidelines=false`)."
    )

    try:
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=CLASSIFIER_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=FileClassificationBatch,
                temperature=0.1,
            ),
        )
    except Exception as exc:
        raise GeminiAgentError(
            f"Error al clasificar archivos contra pautas con Gemini ({resolved_model}): {exc}"
        ) from exc

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, FileClassificationBatch):
        batch = parsed
    elif response.text:
        try:
            batch = FileClassificationBatch.model_validate_json(response.text)
        except Exception as exc:
            raise GeminiAgentError(
                f"Respuesta inválida de clasificación de Gemini: {exc}"
            ) from exc
    else:
        raise GeminiAgentError("Gemini devolvió una clasificación vacía.")

    return {item.file_id: item for item in batch.items}


def generate_post_proposal(
    guidelines_text: str,
    file_metadata: dict[str, Any],
    current_time: datetime | None = None,
    api_key: str | None = None,
    model: str | None = None,
    occupied_slots: list[str] | None = None,
    media_file_path: Path | None = None,
) -> PostProposal:
    """Analyze the actual media file (when provided) + monthly guidelines and propose a post."""
    now = current_time or datetime.now().astimezone()
    file_name = str(file_metadata.get("name", file_metadata.get("id", "")))
    file_meta_json = json.dumps(file_metadata, ensure_ascii=False, indent=2)
    occupied_block = (
        "\n".join(f"- {slot}" for slot in occupied_slots)
        if occupied_slots
        else "Ninguno adicional fuera del documento."
    )

    prompt = (
        f"Fecha y hora actual de referencia: {now.isoformat()}\n"
        f"Archivo nuevo SIN pauta previa: {file_name}\n\n"
        f"=== HORARIOS YA OCUPADOS EN PROPUESTAS/APROBADOS ===\n{occupied_block}\n\n"
        f"=== PAUTAS DEL MES (REFERENCIA DE ESTILO, MARCA Y PUBLICACIONES EXISTENTES) ===\n{guidelines_text}\n\n"
        f"=== METADATOS DEL ARCHIVO EN DRIVE ===\n{file_meta_json}\n\n"
        "Analiza visualmente el archivo multimedia adjunto (si está disponible) y genera una propuesta inteligente:\n"
        "- Verifica primero si es un video grabado por Manuel Martínez (persona hablando a cámara) -> OBLIGATORIAMENTE `MAIN` en Lun/Mié/Vie (`REELS`).\n"
        "- Solo usa `PRO_TIP` (Mar/Jue/Sáb) si es un video corto de textos e imágenes estáticas tipo carrusel automático SIN persona a cámara, o una imagen vertical de texto.\n"
        "- Razona la mejor fecha y hora sin copiar ciegamente horas fijas anteriores, redacta gancho, caption con CTA y hashtags, "
        "y decide si amerita promoción en Meta Ads con presupuesto y duración sugeridos."
    )

    client, resolved_model = _resolve_client_and_model(api_key, model)
    uploaded_file = None
    try:
        contents: Any = prompt
        if media_file_path is not None and media_file_path.is_file():
            try:
                uploaded_file = _upload_media_to_gemini(client, media_file_path)
                contents = [uploaded_file, prompt]
            except Exception as upload_exc:
                logger.warning(
                    "No se pudo analizar '%s' multimodalmente (%s); continuando con metadatos.",
                    file_name,
                    upload_exc,
                )

        logger.info("Generando propuesta inteligente con Gemini para '%s'...", file_name)
        response = client.models.generate_content(
            model=resolved_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=PROPOSAL_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=PostProposal,
                temperature=0.6,
            ),
        )
    except Exception as exc:
        raise GeminiAgentError(f"Error al invocar Gemini ({resolved_model}): {exc}") from exc
    finally:
        _delete_gemini_file_quietly(client, uploaded_file)

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, PostProposal):
        proposal = parsed
    elif response.text:
        try:
            proposal = PostProposal.model_validate_json(response.text)
        except Exception as exc:
            raise GeminiAgentError(
                f"La respuesta de Gemini no cumple el esquema esperado: {exc}"
            ) from exc
    else:
        raise GeminiAgentError("Gemini devolvió una respuesta vacía.")

    return _normalize_proposal(proposal, file_metadata, now)


def revise_post_proposal(
    guidelines_text: str,
    file_metadata: dict[str, Any],
    previous_proposal: PostProposal,
    feedback: str,
    current_time: datetime | None = None,
    api_key: str | None = None,
    model: str | None = None,
    force_apply: bool = False,
    revision_history: list[str] | None = None,
) -> RevisionResult:
    """Critically evaluate admin change requests, reason alternatives, and respect insistence."""
    now = current_time or datetime.now().astimezone()
    file_name = str(file_metadata.get("name", file_metadata.get("id", "")))
    file_meta_json = json.dumps(file_metadata, ensure_ascii=False, indent=2)
    previous_json = previous_proposal.model_dump_json(indent=2)
    history_block = (
        "\n".join(f"- {item}" for item in revision_history)
        if revision_history
        else "Primera solicitud de cambio sobre esta propuesta."
    )

    prompt = (
        f"Fecha y hora actual de referencia: {now.isoformat()}\n"
        f"Archivo multimedia: {file_name}\n"
        f"Modo force_apply (obedecer sin objetar): {force_apply}\n\n"
        f"=== HISTORIAL DE REVISIONES ===\n{history_block}\n\n"
        f"=== PAUTAS DEL MES (GOOGLE DOC) ===\n{guidelines_text}\n\n"
        f"=== METADATOS DEL ARCHIVO EN DRIVE ===\n{file_meta_json}\n\n"
        f"=== PROPUESTA ANTERIOR ===\n{previous_json}\n\n"
        f"=== CAMBIO SOLICITADO POR MANUEL ===\n{feedback.strip()}\n\n"
        "Evalúa el cambio solicitado, razona si conviene o si existe una alternativa mejor "
        "(a menos que force_apply=True o Manuel esté insistiendo), y devuelve la respuesta estructurada."
    )

    client, resolved_model = _resolve_client_and_model(api_key, model)
    logger.info(
        "Evaluando cambio humano para '%s' (force_apply=%s)...",
        file_name,
        force_apply,
    )
    try:
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=REVISION_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=RevisionResult,
                temperature=0.4,
            ),
        )
    except Exception as exc:
        raise GeminiAgentError(f"Error al evaluar revisión con Gemini ({resolved_model}): {exc}") from exc

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, RevisionResult):
        result = parsed
    elif response.text:
        try:
            result = RevisionResult.model_validate_json(response.text)
        except Exception as exc:
            raise GeminiAgentError(
                f"La respuesta de revisión de Gemini no cumple el esquema esperado: {exc}"
            ) from exc
    else:
        raise GeminiAgentError("Gemini devolvió una revisión vacía.")

    normalized = _normalize_proposal(result.proposal, file_metadata, now)
    assessment: Literal["ACCEPTED", "ALTERNATIVE_SUGGESTED"] = (
        "ACCEPTED" if force_apply else result.ai_assessment
    )
    return RevisionResult(
        ai_assessment=assessment,
        reasoning_message=result.reasoning_message.strip(),
        proposal=normalized,
    )


def generate_monthly_template_from_previous(
    previous_guidelines_text: str,
    year_str: str,
    month_str: str,
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    """Create a clean new-month `Pautas` guide document preserving the previous month's style and rules."""
    fallback = (
        f"PAUTAS DE PUBLICACIÓN - @NELHEALTHCOACH ({month_str}/{year_str})\n\n"
        "Guía de estilo, cadencia semanal y criterios de Meta Ads:\n"
        "- Lunes, Miércoles y Viernes: Publicaciones principales (Reels / Feed).\n"
        "- Martes, Jueves y Sábados: Pro Tips (en formato video o historias según el archivo).\n"
        "- La IA analiza cada archivo y sugiere fecha, horario óptimo, copy, hashtags y si aplica Meta Ads.\n"
        "- Formato estándar de cada pauta:\n"
        "  N. Título (o Pro Tip N: Título)\n"
        "  Fecha: Día DD/MM/YYYY\n"
        "  Horario: HH:MM hrs\n"
        "  Pauta en Meta Ads: Orgánico | SÍ (...)\n"
        "  Título / Gancho: ...\n"
        "  Caption:\n"
        "  ...\n"
        "  Hashtags: #...\n\n"
        f"=== PAUTAS DEL MES {month_str}/{year_str} ===\n"
    )

    if not previous_guidelines_text.strip():
        return fallback

    try:
        client, resolved_model = _resolve_client_and_model(api_key, model)
        prompt = (
            f"Estamos iniciando un nuevo mes: {month_str}/{year_str} para @nelhealthcoach.\n\n"
            "A partir del documento de pautas del mes anterior, redacta un documento limpio llamado 'Pautas' "
            "para este nuevo mes que:\n"
            "1. Resuma el tono de marca, cadencia semanal (Lun/Mié/Vie publicaciones principales; Mar/Jue/Sáb Pro Tips en video o historias) y criterios de Meta Ads.\n"
            "2. Muestre exactamente el formato de bloque que debe seguir cada entrada (N. Título / Pro Tip N: Título, Fecha, Horario, Pauta en Meta Ads, Título / Gancho, Caption, Hashtags).\n"
            "3. NO incluya las publicaciones viejas del mes anterior como publicaciones programadas de este mes, para que el conteo del nuevo mes comience limpio desde 1.\n"
            "4. Termine exactamente con el encabezado:\n"
            f"=== PAUTAS DEL MES {month_str}/{year_str} ===\n\n"
            f"=== DOCUMENTO DEL MES ANTERIOR ===\n{previous_guidelines_text}\n"
        )
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        text = (response.text or "").strip()
        if text:
            return text + "\n"
    except Exception as exc:
        logger.warning(
            "No se pudo generar plantilla mensual con Gemini (%s); usando plantilla estándar.",
            exc,
        )

    return fallback
