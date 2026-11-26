"""Gemini AI multimodal filtering, copywriting, scheduling, and reasoning agent."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from config import resolve_timezone

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
MAX_INSTAGRAM_HASHTAGS = 5
TRENDS_CACHE_TTL_SECONDS = 6 * 3600

_TRENDS_CACHE_LOCK = threading.Lock()
_TRENDS_CACHE: dict[str, Any] = {
    "fetched_at": 0.0,
    "brief": "",
}

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
4. **EXCEPCIÓN IMPORTANTE DE HISTORIAS**: Si el archivo es una Historia (`historia ...`, `story ...`, `(HISTORIA)`), **NO** lo emparejes con un Reel o Pro Tip existente aunque compartan tema (por ejemplo, `historia ORINAR.mp4` NO es `8. Orinar`); solo marca `already_in_guidelines = true` si en el documento existe explícitamente una entrada de Historia (`Historia N: ...`) equivalente.
5. Marca `already_in_guidelines = false` ÚNICAMENTE para archivos nuevos cuyo tema/nombre NO aparece en la lista del documento.
"""

PROPOSAL_SYSTEM_INSTRUCTION = """Eres el Director de Contenido, Estratega de Redes y Trafficker de "Nel Health Coach" (@nelhealthcoach).

Tu función:
Analizar DIRECTAMENTE el archivo multimedia (video o imagen) junto con el documento de pautas del mes y las tendencias actualizadas de redes sociales para proponer una pauta inteligente, optimizada para crecimiento orgánico y lista para aprobar.

Reglas editoriales y estratégicas obligatorias:
1. Análisis real del archivo y regla ESTRICTA de `MAIN` vs `PRO_TIP`:
   - Observa/escucha el contenido real del video o imagen (tema exacto, formato visual, si aparece una persona hablando a cámara o si es solo texto/imágenes estáticas).
   - **REGLA CRÍTICA DE HISTORIAS vs REELS/FEED**:
     * Si el nombre del archivo indica que es una Historia (por ejemplo empieza por `historia ...`, `historias ...`, `story ...`, `stories ...`, `(HISTORIA)` o `[HISTORIA]`), es **OBLIGATORIAMENTE `media_type = "STORIES"`** (Historia de Instagram de 24 horas), **NUNCA `"REELS"`**, aunque Manuel aparezca hablando a cámara.
     * Para `"STORIES"`: `promote_in_ads = false`, `publish_to_tiktok = false`, `hashtags = ""`. Las Historias de Instagram NO llevan caption de feed: deja `caption = ""` (o como máximo un comentario muy breve de 1 línea sobre texto/sticker en pantalla si aporta valor).
   - **REGLA CRÍTICA DE CLASIFICACIÓN (`MAIN` vs `PRO_TIP`)**:
     * **Video grabado por Manuel Martínez (persona hablando a cámara / presentador)**: `has_presenter_on_camera = true` -> **SIEMPRE** es `post_category = "MAIN"`, y si NO es una Historia (`STORIES`), su formato es `"REELS"`, sin importar que dure pocos segundos. **NUNCA** clasifiques un video grabado por Manuel hablando a cámara como `PRO_TIP`.
     * **Pro Tips (`post_category = "PRO_TIP"`)**: **SOLO** pueden ser (a) videos cortos sin persona hablando a cámara hechos con textos e imágenes estáticas que cambian automáticamente como un carrusel/presentación, o (b) imágenes de texto en formato vertical (para `"STORIES"` o `"IMAGE"`). En ambos casos `has_presenter_on_camera = false`.
   - Resume brevemente lo que trata el archivo en `content_summary`.
2. Cadencia semanal obligatoria (NO copies ciegamente las horas viejas; razona el mejor día y horario según el contenido):
   - **Lunes, Miércoles y Viernes**: Publicaciones principales (`post_category = "MAIN"`), formato `"REELS"` (video) o `"IMAGE"` (imagen).
   - **Martes, Jueves y Sábados**: ÚNICAMENTE `PRO_TIP` (`post_category = "PRO_TIP"`), en formato `"REELS"` / `"STORIES"` (video carrusel de textos/imágenes) o `"STORIES"` / `"IMAGE"` (imagen vertical de texto).
   - Elige la fecha y horario futuro (`scheduled_time`, ISO 8601 con zona horaria) posterior a la fecha actual y a los slots ya ocupados, respetando estrictamente que los `MAIN` caigan en Lunes/Miércoles/Viernes y los `PRO_TIP` en Martes/Jueves/Sábado, y explica por qué en `schedule_reasoning`.
3. Alineación con tendencias actuales para crecimiento orgánico:
   - Usa el resumen de tendencias actualizadas de redes sociales (Instagram Reels, TikTok, Shorts en salud/metabolismo/bienestar) para potenciar incluso los videos ya grabados por Manuel.
   - Redacta un `hook_title` (gancho de los primeros 3 segundos o texto de portada) y un `caption` con narrativa actual orientada a retención, guardados y compartidos orgánicos (además de seguidores nuevos).
   - Explica brevemente en `trend_alignment` cómo esta pauta aprovecha una tendencia, formato o ángulo viral replicable actual (ej. texto de gancho, formato de audio/edición sugerido en portada/descripción, o enfoque de debate saludable).
4. Tono, ética y límite ESTRICTO de hashtags de @nelhealthcoach:
   - Cercano, empático, educativo, basado en salud integrativa, metabolismo y hábitos.
   - NUNCA hagas promesas médicas milagrosas ni garantías de cura.
   - Redacta un `caption` con gancho claro y llamado a la acción (CTA), SIN hashtags incrustados dentro del caption.
   - En `hashtags`, entrega **EXACTAMENTE 5 hashtags** relevantes (o como máximo absoluto **5 hashtags**). Instagram actualmente solo permite un máximo de 5 hashtags por publicación: **NUNCA pongas 6 ni más**.
5. Pauta en Meta Ads (solo Instagram):
   - Evalúa de forma inteligente si el contenido amerita inversión en Meta Ads (`promote_in_ads = true`) o debe ir orgánico (`false`).
   - El sistema utiliza automáticamente el público guardado `'mujeres'` y la tarjeta configurada en la cuenta publicitaria.
   - Si `promote_in_ads = true`, define `ads_daily_budget_usd` (entre 3.0 y 10.0 USD) y `ads_duration_days` (entre 3 y 7 días), y explica en `ads_recommendation` objetivo y por qué conviene promocionarlo al público `'mujeres'`. Si es orgánico, pon `0.0` y `0`.
6. Automatización de DM por comentario (Instagram):
   - Solo activa `has_keyword_cta = true` si el copy invita explícitamente a comentar una palabra clave para enviar un recurso/enlace por DM (`keyword` en MAYÚSCULAS, `target_link`, `public_reply_text`). Si no aplica, `false` y `""`.
7. Publicación paralela orgánica en TikTok:
   - Para videos (`.mp4`), la publicación también puede enviarse a TikTok de forma **100% ORGÁNICA** (nunca promocionada ni con pauta pagada, incluso si `promote_in_ads = true` en Instagram).
   - Pon `publish_to_tiktok = true` para videos y `false` para imágenes estáticas.
   - Redacta `tiktok_caption` breve, directa y orgánica (con máximo 5 hashtags o sin ellos porque se combinan con `hashtags`). Importante: TikTok NO tiene envío automático de DM por palabra clave en comentarios, así que si el copy de Instagram pide comentar una palabra clave para recibir un DM, adapta `tiktok_caption` para invitar a comentar/guardar/ir al enlace del perfil en lugar de prometer un DM automático.
"""

REVISION_SYSTEM_INSTRUCTION = """Eres el Director de Contenido y Estratega de "Nel Health Coach" (@nelhealthcoach) conversando con Manuel (el administrador).

Cuando Manuel pide un cambio sobre una propuesta:
1. Si `force_apply` es True (o si Manuel está insistiendo claramente después de una sugerencia previa), DEBES obedecer exactamente lo que Manuel pide: pon `ai_assessment = "ACCEPTED"`, aplica sus cambios al pie de la letra en `proposal`, y confirma brevemente en `reasoning_message` (respetando siempre el límite de máximo 5 hashtags de Instagram, salvo que el texto de hashtags ya venga fijado en 5).
2. Si `force_apply` es False:
   - Razona críticamente el cambio pedido por Manuel evaluando: la regla estricta de contenido (videos grabados por Manuel a cámara = `MAIN` Lun/Mié/Vie; `PRO_TIP` Mar/Jue/Sáb = solo carrusel automático de textos/imágenes o imagen vertical de texto), horario óptimo según el tema, retención del copy/CTA, cumplimiento ético en salud, tendencias actuales de crecimiento orgánico, límite estricto de máximo 5 hashtags en Instagram y retorno en Meta Ads.
   - Si el cambio de Manuel es acertado o mejora la propuesta -> `ai_assessment = "ACCEPTED"`, aplícalo en `proposal` y explica en `reasoning_message` por qué es una buena decisión.
   - Si ves que lo que pide Manuel puede perjudicar el rendimiento (ej. mal horario para el tema, romper la cadencia semanal, perder el CTA, promocionar algo que funciona mejor orgánico, etc.) -> `ai_assessment = "ALTERNATIVE_SUGGESTED"`, explica claramente en `reasoning_message` por qué sugieres otra alternativa mejor, y entrega esa alternativa mejorada en `proposal` (para que Manuel decida si acepta tu alternativa o insiste en su cambio).
"""

TRENDS_IDEAS_SYSTEM_INSTRUCTION = """Eres el Director Creativo y Analista de Tendencias Virales de "Nel Health Coach" (@nelhealthcoach).
Tu objetivo es analizar las tendencias ACTUALIZADAS de Instagram Reels, TikTok y YouTube Shorts en el nicho de salud integrativa, metabolismo, nutrición, longevidad y hábitos saludables, y convertirlas en ideas de publicaciones nuevas con fuerza propia para ganar seguidores de forma orgánica.

Reglas obligatorias:
1. NO repitas temas que ya aparecen en el documento `Pautas` del mes actual.
2. En `viral_formats_to_replicate`, propón formatos de video viral concretos que Manuel pueda replicar fácilmente frente a cámara o con B-roll simple (estructura segundo a segundo, gancho visual/texto en pantalla, ritmo/audio).
3. En `new_post_ideas`, propón de 4 a 5 ideas de pautas con alto potencial orgánico por sí mismas (para provocar guardados, compartidos y nuevos seguidores):
   - Respeta la distinción `MAIN` (video con Manuel a cámara para Lun/Mié/Vie) y `PRO_TIP` (carrusel texto/imagen o story sin presentador para Mar/Jue/Sáb).
   - Incluye un gancho de 3 segundos irresistible (`viral_hook`), explicación de por qué funciona según las tendencias actuales (`why_it_trends`), guía práctica de grabación/producción (`recording_guide`), borrador de `suggested_caption` con CTA orgánico, y **exactamente 5 hashtags (máximo estricto 5)** sin promesas médicas milagrosas.
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
        description="Cuerpo del copy optimizado para Instagram con llamada a la acción clara (sin hashtags)."
    )
    hashtags: str = Field(
        description="Exactamente 5 hashtags (máximo estricto de 5 por límite de Instagram), separados por espacios (ej. #NelHealthCoach #SaludMetabolica #Bienestar #HabitosSaludables #Metabolismo)."
    )
    trend_alignment: str = Field(
        default="",
        description="Cómo se adapta la pauta (gancho, ángulo o formato) a las tendencias actuales de redes sociales para crecer orgánicamente.",
    )
    media_type: Literal["REELS", "IMAGE", "STORIES", "CAROUSEL"] = Field(
        description="'REELS' para video en feed/reels, 'IMAGE' para imagen en feed, 'STORIES' para historia, o 'CAROUSEL' para carpeta carrusel multi-imagen."
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
    publish_to_tiktok: bool = Field(
        default=True,
        description="True si es un video elegible para publicarse también orgánicamente en TikTok; False para imágenes estáticas o si se omite TikTok.",
    )
    tiktok_caption: str = Field(
        default="",
        description="Descripción orgánica optimizada para TikTok (sin promesas de DM automático por comentario y con máximo 5 hashtags).",
    )

    @property
    def full_instagram_caption(self) -> str:
        """Return caption combined with hashtags cleanly for Instagram publication."""
        clean_caption, clean_hashtags = _enforce_caption_and_hashtags(
            self.caption,
            self.hashtags,
            max_count=MAX_INSTAGRAM_HASHTAGS,
        )
        if not clean_hashtags:
            return clean_caption
        return f"{clean_caption}\n\n{clean_hashtags}" if clean_caption else clean_hashtags

    @property
    def full_tiktok_caption(self) -> str:
        """Return organic TikTok caption combined with up to 5 hashtags."""
        base_text = self.tiktok_caption.strip() or self.caption.strip()
        clean_caption, clean_hashtags = _enforce_caption_and_hashtags(
            base_text,
            self.hashtags,
            max_count=MAX_INSTAGRAM_HASHTAGS,
        )
        if not clean_hashtags:
            return clean_caption
        return f"{clean_caption}\n\n{clean_hashtags}" if clean_caption else clean_hashtags


class RevisionResult(BaseModel):
    """Structured reasoning and revised proposal when the admin requests changes."""

    ai_assessment: Literal["ACCEPTED", "ALTERNATIVE_SUGGESTED"] = Field(
        description="'ACCEPTED' si se aplica el cambio pedido; 'ALTERNATIVE_SUGGESTED' si el agente propone una opción mejor."
    )
    reasoning_message: str = Field(
        description="Explicación clara y argumentada para Manuel sobre su cambio o la alternativa sugerida."
    )
    proposal: PostProposal


class TrendPostIdea(BaseModel):
    """Actionable new post/reel idea grounded in current social media trends."""

    title: str = Field(description="Título corto y atractivo de la idea de publicación.")
    post_category: Literal["MAIN", "PRO_TIP"] = Field(
        description="'MAIN' para video grabado por Manuel a cámara (Lun/Mié/Vie) o 'PRO_TIP' para carrusel texto/imagen o story (Mar/Jue/Sáb)."
    )
    viral_hook: str = Field(
        description="Gancho exacto de los primeros 3 segundos (texto en pantalla y/o frase de apertura)."
    )
    why_it_trends: str = Field(
        description="Por qué esta idea tiene fuerza orgánica propia según las tendencias actuales."
    )
    recording_guide: str = Field(
        description="Guía práctica de grabación/estructura para replicar el formato viral paso a paso."
    )
    suggested_caption: str = Field(
        description="Borrador del caption con CTA orientado a guardados, compartidos y seguidores."
    )
    hashtags: str = Field(
        description="Exactamente 5 hashtags relevantes (máximo estricto 5), separados por espacios."
    )


class TrendReport(BaseModel):
    """Structured social media trends analysis and standalone post ideas."""

    market_summary: str = Field(
        description="Resumen conciso de qué formatos, ganchos y temas están funcionando hoy orgánicamente en salud/bienestar."
    )
    viral_formats_to_replicate: list[str] = Field(
        description="Lista de 2 a 3 estructuras/formatos virales concretos replicables por Manuel."
    )
    new_post_ideas: list[TrendPostIdea] = Field(
        description="Lista de 4 a 5 ideas nuevas de pautas con alto potencial orgánico."
    )


class VideoScriptResult(BaseModel):
    """Ready-to-film 45-60s spoken script for Manuel Martínez."""

    title: str = Field(description="Título corto del guion.")
    format_type: Literal["MAIN", "PRO_TIP"] = Field(
        description="'MAIN' para video a cámara (Lun/Mié/Vie) o 'PRO_TIP' para texto/carrusel (Mar/Jue/Sáb)."
    )
    hook_0_3s: str = Field(description="Frase exacta o acción de los primeros 3 segundos.")
    body_3_45s: str = Field(description="Guion hablado / desarrollo claro y empático de 3 a 45 segundos.")
    cta_45_60s: str = Field(description="Cierre y llamado a la acción orgánico (45-60s).")
    recording_notes: str = Field(description="Indicaciones prácticas de grabación, ritmo o B-roll (sin textos de portada).")
    caption: str = Field(description="Caption listo para Instagram y TikTok (sin hashtags).")
    hashtags: str = Field(description="Exactamente 5 hashtags relevantes (máximo estricto 5), separados por espacios.")


class WeeklyMetricsReport(BaseModel):
    """Structured weekly analytics report and learned insights for Google Docs & memory."""

    executive_summary: str = Field(description="Resumen ejecutivo breve de la semana.")
    report_document_text: str = Field(
        description="Informe completo en texto plano limpio y bien estructurado en español listo para guardar en Google Docs."
    )
    winning_insights: str = Field(
        description="Aprendizajes clave (temas, formatos, horarios y ganchos ganadores) para mejorar futuras propuestas."
    )


class IgReplySuggestions(BaseModel):
    """AI-suggested reply options for an incoming Instagram comment or DM."""

    intent_summary: str = Field(
        description="Resumen corto de la intención del usuario (ej. 'Consulta de precios', 'Duda metabólica', 'Agradecimiento')."
    )
    options: list[str] = Field(
        description="Lista de 2 a 3 respuestas sugeridas en español, empáticas, claras y sin promesas médicas milagrosas."
    )


def normalize_hashtags(
    raw_hashtags: str,
    max_count: int = MAX_INSTAGRAM_HASHTAGS,
) -> str:
    """Normalize, deduplicate, and strictly cap hashtags to `max_count` (Instagram limit: 5)."""
    if max_count <= 0 or not raw_hashtags.strip():
        return ""

    tokens = re.findall(r"#?[\wáéíóúüñÁÉÍÓÚÜÑ]+", raw_hashtags)
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        clean = token.lstrip("#").strip()
        if not clean or clean.isdigit():
            continue
        tag = f"#{clean}"
        lowered = tag.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(tag)
        if len(result) >= max_count:
            break
    return " ".join(result)


def _enforce_caption_and_hashtags(
    caption: str,
    hashtags: str,
    max_count: int = MAX_INSTAGRAM_HASHTAGS,
) -> tuple[str, str]:
    """Strip trailing hashtag blocks from caption and cap total unique hashtags at `max_count`."""
    lines = caption.strip().splitlines()
    extracted_trailing: list[str] = []

    while lines:
        last_line = lines[-1].strip()
        if not last_line:
            lines.pop()
            continue
        words = last_line.split()
        if words and all(w.startswith("#") for w in words):
            extracted_trailing.insert(0, last_line)
            lines.pop()
            continue
        break

    clean_caption = "\n".join(lines).strip()
    inline_tags = re.findall(r"#[\wáéíóúüñÁÉÍÓÚÜÑ]+", clean_caption)
    inline_seen = {t.lower() for t in inline_tags}
    remaining_budget = max(0, max_count - len(inline_seen))

    combined_raw = " ".join(part for part in [hashtags, *extracted_trailing] if part.strip())
    normalized_pool = normalize_hashtags(combined_raw, max_count=max_count + len(inline_seen))
    filtered_tags: list[str] = []
    for tag in normalized_pool.split():
        if tag.lower() in inline_seen:
            continue
        if len(filtered_tags) >= remaining_budget:
            break
        filtered_tags.append(tag)

    return clean_caption, " ".join(filtered_tags)


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


_STORY_FILENAME_RE = re.compile(
    r"^(?:\d+\s*[\.\-\)]\s*)?(?:[\(\[]\s*)?(?:historias?|story|stories)(?:\s*[\)\]])?(?:[\s._\-:]+|$)",
    re.IGNORECASE,
)


def is_story_filename(filename: str) -> bool:
    """Return True when a Drive filename explicitly marks the asset as an Instagram Story."""
    stem = Path(filename.strip()).stem
    decomposed = unicodedata.normalize("NFKD", stem)
    ascii_stem = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower().strip()
    return bool(_STORY_FILENAME_RE.match(ascii_stem))


def _normalize_media_type(
    proposed_type: Literal["REELS", "IMAGE", "STORIES", "CAROUSEL"],
    file_metadata: dict[str, Any],
    *,
    allow_story_override: bool = False,
) -> Literal["REELS", "IMAGE", "STORIES", "CAROUSEL"]:
    if file_metadata.get("is_carousel"):
        return "CAROUSEL"
    file_name = str(file_metadata.get("name", ""))
    if not allow_story_override and is_story_filename(file_name):
        if proposed_type != "STORIES":
            logger.info(
                "Forzando media_type=STORIES para archivo de historia '%s' (Gemini propuso %s).",
                file_name,
                proposed_type,
            )
        return "STORIES"
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
    *,
    allow_story_override: bool = False,
) -> PostProposal:
    media_type = _normalize_media_type(
        proposal.media_type,
        file_metadata,
        allow_story_override=allow_story_override,
    )

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
    if media_type == "STORIES":
        raw_short = re.sub(r"^(?:historias?|story|stories)\s*[\.\-\s:]*", "", raw_short, flags=re.IGNORECASE)
    if not raw_short:
        raw_short = Path(str(file_metadata.get("name", "Publicacion"))).stem

    promote = bool(proposal.promote_in_ads) and media_type != "STORIES"
    daily_budget = round(max(1.0, proposal.ads_daily_budget_usd), 2) if promote else 0.0
    duration_days = max(1, int(proposal.ads_duration_days)) if promote else 0
    ads_recommendation = (
        "No aplica (Historia de 24h en Instagram)."
        if media_type == "STORIES"
        else proposal.ads_recommendation.strip()
    )

    post_category = proposal.post_category
    if proposal.has_presenter_on_camera and post_category != "MAIN":
        logger.warning(
            "Corrigiendo post_category=%s a MAIN porque '%s' tiene presentador a cámara (regla: Pro Tips nunca son videos grabados por Manuel).",
            post_category,
            file_metadata.get("name"),
        )
        post_category = "MAIN"

    if media_type == "STORIES":
        clean_caption = re.sub(r"(?<!\S)#[\wáéíóúüñÁÉÍÓÚÜÑ]+", "", proposal.caption).strip()
        clean_hashtags = ""
    else:
        clean_caption, clean_hashtags = _enforce_caption_and_hashtags(
            proposal.caption,
            proposal.hashtags,
            max_count=MAX_INSTAGRAM_HASHTAGS,
        )
    is_video = _is_video_file(file_metadata) and not bool(file_metadata.get("is_carousel"))
    publish_to_tiktok = bool(proposal.publish_to_tiktok) and is_video and media_type != "STORIES"
    raw_tiktok_caption = proposal.tiktok_caption.strip()
    if media_type == "STORIES":
        clean_tiktok_caption = ""
    else:
        clean_tiktok_caption, _ = _enforce_caption_and_hashtags(
            raw_tiktok_caption or clean_caption,
            clean_hashtags,
            max_count=MAX_INSTAGRAM_HASHTAGS,
        )

    return PostProposal(
        content_summary=proposal.content_summary.strip(),
        has_presenter_on_camera=bool(proposal.has_presenter_on_camera),
        post_category=post_category,
        short_title=raw_short.strip(),
        hook_title=proposal.hook_title.strip(),
        caption=clean_caption,
        hashtags=clean_hashtags,
        trend_alignment=proposal.trend_alignment.strip(),
        media_type=media_type,
        scheduled_time=parsed_dt.isoformat(),
        schedule_reasoning=proposal.schedule_reasoning.strip(),
        promote_in_ads=promote,
        ads_daily_budget_usd=daily_budget,
        ads_duration_days=duration_days,
        ads_recommendation=ads_recommendation,
        has_keyword_cta=has_cta,
        keyword=keyword if has_cta else "",
        target_link=target_link,
        public_reply_text=public_reply_text,
        publish_to_tiktok=publish_to_tiktok,
        tiktok_caption=clean_tiktok_caption,
    )


def fetch_social_trends_brief(
    current_time: datetime | None = None,
    api_key: str | None = None,
    model: str | None = None,
    force_refresh: bool = False,
) -> str:
    """Research up-to-date social media trends for health/wellness Reels via Google Search grounding."""
    now_mono = time.monotonic()
    with _TRENDS_CACHE_LOCK:
        cached_brief = str(_TRENDS_CACHE.get("brief", ""))
        fetched_at = float(_TRENDS_CACHE.get("fetched_at", 0.0))
        if (
            not force_refresh
            and cached_brief
            and (now_mono - fetched_at) < TRENDS_CACHE_TTL_SECONDS
        ):
            return cached_brief

    now = current_time or datetime.now(resolve_timezone())
    client, resolved_model = _resolve_client_and_model(api_key, model)
    search_prompt = (
        f"Fecha actual: {now.strftime('%Y-%m-%d')} ({now.strftime('%B %Y')}).\n"
        "Investiga en la web las tendencias ACTUALES de Instagram (Reels, imágenes/carruseles e Historias), TikTok y YouTube Shorts "
        "en español (y formatos virales adaptables al mercado hispano/US Hispanic) para el nicho de "
        "salud integrativa, metabolismo, nutrición, longevidad, sueño, estrés y hábitos saludables.\n\n"
        "Entrega un resumen práctico y accionable con:\n"
        "1. Temas y preguntas de salud que están generando más alcance orgánico, guardados y compartidos ahora mismo.\n"
        "2. Formatos y estilos virales replicables tanto en video (Reels/TikTok: ganchos de 3s, POV, mito vs. realidad, storytelling, listas rápidas) "
        "como en piezas estáticas/Pro Tips (imágenes, frases de alto guardado, historias interactivas).\n"
        "3. Tendencias de algoritmo en Instagram (límite actual de 5 hashtags estratégicos, retención, CTAs para "
        "compartir/guardar y ganar seguidores orgánicos).\n"
        "Sé concreto, actualizado y enfocado en qué puede aplicar o grabar Manuel (@nelhealthcoach)."
    )

    brief = ""
    try:
        logger.info("Consultando tendencias actualizadas de redes sociales con Google Search...")
        response = client.models.generate_content(
            model=resolved_model,
            contents=search_prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.4,
            ),
        )
        brief = (response.text or "").strip()
    except Exception as exc:
        logger.warning(
            "No se pudo consultar Google Search para tendencias (%s); intentando análisis sin grounding.",
            exc,
        )
        try:
            fallback_resp = client.models.generate_content(
                model=resolved_model,
                contents=search_prompt,
                config=types.GenerateContentConfig(temperature=0.4),
            )
            brief = (fallback_resp.text or "").strip()
        except Exception as fallback_exc:
            logger.warning("Fallback de tendencias también falló: %s", fallback_exc)
            brief = ""

    if brief:
        with _TRENDS_CACHE_LOCK:
            _TRENDS_CACHE["brief"] = brief
            _TRENDS_CACHE["fetched_at"] = time.monotonic()
    return brief


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
    trends_brief: str | None = None,
    performance_memory: str = "",
    media_file_paths: list[Path] | None = None,
) -> PostProposal:
    """Analyze the actual media file (when provided) + guidelines + live trends and propose a post."""
    now = current_time or datetime.now(resolve_timezone())
    file_name = str(file_metadata.get("name", file_metadata.get("id", "")))
    file_meta_json = json.dumps(file_metadata, ensure_ascii=False, indent=2)
    occupied_block = (
        "\n".join(f"- {slot}" for slot in occupied_slots)
        if occupied_slots
        else "Ninguno adicional fuera del documento."
    )

    if trends_brief is None:
        trends_brief = fetch_social_trends_brief(
            current_time=now,
            api_key=api_key,
            model=model,
            force_refresh=False,
        )
    trends_section = (
        trends_brief.strip()
        if trends_brief and trends_brief.strip()
        else "Aplica mejores prácticas vigentes de ganchos de 3 segundos, retención, CTAs para guardados/compartidos y máximo 5 hashtags."
    )
    memory_section = (
        f"\n=== APRENDIZAJES DE MÉTRICAS REALES PREVIAS ===\n{performance_memory.strip()}\n"
        if performance_memory and performance_memory.strip()
        else ""
    )

    prompt = (
        f"Fecha y hora actual de referencia: {now.isoformat()}\n"
        f"Archivo nuevo SIN pauta previa: {file_name}\n\n"
        f"=== HORARIOS YA OCUPADOS EN PROPUESTAS/APROBADOS ===\n{occupied_block}\n\n"
        f"=== TENDENCIAS ACTUALES EN REDES SOCIALES (REELS/TIKTOK/SHORTS) ===\n{trends_section}\n"
        f"{memory_section}\n"
        f"=== PAUTAS DEL MES (REFERENCIA DE ESTILO, MARCA Y PUBLICACIONES EXISTENTES) ===\n{guidelines_text}\n\n"
        f"=== METADATOS DEL ARCHIVO EN DRIVE ===\n{file_meta_json}\n\n"
        "Analiza visualmente el archivo multimedia adjunto (si está disponible) y genera una propuesta inteligente:\n"
        "- Si el nombre del archivo empieza o viene marcado como Historia (`historia ...`, `story ...`, `(HISTORIA)`), clasifícalo OBLIGATORIAMENTE como `media_type = \"STORIES\"` (NUNCA `REELS`, ni TikTok, ni Ads).\n"
        "- Si NO es una Historia, verifica primero si es un video grabado por Manuel Martínez (persona hablando a cámara) -> OBLIGATORIAMENTE `MAIN` en Lun/Mié/Vie (`REELS`).\n"
        "- Solo usa `PRO_TIP` (Mar/Jue/Sáb) si es un video corto de textos e imágenes estáticas tipo carrusel automático SIN persona a cámara, o una imagen vertical de texto.\n"
        "- Adapta obligatoriamente (tanto para videos como para imágenes nuevas) el gancho (`hook_title`), el enfoque del `caption`, el `trend_alignment` y los hashtags para que esta pauta automática siga las tendencias actuales del momento y gane seguidores orgánicos.\n"
        "- Razona la mejor fecha y hora sin copiar ciegamente horas fijas anteriores.\n"
        "- OBLIGATORIO: Entrega exactamente 5 hashtags (NUNCA más de 5) en `hashtags`, y NO pongas hashtags dentro de `caption`.\n"
        "- Decide si amerita promoción en Meta Ads con presupuesto y duración sugeridos."
    )

    client, resolved_model = _resolve_client_and_model(api_key, model)
    uploaded_files: list[Any] = []
    candidate_paths: list[Path] = []
    if media_file_paths:
        candidate_paths.extend(p for p in media_file_paths if p is not None and p.is_file())
    elif media_file_path is not None and media_file_path.is_file():
        candidate_paths.append(media_file_path)

    try:
        contents: Any = prompt
        if candidate_paths:
            for mpath in candidate_paths[:5]:
                try:
                    uploaded_files.append(_upload_media_to_gemini(client, mpath))
                except Exception as upload_exc:
                    logger.warning(
                        "No se pudo analizar '%s' multimodalmente (%s); continuando.",
                        mpath.name,
                        upload_exc,
                    )
            if uploaded_files:
                contents = [*uploaded_files, prompt]

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
        for uf in uploaded_files:
            _delete_gemini_file_quietly(client, uf)

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
    trends_brief: str | None = None,
) -> RevisionResult:
    """Critically evaluate admin change requests, reason alternatives, and respect insistence."""
    now = current_time or datetime.now(resolve_timezone())
    file_name = str(file_metadata.get("name", file_metadata.get("id", "")))
    file_meta_json = json.dumps(file_metadata, ensure_ascii=False, indent=2)
    previous_json = previous_proposal.model_dump_json(indent=2)
    history_block = (
        "\n".join(f"- {item}" for item in revision_history)
        if revision_history
        else "Primera solicitud de cambio sobre esta propuesta."
    )
    if trends_brief is None:
        with _TRENDS_CACHE_LOCK:
            trends_brief = str(_TRENDS_CACHE.get("brief", ""))
    trends_block = (
        f"\n=== TENDENCIAS ACTUALES EN REDES ===\n{trends_brief.strip()}\n"
        if trends_brief and trends_brief.strip()
        else ""
    )

    prompt = (
        f"Fecha y hora actual de referencia: {now.isoformat()}\n"
        f"Archivo multimedia: {file_name}\n"
        f"Modo force_apply (obedecer sin objetar): {force_apply}\n\n"
        f"=== HISTORIAL DE REVISIONES ===\n{history_block}\n\n"
        f"=== PAUTAS DEL MES (GOOGLE DOC) ===\n{guidelines_text}\n"
        f"{trends_block}\n"
        f"=== METADATOS DEL ARCHIVO EN DRIVE ===\n{file_meta_json}\n\n"
        f"=== PROPUESTA ANTERIOR ===\n{previous_json}\n\n"
        f"=== CAMBIO SOLICITADO POR MANUEL ===\n{feedback.strip()}\n\n"
        "Evalúa el cambio solicitado, razona si conviene o si existe una alternativa mejor "
        "(a menos que force_apply=True o Manuel esté insistiendo), y devuelve la respuesta estructurada. "
        "Recuerda mantener siempre el límite ESTRICTO de máximo 5 hashtags en `hashtags` y ninguno dentro de `caption`."
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

    explicit_format_switch = bool(
        re.search(r"\b(reel|reels|feed|carrusel|imagen|publicaci[oó]n normal|historia|story|stories)\b", feedback, re.IGNORECASE)
    )
    normalized = _normalize_proposal(
        result.proposal,
        file_metadata,
        now,
        allow_story_override=force_apply or explicit_format_switch,
    )
    assessment: Literal["ACCEPTED", "ALTERNATIVE_SUGGESTED"] = (
        "ACCEPTED" if force_apply else result.ai_assessment
    )
    return RevisionResult(
        ai_assessment=assessment,
        reasoning_message=result.reasoning_message.strip(),
        proposal=normalized,
    )


def generate_trend_ideas(
    guidelines_text: str,
    current_time: datetime | None = None,
    api_key: str | None = None,
    model: str | None = None,
    force_refresh: bool = False,
    performance_memory: str = "",
) -> TrendReport:
    """Analyze live social media trends and suggest standalone viral/organic post ideas."""
    now = current_time or datetime.now(resolve_timezone())
    trends_brief = fetch_social_trends_brief(
        current_time=now,
        api_key=api_key,
        model=model,
        force_refresh=force_refresh,
    )
    trends_section = (
        trends_brief
        if trends_brief
        else "No se pudo consultar búsqueda en vivo; utiliza tu conocimiento actualizado sobre formatos virales de Reels/TikTok en salud integrativa."
    )
    memory_section = (
        f"\n=== APRENDIZAJES DE MÉTRICAS REALES PREVIAS ===\n{performance_memory.strip()}\n"
        if performance_memory and performance_memory.strip()
        else ""
    )

    prompt = (
        f"Fecha actual: {now.isoformat()}\n\n"
        f"=== TENDENCIAS ACTUALIZADAS EN REDES SOCIALES ===\n{trends_section}\n"
        f"{memory_section}\n"
        f"=== PAUTAS DEL MES ACTUAL (TEMAS YA CUBIERTOS — NO REPETIR) ===\n"
        f"{guidelines_text or 'Aún no hay pautas registradas este mes.'}\n\n"
        "Con base en las tendencias actuales, genera:\n"
        "1. Un resumen de qué está funcionando orgánicamente ahora mismo (`market_summary`).\n"
        "2. 2 a 3 formatos virales replicables para que Manuel grabe y gane seguidores orgánicos (`viral_formats_to_replicate`).\n"
        "3. 4 a 5 ideas de nuevas publicaciones/pautas con fuerza propia (`new_post_ideas`), sin repetir temas existentes en `Pautas`, "
        "y respetando el límite estricto de máximo 5 hashtags por idea."
    )

    client, resolved_model = _resolve_client_and_model(api_key, model)
    logger.info("Generando reporte de tendencias e ideas virales para @nelhealthcoach...")
    try:
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=TRENDS_IDEAS_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=TrendReport,
                temperature=0.65,
            ),
        )
    except Exception as exc:
        raise GeminiAgentError(
            f"Error al generar ideas de tendencias con Gemini ({resolved_model}): {exc}"
        ) from exc

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, TrendReport):
        report = parsed
    elif response.text:
        try:
            report = TrendReport.model_validate_json(response.text)
        except Exception as exc:
            raise GeminiAgentError(
                f"La respuesta de tendencias de Gemini no cumple el esquema esperado: {exc}"
            ) from exc
    else:
        raise GeminiAgentError("Gemini devolvió un reporte de tendencias vacío.")

    normalized_ideas: list[TrendPostIdea] = []
    for idea in report.new_post_ideas:
        clean_caption, clean_hashtags = _enforce_caption_and_hashtags(
            idea.suggested_caption,
            idea.hashtags,
            max_count=MAX_INSTAGRAM_HASHTAGS,
        )
        normalized_ideas.append(
            TrendPostIdea(
                title=idea.title.strip(),
                post_category=idea.post_category,
                viral_hook=idea.viral_hook.strip(),
                why_it_trends=idea.why_it_trends.strip(),
                recording_guide=idea.recording_guide.strip(),
                suggested_caption=clean_caption,
                hashtags=clean_hashtags,
            )
        )

    return TrendReport(
        market_summary=report.market_summary.strip(),
        viral_formats_to_replicate=[f.strip() for f in report.viral_formats_to_replicate if f.strip()],
        new_post_ideas=normalized_ideas,
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
        "- Hashtags: Máximo estricto de 5 hashtags por publicación (límite actual de Instagram).\n"
        "- La IA analiza cada archivo junto a las tendencias vigentes y sugiere fecha, horario óptimo, copy, 5 hashtags y si aplica Meta Ads.\n"
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
            "1. Resuma el tono de marca, cadencia semanal (Lun/Mié/Vie publicaciones principales; Mar/Jue/Sáb Pro Tips en video o historias), "
            "límite estricto de 5 hashtags por publicación (regla actual de Instagram) y criterios de Meta Ads.\n"
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


def generate_video_script(
    topic_or_idea: str,
    guidelines_text: str = "",
    api_key: str | None = None,
    model: str | None = None,
    performance_memory: str = "",
) -> VideoScriptResult:
    """Generate a ready-to-record 45-60s spoken script for Manuel without cover-text suggestions."""
    client, resolved_model = _resolve_client_and_model(api_key, model)
    memory_section = (
        f"\n=== APRENDIZAJES DE MÉTRICAS PREVIAS ===\n{performance_memory.strip()}\n"
        if performance_memory.strip()
        else ""
    )
    prompt = (
        f"Tema / Idea a convertir en guion: {topic_or_idea.strip()}\n"
        f"{memory_section}\n"
        f"=== REFERENCIA DE TONO Y PAUTAS DEL MES ===\n{guidelines_text or 'N/A'}\n\n"
        "Redacta un guion hablado listo para que Manuel Martínez (@nelhealthcoach) grabe en 45-60 segundos:\n"
        "- Tono cercano, empático, educativo en salud integrativa/metabolismo, sin promesas médicas milagrosas.\n"
        "- Gancho verbal de 0 a 3 segundos, desarrollo claro (3-45s) y cierre con CTA orgánico para guardados/compartidos (45-60s).\n"
        "- En `recording_notes`, da indicaciones prácticas de cámara/ritmo/B-roll, SIN sugerir textos de portada.\n"
        "- Entrega caption (sin hashtags) y exactamente 5 hashtags (máximo estricto 5)."
    )

    try:
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VideoScriptResult,
                temperature=0.6,
            ),
        )
    except Exception as exc:
        raise GeminiAgentError(f"Error generando guion con Gemini ({resolved_model}): {exc}") from exc

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, VideoScriptResult):
        res = parsed
    elif response.text:
        try:
            res = VideoScriptResult.model_validate_json(response.text)
        except Exception as exc:
            raise GeminiAgentError(f"Guion inválido devuelto por Gemini: {exc}") from exc
    else:
        raise GeminiAgentError("Gemini devolvió un guion vacío.")

    clean_caption, clean_hashtags = _enforce_caption_and_hashtags(
        res.caption,
        res.hashtags,
        max_count=MAX_INSTAGRAM_HASHTAGS,
    )
    return VideoScriptResult(
        title=res.title.strip(),
        format_type=res.format_type,
        hook_0_3s=res.hook_0_3s.strip(),
        body_3_45s=res.body_3_45s.strip(),
        cta_45_60s=res.cta_45_60s.strip(),
        recording_notes=res.recording_notes.strip(),
        caption=clean_caption,
        hashtags=clean_hashtags,
    )


def generate_weekly_metrics_report(
    report_title: str,
    period_label: str,
    ig_metrics: dict[str, Any],
    meta_ads_metrics: list[dict[str, Any]],
    tiktok_videos: list[dict[str, Any]],
    guidelines_text: str = "",
    api_key: str | None = None,
    model: str | None = None,
    previous_reports: list[dict[str, str]] | None = None,
    is_month_end_report: bool = False,
    month_label: str = "",
    is_month_start_week1: bool = False,
) -> WeeklyMetricsReport:
    """Analyze weekly IG, Meta Ads, and TikTok metrics matching prior report structure + comparative reflection."""
    client, resolved_model = _resolve_client_and_model(api_key, model)
    metrics_payload = json.dumps(
        {
            "instagram": ig_metrics,
            "meta_ads_campaigns": meta_ads_metrics,
            "tiktok_videos": tiktok_videos,
        },
        ensure_ascii=False,
        indent=2,
    )

    if previous_reports:
        prev_blocks = [
            f"--- {r.get('label', 'Semana previa')} | {r.get('title', '')} ---\n{r.get('text', '').strip()}"
            for r in previous_reports
            if r.get("text", "").strip()
        ]
        prev_section = "\n\n".join(prev_blocks)
    else:
        prev_section = "No se encontraron informes semanales previos en Drive."

    month_start_instruction = (
        "\n- INICIO DE MES (SEMANA 1): Como este informe corresponde a la Semana 1 de un mes nuevo y no hay semanas previas "
        "dentro de este mes, en la sección de análisis cualitativo y reflexivo DEBES comparar los resultados de estos 7 días "
        "directamente contra el cierre del mes anterior y su Resumen Mensual Consolidado (adjuntos arriba)."
        if is_month_start_week1
        else ""
    )

    month_end_instruction = (
        f"\n- CIERRE DE MES OBLIGATORIO: Este informe corresponde a la última semana de {month_label or 'este mes'}. "
        "Además del análisis semanal de los últimos 7 días, DEBES incluir al final una sección titulada "
        f"'RESUMEN MENSUAL CONSOLIDADO ({(month_label or 'DEL MES').upper()})' que compare la evolución semana a semana "
        "(desde la Semana 1 hasta esta última semana), destaque los contenidos ganadores del mes y deje conclusiones "
        "claras para arrancar el próximo mes."
        if is_month_end_report
        else ""
    )

    prompt = (
        f"Título del documento: {report_title}\n"
        f"Periodo evaluado (últimos 7 días / semana): {period_label}\n"
        f"Es Semana 1 de mes nuevo (comparar contra mes anterior): {is_month_start_week1}\n"
        f"Es informe de cierre de mes (incluir resumen mensual): {is_month_end_report}\n\n"
        f"=== INFORMES DE SEMANAS ANTERIORES (PLANTILLA DE ESTRUCTURA Y COMPARATIVA) ===\n{prev_section}\n\n"
        f"=== MÉTRICAS RECOPILADAS DE LOS ÚLTIMOS 7 DÍAS (IG + META ADS + TIKTOK) ===\n{metrics_payload}\n\n"
        f"=== PAUTAS DEL MES (CONTEXTO) ===\n{guidelines_text or 'N/A'}\n\n"
        "Instrucciones obligatorias para redactar `report_document_text` (texto plano limpio para Google Docs, sin Markdown de asteriscos ni HTML):\n"
        "1. ESTRUCTURA BASE: Si existen informes previos arriba (por ejemplo Semana 1, Semana 2 o la última semana del mes anterior), "
        "respeta y sigue fielmente esa misma estructura, encabezados y orden de secciones. Si no hubiera informes previos, usa una estructura clara con: "
        "Resumen Ejecutivo, Desempeño en Instagram, Rendimiento en Meta Ads, Desempeño Orgánico en TikTok, Análisis Cualitativo y Reflexivo, y Recomendaciones.\n"
        "2. ANÁLISIS CUALITATIVO Y REFLEXIVO COMPARATIVO: En la sección cualitativa y reflexiva, analiza las métricas de los 7 días previos "
        "COMPARÁNDOLAS explícitamente con las semanas anteriores (subidas, caídas, retención, guardados, compartidos, respuesta de la audiencia, "
        "y si es Semana 1 de un mes nuevo, compáralo contra el mes anterior usando su última semana y resumen mensual).\n"
        "3. TRANSPARENCIA: Nunca inventes cifras que no estén en el JSON recopilado ni en los informes previos; si una plataforma no reportó datos, indícalo claramente."
        f"{month_start_instruction}"
        f"{month_end_instruction}\n"
        "4. En `winning_insights`, resume en 3-5 líneas concretas qué temas, horarios o ganchos funcionaron mejor para mejorar futuras propuestas."
    )

    try:
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=WeeklyMetricsReport,
                temperature=0.35,
            ),
        )
    except Exception as exc:
        raise GeminiAgentError(
            f"Error generando informe semanal de métricas con Gemini ({resolved_model}): {exc}"
        ) from exc

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, WeeklyMetricsReport):
        rep = parsed
    elif response.text:
        try:
            rep = WeeklyMetricsReport.model_validate_json(response.text)
        except Exception as exc:
            raise GeminiAgentError(
                f"Respuesta inválida de informe semanal de Gemini: {exc}"
            ) from exc
    else:
        raise GeminiAgentError("Gemini devolvió un informe semanal vacío.")

    return WeeklyMetricsReport(
        executive_summary=rep.executive_summary.strip(),
        report_document_text=rep.report_document_text.strip(),
        winning_insights=rep.winning_insights.strip(),
    )


def generate_ig_reply_suggestions(
    *,
    interaction_type: Literal["COMMENT", "DM"],
    sender_label: str,
    message_text: str,
    post_context: str = "",
    previous_options: list[str] | None = None,
    admin_instruction: str = "",
    regenerate: bool = False,
    api_key: str | None = None,
    model: str | None = None,
) -> IgReplySuggestions:
    """Draft, refine, or freshly regenerate 2-3 on-brand reply options for an IG comment or DM."""
    client, resolved_model = _resolve_client_and_model(api_key, model)
    channel_desc = (
        "Comentario público en una publicación de Instagram"
        if interaction_type == "COMMENT"
        else "Mensaje Directo (DM) enviado a la cuenta @nelhealthcoach"
    )
    prev_block = ""
    if previous_options:
        numbered_prev = "\n".join(f"{i}. {opt}" for i, opt in enumerate(previous_options, start=1))
        prev_block = f"\nOpciones anteriores:\n{numbered_prev}\n"

    mode_block = ""
    if admin_instruction.strip():
        mode_block = (
            f"\nINSTRUCCIÓN DEL ADMINISTRADOR PARA REFINAR LAS OPCIONES:\n"
            f"{admin_instruction.strip()}\n"
            "Ajusta o reescribe las opciones obedeciendo fielmente esta instrucción.\n"
        )
    elif regenerate:
        mode_block = (
            "\nINSTRUCCIÓN ESPECIAL — REGENERAR DESDE CERO:\n"
            "Al administrador no le convencieron las opciones anteriores. DESCÁRTALAS por completo y "
            "propón 2 a 3 respuestas totalmente distintas en enfoque, apertura y estructura.\n"
        )

    prompt = (
        f"Canal: {channel_desc}\n"
        f"Remitente: {sender_label}\n"
        f"Contexto de la publicación (si aplica): {post_context or 'N/A'}\n"
        f"Mensaje exacto recibido:\n{message_text.strip()}\n"
        f"{prev_block}"
        f"{mode_block}\n"
        "Redacta 2 a 3 opciones de respuesta listas para enviar en nombre de Nel Health Coach (@nelhealthcoach):\n"
        "- Tono cercano, empático, profesional y claro en español.\n"
        "- Si es comentario público, mantén respuestas breves y amables (1-3 líneas); si es DM, pueden ser un poco más completas.\n"
        "- NUNCA hagas diagnósticos médicos ni promesas milagrosas de cura.\n"
        "- Si pregunta por precios, asesorías o detalles que debe confirmar el equipo, ofrece una respuesta cordial invitando a continuar por DM o dando el siguiente paso sin inventar tarifas que no conozcas."
    )

    try:
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=IgReplySuggestions,
                temperature=0.8 if regenerate else 0.5,
            ),
        )
    except Exception as exc:
        logger.warning("No se pudieron generar sugerencias de respuesta con Gemini: %s", exc)
        return IgReplySuggestions(
            intent_summary="Mensaje de seguidor",
            options=[
                "¡Hola! 💚 Gracias por escribirnos. En un momento te damos todos los detalles por aquí.",
                "¡Hola! Qué gusto saludarte 🙌 Cuéntanos un poco más para poder orientarte mejor.",
            ],
        )

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, IgReplySuggestions):
        res = parsed
    elif response.text:
        try:
            res = IgReplySuggestions.model_validate_json(response.text)
        except Exception:
            res = IgReplySuggestions(
                intent_summary="Mensaje de seguidor",
                options=[
                    "¡Hola! 💚 Gracias por escribirnos. En un momento te damos todos los detalles.",
                ],
            )
    else:
        res = IgReplySuggestions(
            intent_summary="Mensaje de seguidor",
            options=["¡Hola! 💚 Gracias por escribirnos. En un momento te respondemos."],
        )

    clean_opts = [opt.strip() for opt in res.options if opt.strip()][:3]
    if not clean_opts:
        clean_opts = ["¡Hola! 💚 Gracias por escribirnos. En un momento te damos más detalles."]
    return IgReplySuggestions(
        intent_summary=res.intent_summary.strip() or "Consulta de seguidor",
        options=clean_opts,
    )
