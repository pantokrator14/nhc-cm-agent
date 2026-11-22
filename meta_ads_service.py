"""Meta Marketing API v21.0 service for automated Instagram post promotions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from meta_publisher import GRAPH_API_BASE_URL, _extract_meta_error

logger = logging.getLogger(__name__)

DEFAULT_ADS_COUNTRIES = ["US", "MX", "CO", "ES", "AR", "CL", "DO", "VE"]
DEFAULT_SAVED_AUDIENCE_NAME = "mujeres"


class MetaAdsError(RuntimeError):
    """Raised when creating a Meta Ads promotion fails."""


@dataclass(frozen=True, slots=True)
class MetaAdResult:
    """Identifiers and budget summary of a created Meta Ads promotion."""

    ad_account_id: str
    campaign_id: str
    adset_id: str
    creative_id: str
    ad_id: str
    daily_budget_usd: float
    duration_days: int
    status: str
    audience_used: str = ""
    payment_method_summary: str = ""


async def list_accessible_ad_accounts(access_token: str) -> list[dict[str, Any]]:
    """Return active/accessible Meta Ad Accounts for the configured token."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
        response = await client.get(
            f"{GRAPH_API_BASE_URL}/me/adaccounts",
            params={
                "fields": "id,name,account_status,currency",
                "access_token": access_token,
            },
        )
        if response.is_error:
            raise MetaAdsError(
                f"No se pudieron listar cuentas publicitarias: {_extract_meta_error(response)}"
            )
        return list(response.json().get("data", []))


async def _resolve_facebook_page_id(
    client: httpx.AsyncClient,
    access_token: str,
    ig_user_id: str,
) -> str:
    response = await client.get(
        f"{GRAPH_API_BASE_URL}/me/accounts",
        params={
            "fields": "id,name,instagram_business_account",
            "access_token": access_token,
        },
    )
    if response.is_error:
        raise MetaAdsError(
            f"Error obteniendo páginas de Facebook vinculadas: {_extract_meta_error(response)}"
        )

    pages = response.json().get("data", [])
    for page in pages:
        ig_acc = page.get("instagram_business_account") or {}
        if str(ig_acc.get("id", "")) == str(ig_user_id):
            return str(page["id"])

    if pages:
        return str(pages[0]["id"])
    raise MetaAdsError(
        "No se encontró ninguna Página de Facebook vinculada a IG_USER_ID para crear el AdCreative."
    )


async def _check_boost_eligibility(
    client: httpx.AsyncClient,
    ig_media_id: str,
    access_token: str,
) -> None:
    response = await client.get(
        f"{GRAPH_API_BASE_URL}/{ig_media_id}",
        params={
            "fields": "boost_eligibility_info",
            "access_token": access_token,
        },
    )
    if response.is_error:
        logger.warning(
            "No se pudo verificar boost_eligibility_info de media_id=%s: %s",
            ig_media_id,
            _extract_meta_error(response),
        )
        return

    info = response.json().get("boost_eligibility_info") or {}
    eligible = info.get("eligible_to_boost")
    if eligible is False:
        reasons = info.get("boost_ineligible_reason") or "Sin razón especificada"
        raise MetaAdsError(
            f"La publicación {ig_media_id} no es elegible para promocionar en Meta Ads: {reasons}"
        )


def _default_fallback_targeting() -> dict[str, Any]:
    return {
        "geo_locations": {"countries": DEFAULT_ADS_COUNTRIES},
        "age_min": 25,
        "age_max": 54,
        "publisher_platforms": ["instagram"],
        "instagram_positions": ["stream", "reels", "story", "explore"],
    }


async def _inspect_ad_account_billing(
    client: httpx.AsyncClient,
    normalized_act: str,
    access_token: str,
) -> str:
    """Verify ad account is active and inspect its configured payment method (`funding_source`)."""
    response = await client.get(
        f"{GRAPH_API_BASE_URL}/{normalized_act}",
        params={
            "fields": "id,name,account_status,disable_reason,funding_source,funding_source_details,currency",
            "access_token": access_token,
        },
    )
    if response.is_error:
        logger.warning(
            "No se pudo inspeccionar facturación de %s: %s",
            normalized_act,
            _extract_meta_error(response),
        )
        return "Tarjeta predeterminada de la cuenta (no verificable por permisos)"

    payload = response.json()
    account_status = payload.get("account_status")
    if account_status is not None and int(account_status) != 1:
        disable_reason = payload.get("disable_reason", "N/A")
        raise MetaAdsError(
            f"La cuenta publicitaria {normalized_act} no está activa "
            f"(account_status={account_status}, disable_reason={disable_reason}). "
            "Verifica la tarjeta o el estado de facturación en Meta Ads Manager."
        )

    details = payload.get("funding_source_details")
    funding_source = payload.get("funding_source")
    if isinstance(details, dict) and details.get("display_string"):
        return str(details["display_string"]).strip()
    if funding_source:
        return f"Método de pago activo en cuenta (id={funding_source})"

    logger.warning(
        "La cuenta %s no devolvió funding_source (puede requerir permiso ads_read/finance o no tener tarjeta vinculada).",
        normalized_act,
    )
    return "Sin detalle visible (Meta cobrará al método principal configurado si existe)"


async def _resolve_saved_audience_targeting(
    client: httpx.AsyncClient,
    normalized_act: str,
    audience_name: str,
    access_token: str,
) -> tuple[dict[str, Any], str]:
    """Fetch Saved Audience by name (e.g. 'mujeres') and extract its `targeting` spec for Instagram."""
    target_norm = audience_name.strip().lower()
    if not target_norm:
        return _default_fallback_targeting(), "Predeterminado (sin nombre de público configurado)"

    response = await client.get(
        f"{GRAPH_API_BASE_URL}/{normalized_act}/saved_audiences",
        params={
            "fields": "id,name,targeting",
            "limit": 100,
            "access_token": access_token,
        },
    )
    if response.is_error:
        logger.warning(
            "No se pudieron consultar saved_audiences de %s (%s); usando targeting predeterminado.",
            normalized_act,
            _extract_meta_error(response),
        )
        return _default_fallback_targeting(), f"Predeterminado (error consultando '{audience_name}')"

    audiences = response.json().get("data", [])
    matched = next(
        (a for a in audiences if str(a.get("name", "")).strip().lower() == target_norm),
        None,
    )
    if matched is None:
        matched = next(
            (a for a in audiences if target_norm in str(a.get("name", "")).strip().lower()),
            None,
        )

    if matched is None or not isinstance(matched.get("targeting"), dict):
        logger.warning(
            "No se encontró el público guardado '%s' en %s; usando targeting predeterminado.",
            audience_name,
            normalized_act,
        )
        return _default_fallback_targeting(), f"Predeterminado (no se encontró '{audience_name}')"

    targeting = dict(matched["targeting"])
    for non_ig_key in (
        "facebook_positions",
        "messenger_positions",
        "audience_network_positions",
        "device_platforms",
    ):
        targeting.pop(non_ig_key, None)

    targeting["publisher_platforms"] = ["instagram"]
    targeting.setdefault("instagram_positions", ["stream", "reels", "story", "explore"])

    matched_name = str(matched.get("name", audience_name))
    matched_id = str(matched.get("id", ""))
    label = f"Público guardado '{matched_name}' (id={matched_id})"
    logger.info("Usando %s para Meta Ads en %s.", label, normalized_act)
    return targeting, label


async def inspect_meta_ads_setup(
    access_token: str,
    ad_account_id: str,
    saved_audience_name: str = DEFAULT_SAVED_AUDIENCE_NAME,
) -> tuple[str, str]:
    """Return `(payment_method_summary, audience_summary)` for diagnostics/previews."""
    if not ad_account_id:
        return "META_AD_ACCOUNT_ID no configurado", "N/A"
    normalized_act = (
        ad_account_id if ad_account_id.startswith("act_") else f"act_{ad_account_id}"
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
        payment_summary = await _inspect_ad_account_billing(client, normalized_act, access_token)
        _, audience_label = await _resolve_saved_audience_targeting(
            client, normalized_act, saved_audience_name, access_token
        )
        return payment_summary, audience_label


async def promote_instagram_post(
    access_token: str,
    ig_user_id: str,
    ad_account_id: str,
    ig_media_id: str,
    post_name: str,
    daily_budget_usd: float,
    duration_days: int,
    target_link: str = "",
    saved_audience_name: str = DEFAULT_SAVED_AUDIENCE_NAME,
) -> MetaAdResult:
    """Create Campaign -> AdSet (using saved audience + account card) -> AdCreative -> Ad in Meta Ads."""
    normalized_act = (
        ad_account_id if ad_account_id.startswith("act_") else f"act_{ad_account_id}"
    )
    safe_daily_usd = round(min(max(daily_budget_usd or 4.0, 1.0), 25.0), 2)
    safe_days = min(max(int(duration_days or 5), 1), 30)
    daily_budget_cents = int(round(safe_daily_usd * 100))

    now_utc = datetime.now(timezone.utc)
    start_time = (now_utc + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S+0000")
    end_time = (now_utc + timedelta(days=safe_days)).strftime("%Y-%m-%dT%H:%M:%S+0000")
    clean_name = post_name.strip()[:80] or f"IG_{ig_media_id}"

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        await _check_boost_eligibility(client, ig_media_id, access_token)
        payment_summary = await _inspect_ad_account_billing(client, normalized_act, access_token)
        targeting, audience_used = await _resolve_saved_audience_targeting(
            client,
            normalized_act,
            saved_audience_name,
            access_token,
        )
        page_id = await _resolve_facebook_page_id(client, access_token, ig_user_id)

        # 1. Campaign
        camp_resp = await client.post(
            f"{GRAPH_API_BASE_URL}/{normalized_act}/campaigns",
            data={
                "name": f"NHC - {clean_name}",
                "objective": "OUTCOME_ENGAGEMENT",
                "status": "ACTIVE",
                "special_ad_categories": "[]",
                "is_adset_budget_sharing_enabled": "false",
                "access_token": access_token,
            },
        )
        if camp_resp.is_error:
            raise MetaAdsError(
                f"Error creando Campaign en Meta Ads: {_extract_meta_error(camp_resp)}"
            )
        campaign_id = str(camp_resp.json()["id"])

        # 2. AdSet
        adset_resp = await client.post(
            f"{GRAPH_API_BASE_URL}/{normalized_act}/adsets",
            data={
                "name": f"NHC AdSet - {clean_name}",
                "campaign_id": campaign_id,
                "daily_budget": str(daily_budget_cents),
                "billing_event": "IMPRESSIONS",
                "optimization_goal": "POST_ENGAGEMENT",
                "bid_strategy": "LOWEST_COST_WITHOUT_CAP",
                "start_time": start_time,
                "end_time": end_time,
                "targeting": json.dumps(targeting),
                "status": "ACTIVE",
                "access_token": access_token,
            },
        )
        if adset_resp.is_error:
            raise MetaAdsError(
                f"Error creando AdSet en Meta Ads (campaign_id={campaign_id}): "
                f"{_extract_meta_error(adset_resp)}"
            )
        adset_id = str(adset_resp.json()["id"])

        # 3. AdCreative from published IG media
        creative_resp = await client.post(
            f"{GRAPH_API_BASE_URL}/{normalized_act}/adcreatives",
            data={
                "name": f"NHC Creative - {clean_name}",
                "object_id": page_id,
                "instagram_user_id": ig_user_id,
                "source_instagram_media_id": ig_media_id,
                "access_token": access_token,
            },
        )
        if creative_resp.is_error:
            raise MetaAdsError(
                f"Error creando AdCreative en Meta Ads (adset_id={adset_id}): "
                f"{_extract_meta_error(creative_resp)}"
            )
        creative_id = str(creative_resp.json()["id"])

        # 4. Ad
        ad_resp = await client.post(
            f"{GRAPH_API_BASE_URL}/{normalized_act}/ads",
            data={
                "name": f"NHC Ad - {clean_name}",
                "adset_id": adset_id,
                "creative": json.dumps({"creative_id": creative_id}),
                "status": "ACTIVE",
                "access_token": access_token,
            },
        )
        if ad_resp.is_error:
            raise MetaAdsError(
                f"Error creando Ad en Meta Ads (creative_id={creative_id}): "
                f"{_extract_meta_error(ad_resp)}"
            )
        ad_id = str(ad_resp.json()["id"])

    logger.info(
        "Promoción Meta Ads creada: ad_account=%s, campaign=%s, adset=%s, ad=%s ($%.2f/día x %d días, audience=%s, pago=%s).",
        normalized_act,
        campaign_id,
        adset_id,
        ad_id,
        safe_daily_usd,
        safe_days,
        audience_used,
        payment_summary,
    )
    return MetaAdResult(
        ad_account_id=normalized_act,
        campaign_id=campaign_id,
        adset_id=adset_id,
        creative_id=creative_id,
        ad_id=ad_id,
        daily_budget_usd=safe_daily_usd,
        duration_days=safe_days,
        status="ACTIVE",
        audience_used=audience_used,
        payment_method_summary=payment_summary,
    )


async def fetch_meta_ads_weekly_insights(
    access_token: str,
    ad_account_id: str,
    since_date: str,
    until_date: str,
) -> list[dict[str, Any]]:
    """Fetch campaign-level Meta Ads performance metrics for `[since_date, until_date]`."""
    if not ad_account_id:
        return []

    normalized_act = (
        ad_account_id if ad_account_id.startswith("act_") else f"act_{ad_account_id}"
    )
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
            response = await client.get(
                f"{GRAPH_API_BASE_URL}/{normalized_act}/insights",
                params={
                    "level": "campaign",
                    "fields": "campaign_id,campaign_name,spend,impressions,reach,clicks,cpc,cpm,actions",
                    "time_range": json.dumps({"since": since_date, "until": until_date}),
                    "limit": 50,
                    "access_token": access_token,
                },
            )
            if response.is_error:
                logger.warning(
                    "No se pudieron consultar métricas de Meta Ads (%s): %s",
                    normalized_act,
                    _extract_meta_error(response),
                )
                return []
            return list(response.json().get("data", []))
    except Exception as exc:
        logger.warning("Error consultando insights de Meta Ads (%s): %s", normalized_act, exc)
        return []
