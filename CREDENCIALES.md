# Guía de Credenciales y Despliegue (`nelhealthcoach-agent`)

## 1. Google Drive / Docs OAuth (`credentials/oauth_token.json`)

Cuando la carpeta raíz está en **Mi unidad** (no en una *Unidad compartida*), la *Service Account* tiene cuota `0 GB` y Google rechaza la creación de Google Docs nuevos (`403 storageQuotaExceeded`). Por eso se usa `GOOGLE_OAUTH_CREDENTIALS=./credentials/oauth_token.json` autenticado como `ceo@nelhealthcoach.com`.

### Cómo generar o renovar `credentials/oauth_token.json`
1. Asegúrate de tener `credentials/oauth_client.json` (descargado de Google Cloud Console → *Google Auth Platform → Clientes*).
2. Ejecuta en tu PC:
   ```bash
   .venv/bin/python scripts/google_oauth_setup.py
   ```
   *(Por SSH en servidor, usa `--manual` para pegar la URL de redirección).*
3. Abre la URL impresa, inicia sesión con `ceo@nelhealthcoach.com` y acepta los permisos de Drive y Docs.
4. En `.env` (local y servidor), verifica que esté configurado:
   ```env
   GOOGLE_OAUTH_CREDENTIALS=./credentials/oauth_token.json
   ```

> **Nota sobre expiración de 7 días (modo Prueba):**
> Si la app en Google Cloud Console sigue en estado **Prueba**, Google caduca el `refresh_token` cada **7 días**. El bot revisa diariamente la antigüedad del token y enviará un recordatorio por Telegram a partir del **día 5** (también puedes consultar el estado en cualquier momento con `/token`).

---

## 2. Sincronización con DigitalOcean tras cambios o renovaciones

El repositorio **no** sube `.env` ni `credentials/` (están en `.gitignore`). Cuando actualices credenciales localmente:

```bash
scp .env root@TU_IP_DROPLET:/root/nhc-cm-agent/.env
scp credentials/oauth_token.json root@TU_IP_DROPLET:/root/nhc-cm-agent/credentials/oauth_token.json
ssh root@TU_IP_DROPLET "systemctl restart nel-agent"
```

---

## 3. Claves de TikTok (`TIKTOK_*`)

1. Entra a [TikTok for Developers](https://developers.tiktok.com/) y crea/abre una app.
2. Agrega el producto **Login Kit** y **Content Posting API**.
3. Solicita los scopes:
   - `video.upload` (subida a borradores / inbox)
   - `video.publish` (publicación directa; requiere auditoría de TikTok)
   - `video.list` (para incluir métricas orgánicas de TikTok en `/metricas`)
4. Obtén `Client Key`, `Client Secret` y genera un `access_token` + `refresh_token` mediante el flujo OAuth v2 de TikTok.
5. Completa en `.env`:
   ```env
   TIKTOK_CLIENT_KEY=...
   TIKTOK_CLIENT_SECRET=...
   TIKTOK_REFRESH_TOKEN=...
   TIKTOK_ACCESS_TOKEN=
   # Usa INBOX mientras TikTok aprueba la auditoría; luego cambia a DIRECT
   TIKTOK_PUBLISH_MODE=INBOX
   TIKTOK_PRIVACY_LEVEL=PUBLIC_TO_EVERYONE
   ```
