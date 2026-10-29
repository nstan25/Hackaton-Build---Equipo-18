# Railway: de Config as Code (`railway.json`) a Infrastructure as Code (`.railway/`)

Railway deja de leer `railway.json` el **1/12/2026**. Este documento describe el archivo nuevo
(`.railway/railway.py`), cómo pasar los entornos **dev** y **production** del proyecto
`hopeful-cooperation` (servicio `bcs`) y cómo volver atrás. Preparado en T29 (29/10/2026); el
cutover lo hace el orquestador.

## Qué cambia

| | Antes (`railway.json`) | Después (`.railway/railway.py`) |
|---|---|---|
| Dónde vive la config | En el archivo subido con cada deploy; se aplica en el build | En los *settings* del servicio, escritos por `railway config apply` |
| Build | Dockerfile, watch patterns | Igual (`builder: DOCKERFILE`, `dockerfilePath: Dockerfile`, mismos watch patterns) |
| Deploy | `/health` 120 s, ON_FAILURE ×5, 1 réplica, `sleepApplication` | Igual (1 réplica en `iad`, Serverless) |
| Volumen `bcs-volume` en `/data` | No lo manejaba | Declarado con su estado actual (`iad`, 500 MB) para que IaC **no lo desmonte** |
| Dominios | No los manejaba | Fuera del archivo; el plan no los toca |
| Variables (`BCS_APP_TOKEN`, `OPENAI_API_KEY`, `BCS_SCALE`) | En Railway | **Unmanaged**: siguen solo en Railway; IaC no las lee ni las escribe |

Generado con `railway config migrate --lang py --variables unmanaged` (CLI 5.71.0) y ajustado a mano
en dos puntos que el generador perdía:

- **Reinicio ON_FAILURE ×5**: el generador no lo emitía; sin él el servicio quedaría con el default de Railway.
- **Volumen**: sin declararlo, el plan proponía desmontar `/data` (cambio destructivo: la base SQLite y
  los medios viven ahí). Declarado sin región ni tamaño, proponía borrarlos (`iad → null`, `500 → null`).
  Con `volume("bcs-volume", region="iad", sizeMB=500)` el plan no lo toca. Consecuencia: un resize del
  volumen a partir de ahora se hace en el archivo, no en el dashboard.

`PARTIAL = "bcs"`: el archivo solo gestiona sus propios recursos; nada más del proyecto queda bajo IaC.

### Por qué Python

`ts` (default) necesita node y en la VM no hay; tampoco hay `go`. `py` necesita el paquete PyPI
`railway-sdk`, que **no** es dependencia del proyecto (no está en `pyproject.toml` ni `uv.lock`): se
instala en un venv aparte, fuera del repo. `.railway/requirements.txt` lo declara para quien corra el
archivo en otra máquina.

## Requisito: venv de IaC (para `plan`, `apply`, `cutover`)

La CLI ejecuta `.railway/railway.py` con el `python` del `PATH`, que tiene que poder importar `railway_sdk`.

```bash
uv venv ~/bcs-shared/runs/T29/.venv-iac --python 3.12
VIRTUAL_ENV=~/bcs-shared/runs/T29/.venv-iac uv pip install 'railway-sdk==0.7.0'
export PATH=~/bcs-shared/runs/T29/.venv-iac/bin:$PATH     # en cada shell que use railway config
```

`railway-sdk` 0.7.0 · wheel `railway_sdk-0.7.0-py3-none-any.whl`
sha256 `2cabab22864dd36ea4538a5d14bc6c41f871cbfd05344182ec1a8a98b836b06c`.

Sin el venv, `railway config plan` falla con *"The Railway Python SDK is not installed"*.

## Estado actual (plan del 29/10/2026)

Salidas completas en `~/bcs-shared/runs/T29/plan-{production,dev}.txt` (sin valores de variables).
Los dos entornos dan lo mismo:

```
Plan: 0 to add, 2 to change, 0 to destroy
  ~ Update bcs build.builder, build.watchPatterns            (null → DOCKERFILE / lista de railway.json)
  ~ Update bcs deploy.healthcheckPath, healthcheckTimeout,
               restartPolicyMaxRetries, restartPolicyType    (null → /health, 120, 5, ON_FAILURE)
```

Son **cambios de forma**: hoy esos valores no están en los settings del servicio porque los pone
`railway.json` en cada build; `apply` los copia a los settings con los mismos valores. Comportamiento
efectivo: idéntico. Volumen, dominios, réplicas, Serverless y variables: sin cambios. Nada destructivo.
Después del `apply` el plan tiene que dar `No changes` (exit 0 con `--detailed-exit-code`).

`railway config migrate status` da *"No services in production/dev read Config as Code"* en los dos
entornos: el servicio no tiene un *Config File path* cargado y **no está conectado a GitHub** (deploys
con `railway up`), así que Railway lee el `railway.json` que viene en cada upload. Por eso:

- `railway config migrate cutover` probablemente no tenga ningún path que limpiar (no-op). El paso que
  de verdad saca a `railway.json` del medio es **borrarlo** del repo (ver abajo).
- Mientras `railway.json` siga en el repo, conviven las dos fuentes con **los mismos valores**: no hay
  conflicto. Si alguien cambia una, tiene que cambiar la otra hasta que se borre `railway.json`.
- Los watch patterns no tienen efecto sin fuente GitHub; se conservan por fidelidad.

## Cutover (primero dev, después production)

Siempre con el venv en el `PATH`. Desde un checkout de la rama correspondiente (`dev` para el entorno
dev, `main` para production) con `.railway/` ya mergeado.

### 1. dev

```bash
railway config plan --environment dev --verbose     # tiene que coincidir con el de arriba
railway config migrate status                       # con el link en dev: railway environment dev
railway config migrate cutover                      # si avisa que no hay nada que cortar, seguir
railway config apply --environment dev              # revisar el resumen antes de confirmar
railway config plan --environment dev --detailed-exit-code   # exit 0: sin cambios
```

Nota de la CLI: el cutover limpia el *Config File path* y **redeploya**. Si en este proyecto sí
encuentra algo que cortar, correr el `apply` enseguida: entre el cutover y el apply el servicio quedaría
sin healthcheck ni reinicio configurados.

### 2. Verificar dev

1. `railway config migrate status` → ningún servicio lee Config as Code.
2. Deploy: `railway up --environment dev --service bcs` desde la rama `dev`.
3. En el dashboard (Settings del servicio en dev): builder Dockerfile, healthcheck `/health` 120 s,
   restart On Failure ×5, 1 réplica en `iad`, **Serverless** activado, volumen en `/data`, dominio
   `bcs-dev.up.railway.app`.
4. `curl https://bcs-dev.up.railway.app/health` → 200 con el modelo real.
5. Serverless: después de ~10 min sin tráfico, `railway status --environment dev` muestra `Sleeping`;
   el primer request lo despierta.
6. Los datos siguen ahí (`/cows` con el token de dev devuelve el rodeo de antes).

### 3. production

Lo mismo con `--environment production` (y `railway environment production` para el link), deploy
desde `main`, dominio `bcs-production.up.railway.app`. Hacerlo solo con dev verificado.

## Rollback

- **Antes de borrar `railway.json`**: el `apply` solo copió a los settings los mismos valores que
  `railway.json`, así que no hay nada urgente que deshacer. Para volver al estado previo:
  `railway config migrate undo` (restaura los *Config File paths* guardados por el último cutover, si
  había alguno) y, si hace falta, vaciar en el dashboard los settings que puso el `apply`. Redeployar con
  `railway up` desde la rama del entorno.
- **Después de borrar `railway.json`**: revertir ese commit (`git revert`), `railway config migrate undo`
  y `railway up`. Hasta el 1/12/2026 Railway lo vuelve a leer; después, el único camino es arreglar
  `.railway/railway.py` y hacer `apply`.
- El volumen nunca debería verse afectado: si un plan llegara a mostrar `volumeAttachments` o
  `bcs-volume` con `→ null`, **no aplicar**.

## Cuándo borrar `railway.json`

Solo cuando **los dos** entornos estén aplicados y verificados (plan sin cambios en dev y production,
`/health` OK, Serverless OK). Borrarlo en un commit aparte, en `dev` y en `main`, antes del 1/12/2026
(`railway config migrate --delete-files` lo borra del disco; pushear/deployar solo después del cutover).
Desde ese momento la configuración del servicio se cambia **solo** en `.railway/railway.py` + `apply`.

## dev vs. production

El mismo archivo sirve para los dos entornos: la configuración es idéntica (mismo servicio, mismo
volumen lógico, 1 réplica, Serverless). Lo único que difiere es lo que IaC no maneja: dominio
(`bcs-dev` / `bcs-production`), valores de las variables y de qué rama se hace `railway up`. Como el
servicio no tiene fuente GitHub, no hay rama de origen que configurar. Si en algún momento se conecta
GitHub, cada entorno tendría que declarar su rama (`ctx.is_environment("dev")` en el archivo).
