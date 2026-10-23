# Cámara para el tambo: prueba de concepto punta a punta de ARGOS

Tarea: T22 (investigación). Rama: `investigacion-camaras`. Consulta web: **22/10/2026**.
No se compró nada, no se creó ninguna cuenta y no se tocó código.

**Convenciones**
- Los precios son fotos del día de consulta, tomados de buscadores y de publicaciones. MercadoLibre Uruguay
  bloquea los accesos automáticos con una verificación de cuenta, así que sus precios salen de los fragmentos
  que muestra el buscador. Hay que confirmarlos en la publicación antes de comprar.
- Tipo de cambio de referencia: **1 USD ≈ 40,3 UYU**. Es el cierre de agosto de 2026 (Infobae): no encontré
  la cotización de octubre.
- **(est.)** marca una estimación mía. **(sin verificar)** marca un dato que dice una sola fuente no oficial.

## 0. Resumen

- **Recomendación para la PoC:** cámara PoE **Reolink RLC-510A** (5 MP, FTP + RTSP + ONVIF, IP66) y un
  **router 4G con chip Antel**. La cámara se alimenta desde la sala de ordeñe, que ya tiene 220 V. En cada
  movimiento sube la foto por FTP a nuestro servidor. La conexión es saliente, así que funciona aunque el 4G
  esté detrás de CGNAT. Cuesta unos **US$ 130–150 si se importa** y **US$ 250–280 si se compra en Uruguay** (est.).
- **Si en el punto de captura no hay corriente:** una cámara 4G a batería + solar, la **Reolink Go Ultra en
  versión "America"** (B2/B4: sirve con Antel, Claro y Movistar). Reolink documenta que sube por FTP. Unos
  US$ 160–200 importada.
- **Alternativa sin SIM:** una **cámara de caza wifi/Bluetooth** (Usogood, Campark: US$ 36–57). Las fotos se
  descargan al celular con la app y desde ahí se suben a ARGOS con la web actual (`/predict`).
- **Bandas de Uruguay:** el LTE usa **700 MHz (B28)** y **AWS 1700/2100 (B4)** en Antel y Claro, y **700 (B28)**
  y **1900 (B2)** en Movistar. Un equipo "versión Europa" casi siempre trae B28, pero no B4 ni B2. Un equipo
  "versión EE. UU." trae B2/B4, pero no B28. Las cámaras chinas genéricas (B1/B3/B5/B7/B8/B20/B38–41) quedan
  **en riesgo**.
- **El `cow_id`:** la cámara no sabe qué vaca es. Para la PoC propongo una **bandeja de fotos "sin vaca"**: la
  API de ingesta guarda la foto y alguien le asigna el ID (costo US$ 0). La salida definitiva es un **lector de
  la caravana electrónica SNIG** (botón **HDX**, 134,2 kHz) en el brete, que se cruza con la foto por la hora.
  Un bastón Allflex RS420 sale **≈ US$ 1.161** en Uruguay.

## 1. Requisitos que salen del modelo

- **Dominio de entrenamiento:** `sdb_bcs` son frames de 1024×576 (16:9) de cámara fija, desde atrás y arriba
  (SPEC §1.2 y §0.2). Cualquier cámara de **2 MP o más** alcanza en resolución. Lo que pesa es **el encuadre**:
  el anca tiene que ocupar buena parte del cuadro. Con 2,5 m de distancia (est.):
  - un lente de 4 mm con ~80–90° horizontales cubre ~4–5 m de ancho;
  - un lente de 105–120° (Go Ultra, Camera Module 3 Wide) cubre ~6–8 m.
  Con un lente gran angular hay que **recortar** al bbox de la vaca, cosa que el pipeline ya hace (SPEC §1.1),
  y por eso conviene que la cámara tenga mucha resolución.
- **Color:** de noche, con IR, la imagen sale en **escala de grises**. El modelo se entrenó con color, así que
  es un salto de dominio nuevo, sin medir. Para la PoC: ordeñes con luz o un reflector blanco.
- **Texto sobreimpreso:** **apagar la marca de fecha (OSD)** de la cámara. Es la trampa nº 1 de `sdb_bcs`
  (SPEC §1.2). La hora tiene que viajar en los metadatos, no pintada en la imagen.

## 2. Tabla comparativa

UY = Uruguay. "Bandas UY" indica con qué operador puede funcionar según la §2.1.

| # | Modelo | Resolución / lente | Conectividad (bandas UY) | Cómo salen las imágenes | Alimentación | IP | Noche | Precio | Dónde |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Reolink RLC-510A** (PoE) + router 4G | 5 MP (2560×1920), lente fijo 4 mm | Ethernet; el 4G lo pone el router (ver 2.2) | **FTP** (foto/clip por evento), **RTSP**, ONVIF, API HTTP, microSD | PoE 802.3af (12 W) | IP66 | IR 18 LEDs (~30 m) | UY: **$ 5.900–6.900** (~US$ 147–171). Amazon: ~US$ 45–52 c/u (pack de 2 a US$ 84–105) | [ML UY](https://listado.mercadolibre.com.uy/reolink), [Slickdeals/Amazon](https://slickdeals.net/f/19080295-89-24-reolink-5mp-poe-security-camera-outdoor-indoor-surveillance-for-home-person-vehicle-pet-detection-100ft-ir-night-vision-work-with-smart-home-rlc-510a-pack-of-2), [specs](https://www.123securityproducts.com/cameras/rlc-510a.html) |
| 2 | **Reolink Go Ultra** (4G, batería) | 8 MP (3840×2160, 15 fps), 105° H | 4G nano-SIM. **Versión America:** B2/B4/B5/B12/B13/B14/B66/B71 → ✔ Antel/Claro (B4), ✔ Movistar (B2). **Versión Europa:** B1/B3/B7/B8/B20/B28 → B28 sí, B4 no | App/nube Reolink, **FTP** (figura en la lista oficial), email SMTP. **Sin RTSP** | Batería 28 Wh + panel solar 6 W o USB 5 V | IP65 | IR 10 m. PIR + detección de personas/animales | Walmart: **US$ 161,99** | [Reolink](https://reolink.com/product/reolink-go-ultra/), [FTP](https://support.reolink.com/hc/en-us/articles/900000625446-Which-Cameras-NVRs-Support-FTP-Uploading), [Walmart](https://www.walmart.com/ip/13357166146) |
| 3 | **Reolink Go PT Plus / Go Plus** (4G) | 5 MP PT / 4 MP fija | 4G. Las mismas variantes regionales que la #2 | App/nube, **FTP** (lista oficial) | Batería + solar | IP64–65 | IR | IT: €180–220 con solar. **UY: Go PT 2 MP US$ 439; "Go-LTE" US$ 369–379** | [ML UY](https://listado.mercadolibre.com.uy/reolink-argus-2), [smartworld](https://www.smartworld.it/tecnologia/reolink-go-pt-plus-4g-sim-pannello-solare-offerta.html) |
| 4 | **EZVIZ EB8 4G** (kit con panel solar) | 3 MP (2K), PT 360° | 4G. Bandas para UY **(sin verificar)** | **Solo nube/app EZVIZ**: no encontré FTP ni RTSP documentados | Batería 10.400 mAh + solar | IP65 (est.) | IR + color | **UY: US$ 344** (kit) | [ML UY](https://listado.mercadolibre.com.uy/camara-4g) |
| 5 | **Cámara solar 4G genérica** ("5 MP + SIM Antel + microSD") | 5 MP declarados | 4G, **se vende con chip Antel** | App P2P china (UBox/V380/CamHi, según el modelo). RTSP/ONVIF **a veces** | Batería + solar integrada | IP65–66 declarado | IR / color | UY: ~US$ 80–150 (est., por publicaciones parecidas) | [ML UY](https://www.mercadolibre.com.uy/shorts/clips/IVlFC2/camara-solar-4g-autonoma-5mp-sim-antel-microsd-64gb-gris) |
| 6 | **Suntek HC-940 Pro-Li 4G** (cámara de caza) | Video 4K, foto interpolada (marketing "36–120 MP") | 4G, con cualquier SIM. Bandas de modelos hermanos: B1/B3/B5/B7/B8/B20/B38/B40/B41 → **riesgo**: solo B3/B7, que Antel tendría según Kimovil **(sin verificar)** | **SMTP/email, FTP**, app | Pilas AA / 18650 / litio 8000 mAh; 12 V externo | IP65–66 | IR 25 m, 56 LEDs, disparo en 0,1 s | **UY: US$ 329** | [ML UY](https://listado.mercadolibre.com.uy/camara-trampa-caza), [specs](https://catalog.onliner.by/camera_traps/suntek/hc940pro2), [bandas HC-900LTE](https://www.hktdc.com/event/hkelectronicsfairae/en/product/1Z03E8976) |
| 7 | **TP-Link Tapo C320WS / C510W** + router 4G + equipo "edge" | 2K (2560×1440 / 2304×1296) | Wifi o Ethernet; el 4G lo pone el router | **RTSP/ONVIF** local, app Tapo. **No sube por FTP**: hace falta un equipo en el tambo que lea el stream y lo suba | 12 V / 9 V | IP66 / IP65 | IR + color | **UY: US$ 66–76** | [ML UY](https://listado.mercadolibre.com.uy/tplink-tapo-c320ws), [RS](https://th.rs-online.com/web/p/cctv-cameras/0638004) |
| 8 | **DIY Raspberry Pi** + Camera Module 3 Wide (NoIR) + HAT SIM7600G-H | 12 MP IMX708, 120°, HDR, autofoco | HAT 4G global: **B2/B4/B28 ✔** los tres operadores | Lo que programemos: **HTTPS directo a `/ingest`**, buffer offline | 5 V/5 A (red) o solar + batería (est.) | Depende del gabinete (IP65, ~US$ 30, est.) | Con NoIR + reflector IR (~US$ 20, est.) | Cámara US$ 35–60; HAT US$ 76–94; Pi 5 8 GB UY US$ 199–280. **Total UY ~US$ 350–420; importado ~US$ 200–250** (est.) | [Waveshare](https://www.waveshare.com/sim7600g-h-4g-hat.htm), [Pi Cam 3](https://www.sparkfun.com/raspberry-pi-camera-module-3-wide-angle.html), [Pi 5 ML UY](https://www.mercadolibre.com.uy/raspberry-pi-5-8-gb/p/MLU32625223) |
| 9 | **DIY ESP32**: LilyGO T-A7670SA + cámara OV2640/OV5640 | 2 MP (1600×1200) / 5 MP; poco rango dinámico | A7670SA: **B1/B2/B3/B4/B5/B7/B8/B28/B66 ✔** | HTTPS POST programado | 5 V; posible con solar | Gabinete aparte | Mala con poca luz | Placa US$ 33–49 + cámara ~US$ 10 (est.) | [LilyGO wiki](https://wiki.lilygo.cc/get_started/en/SIM/T-A7670/T-A7670SA/T-A7670SA.html) |
| 10 | **Cámara de caza wifi/Bluetooth sin SIM** (Usogood, Campark, Ceyomur) | "4K / 48 MP" de marketing (sensor real más chico, est.) | Wifi directo + BT con el celular (~10–20 m) | microSD → **app al celular** → se sube a ARGOS desde el celular | Pilas AA / panel solar (modelos solar) | IP66 | IR sin destello | Walmart/eBay: **US$ 36–57**. UY: cámaras de caza sin wifi desde $ 3.990 | [Usogood](https://www.ypc.co.uk/product/usogood-wildlife-camera-48mp-4k-30fps-wifi-trail-camera-with-night-vision-motion-activated-ip66-waterproof-120detection-65ft-range-connect-hunting-cam-with-phone-app-for-outdoor-garden-green/), [Campark/Walmart](https://www.walmart.com/ip/110448331), [ML UY](https://listado.mercadolibre.com.uy/camara-trampa-caza) |

**Descartadas**
- **Spypoint, Tactacam, Browning y otras cámaras de caza celulares de EE. UU.:** vienen con SIM propia,
  atada a su plan y a redes de EE. UU./Canadá (o UK/UE). No encontré soporte para Uruguay. Ejemplo: la
  Spypoint 4G "works with SIMs in the US, Canada and UK only"
  ([OpticsPlanet](https://www.opticsplanet.com/spy-point-10mp-4g-cellular-trail-camera.html)). Además, no
  exportan por FTP a un servidor propio.
- **Imou Cell Go:** es solo wifi. La Cell PT 4G existe, pero no encontré sus bandas completas ni si tiene FTP
  ([PriceSpy](https://pricespy.co.uk/s/imou-cameras/)).

### 2.1 Bandas LTE en Uruguay

| Operador | LTE (fuente: Telesemana con datos de URSEC, act. 15/10/2026) | Números de banda | Cuota |
|---|---|---|---|
| Antel | 700 MHz, 1700/2100 MHz | **B28, B4** (Kimovil agrega B1/B3/B7, sin verificar) | ~61 % |
| Claro | 700 MHz, 1700/2100 MHz | **B28, B4** | ~16,5 % |
| Movistar | 700 MHz, 1900 MHz | **B28, B2** | ~22,6 % |

Fuentes: [Telesemana](https://www.telesemana.com/panorama-de-mercado/uruguay/),
[Kimovil](https://www.kimovil.com/es/frequency-checker/UY), [cosmovil](https://cosmovil.com/bandas-y-frecuencias-de-operadores-moviles-uruguay/),
[Prensario, subasta 2017](https://www.prensario.net/20930-Uruguay-Claro-Movistar-y-Antel-reciben-nuevas-frecuencias-4G-LTE.note.aspx).
Las fuentes no coinciden del todo. No encontré la tabla oficial de URSEC.

**Regla práctica:** el equipo tiene que traer **B4 y B28** para Antel y Claro, o **B2 y B28** para Movistar.
Con una sola de las dos bandas también funciona, pero con menos cobertura.
- En el campo manda **B28 (700 MHz)**, que llega más lejos. Por eso la versión America de Reolink (sin B28)
  puede andar peor en un tambo alejado que en la ciudad.
- Los módulos "globales" (SIM7600G-H, A7670SA, Teltonika RUT241) traen las tres bandas.

### 2.2 Routers 4G para las opciones 1 y 7

| Router | Bandas | Precio | Fuente |
|---|---|---|---|
| Router 4G exterior IP65 "para Antel, Movistar, Claro" | Dice que funciona con los tres (sin verificar) | UY ~$ 4.200 (~US$ 104) | [ML UY](https://listado.mercadolibre.com.uy/router-4g-sim) |
| Cudy LT500 Outdoor (IP65, SMA para antena externa) | Confirmar la variante | UY ~US$ 85–230 (rango de la categoría) | [ML UY](https://www.mercadolibre.com.uy/router-lte-4g-para-intemperie-cudy-lt500-outdoor/p/MLU45816823) |
| Teltonika RUT241 (industrial) | B1–B5, B7, B8, B12, B13, B18–B20, B26, **B28** | ~£ 143 (UK) | [wlan-shop24](https://www.wlan-shop24.de/teltonika-rut241-global-4G-router) |
| TP-Link TL-MR100 | Versión TH: B1/B3/B5/B7/B8/B20 → **riesgo** | €42–54 | [Tapo TH](https://th.store.tapo.com/) |

Si la cámara va por PoE, además del router hace falta un inyector PoE 802.3af (~US$ 15–25, est.). Si el router
queda dentro de la sala, el cable Ethernet aguanta hasta 100 m.

### 2.3 Importar o comprar en Uruguay

- Desde el **1/5/2026** rige el Decreto 50/2026: franquicia de **3 envíos por año y US$ 800 acumulados por
  persona**. Fuera de la franquicia se paga una tasa única del 60 % (mínimo US$ 20). Los envíos desde EE. UU.
  de hasta US$ 200 cada uno están exentos de IVA por el acuerdo TIFA.
  ([FedEx, alerta regulatoria UY](https://www.fedex.com/content/dam/fedex/lac-latin-america/downloads/regulatory/UY_ES_2026_5_pdf_Regulatory-Alert-updated-28-April.pdf),
  [hacecuentas](https://hacecuentas.com/uy/finanzas/cuentas-y-compras)).
- El courier (Miami → Montevideo) se cobra por kilo. Para una cámara de < 1 kg calculo unos **US$ 15–30**
  (est.: no encontré tarifas publicadas).
- En Reolink, comprar en **Amazon EE. UU. trae la versión America** de bandas, que es la que sirve con B2/B4.
  Hay que confirmar en la publicación que diga "US version".

## 3. Recomendación

### 3.1 La más apropiada y barata para la PoC: RLC-510A + router 4G

| Ítem | Importado (est.) | En Uruguay (est.) |
|---|---|---|
| Reolink RLC-510A | US$ 45–52 | US$ 147–171 |
| Router 4G con B4/B28 (Cudy o similar; no el TL-MR100) | US$ 60–85 | US$ 85–104 |
| Inyector PoE + cable exterior 20 m + soporte | US$ 25–35 | US$ 35–50 |
| microSD 64 GB (respaldo local) | US$ 8 | US$ 12 |
| **Total** | **~US$ 140–180** + courier | **~US$ 280–340** |
| Chip Antel con datos | — | $ 888–1.390 por mes (§3.4) |

**Por qué esta opción**
- **Empuja las fotos a nuestro servidor:** FTP por evento, sin la nube del fabricante. También tiene
  **RTSP/ONVIF/HTTP** para cuando haya un equipo en el tambo: no ata a un fabricante.
- **Corriente fija de la sala de ordeñe:** captura sin límite de batería, con ráfagas o fotos por intervalo
  durante el ordeñe. El PIR de una cámara a batería, con 200 vacas por turno, gasta mucho.
- **IP66 y lente fijo:** encuadre estable, lo mismo que en `sdb_bcs`.
- **Repuestos baratos:** si falla, se cambia por cualquier cámara ONVIF con FTP (Hikvision, Dahua).

**Lo que no resuelve**
- El lente de 4 mm puede quedar abierto de más. Si sobra campo, conviene una **varifocal** (p. ej. Reolink
  RLC-511WA, 4× óptico, UY US$ 354 en
  [ML UY](https://articulo.mercadolibre.com.uy/MLU-681899150-reolink-rlc-511wa-camara-de-seguridad-de-5mp-para-exterior-_JM)).
- Detecta "persona/vehículo/mascota", **no vacas**. Para disparar hay que usar la detección de movimiento por
  píxeles con una zona marcada sobre el brete.

### 3.2 Sin corriente en el punto de captura: Reolink Go Ultra (versión America)

Es un solo equipo: SIM + batería + solar, IP65, PIR, 4K, y sube por FTP según la lista oficial de Reolink.
Cuesta ~US$ 162 más el panel solar (si no viene en kit) y el courier. Tiene dos riesgos:
- no trae B28 en la versión America;
- no confirmé si el FTP de las cámaras a batería sube **fotos** o solo clips.
Las dos cosas se prueban con la primera unidad.

### 3.3 Sin SIM: cámara de caza wifi, descarga al celular

- **Equipo:** Usogood o Campark con wifi + Bluetooth (US$ 36–57 importada), pilas AA, colgada del brete.
- **Flujo:** el operario pasa cerca con el celular, abre la app (TrailCam GO / Trail Camera Pro) y baja las
  fotos del día. Las sube a ARGOS desde la web actual (`index.html` → `/predict`), escribiendo el `cow_id` en
  cada una. Así **resuelve también el problema del ID** a mano, sin cambios en la API.
- **Contras:** es manual, la resolución real es dudosa (la "4K/48 MP" es de marketing) y la calidad de imagen
  no se puede controlar. Sirve para juntar fotos de dominio y medir el salto de dominio, no para producción.

### 3.4 Datos móviles en Uruguay

No encontré planes IoT/M2M publicados: se cotizan con el área de empresas. Opciones de consumo:

| Plan | Precio | Fuente |
|---|---|---|
| Antel "Internet móvil 150 Gigas" (con módem Alcatel, contrato 2 años) | **$ 1.390 por mes** (~US$ 34) | [tienda Antel](https://tienda.antel.com.uy/plan/12199/internet-movil-150-gigas) |
| Antel "90 GIGAS con límite" | $ 1.110 por mes ($ 888 los primeros 12 meses) | [tienda Antel](https://tienda.antel.com.uy/plan/8592/90-gigas-con-limite) |
| Antel prepago | Recarga inicial mínima $ 220 | [condiciones prepago 2026](https://www.antel.com.uy/documents/37544/1201919/condiciones-servicio-prepago-enero-2026-v2.pdf) |
| eSIM de reventa sobre la red de Movistar | ~US$ 1,7 por GB (30 GB / 30 días) | [esims.io](https://esims.io/providers/wiiline/516424230720308bc9) |

Para la PoC alcanza un plan de 40–90 GB (ver el consumo en §6).

## 4. Arquitectura de la PoC punta a punta

```
 TAMBO                                         NUBE                                  ARGOS (Railway)
┌──────────────────────────┐          ┌──────────────────────────┐          ┌──────────────────────────────┐
│ RLC-510A sobre el brete  │  FTP(S)  │ Receptor de ingesta       │  HTTPS   │ POST /ingest   (nuevo)       │
│  zona de movimiento ->   ├─────────►│ (VPS chico o contenedor)  ├─────────►│   -> inbox (SQLite)          │
│  ráfaga de 3 JPEG        │ saliente │  - FTP solo de escritura  │ X-Device │   -> asociación de cow_id    │
│  OSD apagado, NTP        │ por 4G   │  - script de vigilancia   │ -Token   │      (RFID / hora / manual)  │
│ Router 4G (chip Antel)   │ (CGNAT   │  - reintentos             │          │   -> inference.predict(...)  │
│ [opcional] lector RFID   │  no      │                           │          │   -> Store (mismo registro   │
│  -> log con hora         │  molesta)│                           │          │      que /predict)           │
└──────────────────────────┘          └──────────────────────────┘          │ GET /inbox, POST /inbox/{id}/ │
                                                                             │  assign (bandeja web)         │
                                                                             └──────────────────────────────┘
```

**Variante con equipo en el tambo** (opción 8, o un mini-PC en la sala): el equipo lee la cámara por RTSP o
con la foto HTTP de Reolink (`/cgi-bin/api.cgi?cmd=Snap`), elige el mejor frame y hace `POST /ingest` por
HTTPS directo. Guarda localmente si no hay señal. Con esta variante no hace falta el receptor FTP en la nube.
Es el camino para después de la PoC.

### 4.1 Por qué FTP → receptor → HTTPS, y no la cámara contra ARGOS directo

- Las cámaras baratas no hacen `POST` HTTP con la foto (no tienen webhooks con imagen), pero casi todas suben
  por **FTP**.
- **Railway expone HTTP(S)**. Un FTP en modo pasivo necesita varios puertos TCP, y el proxy TCP de Railway da
  uno solo (a confirmar, est.). Por eso el receptor FTP va **aparte**: un VPS de ~US$ 5 por mes (est.) o un
  contenedor en otro proveedor.
- El receptor hace de amortiguador: si ARGOS se cae o se redeploya, las fotos quedan en disco y se reintentan.
- Preferir **FTPS** (FTP con TLS) si el firmware lo ofrece (sin verificar en el RLC-510A). Si no, FTP con un
  usuario **solo de escritura** en una carpeta por cámara, con la contraseña rotada. Las fotos no son datos
  sensibles, pero la contraseña viaja en claro.

### 4.2 Propuesta de `POST /ingest`

Tiene que ser un archivo nuevo (p. ej. `src/bcs/api_ingest.py`). Montarlo toca `src/bcs/api.py`, que es un
archivo caliente: lo hace el orquestador o una tarea propia.

| Campo | Tipo | Notas |
|---|---|---|
| `image` | multipart, obligatorio | JPEG tal cual sale de la cámara |
| `device_id` | obligatorio | `^[A-Z0-9][A-Z0-9._-]{0,63}$`, el mismo formato que `cow_id` |
| `captured_at` | ISO 8601, obligatorio | Hora de la cámara (sincronizada por NTP) o del archivo en el FTP. Se convierte en `taken_at` |
| `burst_id` | opcional | Agrupa los frames de una misma pasada |
| `cow_id` | **opcional** | Solo si el equipo ya lo sabe (RFID local) |

**Diseño**
- **Autenticación:** un `X-Device-Token` **por dispositivo**, distinto de `BCS_APP_TOKEN`. Así se puede revocar
  una cámara sin cortar la app. Se guarda hasheado y se registra `last_seen`.
- **Deduplicación:** se usa `image_sha256`, que el contrato ya tiene. Si el mismo `(device_id, sha256)` llega
  dos veces, se responde `200 {"duplicate": true}` sin reprocesar, porque el reintento de FTP es normal.
  Además, **de cada `burst_id` se elige un frame**: el de mejor puntaje del filtro de calidad. Así no se
  registran 3 copias casi iguales (el espíritu de SPEC §1.1).
- **Cola:** para la PoC basta una tabla `inbox` en el mismo SQLite (`status`: `pending_id` → `queued` →
  `done` / `rejected`) y un worker en el mismo proceso. Así no se suma Redis. Responde `202 Accepted` con el
  `inbox_id`.
- **Asociación del `cow_id` (§5):** un ítem pasa de `pending_id` a `queued` cuando tiene `cow_id`. Lo asigna
  el lector RFID por la hora, o una persona desde la bandeja (`GET /inbox`, `POST /inbox/{id}/assign`).
- **Predicción:** el worker llama a `inference.predict(image, cow_id)` y guarda en el `Store` con
  `taken_at = captured_at` y `notes = "ingest:<device_id>"`. Usa el mismo contrato que `/predict`, así que la
  vista de rodeo no cambia. El `cow_id` sigue sin llegar al modelo (SPEC §0).
- **Límites:** tamaño máximo (p. ej. 5 MB), solo JPEG, tasa máxima por dispositivo.

### 4.3 Pasos de la PoC

1. Comprar 1 RLC-510A y 1 router con B4/B28, y conseguir un chip Antel.
2. Probar en casa: zona de movimiento, OSD apagado, FTP a un receptor de prueba, consumo de datos en 24 h.
3. Montar en el tambo, a la salida de la sala (§6), y juntar una semana de fotos **sin predecir**. Mirarlas:
   encuadre, luz, suciedad.
4. Pasar 50 fotos a mano por la web actual (`/predict` con `cow_id`) para ver si el filtro las acepta y el BCS
   es razonable.
5. Recién ahí, programar `/ingest` y la bandeja.

## 5. El problema del ID de la vaca

La app necesita un `cow_id` en cada evaluación y la cámara no lo sabe. Según SPEC §7, reconocer la vaca por la
foto está **fuera de alcance**, así que el ID tiene que venir de otro lado.

| Opción | Cómo funciona | Costo (aprox.) | Pros / contras |
|---|---|---|---|
| **A. Bandeja manual** (PoC) | `/ingest` guarda la foto en `pending_id`. Desde el celular, alguien que conoce las vacas le pone el número, o lo copia de la planilla del ordeñe por la hora | US$ 0 | Funciona desde el día 1, pero es trabajo humano y no escala a 200 vacas por día |
| **B. RFID de la caravana electrónica SNIG** | Uruguay identifica a los bovinos con un **par de caravanas**: una visual y un **botón electrónico HDX** (Allflex, color salmón, homologado por el MGAP; norma ISO 11784/11785, 134,2 kHz). Un lector en el brete, a la altura de la oreja, registra `(hora, número SNIG)`. El backend junta foto y lectura por cercanía de hora (±2–3 s) en el paso de a una | Bastón **Allflex RS420** con Bluetooth: **US$ 1.161** ([Spezial UY](https://spezial.com.uy/producto/lector-allflex-rs420/), precio de hace ~2 años). Tru-Test XRS2i: ~US$ 2.159 (AR). Lector fijo de panel para manga: cotizar (est. US$ 1.500–3.000). DIY: módulos de 134,2 kHz desde US$ 20–50, pero **la mayoría lee solo FDX-B**, no HDX, y alcanza 20–25 cm | Es la solución robusta: el número es oficial y único. **El lector tiene que leer HDX.** Un bastón requiere a alguien que lo pase; un lector fijo es caro y su alcance es corto |
| **C. Registro de la sala de ordeñe** | Si el tambo tiene identificación en la sala (DeLaval DelPro, GEA, Afimilk), exportar la lista de vacas por turno con la hora de entrada o salida y alinearla con el orden de las fotos | US$ 0 en hardware; integración distinta para cada marca (DelPro exporta informes, no encontré una API pública) | Barato si ya existe. **Frágil:** en espina de pescado las vacas salen en tandas, no en el orden de entrada. GEA vende verificación por cámara justamente porque una sola lectura RFID falla |
| **D. OCR del número de la caravana visual** | Una **segunda cámara** a la altura de la cabeza (las caravanas van en las orejas y **no se ven desde atrás y arriba**) lee el número con OCR | Segunda cámara US$ 50–150; OCR con modelos abiertos (US$ 0). Precisión incierta en exteriores (est.) | No requiere lector, pero las caravanas se ensucian, giran o se pierden |
| **E. Número pintado en el anca** | Pintar o marcar a fuego el número en el anca y leerlo con OCR desde **la misma cámara** | Pintura ~US$ 10 por mes (est.) | Barato, pero hay que repintar. Además **contamina la imagen de BCS**: el modelo ve la pintura en la zona que evalúa |

**Recomendación:** **A** para la PoC. Si se prueba con usuarios, pasar a **B** con un bastón Bluetooth: el
operario lo pasa en el brete y el celular manda `(hora, número SNIG)` a ARGOS, que asocia. Usar el número SNIG
como `cow_id` (cumple el formato de SPEC §2). Fuentes:
[MGAP, dispositivos autorizados](https://www.gub.uy/ministerio-ganaderia-agricultura-pesca/comunicacion/publicaciones/dispositivos-identificacion-autorizados),
[GEA Cow Verification](https://www.gea.com/es/products/milking-farming-barn/gea-herd-management/gea-dairynet/cow-verification/),
[lectores LF (Montana DoL)](https://liv.mt.gov/_docs/Animal-Health/import/ImportDocs2024/Common-RFID-Reader-Options.pdf).

## 6. Riesgos y pendientes

- **Cobertura 4G rural:** mirar el mapa de cobertura de Antel para el padrón y medir la señal con un celular
  en el lugar exacto antes de comprar. Si es débil, usar un router con conector SMA y una antena direccional
  (~US$ 30–60, est.). La banda que más llega es B28 (700 MHz).
- **Consumo de datos** (est., 200 vacas en ordeñe, 2 ordeñes por día):

  | Qué se sube | Por día | Por mes |
  |---|---|---|
  | 1 frame por pasada, recortado a 1024×576 (~100 KB) | ~40 MB | ~1,2 GB |
  | Ráfaga de 3 frames de 5 MP (~1 MB c/u) | ~1,2 GB | ~36 GB |
  | Clips de video de 10 s, 2–4 Mbps (~4 MB) | ~1,6 GB | ~48 GB |

  Para la PoC sirve la ráfaga cruda con un plan de 90–150 GB. Para producción, recortar en un equipo en el
  tambo.
- **Luz:** a la salida de la sala hay **contraluz** (afuera claro, adentro oscuro) y sol bajo de mañana. Hace
  falta una cámara con WDR o HDR y orientarla con el sol de espaldas. El ordeñe de madrugada con IR da
  imágenes en gris, un dominio sin medir: conviene un **reflector LED blanco** con sensor (~US$ 20, est.).
- **Suciedad del lente:** salpicaduras de bosta y de las mangueras de lavado. Usar visera, montar la cámara
  fuera del alcance de la manguera y limpiarla una vez por semana. El filtro de calidad tiene que rechazar las
  fotos borrosas, y conviene una alerta si la tasa de rechazo sube (señal de lente sucio).
- **Montaje:** a 2,5–3 m, centrada sobre el brete o pasillo de salida, mirando hacia abajo con 30–60° y hacia
  la cola (SPEC §0.2). Con soporte metálico firme: las vacas empujan los caños y vibra. El cable tiene que
  quedar fuera del alcance de los animales. Necesita **paso de a una vaca**: si pasan dos juntas, la
  asociación con el RFID falla.
- **Hora:** NTP activado en la cámara y en el receptor. Un desfase de segundos rompe la asociación con el RFID.
- **Privacidad:** la cámara también filma a los operarios. Rige la Ley 18.331 de protección de datos
  personales. Hace falta cartel, consentimiento del productor y del personal, zona de movimiento solo sobre el
  brete, recorte a la vaca, retención limitada de las fotos crudas, y `BCS_APP_TOKEN` en la app (SPEC §2).
- **Proveedor:** Reolink y EZVIZ empujan su nube. Desactivar la nube y el P2P si no se usan, y cambiar la
  contraseña por defecto.
- **Pendientes que no pude verificar:**
  1. Si el FTP de la Go Ultra sube fotos o solo video.
  2. Si el RLC-510A tiene FTPS.
  3. Las bandas reales de la EZVIZ EB8 y de la Suntek HC-940 Pro.
  4. La tabla oficial de bandas de URSEC.
  5. El precio de un plan M2M de Antel (cotizar con empresas).
  6. El precio actual del RS420 y de un lector fijo HDX en Uruguay (Spezial, Farmquip).
  7. Si el proxy TCP de Railway sirve para un FTP.
