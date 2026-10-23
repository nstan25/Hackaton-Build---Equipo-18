# Mails de solicitud de datasets de BCS (uso comercial)

Borradores para pedir datos a los autores de los dos mejores candidatos con vacas flacas (ver
`docs/investigacion/datasets-bcs.md` y `datasets-bcs-2.md`). Indican claramente que **ARGOS puede usarse
comercialmente**, para que el permiso que den cubra entrenar y servir el modelo en un producto.

Antes de enviar, completá los datos entre corchetes. Los manda una persona del equipo (los agentes no
escriben a terceros).

---

## 1. Nagy et al. 2023 (Hungría)

**Para:** solymosi.norbert@gmail.com

**Subject:** Data request for commercial use – rump images from "Animals 2023, 13(2):194" (dairy cow BCS)

Dear Dr. Solymosi,

My name is [Nombre Apellido], and I am part of a team in Uruguay ([empresa / institución / equipo]) developing
**ARGOS**, a product that estimates dairy cow Body Condition Score (BCS, 1–5 scale) from rear-view photographs,
to help producers and veterinarians monitor their herds. ARGOS started as a hackathon prototype and **we intend
to offer it as a commercial service**.

We read your paper by Nagy, Kilim, Csabai, Gábor and Solymosi (*Animals* 2023, 13(2):194,
DOI: 10.3390/ani13020194) with great interest. Our current model was trained on public data covering only
BCS 3.0–5.0, so it cannot recognise thin cows, which is the case that matters most for animal welfare. As far as
we know, your dataset is the only one with rear-view RGB images covering BCS 1.0–2.5.

We would like to ask whether you would be willing to share, **with permission to use it for training and
evaluating a model that will be deployed in a commercial product**:
1. the images (or video frames) with their BCS labels;
2. the rump bounding boxes, if available;
3. a cow or video identifier per image, so that we can split training and test sets by animal.

We would not redistribute the data, we would cite your paper and acknowledge your group in our documentation and
any publication, and we are happy to sign a data-use or licensing agreement under the terms you consider
appropriate, including discussing compensation or a research collaboration. We would also gladly share our
results on your data with you.

Please let us know whether this is possible and under which conditions. Thank you very much for your time.

Kind regards,
[Nombre Apellido]
[Empresa / institución / equipo] – Uruguay
[mail] · [teléfono, opcional]

---

## 2. Shriramulu, Pushpadass et al. 2024 (India)

**Para:** heartwin1@gmail.com, P.Dhass@icar.gov.in

**Subject:** Data request for commercial use – dorsal BCS images from "Indian J. Dairy Sci. 77(3):243–251"

Dear Dr. Shriramulu and Dr. Pushpadass,

My name is [Nombre Apellido], and I am part of a team in Uruguay ([empresa / institución / equipo]) developing
**ARGOS**, a product that estimates dairy cow Body Condition Score (BCS, 1–5 scale) from photographs, to help
producers and veterinarians monitor their herds. ARGOS started as a hackathon prototype and **we intend to
offer it as a commercial service**.

We read your 2024 article in the *Indian Journal of Dairy Science* (77(3):243–251,
DOI: 10.33785/IJDS.2024.v77i03.008) with great interest. Our current model was trained on public data covering
only BCS 3.0–5.0, so it cannot yet recognise thin cows, which is the case that matters most for animal welfare.
Your dataset of dorsal photographs of crossbred Holstein cows, scored in 0.25 steps and with many animals below
BCS 2.75, would be extremely valuable to us.

We would like to ask whether you would be willing to share, **with permission to use it for training and
evaluating a model that will be deployed in a commercial product**:
1. the 503 dorsal photographs with their BCS;
2. the individual scores given by each of the three veterinarians (not only the consensus), if available;
3. a cow and farm identifier for each photograph, so that we can split training and test sets by animal.

We would not redistribute the data, we would cite your article and acknowledge your group in our documentation and
any publication, and we are happy to sign a data-use or licensing agreement under the terms you consider
appropriate, including discussing compensation or a research collaboration. We would also gladly share our
results on your data with you.

Please let us know whether this is possible and under which conditions. Thank you very much for your time.

Kind regards,
[Nombre Apellido]
[Empresa / institución / equipo] – Uruguay
[mail] · [teléfono, opcional]

---

## Notas
- Si un autor acepta solo uso no comercial, ese dato **no** puede entrar al modelo del producto; como mucho a una
  evaluación interna. Guardar por escrito el permiso que den y su alcance.
- El dataset de Shriramulu/Pushpadass es © Indian Dairy Association y no tiene licencia pública: el permiso tiene
  que venir por escrito de los autores (y quizás de la institución, ICAR-NDRI).
- El modelo actual se entrenó con `sdb_bcs` (CC BY 4.0, permite uso comercial con atribución) y `zen_rgbd`
  (CC BY 4.0). El detector usa Ultralytics YOLO (AGPL-3.0): revisar esa licencia antes de lanzar comercialmente.
