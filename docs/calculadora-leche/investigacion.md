# Calculadora de pérdidas por condición corporal: investigación y modelo

Rama: `feature/mejoras-pagina`. Código: [`web/perdidas.js`](../../web/perdidas.js), un módulo compartido por las dos páginas.
- **`index.html`:** en el resultado de una vaca, el bloque "¿Cuánto te cuesta esta condición?".
- **`rodeo.html`:** debajo de la distribución, la tarjeta "¿Cuánto le cuesta la condición corporal al rodeo?".

El productor ve cuánta leche y cuánta preñez pierde una vaca, o el rodeo, por tener una condición corporal
(BCS, escala 1–5) por debajo de la meta del momento. El resultado se da en litros, días abiertos y dólares.

## 1. Fuentes

### Uruguay y la región

| Fuente | Qué aporta | Uso en el modelo |
|---|---|---|
| Meikle et al. (2001), citado en [de los Santos et al. 2007](https://produccion-animal.com.ar/informacion_tecnica/cria_condicion_corporal/19-lechera.pdf) | Óptimo al parto **3,25–3,75** en Holando, Uruguay | Meta al parto (piso: 3,25) |
| de los Santos, Etchevers y Gatti (2007), [PDF](https://produccion-animal.com.ar/informacion_tecnica/cria_condicion_corporal/19-lechera.pdf). 9 tambos de CONAPROLE, parición de otoño | CC media al parto 3,13 ± 0,37. **CC 2,75 entre los 30 y 60 días posparto es el mínimo aceptable**: por debajo empeoran la leche a 120 d, el intervalo parto-1er servicio y el intervalo parto-concepción | Meta en el nadir (2,75) |
| Gastal, Lemaire, Cavestany y Hirigoyen (2021), XLVIII Jornadas Uruguayas de Buiatría, [PDF](https://inia.uy/sites/default/files/publications/2024-10/JUB-no.48-2021.p95-198.Gastal.pdf). 312 tambos, 81.060 vacas | Vacas adultas: preñez al 1er servicio 42/43/41/**46%** (2010/11/13/18); intervalo parto-concepción 138–157 d; 2,0–2,4 servicios por preñez | Probabilidad base de preñez: 46% |
| Cruz Freitas et al. (2024), PANVET, INIA La Estanzuela, [PDF](https://ainfo.inia.uy/digital/bitstream/item/18175/1/Cruz-Freitas-PANVET-2024-p145-146.pdf) | Holando en pastoreo: CC preparto 3,25, a los 15 y 40 días posparto 3,0. Pérdida hasta el servicio de −0,25 a −0,50. Concepción al 1er servicio 35–64% | Valida la forma de la curva objetivo |
| Chilibroste et al. (2012), *Anim Feed Sci Technol* 173:201 (solo el resumen) | Primíparas en pastoreo: con baja asignación de forraje, 19,2 vs 24,3 L/d y ~1 punto de CC perdido en 3 semanas | Contexto: la pérdida de CC acompaña la caída de leche |
| Meikle et al. (2004), *Reproduction* 127:727, y Adrien et al. (2012), *Animal* 6:292 (solo los resúmenes) | Vacas que paren con CC < 3,0 tardan más en volver a ciclar | Contexto, sin cifras utilizables |
| Krall et al. (2007), *Veterinaria* (Montevideo) 42:15, [PDF](https://www.produccion-animal.com.ar/informacion_tecnica/cria_condicion_corporal/102-cc_leche.pdf) | El efecto de la CC al parto sobre la leche es **mayor en pastoreo** que con raciones totales mezcladas de alta energía | Justifica usar una curva de un sistema pastoril |
| Hernández-Gotelli et al. (2024), *JDS Communications* 6:110, [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11770315/). Chile, pastoreo | Nadir a los 76–81 días en leche, pérdida de 0,36–0,43 | Día del nadir (~70) |
| [INALE, Situación y perspectivas 2026](https://www.inale.org/wp-content/uploads/2026/06/Situacion-y-perspectivas-2026.pdf) | Precio al productor: US$ 0,42/L (junio 2026) | Precio por defecto |

### Internacionales (con ecuación o tamaño de efecto)

| Fuente | Qué aporta |
|---|---|
| Berry, Buckley y Dillon (2007), *Animal* 1:1351, [doi](https://doi.org/10.1017/S1751731107000419). Irlanda, pastoreo, 9.886 lactancias | **Leche 305 d (kg) = 935,8·BCS − 109,0·BCS²**, según la CC al parto. Curva de lactancia de Wilmink: a = 36,5, b = −16,7, c = −0,092 |
| Carvalho et al. (2014), *J Dairy Sci* 97:3666 (solo el resumen) | Preñez por IA de 40,4% con BCS ≤ 2,50 frente a 49,2% con ≥ 2,75. Según el cambio de CC en las primeras 3 semanas: perdió 22,8%, mantuvo 36,0% |
| López-Gatius et al. (2003), *Theriogenology* 59:801, metaanálisis (solo el resumen) | CC < 2,5 al parto: unos 10% menos de preñez. Perder más de 1 punto suma 10,6 días abiertos |
| Roche et al. (2009), *J Dairy Sci* 92:5769 | Relaciones no lineales. Óptimo al parto 3,0–3,25 |
| ICBF/Teagasc, Berry et al. (2007), [EBI](https://www.icbf.com/wp-content/uploads/2013/06/economic_breeding_index.pdf) | Cada día más de intervalo entre partos cuesta unos €7 (Irlanda, pastoreo) |

No encontré un valor publicado en Uruguay para el costo de un día abierto.

## 2. Modelo (`web/perdidas.js`)

**Meta de CC según el momento**, interpolada linealmente: parto 3,25 → 70 días 2,75 → 200 días 3,0 → secado (305) 3,25.
Las entradas son los días que faltan para el parto, o los días en leche.

```
brecha       = max(0, meta(días) − BCS de la foto)
leche_305    = [f(3,25) − f(3,25 − brecha)] × producción/6877      f(b) = 935,8·b − 109,0·b²   (Berry)
leche_perd   = leche_305 × fracción de la lactancia que queda        (Wilmink, Berry; 1 antes del parto)
p_preñez     = 0,46 × max(0,64; 1 − 0,72·brecha)                     (Gastal + Carvalho)
días_extra   = 21 × (1/p_preñez − 1/0,46)                            (un ciclo de 21 d por servicio fallido)
US$          = leche_perd × precio + días_extra × costo del día abierto
```

- **El tiempo cuenta dos veces.** La meta cambia con los días en leche: una vaca con 2,75 a los 70 días está
  bien, y a los 10 días le faltan 0,5. Además, la leche perdida solo cuenta lo que resta de la lactancia: la misma
  brecha cuesta mucho al principio y poco cerca del secado.
- **Antes del parto** se calcula la próxima lactancia completa y el próximo servicio. Es el momento en que todavía
  se puede corregir.
- **Preñada:** no hay costo reproductivo en esta lactancia.
- **Factor de preñez.** Carvalho da ×0,82 con 0,25 puntos de diferencia, que da la pendiente de 0,72. El piso de
  ×0,64 coincide con el ×0,63 de "perdió condición". Con una brecha de 0,5 la preñez baja a ~29% y suma ~26 días
  abiertos, el mismo orden que López-Gatius (+10,6 a +24 d).
- **Rango:** con el intervalo del 90% del modelo de BCS se muestran los litros mínimos y máximos.

Supuestos editables en la página, que se recuerdan en el navegador: producción por lactancia (6.500 L),
precio (US$ 0,42/L) y costo del día abierto (US$ 4/día). Este último sale de unos 8 L/día de diferencia entre el
promedio y la cola de la lactancia a US$ 0,42, más una parte del valor del ternero. Está por debajo de la cifra
irlandesa (~€7), así que la estimación es conservadora.

**Avisos en la página:**
- Condición al parto equivalente por debajo de 2,5: fuera de los datos de Berry, la cifra es una extrapolación.
- Más de 0,5 por encima de la meta: sobrecondición. No suma leche y aumenta el riesgo metabólico.

Ejemplos con los valores por defecto (BCS 2,5, intervalo 2,25–2,75):

| Momento | Meta | Leche perdida | Preñez | Total |
|---|---|---|---|---|
| 20 días antes del parto | 3,25 | 219 L | 29% (+26 d) | US$ 195 |
| 10 días en leche | 3,18 | 186 L | 29% (+26 d) | US$ 181 |
| 70 días en leche | 2,75 | 42 L | 38% (+10 d) | US$ 58 |
| 200 días en leche | 3,00 | 28 L | 29% (+26 d) | US$ 115 |

## 3. Límites

- Es una **estimación poblacional**: asociaciones estadísticas de estudios, no una predicción para esa vaca.
- Llevar una brecha a mitad de lactancia a la curva de "CC al parto" de Berry es una aproximación nuestra.
  También lo es el factor de preñez lineal derivado de Carvalho (EE.UU., confinamiento).
- El BCS de Argos está validado entre 3,25 y 4,25. Fuera de ese rango se suman dos incertidumbres.
- En el rodeo, el momento y los días son un promedio del rodeo: el registro todavía no guarda la fecha de parto de cada vaca.
  Un buen próximo paso sería guardar la fecha de parto con `cow_id` en el backend.
