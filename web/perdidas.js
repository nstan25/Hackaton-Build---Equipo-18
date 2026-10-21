// Calculadora de pérdidas por condición corporal (docs/calculadora-leche/investigacion.md).
// Módulo sin dependencias: lo importan index.html (una vaca) y rodeo.html (el rodeo).
// Todo es estimación poblacional: los coeficientes vienen de estudios publicados, no de esta vaca.

// ---------- trayectoria objetivo del BCS en la lactancia (escala 1–5, metas uruguayas) ----------
// Parto 3,25: piso del óptimo de Meikle et al. (2001), 3,25–3,75, y techo del de Roche et al. (2009), 3,0–3,25.
// Mínimo 2,75 hacia los 70 días: de los Santos et al. (2007) la marcan como el mínimo aceptable entre los 30 y
// los 60 días, en tambos de CONAPROLE. El nadir cae a los 65–96 días en pastoreo (Hernández-Gotelli et al. 2024).
// Después se recupera hacia el secado.
export const OBJETIVO = [[0, 3.25], [70, 2.75], [200, 3.0], [305, 3.25]];
export function objetivoEn(dias) {
  if (dias <= 0) return OBJETIVO[0][1];
  for (let i = 1; i < OBJETIVO.length; i++) {
    const [d0, v0] = OBJETIVO[i - 1], [d1, v1] = OBJETIVO[i];
    if (dias <= d1) return v0 + (v1 - v0) * (dias - d0) / (d1 - d0);
  }
  return OBJETIVO[OBJETIVO.length - 1][1];
}
const PARTO = OBJETIVO[0][1];

// ---------- leche: Berry, Buckley y Dillon (2007), Animal 1:1351 (Tabla 3) ----------
// Leche 305 d (kg) según BCS al parto (escala 1–5), Holstein-Friesian en pastoreo: 935,8·b − 109,0·b².
const BERRY = { b1: 935.8, b2: 109.0, mediaKg: 6877 };
const berry = (b) => BERRY.b1 * b - BERRY.b2 * b * b;
const DATOS_MIN = 2.5;  // media 3,2 − 2 DE del BCS al parto en Berry: más abajo es extrapolación

// Curva de lactancia (Wilmink) con los parámetros medios de Berry (Tabla 1): qué parte de la leche de la
// lactancia queda por delante a los `dias` en leche.
const WILMINK = { a: 36.5, b: -16.7, c: -0.092, k: 0.05 };
const LACTANCIA = 305;
const lecheAcum = (t) => WILMINK.a * t + (WILMINK.b / WILMINK.k) * (1 - Math.exp(-WILMINK.k * t)) + WILMINK.c * t * t / 2;
export function fraccionRestante(dias) {
  if (dias <= 0) return 1;
  return 1 - lecheAcum(Math.min(LACTANCIA, dias)) / lecheAcum(LACTANCIA);
}

// ---------- preñez ----------
// Base: preñez al 1er servicio en vacas adultas de 312 tambos uruguayos, 46% (Gastal et al. 2021, datos 2018).
// Efecto del BCS: Carvalho et al. (2014) encontraron 40,4% de preñez por IA con BCS ≤ 2,50 frente a 49,2% con
// BCS ≥ 2,75 (×0,82, con 0,25 de diferencia), y 22,8% frente a 36,0% (×0,63) en vacas que perdieron
// condición. Se toma ×(1 − 0,72·brecha) con un piso de 0,64.
// Si un servicio falla, la vaca espera un ciclo más (21 días): los días abiertos esperados son 21·(1/p).
const PRENEZ = { base: 0.46, pendiente: 0.72, piso: 0.64, ciclo: 21 };
export function probPrenez(brecha) {
  return PRENEZ.base * Math.max(PRENEZ.piso, 1 - PRENEZ.pendiente * Math.max(0, brecha));
}

// Supuestos del tambo, editables en la página. Precio: INALE, junio 2026. Día abierto: no hay un valor
// uruguayo publicado. Se toman ~8 L/día de diferencia entre el promedio de la lactancia y su cola, a
// US$ 0,42, más una parte del valor del ternero, dentro del rango internacional (ICBF, Irlanda: ~€7/día).
export const DEFAULTS = { produccion: 6500, precio: 0.42, costoDia: 4 };
export const FUENTES_TEXTO =
  "Estimación orientativa. Leche: Berry et al. 2007 (Holando en pastoreo). Preñez: Carvalho et al. 2014 " +
  "sobre la base uruguaya de Gastal et al. 2021 (46% al 1er servicio). Metas de condición: Meikle et al. 2001 y " +
  "de los Santos et al. 2007 (Uruguay). Precio: INALE, junio 2026.";
const DENSIDAD = 1.03;  // kg por litro de leche

// La foto se compara con la meta del momento (`dias` < 0: días que faltan para el parto; ≥ 0: días en leche).
// La brecha se lleva a la curva de Berry como condición al parto, y solo cuenta la leche que queda por producir.
function leche(bcs, dias, produccion) {
  const brecha = Math.max(0, objetivoEn(dias) - bcs);
  const alParto = PARTO - brecha;
  const kg305 = (berry(PARTO) - berry(alParto)) * (produccion * DENSIDAD / BERRY.mediaKg);
  return { litros: kg305 / DENSIDAD * fraccionRestante(dias), brecha, alParto };
}

// Estimación completa de una vaca. `interval`: el intervalo del 90% del modelo, que da el rango de litros.
// `prenada`: si ya está preñada no hay costo reproductivo en esta lactancia. Antes del parto se evalúa
// el próximo servicio, suponiendo que la brecha de hoy se mantiene hasta entonces.
export function estimar({ bcs, interval, dias, prenada, produccion, precio, costoDia }) {
  const l = leche(bcs, dias, produccion);
  const lo = interval ? leche(interval[1], dias, produccion).litros : l.litros;
  const hi = interval ? leche(interval[0], dias, produccion).litros : l.litros;
  const avisos = [];
  if (l.brecha > 0 && l.alParto < DATOS_MIN) {
    avisos.push("Condición muy por debajo de los datos de los estudios: la cifra es una extrapolación.");
  }
  if (bcs - objetivoEn(dias) > 0.5) {
    avisos.push("Condición alta para este momento: no suma leche y aumenta el riesgo de cetosis, hígado graso y problemas al parto.");
  }
  let prenez = null, usdRepro = 0;
  if (dias < 0 || !prenada) {
    const p = probPrenez(l.brecha);
    const diasExtra = PRENEZ.ciclo * (1 / p - 1 / PRENEZ.base);
    prenez = { p, pRef: PRENEZ.base, diasExtra };
    usdRepro = diasExtra * costoDia;
  }
  const usdLeche = l.litros * precio;
  const cuando = dias < 0 ? "en la próxima lactancia" : `en los ${Math.max(0, LACTANCIA - dias)} días que quedan`;
  return { leche: { litros: l.litros, lo, hi, cuando }, brecha: l.brecha, objetivo: objetivoEn(dias),
           usdLeche, prenez, usdRepro, total: usdLeche + usdRepro, avisos };
}
