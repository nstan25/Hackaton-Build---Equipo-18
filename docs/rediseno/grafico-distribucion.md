# ARGOS — Gráfico de distribución de condición corporal del rodeo

Quiero que implementes una nueva funcionalidad para el dashboard de Argos, nuestro proyecto AgriTech que utiliza inteligencia artificial para estimar la condición corporal (Body Condition Score, BCS) de vacas a partir de fotografías.

**IMPORTANTE: Todo el desarrollo debe realizarse en un worktree Git independiente, utilizando una nueva rama llamada `graficas-rodeo`. No debes modificar la rama ni el worktree en el que estamos trabajando actualmente.**

## 1. Preparación del entorno Git

Antes de modificar cualquier archivo:

1. Inspecciona el estado del repositorio con `git status`, `git branch` y `git worktree list`.
2. Identifica la rama y el commit actual.
3. Crea un worktree independiente basado en el HEAD actual, con una rama llamada `graficas-rodeo`.
4. Puedes utilizar un directorio hermano del proyecto, como `../argos-graficas-rodeo`, siempre que no exista y sea una ubicación apropiada.
5. Si la rama o el worktree ya existen, inspecciona su estado y reutilízalos cuando corresponda, sin sobrescribir cambios.
6. Si hay cambios sin confirmar en el worktree original que sean necesarios para implementar la funcionalidad, avísame antes de continuar. No los copies ni los descartes automáticamente.

El comando de referencia, si la rama y la ubicación todavía no existen, sería:

```bash
git worktree add -b graficas-rodeo ../argos-graficas-rodeo HEAD
```

A partir de ese momento, **todas las modificaciones, instalaciones de dependencias, compilaciones y pruebas deben hacerse dentro del nuevo worktree**.

No hagas merge, rebase, push ni operaciones destructivas sobre las ramas existentes.

## 2. Nueva funcionalidad solicitada

Queremos incorporar al dashboard una visualización que permita conocer **cómo se distribuye la condición corporal entre los animales del rodeo**.

Actualmente, el dashboard contempla el registro histórico de las mediciones de BCS y la evolución individual de los animales, donde cada medición puede representarse como un punto dentro de una secuencia temporal.

Queremos conservar esa funcionalidad y agregar una nueva perspectiva: **la distribución global del rodeo**.

El objetivo es poder responder rápidamente preguntas como:

- ¿Cuántas vacas tienen BCS 2.0?
- ¿Cuántas tienen BCS 2.5?
- ¿Cuántas tienen BCS 3.0?
- ¿En qué valores de BCS se concentra la mayor cantidad de animales?
- ¿Cómo está distribuida la condición corporal de los animales evaluados?

Es una funcionalidad relativamente sencilla, pero que agrega mucho valor al dashboard.

## 3. Visualización propuesta

Implementa un gráfico de barras moderno, claro y responsive.

Título sugerido:

**Distribución de condición corporal del rodeo**

Subtítulo:

"Cantidad de animales según su condición corporal estimada".

### Eje horizontal (X)

Valores o categorías de BCS utilizados realmente por el modelo.

Por ejemplo, si el sistema utiliza la escala de 1 a 5 con incrementos de 0.25, el gráfico deberá respetar esos valores.

No asumas una escala o una granularidad sin verificar antes cómo funciona el sistema actual.

### Eje vertical (Y)

Cantidad de animales que se encuentran en cada valor de BCS.

Cada barra representará la cantidad de vacas correspondientes a esa condición corporal.

### Ejemplo ilustrativo

| BCS | Cantidad de vacas |
|---|---:|
| 2.0 | 3 |
| 2.5 | 7 |
| 3.0 | 15 |
| 3.5 | 9 |
| 4.0 | 4 |

Estos números son únicamente ilustrativos. La implementación debe utilizar datos reales de la aplicación.

Si el BCS es continuo, analiza cómo conviene agrupar los valores en intervalos significativos sin introducir una discretización arbitraria.

## 4. Regla fundamental: una vaca debe contarse una sola vez

Esto es muy importante.

Como Argos puede registrar varias mediciones de condición corporal para un mismo animal, el gráfico de distribución del rodeo no debe contar todas las mediciones históricas como si fueran animales diferentes.

**Cada vaca debe aparecer una única vez, utilizando su última medición válida de BCS.**

Por ejemplo:

- Vaca 001: mediciones 2.5, 2.75 y 3.0 → contar una vaca con BCS 3.0.
- Vaca 002: mediciones 3.25 y 3.5 → contar una vaca con BCS 3.5.
- Vaca 003: medición 2.5 → contar una vaca con BCS 2.5.

Resultado: tres animales distribuidos entre los valores 2.5, 3.0 y 3.5.

No cinco ni seis mediciones.

Para ello, investiga cómo se identifican los animales en el sistema y cómo se almacenan las mediciones.

Si la aplicación todavía no cuenta con identificadores persistentes de animales, propón una solución coherente con la arquitectura existente y explícame la limitación antes de introducir cambios importantes al modelo de datos.

No inventes identificadores ni consideres automáticamente que cada imagen representa un animal diferente.

## 5. Integración con el dashboard existente

Antes de implementar, revisa la estructura actual del dashboard y sus componentes.

El gráfico debe integrarse con la estética de Argos:

- Verde bosque y verde campo.
- Acentos dorados.
- Fondos claros y tonos naturales.
- Tipografía consistente con el resto de la aplicación.
- Bordes suaves y espaciado generoso.
- Aspecto moderno, elegante y profesional.

Evita crear un gráfico que parezca visualmente separado del resto del sistema.

### Interacciones deseadas

- Mostrar el valor de BCS y la cantidad exacta de animales al pasar el cursor sobre una barra.
- En dispositivos táctiles, permitir consultar la misma información.
- Mostrar correctamente los valores aunque haya categorías sin animales.
- Adaptarse a pantallas pequeñas sin perder legibilidad.
- Actualizarse cuando cambien los datos disponibles.
- Mostrar un estado vacío claro cuando no haya información suficiente.

Si encaja con el dashboard, agrega encima del gráfico un pequeño resumen con:

- **Animales evaluados:** cantidad de animales únicos con una medición válida.
- **BCS promedio:** promedio de las últimas mediciones válidas de esos animales.

No agregues demasiadas métricas ni conviertas esta sección en un dashboard complejo.

Si el sistema distingue distintos rodeos, establecimientos o grupos de animales, respeta el contexto y los filtros existentes para no mezclar animales de diferentes grupos.

## 6. Datos y arquitectura

Analiza primero la arquitectura de la aplicación.

Identifica:

1. Dónde se almacenan las mediciones.
2. Cómo se identifica cada animal.
3. Cómo se obtiene el historial de BCS.
4. Cómo se comunican frontend y backend.
5. Qué librerías de gráficos están disponibles.
6. Qué estructura utiliza actualmente el dashboard.

A partir de eso, selecciona una solución sencilla y mantenible.

Prioriza reutilizar endpoints, servicios y componentes existentes.

Si hace falta agregar un endpoint o modificar una consulta, hazlo respetando las convenciones del proyecto.

Evita transferir historiales completos al frontend únicamente para calcular una distribución, cuando sea más razonable realizar la agregación en el backend.

Reutiliza la infraestructura disponible cuando sea posible.

No agregues dependencias pesadas si existe una alternativa ligera y compatible con el stack.

## 7. Manejo de datos y casos especiales

Considera:

- Animales con una sola medición.
- Animales con múltiples mediciones.
- Mediciones sin un BCS válido.
- Mediciones que tengan exactamente la misma fecha u hora.
- Valores decimales de BCS.
- Categorías sin animales.
- Rodeos sin información.
- Errores al recuperar datos.
- Nuevas mediciones realizadas mientras se utiliza la aplicación.
- Diferentes rodeos o grupos, si el sistema los soporta.

Define claramente el criterio para seleccionar la última medición cuando haya empates temporales, utilizando los datos realmente disponibles.

Si solo existen mediciones y no un registro completo del rodeo, el gráfico debe indicar que la distribución corresponde a los **animales evaluados**, no afirmar que representa animales para los que no existen datos.

## 8. Experiencia visual y presentación para la hackathon

Argos se presentará en una hackathon, así que el resultado debe ser visualmente atractivo y fácil de comprender durante una demostración.

No quiero una visualización excesivamente compleja.

Quiero que, al mirar el gráfico durante unos segundos, cualquier integrante del jurado pueda comprender cómo está distribuida la condición corporal del rodeo.

Utiliza animaciones de entrada sutiles si mejoran la presentación, respetando las preferencias de movimiento reducido.

Evita elementos decorativos que no aporten información.

El gráfico debe transmitir claridad, confianza y profesionalismo.

## 9. Implementación y pruebas

Realiza los cambios directamente en el worktree `graficas-rodeo`.

No te limites a explicar cómo implementar la funcionalidad: quiero que quede efectivamente incorporada al proyecto.

Verifica que:

- El proyecto compile correctamente.
- El dashboard existente continúe funcionando.
- El gráfico represente correctamente los datos.
- Cada animal se cuente una única vez.
- El BCS promedio y los totales coincidan con los datos.
- La interfaz funcione en desktop y mobile.
- Los estados de carga, error y ausencia de datos estén contemplados.

Agrega o adapta pruebas para la lógica de agregación cuando el proyecto disponga de infraestructura de testing.

No alteres la lógica de inferencia del modelo de IA.

## 10. Entrega

Al finalizar:

1. Indica la ruta del nuevo worktree.
2. Confirma que el trabajo está en la rama `graficas-rodeo`.
3. Resume los archivos modificados.
4. Explica cómo obtienes y agrupas los valores de BCS.
5. Confirma que cada vaca se contabiliza solamente una vez.
6. Enumera las pruebas realizadas y sus resultados.
7. Indica cualquier limitación del modelo de datos actual.

No integres los cambios a la rama principal. Quiero poder revisar el resultado de forma independiente antes de decidir si lo incorporamos.

**Objetivo final:** agregar una visualización sencilla, elegante y útil que permita comprender de un vistazo la distribución de condición corporal de los animales evaluados, complementando las mediciones individuales y el seguimiento temporal que ya existen en Argos.
