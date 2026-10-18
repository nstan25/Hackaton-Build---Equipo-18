# Rediseño integral UX/UI — ARGOS | AgriTech & AI

Actúa como un **Senior Product Designer, UX/UI Expert y Senior Frontend Engineer**, especializado en productos AgriTech, aplicaciones de inteligencia artificial y experiencias digitales de alto impacto visual.

## 1. Contexto del proyecto

Estamos desarrollando **ARGOS**, una plataforma AgriTech para presentar en una hackathon ante un jurado.

La aplicación permite que un productor ganadero o veterinario cargue una fotografía de la parte posterior de una vaca y que un modelo de inteligencia artificial estime su **Body Condition Score (BCS)**, o condición corporal.

El objetivo es que evaluar la condición corporal de un animal resulte rápido, sencillo e intuitivo, utilizando únicamente una fotografía y un modelo de visión artificial.

Es un MVP funcional para una hackathon, pero queremos que visualmente tenga calidad de producto comercial.

**Queremos que ARGOS se sienta como una startup AgriTech profesional, innovadora y confiable.**

## 2. Identidad visual

Adjunto dos imágenes que debes utilizar como referencia principal:

- **Imagen 1 — Logo ARGOS:** contiene un símbolo que combina un sol, un ojo y campos de cultivo, con tipografía verde y una identidad orgánica.
- **Imagen 2 — Imagen hero:** una vaca Holstein pastando en un campo al atardecer, con el logo ARGOS y elementos gráficos que representan visión artificial.

Ambas imágenes definen la identidad de la plataforma.

### Estética deseada

Quiero una combinación equilibrada de:

- Agricultura moderna y naturaleza.
- Tecnología avanzada e inteligencia artificial.
- Diseño minimalista, elegante y profesional.
- Calidez, confianza y cercanía con el productor rural.

Piensa en una combinación de **AgriTech premium + modern SaaS + diseño editorial natural**.

Utiliza como punto de partida una paleta de verdes bosque, verdes suaves, dorado solar, marfil y blanco cálido. Extrae los colores definitivos del logo.

Evita una estética excesivamente corporativa, futurista o fría.

No quiero fondos negros, excesivas luces neón ni una estética genérica de dashboard de inteligencia artificial.

### Tipografía

Utiliza una combinación cuidadosamente seleccionada:

- Tipografía de títulos elegante, con personalidad y cierta inspiración editorial, coherente con el logo.
- Tipografía sans-serif moderna y sumamente legible para controles, explicaciones, resultados y navegación.

Genera una jerarquía visual clara con buen contraste y espacios en blanco generosos.

## 3. Experiencia de usuario: prioridad absoluta

La funcionalidad principal es extremadamente simple:

**Fotografiar → Analizar → Obtener resultado.**

La experiencia debe estar optimizada para que un usuario nuevo pueda completar este flujo sin necesidad de recibir instrucciones personales.

El usuario podría ser un productor trabajando en el campo, usando el celular, posiblemente bajo luz solar intensa.

Por eso:

- Diseña primero para celulares (mobile-first).
- Utiliza botones grandes y cómodos para tocar.
- Prioriza texto legible y contraste alto.
- Evita menús complejos y pasos innecesarios.
- Mantén siempre visible la acción principal.
- Usa lenguaje claro, evitando tecnicismos innecesarios.
- Reduce al mínimo los campos obligatorios.
- Proporciona feedback inmediato para cada interacción.
- Asegura que todos los estados de la interfaz sean comprensibles.

La pantalla no debe parecer un formulario técnico ni un dashboard complejo.

La sensación debería ser: **"Saco una foto de mi vaca y Argos me ayuda a evaluar su condición corporal".**

## 4. Página principal y Hero

Quiero una página inicial memorable, especialmente pensada para una presentación en una hackathon.

Utiliza la segunda imagen adjunta como referencia principal del hero.

### Composición

- Header limpio con el logo ARGOS y navegación sencilla.
- Hero amplio, visualmente atractivo y perfectamente responsive.
- Una propuesta de valor inmediatamente comprensible.
- Un botón CTA principal claramente destacado.
- Una transición elegante hacia la funcionalidad de análisis.

### Copywriting propuesto

Título principal:

**La inteligencia artificial llega al campo.**

Subtítulo:

"Evaluá la condición corporal de tu ganado a partir de una fotografía. Simple, rápido e inteligente."

CTA principal:

**Analizar una vaca**

CTA secundario opcional:

**¿Cómo funciona?**

Puedes mejorar el texto si encuentras una alternativa más clara, profesional y persuasiva.

### Animación del hero

Me gustaría que el hero transmita tecnología en acción, aprovechando la imagen de la vaca y las líneas gráficas de escaneo.

Puedes explorar:

- Movimiento muy sutil de acercamiento o parallax.
- Líneas de visión artificial que aparezcan suavemente.
- Pulsos discretos de detección alrededor de la vaca.
- Animaciones de entrada elegantes para los elementos de interfaz.

IMPORTANTE:

- No deformes ni modifiques innecesariamente la imagen original.
- La imagen ya contiene el logo y el nombre ARGOS; evita duplicaciones visuales cuando se utilice completa.
- Cuida especialmente los recortes en teléfonos para no cortar la vaca o elementos importantes.
- Mantén la animación ligera para no perjudicar el rendimiento.
- Respeta `prefers-reduced-motion`.
- No utilices animaciones exageradas ni efectos que dificulten la lectura.

La estética debe resultar sofisticada, no sobrecargada.

## 5. Experiencia de análisis de condición corporal

Esta es la parte más importante del producto.

Diseña una experiencia de carga y análisis de imágenes excepcionalmente simple.

### Paso 1 — Capturar o subir fotografía

Presenta una sección o tarjeta protagonista con un título como:

**Conocé la condición corporal de tu vaca**

Texto de apoyo:

"Tomá una foto desde atrás o seleccioná una imagen de tu dispositivo."

Incluye dos opciones claramente diferenciadas:

- **Tomar fotografía:** abre la cámara en dispositivos compatibles.
- **Subir imagen:** permite seleccionar una fotografía existente.

En escritorio, permite drag & drop si aporta valor.

### Guía visual de captura

Es fundamental que el usuario comprenda cómo debe tomar la fotografía.

Muestra una pequeña guía visual con:

- Una vaca vista desde atrás.
- La región posterior correctamente encuadrada.
- Indicaciones sobre iluminación y enfoque.

Mensajes cortos:

- "Ubicate detrás de la vaca, a una distancia segura."
- "Asegurate de que la grupa y las caderas sean visibles."
- "Evitá imágenes borrosas u oscuras."

Si existen criterios específicos del modelo de IA, prioriza esos requisitos sobre las indicaciones generales.

No utilices la imagen del hero como ejemplo de fotografía válida, porque muestra al animal de costado.

### Paso 2 — Vista previa y análisis

Una vez seleccionada la foto:

- Muestra una vista previa grande y clara.
- Permite reemplazar o eliminar la fotografía.
- Muestra un CTA destacado: **Analizar condición corporal**.
- Valida tipo de archivo y tamaño.
- Informa errores con mensajes fáciles de entender.

Durante el análisis, muestra un estado visual atractivo:

"Argos está analizando la imagen..."

Puedes incluir un indicador animado inspirado en visión artificial, siempre que no resulte invasivo.

No inventes porcentajes de progreso si el backend no los proporciona.

### Paso 3 — Resultado

La pantalla de resultados debe ser el momento visualmente más satisfactorio de la experiencia.

Presenta:

**Condición corporal estimada**

Un número grande, visualmente destacado, correspondiente a la predicción real del modelo.

Si el modelo devuelve BCS en escala de 1 a 5, representa esa escala de forma elegante mediante un indicador visual, por ejemplo una barra segmentada o un medidor horizontal.

Ejemplo puramente ilustrativo:

BCS: 3.0 / 5

Acompaña el resultado con:

- Una breve interpretación comprensible.
- Un indicador de nivel dentro de la escala.
- La fotografía original del animal.
- La opción de analizar otra fotografía.
- La confianza del modelo solamente cuando esté disponible y tenga significado técnico.

El resultado debe ser comprensible en tres segundos.

Evita el uso de indicadores alarmistas.

No declares que una vaca está enferma ni prescribas tratamientos basándote únicamente en el BCS.

Los rangos recomendados deben depender del contexto productivo cuando corresponda; no presupongas un rango óptimo universal.

Incluye una nota discreta indicando que el resultado es una estimación automatizada de apoyo y no sustituye una evaluación profesional.

No fabriques valores ni simules una predicción real. Si existe un modo demostración, identifícalo claramente como tal.

## 6. Componentes visuales

Utiliza un lenguaje visual coherente:

- Tarjetas con esquinas suavemente redondeadas.
- Sombras muy sutiles.
- Fondos claros con tonos naturales.
- Bordes discretos.
- Botones verdes protagonistas.
- Acentos dorados para destacar interacciones y detalles importantes.
- Iconografía moderna y consistente.
- Espaciado generoso.
- Microinteracciones delicadas.
- Transiciones fluidas entre captura, análisis y resultado.

Evita el exceso de glassmorphism, gradientes llamativos, tarjetas innecesarias y efectos visuales repetitivos.

El contenido debe tener suficiente contraste y ser legible bajo luz solar.

## 7. Estructura sugerida

Propón una experiencia con estas áreas:

**A. Header**

Logo ARGOS, navegación mínima y acceso directo a analizar.

**B. Hero**

Imagen principal, mensaje de valor y CTA.

**C. Analizar condición corporal**

Interfaz de captura y carga de imágenes. Debe tener máxima importancia dentro de la experiencia.

**D. ¿Cómo funciona?**

Tres pasos visuales y sencillos:

1. Tomá una fotografía.
2. Argos la analiza con IA.
3. Conocé la condición corporal estimada.

**E. Sobre Argos**

Un bloque breve y atractivo que explique cómo la tecnología puede asistir al manejo ganadero.

**F. Footer**

Identidad visual, información mínima del proyecto y referencia a la hackathon cuando corresponda.

No agregues apartados vacíos ni funcionalidades ficticias solo para rellenar la página.

Puedes reorganizar esta estructura si eso mejora el recorrido del usuario.

## 8. Experiencia durante el pitch

La página será utilizada en una demostración ante un jurado.

Necesitamos que en muy pocos segundos se entienda:

- Qué problema resuelve Argos.
- Qué debe hacer el usuario.
- Cómo interviene la inteligencia artificial.
- Qué resultado recibe.

Quiero que la experiencia tenga un efecto "wow" por su claridad, belleza y fluidez, no por acumular animaciones.

El recorrido completo debe sentirse como un producto listo para utilizar.

Si existe una imagen de prueba representativa y una inferencia real disponible, facilita ejecutar una demostración sin pasos innecesarios.

Si la inferencia falla, muestra un estado de error claro y la posibilidad de reintentar; nunca presentes un resultado inventado como si fuera real.

## 9. Accesibilidad, responsive y rendimiento

Requisitos:

- Diseño mobile-first real, no simplemente una versión de escritorio comprimida.
- Excelente adaptación a celulares, tablets y computadoras.
- Accesibilidad WCAG 2.2 AA como objetivo.
- Contraste mínimo de 4.5:1 para texto normal.
- Áreas táctiles cómodas.
- Estados de foco visibles.
- Navegación por teclado.
- Etiquetas accesibles para carga de archivos y botones.
- Animaciones respetuosas del movimiento reducido.
- Imágenes optimizadas, responsive y sin saltos de layout.
- Evitar cargas pesadas o animaciones que afecten el tiempo de respuesta.

## 10. Implementación

Antes de comenzar:

1. Inspecciona el proyecto existente y determina el framework, estructura, componentes y estilos utilizados.
2. Identifica el flujo actual de carga de imágenes, inferencia y presentación de resultados.
3. Revisa los recursos gráficos adjuntos.
4. Propón una dirección de diseño coherente con el proyecto.

Después, **implementa realmente el rediseño en el código existente**.

Reglas importantes:

- Respeta el framework y la arquitectura del proyecto.
- Reutiliza los componentes y librerías existentes cuando sea razonable.
- No agregues dependencias grandes sin justificación.
- No rompas el flujo de análisis ni cambies innecesariamente las llamadas al backend.
- Mantén la funcionalidad actual.
- No sustituyas la inferencia por resultados simulados.
- Evita código duplicado.
- Construye componentes reutilizables donde corresponda.
- Prioriza código limpio, mantenible y fácil de modificar.

Si tienes acceso a la ejecución del proyecto, verifica los estados de la interfaz y su comportamiento responsive.

## 11. Resultado esperado

Quiero que entregues:

1. Una interfaz moderna, atractiva, consistente y funcional.
2. Una experiencia de usuario excepcionalmente simple.
3. Una integración cuidada de las imágenes y del logo de ARGOS.
4. Un hero impactante pero elegante.
5. Una experiencia pulida de captura, análisis y resultados.
6. Un diseño responsive de alta calidad.
7. El código necesario para llevarlo a cabo.

Al terminar, explica brevemente qué cambiaste, qué decisiones importantes de UX/UI tomaste y qué puntos pendientes requieren información adicional.

**Criterio final:**

Imagina que ARGOS es una startup AgriTech que mañana va a presentar su producto frente a inversores y productores ganaderos.

La página debe transmitir innovación, confianza y simplicidad desde el primer segundo.

Pero recuerda que no estamos diseñando solamente una landing page bonita: estamos diseñando una herramienta que un productor debe poder usar fácilmente en el campo.

**La prioridad es una estética excepcional al servicio de una experiencia realmente útil.**
