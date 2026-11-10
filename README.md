

# Taller de MLOps: Ciclo de Vida de un Modelo de Riesgo y Retención

Bienvenido al repositorio del taller **"El Ciclo de Vida de un Modelo de Machine Learning"**. 

Este proyecto está diseñado para estudiantes de la carrera de Actuaría. Aquí no nos limitaremos a ajustar un algoritmo en un notebook: recorreremos el camino completo que vive un modelo en el mundo real, desde la concepción del problema de negocio y la calibración estadística, hasta su despliegue automatizado en producción y el monitoreo de su degradación ante perturbaciones del mercado.

---

## 1. El Problema de Negocio y Enfoque Actuarial

Imagina una compañía de servicios de suscripción mensual (telefonía y conectividad de fibra óptica, o seguros masivos de autos/vida con cobro recurrente).

* **El reto:** Adquirir un cliente nuevo cuesta mucho dinero (marketing, instalación de infraestructura, comisiones de suscripción). Si el cliente cancela su contrato en los primeros meses, la compañía incurre en una pérdida neta.
* **El objetivo:** Predecir con anticipación qué clientes tienen una alta probabilidad de marcharse (*Churn* o Cancelación) durante el próximo mes.
* **La estrategia comercial/actuarial:** Con estas alertas, el área de retención ofrece beneficios o promociones personalizadas antes de que el cliente abandone la compañía.
* **La matriz de costos del error:**
  * **Falso Positivo ($FP$):** Creer que el cliente se va a ir cuando en realidad planeaba quedarse. *Costo:* Regalar un descuento innecesario (pérdida de margen).
  * **Falso Negativo ($FN$):** No detectar a un cliente que decide marcharse. *Costo:* Pérdida total del flujo futuro de primas/pagos y del valor de vida del cliente (*Customer Lifetime Value*). En este contexto, un Falso Negativo suele ser significativamente más perjudicial que un Falso Positivo.

---

## 2. ¿Qué Modelo Usamos y Por Qué?

En este taller empleamos un **Pipeline con Regresión Logística Balanceada**:

1. **¿Por qué Regresión Logística?**  
   Para un actuario, la Regresión Logística es la base de los **Modelos Lineales Generalizados (GLM)**, el estándar regulatorio por excelencia en tarificación y reservas. No es una "caja negra": permite auditar directamente los coeficientes y entender cómo impacta cada variable en la razón de momios (*Odds Ratio*).
2. **¿Por qué `class_weight='balanced'`?**  
   En la base histórica, solo cerca del 26% de los clientes cancelan (`Churn = 1`). Si entrenáramos un modelo estándar sin ajustes, el algoritmo tendería a predecir casi siempre que nadie se va para obtener un "Accuracy" engañoso del 74%. Al ponderar las clases, forzamos matemáticamente al estimador a penalizar con mayor severidad los errores en la clase minoritaria.
3. **¿Por qué usamos un `Pipeline` con `ColumnTransformer`?**  
   En producción real, los datos nuevos llegan "en bruto". Un Pipeline empaqueta juntos tanto la estandarización numérica (`StandardScaler`) como la codificación de variables cualitativas (`OneHotEncoder`) junto con el modelo. Esto previene la fuga de información (*Data Leakage*) y asegura que el sistema en producción procese cualquier entrada sin requerir limpieza manual.

---

## 3. Glosario de Métricas: ¿Cómo Evaluamos el Modelo?

Cuando trabajamos con eventos de riesgo o eventos desbalanceados, la precisión global (**Accuracy**) no es la mejor métrica. En este taller nos enfocamos en tres indicadores actuariales:

### A) ROC-AUC (Área Bajo la Curva ROC) 
* **¿Qué mide?** La habilidad del modelo para ordenar a los clientes de menor a mayor riesgo.
* **Interpretación intuitiva:** Si tomas al azar a un cliente que canceló y a uno que se quedó, el AUC representa la probabilidad de que tu modelo le asigne un puntaje de riesgo más alto al que efectivamente se marchó.
* **Escala:**
  * `0.50`: Equivalente a lanzar una moneda al aire (azar puro, modelo inútil).
  * `0.70 - 0.79`: Discriminación aceptable.
  * `0.80 - 0.89`: **Discriminación sólida** (nuestro objetivo en el entrenamiento es $pprox 0.85$).
  * `1.00`: Separación perfecta (suele indicar sobreajuste o fuga de información).

### B) Brier Score 
* **¿Qué mide?** Qué tan cerca están las probabilidades numéricas de la realidad observada. Se calcula como el error cuadrático medio entre la probabilidad predicha ($p_i \in [0, 1]$) y el desenlace real ($y_i \in \{0, 1\}$):
  $$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$
* **Interpretación intuitiva:** Un modelo puede tener buen AUC pero asignar probabilidades irreales (por ejemplo, predecir 0.99 a todo el mundo). Para un actuario, el valor de la probabilidad es vital porque se usa para calcular primas de riesgo esperadas.
* **Escala:** Entre más cercano a `0.0`, mejor calibrado está el modelo. Un valor por encima de `0.25` suele indicar un modelo descalibrado.

### C) Recall (Sensibilidad) 
* **¿Qué mide?** Del total de clientes que realmente cancelaron, ¿qué porcentaje logró cazar el modelo?
  $$\text{Recall} = \frac{TP}{TP + FN}$$
* En retención de cartera, buscamos un Recall alto (e.g. $> 75\%$) para no dejar escapar asegurados en riesgo.

---

## 4. Estructura del Repositorio

```text
├── .github/
│   └── workflows/
│       ├── 01_inferencia_batch.yml       <- Acción que simula la predicción mensual automática
│       └── 02_monitoreo_desempeno.yml     <- Acción que evalúa si el modelo se degradó con el tiempo
├── data/
│   ├── telco_train_alumnos.csv           <- 80% histórico para entrenar 
│   ├── lote_mes_1_inferencia.csv         <- Clientes del Mes 1 sin etiquetar (Producción)
│   ├── ground_truth_mes_1.csv            <- Realidad del Mes 1 (para monitoreo estable)
│   ├── lote_mes_2_inferencia.csv         <- Clientes del Mes 2 sin etiquetar (Producción)
│   └── ground_truth_mes_2.csv            <- Realidad del Mes 2 con Shock de Mercado
├── models/
│   └── .gitkeep                          <- Carpeta donde colocarás tu modelo entrenado
├── notebooks/
│   └── 01_entrenamiento_pipeline.ipynb   <- Notebook de trabajo
├── scripts/
│   ├── predict_batch.py                  <- Código Python de inferencia 
│   └── monitor_performance.py            <- Código Python de auditoría de métricas y alertas
├── requirements.txt                      <- Dependencias exactas del entorno de ejecución
└── README.md                             <- Guía del taller
```

---

## 5. Guía de Ejecución Paso a Paso

### Paso 1: Hacer Fork del Repositorio
1. En la esquina superior derecha de esta pantalla de GitHub, haz clic en el botón **Fork**.
2. Confirma crear una copia en tu cuenta personal. A partir de este momento, trabajarás sobre tu propio repositorio forkeado.

---

### Paso 2: Abrir el Notebook y Entrenar el Modelo
1. Entra a la carpeta `notebooks/` y abre `01_entrenamiento_pipeline.ipynb`.
2. Completa los 4 bloques marcados con `# TODO`:
   * **TODO 1:** Partición estratificada Train/Test (`test_size=0.20`, `stratify=y`).
   * **TODO 2:** Configuración del estimador balanceado (`class_weight='balanced'`).
   * **TODO 3:** Cálculo de métricas base (**ROC-AUC** y **Brier Score**).
   * **TODO 4:** Serialización del pipeline completo con `joblib.dump()`.
3. Al ejecutar la última celda, descarga a tu computadora el archivo `modelo_retencion.joblib`.

---

### Paso 3: Subir el Modelo a tu Repositorio
Para que GitHub Actions pueda usar tu modelo, debes colocarlo en la carpeta `models/`:
1. En tu repositorio forkeado de GitHub, entra a la carpeta `models/`.
2. Haz clic en **Add file** -> **Upload files**.
3. Arrastra tu archivo `modelo_retencion.joblib`.
4. En la parte inferior, escribe un mensaje de confirmación (ej. `"Subir modelo calibrado v1"`) y presiona el botón verde **Commit changes**.

---

### Paso 4: Simulación de Producción (Inferencia por Lotes)
Imaginemos que es el **Día 1 del Mes 1**. Entran 704 clientes a renovar y el área comercial necesita saber a quién contactar.
1. En tu repositorio, haz clic en la pestaña **Actions** (barra superior).
2. En el menú lateral izquierdo, haz clic en **01 - Inferencia Batch**.
3. En el botón desplegable **Run workflow** a la derecha, mantén el archivo por defecto: `data/lote_mes_1_inferencia.csv`.
4. Haz clic en el botón verde **Run workflow**.
5. Espera unos 30 segundos hasta ver la palomita verde.
6. Haz clic en la ejecución completada y luego entra a la pestaña **Summary** (arriba a la izquierda):
   * Verás la tabla con el porcentaje de clientes en alerta y sus probabilidades individuales.
   * Al pie de la página, en la sección **Artifacts**, puedes descargar el archivo CSV con las calificaciones.

---

### Paso 5: Monitoreo y Auditoría Actuarial (Mes 1 vs. Mes 2)

En la práctica aseguradora transcurren semanas para conocer el desenlace real: quién pagó y quién canceló (*Ground Truth Lag*).

#### A) Evaluación del Mes 1 (Comportamiento Habitual):
1. En la pestaña **Actions**, selecciona el workflow **02 - Monitoreo de Desempeño**.
2. Haz clic en **Run workflow** dejando los valores por defecto:
   * *Lote evaluado:* `data/lote_mes_1_inferencia.csv`
   * *Archivo con la realidad:* `data/ground_truth_mes_1.csv`
3. Presiona **Run workflow**.
4. Entra al **Summary** de la ejecución al finalizar:
   * Comprueba que el ROC-AUC se mantenga en ~0.85 y el Brier Score en ~0.16.
   * El sistema emitirá el dictamen: **MODELO SALUDABLE**.

#### B) Evaluación del Mes 2 (Shock de Mercado y Descalibración):
Imaginemos que entra un competidor agresivo con tarifas de fibra óptica a mitad de precio, rompiendo los patrones históricos de permanencia.
1. En la pestaña **Actions**, selecciona nuevamente **02 - Monitoreo de Desempeño**.
2. Haz clic en **Run workflow**, pero esta vez **cambia ambos parámetros por los datos del Mes 2**:
   * *Lote evaluado:* `data/lote_mes_2_inferencia.csv`
   * *Archivo con la realidad:* `data/ground_truth_mes_2.csv`
3. Presiona **Run workflow**.
4. Entra al **Summary** al terminar:
   * Observa el impacto: el ROC-AUC se desplomará a ~0.52 (discriminación nula, como un volado) y el Brier Score subirá a ~0.35.
   * El sistema disparará la advertencia de gobierno de modelos: **ACCIÓN REQUERIDA: REENTRENAMIENTO INMEDIATO**.

---

## 6. Conclusiones del Ciclo de Vida (MLOps)

1. **Un modelo no termina cuando el código compila en el notebook:** Ajustar el algoritmo representa apenas una fracción del trabajo.
2. **Los modelos se degradan:** Las preferencias del cliente y el entorno macroeconómico cambian con el tiempo (*Concept Drift*).
3. **El monitoreo continuo es indispensable:** Sin un pipeline automatizado de auditoría, la compañía continuaría tomando decisiones comerciales y de reservas basándose en probabilidades ficticias.
