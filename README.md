# Taller de MLOps: Ciclo de Vida de un Modelo de Riesgo y Retención

Bienvenido al repositorio del taller **"El Ciclo de Vida de un Modelo de Machine Learning"**. 

Este proyecto está diseñado para estudiantes de la carrera de Actuaría. Recorreremos el flujo de trabajo profesional de un equipo de ciencia de datos: clonación de repositorio, aislamiento de dependencias en entornos virtuales, entrenamiento en VS Code, versionamiento con Git y despliegue automatizado en GitHub Actions.

---

##  1. El Problema de Negocio y Enfoque Actuarial

Imagina una compañía de servicios de suscripción mensual (telefonía y conectividad de fibra óptica, o seguros masivos de autos/vida con cobro recurrente).

* **El reto:** Adquirir un cliente nuevo cuesta mucho dinero (marketing, comisiones de venta, emisión). Si el cliente cancela en los primeros meses, la compañía incurre en una pérdida neta.
* **El objetivo:** Predecir con anticipación qué clientes tienen alta probabilidad de marcharse (*Churn* o Cancelación) durante el próximo mes.
* **La estrategia actuarial y comercial:** Con estas alertas, el área de retención ofrece beneficios o intervenciones oportunas antes de que el cliente abandone.
* **La matriz de costos del error:**
  * **Falso Positivo ($FP$):** Creer que el cliente se va a ir cuando en realidad planeaba quedarse. *Costo:* Gasto innecesario de retención o pérdida de margen por descuento.
  * **Falso Negativo ($FN$):** No detectar a un cliente que decide marcharse. *Costo:* Pérdida total del flujo futuro de primas/ingresos y del valor de vida del cliente (*LTV*). En este contexto, un Falso Negativo suele ser mucho más costoso que un Falso Positivo.

---

## 2. ¿Qué Modelo Usamos y Por Qué?

En este taller empleamos un **Pipeline con Regresión Logística Balanceada**:

1. **¿Por qué Regresión Logística?**  
   Para un actuario, la Regresión Logística es la base de los **Modelos Lineales Generalizados (GLM)**, el estándar regulatorio por excelencia en tarificación y reservas. Permite auditar directamente los coeficientes y entender cómo impacta cada variable en la razón de momios (*Odds Ratio*).
2. **¿Por qué `class_weight='balanced'`?**  
   En la base histórica, solo cerca del 26% de los clientes cancelan (`Churn = 1`). Al balancear las clases, forzamos al estimador a penalizar con mayor rigor los errores cometidos sobre los clientes que sí cancelan.
3. **¿Por qué usamos un `Pipeline` con `ColumnTransformer`?**  
   En producción real, los datos nuevos llegan "en bruto". Un Pipeline empaqueta juntos tanto la estandarización numérica (`StandardScaler`) como la codificación cualitativa (`OneHotEncoder`) junto con el modelo, evitando la fuga de información (*Data Leakage*).

---

## 3. Glosario de Métricas: ¿Cómo Evaluamos el Modelo?

### A) ROC-AUC (Área Bajo la Curva ROC) 
* **¿Qué mide?** La habilidad del modelo para ordenar a los clientes de menor a mayor riesgo. Si tomas al azar a un cliente que canceló y a uno que se quedó, el AUC representa la probabilidad de que tu modelo le asigne un puntaje más alto al que se marchó.
* **Escala:** `0.50` es azar puro (un volado); `0.80 - 0.89` es **discriminación sólida** (nuestro objetivo es $\approx 0.85$).

### B) Brier Score 
* **¿Qué mide?** El error cuadrático medio entre la probabilidad predicha ($p_i \in [0, 1]$) y el resultado real ($y_i \in \{0, 1\}$):
  $$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$
  Para un actuario, el valor exacto de la probabilidad es vital para calcular primas de riesgo y reservas.
* **Escala:** Entre más cercano a `0.0`, mejor calibrado está. Un valor mayor a `0.22` indica descalibración crítica.

### C) Recall (Sensibilidad) 
* **¿Qué mide?** Del total de clientes que realmente cancelaron, ¿qué porcentaje logró capturar el modelo?
  $$\text{Recall} = \frac{TP}{TP + FN}$$

---

##  4. Estructura del Repositorio

```text
├── .github/
│   └── workflows/
│       ├── 01_inferencia_batch.yml       <- Simula la calificación mensual automática
│       └── 02_monitoreo_desempeno.yml     <- Audita la calidad de las predicciones
├── data/
│   ├── telco_train_alumnos.csv           <- 80% histórico para entrenar en VS Code
│   ├── lote_mes_1_inferencia.csv         <- Clientes del Mes 1 sin etiquetar (Producción)
│   ├── ground_truth_mes_1.csv            <- Realidad observada del Mes 1 (Estable)
│   ├── lote_mes_2_inferencia.csv         <- Clientes del Mes 2 sin etiquetar (Producción)
│   └── ground_truth_mes_2.csv            <- Realidad observada del Mes 2 (Shock de Mercado)
├── models/
│   └── .gitkeep                          <- Carpeta donde colocarás tu modelo entrenado
├── notebooks/
│   └── 01_entrenamiento_pipeline.ipynb   <- Libreta interactiva de trabajo
├── scripts/
│   ├── predict_batch.py                  <- Script de inferencia desatendida
│   └── monitor_performance.py            <- Script de cálculo de métricas actuariales y alertas
├── requirements.txt                      <- Dependencias exactas para ejecución local y en nube
└── README.md                             <- Guía del taller
```

---

##  5. Guía de Ejecución Paso a Paso

### Paso 1: Hacer Fork y Clonar en tu Computadora
1. En la esquina superior derecha de esta página en GitHub, haz clic en el botón **Fork** y crea una copia en tu cuenta.
2. En **tu repositorio forkeado**, haz clic en el botón verde **`<> Code`** y copia la URL HTTPS.
3. Abre **VS Code**, abre una terminal (`Ctrl + ñ` o menú *Terminal > New Terminal*), ubícate en la carpeta donde guardas tus proyectos y ejecuta:
   ```bash
   git clone <URL_DE_TU_FORK>
   cd taller-intro-mlops-actuaria
   ```

---

### Paso 2: Crear Entorno Virtual e Instalar Dependencias
Para evitar conflictos de versiones entre tu computadora y el servidor de producción, crearemos un entorno aislado:

**En Windows (PowerShell / CMD):**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**En Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Paso 3: Entrenar y Exportar el Modelo en VS Code
1. Abre el archivo `notebooks/01_entrenamiento_pipeline.ipynb` en VS Code.
2. Arriba a la derecha del notebook, haz clic en **Select Kernel** $\rightarrow$ **Python Environments** y selecciona el entorno **`.venv`** que acabas de crear.
3. Ejecuta las primeras celdas para validar las versiones y rutas.
4. Completa los 4 bloques marcados con `# TODO`:
   * **TODO 1:** Partición estratificada (`test_size=0.20`, `stratify=y`).
   * **TODO 2:** Configuración de `LogisticRegression` con `class_weight="balanced"`.
   * **TODO 3:** Cálculo de `roc_auc_score` y `brier_score_loss`.
   * **TODO 4:** Guardar el modelo usando `joblib.dump(pipeline_completo, MODEL_PATH)`.
5. Verifica en el explorador de archivos de VS Code que se haya generado el archivo `models/modelo_retencion.joblib`.

---

### Paso 4: Subir tu Modelo a GitHub (`git push`)
En la terminal de VS Code, registra y envía tu modelo entrenado a tu repositorio remoto:

```bash
git status
git add models/modelo_retencion.joblib
git commit -m "feat: entrenar y desplegar modelo de retencion v1"
git push origin main
```

---

### Paso 5: Producción 
1. Ve a tu repositorio forkeado en el navegador y haz clic en la pestaña **Actions**.
2. Si aparece un aviso amarillo, haz clic en el botón verde **`I understand my workflows, go ahead and enable them`**.
3. En el menú izquierdo, selecciona **`01 - Inferencia Batch`**.
4. Haz clic en **Run workflow** (manteniendo `data/lote_mes_1_inferencia.csv`) y presiona el botón verde **Run workflow**.
5. Al terminar (~30 seg), entra a la ejecución y revisa la pestaña **Summary** para ver los clientes calificados y descargar el reporte CSV en la sección *Artifacts*.

---

### Paso 6: Monitoreo y Auditoría Actuarial (Mes 1 vs. Mes 2)

#### A) Mes 1 (Comportamiento Estable):
1. En **Actions**, selecciona **`02 - Monitoreo de Desempeño`**.
2. Ejecuta **Run workflow** con los valores por defecto (`lote_mes_1_inferencia.csv` y `ground_truth_mes_1.csv`).
3. Revisa en **Summary** que el ROC-AUC ($\approx 0.85$) y el Brier Score ($\approx 0.16$) estén en verde (**✅ MODELO SALUDABLE**).

#### B) Mes 2 (Shock de Mercado y Descalibración):
1. Vuelve a ejecutar **`02 - Monitoreo de Desempeño`**, pero **cambia ambos parámetros por los del Mes 2**:
   * *Lote evaluado:* `data/lote_mes_2_inferencia.csv`
   * *Archivo con la realidad:* `data/ground_truth_mes_2.csv`
2. Revisa en **Summary** cómo el ROC-AUC cae a $\approx 0.52$ y el Brier Score sube a $\approx 0.35$, disparando la alerta: **`🚨 ACCIÓN REQUERIDA: REENTRENAMIENTO INMEDIATO`**.

---

## 6. Conclusiones del Ciclo de Vida (MLOps)

1. **Un modelo no termina cuando el código compila en el notebook:** Ajustar el algoritmo representa apenas una fracción del trabajo.
2. **Los modelos se degradan:** Las preferencias del cliente y el entorno macroeconómico cambian con el tiempo (*Concept Drift*).
3. **El monitoreo continuo es indispensable:** Sin un pipeline automatizado de auditoría, la compañía continuaría tomando decisiones comerciales y de reservas basándose en probabilidades ficticias.
