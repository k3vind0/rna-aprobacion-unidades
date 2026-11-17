# Bitácora técnica y sustento del desarrollo

Este documento resume las decisiones de diseño, los experimentos comparativos y las pruebas realizadas para resolver el ejercicio de clasificación académica mediante una Red Neuronal Artificial (RNA) en `Ejercicio_RNA_Unidades`.

---

## 1. Análisis del problema y criterios considerados

El ejercicio plantea predecir si un estudiante aprueba (`1`) o desaprueba (`0`) una experiencia curricular a partir de tres promedios de unidad (`U1`, `U2`, `U3`), cumpliendo cuatro condiciones:

1. Exactamente tres entradas numéricas correspondientes a las tres unidades.
2. Mínimo 2 de las 3 unidades aprobadas para aprobar el curso.
3. No utilizar el promedio aritmético global entre las tres notas para decidir la clase final.
4. Rango válido de `0` a `20` para cada unidad (`NOTA_MINIMA = 0.0`, `NOTA_MAXIMA = 20.0`).

De acuerdo con la regla académica aplicada, el corte por unidad es estricto en `14.0` (`UMBRAL_UNIDAD = 14.0`). No existe redondeo de medio punto dentro de una unidad individual: notas como `13.5`, `13.6` o `13.9` dejan desaprobada esa unidad. El valor `UMBRAL_PROMEDIO_REF = 13.5` se conservó únicamente en las tablas comparativas (`Diria_Promedio` en el notebook y `Criterio promedio` en `app.py`) para mostrar cuándo el promedio tradicional con redondeo daría un resultado distinto al conteo de unidades.

En términos formales, la función objetivo queda definida sobre el cubo $[0, 20]^3$ como:

$$y(U_1, U_2, U_3) = \mathbb{I}\Big(\mathbb{I}(U_1 \ge 14.0) + \mathbb{I}(U_2 \ge 14.0) + \mathbb{I}(U_3 \ge 14.0) \ge 2\Big)$$

Esta regla aparece únicamente dentro de `etiquetar_aprobacion` para construir las etiquetas `y` del conjunto sintético de entrenamiento y evaluación. En la etapa de inferencia, tanto en el notebook como en `app.py`, descartamos cualquier condicional `if` manual: la clase predicha y la probabilidad salen directamente de `modelo_rna.predict` y `modelo_rna.predict_proba`, usando los pesos aprendidos por la red.

---

## 2. Construcción del dataset y manejo de casos de frontera

Al diseñar `generar_dataset_balanceado` (con semilla fija `SEMILLA = 42`), comprobamos que un muestreo puramente uniforme en $[0, 20]^3$ no bastaba. Por volumen geométrico, en `[0, 20]^3` solo alrededor del 21.6 % de los puntos aleatorios cumplen tener al menos 2 unidades `>= 14.0`, y muy pocos caen justo en el salto de `0.1` puntos entre `13.9` y `14.0`. Si la red no ve suficientes ejemplos cerca del corte, tiende a suavizar las esquinas o desplazar el umbral real.

Por eso armamos el dataset combinando tres bloques:

- **Cobertura general (`muestras_uniformes`):** `4,500` registros generados de forma uniforme en `[0, 20]^3` con un decimal.
- **Muestreo denso simétrico en la frontera (`muestras_borde`):** `7,500` registros donde `1`, `2` o `3` unidades (con probabilidades `0.20`, `0.45` y `0.35`) se toman alternadamente de `banda_umbral` (`13.3` a `14.6`) y `banda_fina` (`13.5` a `14.4`). Mantener estas bandas simétricas respecto al punto medio `13.95` evita sesgar el límite aprendido hacia arriba o hacia abajo de `14.0`.
- **Casos de contraste (`muestras_trampa`):** `3,000` registros divididos en dos grupos: mitad con una unidad alta en `[16.0, 20.0]` y dos desaprobadas (con 60 % de probabilidad en `[13.5, 13.9]` y 40 % en `[0.0, 13.9]`), y mitad con dos unidades aprobadas (60 % en `[14.0, 14.4]` y 40 % en `[14.0, 17.5]`) acompañadas de una nota baja en `[0.0, 8.0]`.

Después de unir los tres bloques, equilibramos ambas clases tomando la misma cantidad de ejemplos (`n_por_clase`), obteniendo un dataset final de **`11,264` registros** (`5,632` de clase `0` y `5,632` de clase `1`). Este conjunto se dividió de manera estratificada (`train_test_split`, `test_size=0.20`, `stratify=y_datos`) en **`9,011` registros de entrenamiento** (`X_train`) y **`2,253` registros de prueba** (`X_test`).

### Preprocesamiento aplicado

Usamos dos pasos encadenados antes de entrar a la red neuronal:

1. `escalador_01 = MinMaxScaler(feature_range=(0, 1))`, ajustado explícitamente sobre los extremos teóricos `[[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]]` para mapear el rango vigesimal a `[0, 1]`.
2. `estandarizador = StandardScaler()`, ajustado únicamente con `X_train` (`fit_transform` sobre entrenamiento y `transform` sobre prueba) para evitar fuga de datos.

En las pruebas iniciales sin `StandardScaler`, dejar las notas solo en `[0, 1]` ubicaba el umbral `14.0` en `0.70` y reducía la separación entre `13.9` (`0.695`) y `14.0` (`0.700`) a apenas `0.005`, lo que hacía que los gradientes oscilaran y algunas activaciones `ReLU` quedaran mal posicionadas respecto al origen. Al centrar las entradas alrededor de `0` con varianza unitaria mediante `StandardScaler`, los gradientes se estabilizaron y la optimización ubicó el corte exacto en `14.0` en menos épocas.

---

## 3. De Perceptrón Simple a Perceptrón Multicapa (proceso experimental)

Siguiendo el desarrollo teórico visto en la Sesión 12, evaluamos tres modelos en orden creciente de capacidad sobre exactamente los mismos conjuntos `X_train_norm` y `X_test_norm`:

### 3.1. Perceptrón simple (modelo lineal)
Probamos `Perceptron(max_iter=1000, random_state=42)` como línea base. Su exactitud se quedó en **`0.6334` (63.34 %) en entrenamiento** y **`0.6201` (62.01 %) en prueba**.

La explicación de este límite es geométrica. Un perceptrón simple calcula una única suma ponderada $z = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$ y separa el espacio con un solo plano. Todo conjunto clasificado como positivo por un plano único debe ser convexo. En nuestro problema no ocurre eso: los tres puntos `(14, 14, 0)`, `(14, 0, 14)` y `(0, 14, 14)` pertenecen a la clase `1` (tienen 2 unidades aprobadas), pero su promedio geométrico o centroide `(9.33, 9.33, 9.33)` pertenece a la clase `0` (tiene 0 unidades aprobadas). Además, al fijar una unidad aprobada (por ejemplo `U3 = 16.0`), la región válida de `(U1, U2)` forma una esquina en L (`U1 >= 14.0` o `U2 >= 14.0`), que tampoco es linealmente separable.

### 3.2. MLP de una capa oculta (`MLP (16)`)
Añadimos una capa oculta con `16` neuronas y activación `ReLU` (`MLPClassifier(hidden_layer_sizes=(16,), activation="relu", solver="adam", alpha=1e-6, batch_size=64, learning_rate_init=0.005, max_iter=1200, tol=1e-6, n_iter_no_change=25)`). La exactitud subió a **`0.9725` (97.25 %) en entrenamiento** y **`0.9694` (96.94 %) en prueba**. Esto confirma que las funciones de activación no lineales permiten aproximar las esquinas ortogonales de la regla, aunque con una sola capa oculta aún quedaba un margen de error cercano al 3 % en casos ajustados.

### 3.3. Arquitectura final: MLP de dos capas ocultas (`MLP (32, 16)`)
Para componer con mayor precisión las tres regiones de aprobación por pares (`(U1, U2)`, `(U1, U3)` y `(U2, U3)`), definimos la arquitectura final **`3 -> 32 (ReLU) -> 16 (ReLU) -> 1 (Sigmoide)`** mediante:

```python
red_mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    alpha=1e-6,
    batch_size=64,
    learning_rate_init=0.005,
    max_iter=600,
    tol=1e-6,
    n_iter_no_change=40,
    random_state=SEMILLA,
)
```

- **Función de pérdida y salida:** al tratarse de clasificación binaria (`0` y `1`), `MLPClassifier` emplea automáticamente activación logística (`logistic` / sigmoide) en la neurona de salida y minimiza la entropía cruzada binaria (`log_loss`).
- **Regularización L2 (`alpha=1e-6`):** usamos una penalización baja porque valores altos de `alpha` castigan las pendientes pronunciadas y suavizan la transición entre `13.9` y `14.0`.
- **Convergencia y métricas:** el entrenamiento finalizó en **`280` épocas** con pérdida final `0.00011`, alcanzando **`1.0000` (100.00 %) en entrenamiento**, **`1.0000` (100.00 %) en prueba** (`2,253` casos), **`ROC-AUC = 1.0000`** y matriz de confusión `[[1127, 0], [0, 1126]]`.

---

## 4. Validación exhaustiva y puesta en producción (`app.py`)

Antes de dar por terminado el modelo, aplicamos tres niveles de verificación independientes en `RNA_Aprobacion_Estudiante.ipynb`:

1. **Barrido completo de notas enteras (`rejilla_enteros`):** evaluamos las `21^3 = 9,261` combinaciones enteras posibles de `(U1, U2, U3)` entre `0` y `20`. La red obtuvo **`1.0000` (`0` errores)**.
2. **Casos trampa y permutaciones límite:** evaluamos `13` casos críticos (`13/13`) y sus `33` permutaciones únicas (`33/33`), verificando ejemplos concretos donde el promedio tradicional engaña:
   - `[13.5, 13.5, 5.0]` -> `prob = 0.0000` (`Desaprueba`).
   - `[13.8, 13.8, 13.8]` -> `prob = 0.0000` (`Desaprueba`, pese a que su promedio `13.80 >= 13.5` marcaría aprobado por promedio).
   - `[20.0, 13.9, 13.9]` -> `prob = 0.0031` (`Desaprueba`, pues solo tiene una unidad `>= 14.0`, aunque promedie `15.93`).
   - `[14.0, 14.0, 2.0]` -> `prob = 0.9997` (`Aprueba`, pues cumple 2 unidades `>= 14.0`, aunque su promedio `10.00` repruebe en el criterio clásico).
   - `[13.9, 14.0, 14.0]` -> `prob = 0.9999` (`Aprueba`).
   - `[13.9, 13.9, 14.0]` -> `prob = 0.0003` (`Desaprueba`).
3. **Inspección visual de cortes 2D:** graficamos los planos `U3 = 5.0` (donde solo aprueba el cuadrante `U1 >= 14.0` y `U2 >= 14.0`) y `U3 = 16.0` (donde aprueba la región en L `U1 >= 14.0` o `U2 >= 14.0`) para confirmar que las fronteras aprendidas coinciden con la regla.

### Separación del entrenamiento y la interfaz

Para no reentrenar la red cada vez que se consulta un alumno en vivo, empaquetamos todo el `Pipeline` (`escalador_01`, `estandarizador` y `red_mlp`) junto con `umbral_unidad`, `umbral_promedio_ref`, `loss_curve`, `comparacion_modelos` y `metricas` dentro de `modelo_rna_aprobacion.pkl` usando `joblib.dump`.

En `app.py`, la función `cargar_paquete_modelo(mtime_ns, tamano_bytes)` está decorada con `@st.cache_resource` y recibe la marca de modificación (`st_mtime_ns`) y el tamaño (`st_size`) del archivo `.pkl`. De este modo, el modelo permanece cargado en memoria para responder al instante en la interfaz de Streamlit y, si en algún momento se sobrescribe `modelo_rna_aprobacion.pkl` en disco, la caché se invalida automáticamente y recarga los pesos actualizados sin reiniciar el servidor.
