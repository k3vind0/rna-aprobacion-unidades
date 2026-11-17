# Predicción de aprobación por unidades con Red Neuronal Artificial (RNA)

Ejercicio práctico PERCEPTRON, curso **Sistemas Inteligentes (Semana 13)**.

## Reglas del problema
El modelo recibe 3 promedios de unidad (`U1`, `U2`, `U3`) en escala de `0` a `20` y predice si el estudiante aprueba (`1`) o desaprueba (`0`) la experiencia curricular bajo dos condiciones:
- Debe tener **por lo menos 2 unidades aprobadas**.
- No se usa promedio aritmético global (por ejemplo, `(20, 13.9, 13.9)` y `(13.8, 13.8, 13.8)` desaprueban; `(14, 14, 2)` y `(13.9, 14, 14)` aprueban).

### Umbral estricto por unidad y promedio referencial
Cada unidad individual se aprueba estrictamente con `>= 14.0` (no aplica medio punto ni redondeo por unidad). El `13.5` aplica únicamente al promedio final tradicional, usado en las tablas solo como columna de contraste (`Criterio promedio`):

```python
UMBRAL_UNIDAD = 14.0
UMBRAL_PROMEDIO_REF = 13.5
```

## Estructura de la carpeta
- `RNA_Aprobacion_Estudiante.ipynb`: generación balanceada del dataset (incluyendo casos límite entre `13.4` y `14.2`), escalamiento `[0, 20] -> [0, 1]` con estandarización, comparación con Perceptrón simple, entrenamiento del MLP (`3 -> 32 (ReLU) -> 16 (ReLU) -> 1 (Sigmoide)`), gráficas, pruebas trampa y exportación del modelo.
- `modelo_rna_aprobacion.pkl`: pipeline entrenado (`MinMaxScaler` + `StandardScaler` + `MLPClassifier`) y métricas guardadas.
- `app.py`: interfaz web en Streamlit para probar predicciones en vivo sin reentrenar.
- `EXPLICACION_DESARROLLO.md`: bitácora y sustento técnico del diseño del dataset, escalamiento, evolución desde Perceptrón simple hasta MLP y pruebas de validación.
- `requirements.txt`: dependencias del proyecto.

## Cómo ejecutar

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Opcional: abrir y ejecutar `RNA_Aprobacion_Estudiante.ipynb` (en VS Code o Jupyter) para revisar el entrenamiento o regenerar `modelo_rna_aprobacion.pkl`.

3. Iniciar la interfaz en Streamlit desde esta carpeta:
```bash
streamlit run app.py
```
Si `streamlit` no figura en el PATH del sistema, ejecutar:
```bash
python -m streamlit run app.py
```
