from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

RUTA_MODELO = Path(__file__).resolve().parent / "modelo_rna_aprobacion.pkl"

st.set_page_config(
    page_title="Evaluación por Unidades - RNA",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }
        hr {
            margin-top: 2.1rem !important;
            margin-bottom: 2.1rem !important;
            border-color: #E2E8F0 !important;
        }
        h1 {
            letter-spacing: -0.015em;
            margin-bottom: 0.25rem !important;
        }
        h2, h3 {
            margin-top: 0.35rem !important;
            margin-bottom: 0.85rem !important;
        }
        .subtitulo-curso {
            font-size: 0.96rem;
            color: #475569;
            margin-bottom: 1.2rem;
        }
        .caja-resultado {
            padding: 1.25rem 1.45rem;
            border-radius: 8px;
            border: 1px solid #CBD5E1;
            background-color: #F8FAFC;
            margin-top: 0.35rem;
        }
        .caja-aprobado {
            border-left: 5px solid #1B6B47;
            background-color: #F3FAF6;
        }
        .caja-desaprobado {
            border-left: 5px solid #991B1B;
            background-color: #FEF5F5;
        }
        .etiqueta-estado {
            font-size: 0.88rem;
            color: #475569;
            margin-bottom: 0.25rem;
        }
        .valor-aprobado {
            font-size: 1.65rem;
            font-weight: 700;
            color: #1B6B47;
            margin: 0;
        }
        .valor-desaprobado {
            font-size: 1.65rem;
            font-weight: 700;
            color: #991B1B;
            margin: 0;
        }
        .nota-pie {
            font-size: 0.9rem;
            color: #334155;
            line-height: 1.55;
            margin-top: 0.55rem;
        }
        .titulo-grafica {
            font-size: 0.98rem;
            font-weight: 600;
            color: #1E293B;
            margin-bottom: 0.35rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cargar_paquete_modelo(mtime_ns: int, tamano_bytes: int):
    return joblib.load(RUTA_MODELO)


def aplicar_caso_rapido(u1: float, u2: float, u3: float) -> None:
    st.session_state["u1_input"] = float(u1)
    st.session_state["u2_input"] = float(u2)
    st.session_state["u3_input"] = float(u3)


if RUTA_MODELO.exists():
    info_archivo = RUTA_MODELO.stat()
    paquete = cargar_paquete_modelo(info_archivo.st_mtime_ns, info_archivo.st_size)
else:
    paquete = None

st.title("Predicción de aprobación por unidades mediante RNA")
st.markdown(
    '<div class="subtitulo-curso">Curso: Sistemas Inteligentes | Implementación MLP con scikit-learn coherente con la Sesión 12</div>',
    unsafe_allow_html=True,
)

if paquete is None:
    st.error(
        "No se encontró el archivo 'modelo_rna_aprobacion.pkl'. "
        "Ejecute primero todas las celdas de 'RNA_Aprobacion_Estudiante.ipynb' para entrenar y exportar el modelo."
    )
    st.stop()

modelo_rna = paquete["modelo"]
umbral_unidad = float(paquete["umbral_unidad"])
umbral_promedio_ref = float(paquete.get("umbral_promedio_ref", 13.5))
arquitectura = paquete["arquitectura"]
metricas = paquete["metricas"]
loss_curve = paquete["loss_curve"]
comparacion_modelos = paquete["comparacion_modelos"]

with st.expander("Reglas del problema y configuración del modelo", expanded=False):
    st.markdown(
        f"""
        - **Regla a:** Se ingresan 3 promedios de unidad (`U1`, `U2`, `U3`).
        - **Regla b:** El estudiante aprueba si obtiene al menos **2 unidades aprobadas**.
        - **Regla c:** No se usa promedio aritmético global; la RNA aprendió esta frontera no lineal desde los datos.
        - **Regla d:** Rango válido de cada unidad entre `0.0` y `20.0` (escalado a `[0, 1]`).
        - **Umbral estricto por unidad:** `{umbral_unidad}` (cada unidad requiere `>= 14.0`; no hay redondeo por unidad).
        - **Promedio final referencial:** `{umbral_promedio_ref}` (el medio punto a favor solo aplica al promedio tradicional en la columna comparativa).
        - **Arquitectura cargada:** `{arquitectura}` ({paquete['epocas']} épocas de entrenamiento).
        """
    )

if "u1_input" not in st.session_state:
    st.session_state["u1_input"] = 14.0
    st.session_state["u2_input"] = 14.0
    st.session_state["u3_input"] = 2.0

st.subheader("1. Evaluación individual con la Red Neuronal Artificial")

col_izq, col_der = st.columns([1.05, 1.15], gap="large")

with col_izq:
    c1, c2, c3 = st.columns(3)
    u1 = c1.number_input("Unidad 1 (U1)", min_value=0.0, max_value=20.0, step=0.1, format="%.1f", key="u1_input")
    u2 = c2.number_input("Unidad 2 (U2)", min_value=0.0, max_value=20.0, step=0.1, format="%.1f", key="u2_input")
    u3 = c3.number_input("Unidad 3 (U3)", min_value=0.0, max_value=20.0, step=0.1, format="%.1f", key="u3_input")

    st.markdown("**Casos rápidos para demostración en vivo:**")
    r1_c1, r1_c2, r1_c3 = st.columns(3, gap="small")
    r1_c1.button("20 | 13.5 | 13.5", on_click=aplicar_caso_rapido, args=(20.0, 13.5, 13.5), width="stretch")
    r1_c2.button("20 | 13.9 | 13.9", on_click=aplicar_caso_rapido, args=(20.0, 13.9, 13.9), width="stretch")
    r1_c3.button("13.8 | 13.8 | 13.8", on_click=aplicar_caso_rapido, args=(13.8, 13.8, 13.8), width="stretch")

    r2_c1, r2_c2, r2_c3 = st.columns(3, gap="small")
    r2_c1.button("14 | 14 | 2", on_click=aplicar_caso_rapido, args=(14.0, 14.0, 2.0), width="stretch")
    r2_c2.button("13.9 | 14 | 14", on_click=aplicar_caso_rapido, args=(13.9, 14.0, 14.0), width="stretch")
    r2_c3.button("20 | 12 | 12", on_click=aplicar_caso_rapido, args=(20.0, 12.0, 12.0), width="stretch")

with col_der:
    # La predicción proviene exclusivamente de los pesos aprendidos del MLP
    entrada = np.array([[u1, u2, u3]], dtype=float)
    pred_clase = int(modelo_rna.predict(entrada)[0])
    prob_aprobar = float(modelo_rna.predict_proba(entrada)[0, 1])
    promedio_ref = float(np.mean(entrada))

    if pred_clase == 1:
        clase_css = "caja-aprobado"
        valor_css = "valor-aprobado"
        texto_estado = "APROBADO (1)"
    else:
        clase_css = "caja-desaprobado"
        valor_css = "valor-desaprobado"
        texto_estado = "DESAPROBADO (0)"

    st.markdown(
        f"""
        <div class="caja-resultado {clase_css}">
            <div class="etiqueta-estado">Predicción directa de la RNA</div>
            <p class="{valor_css}">{texto_estado}</p>
            <div class="nota-pie">
                Probabilidad estimada de aprobación (salida sigmoide): <strong>{prob_aprobar * 100:.2f}%</strong>
                (<code>{prob_aprobar:.4f}</code>)<br>
                Promedio aritmético referencial (no usado por el modelo): <strong>{promedio_ref:.2f}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.subheader("2. Demostración con casos extremos y de frontera")
st.write(
    f"Comparación de casos donde el promedio final referencial (>= {umbral_promedio_ref}) falla o donde las notas "
    f"bordean el umbral estricto por unidad ({umbral_unidad}) frente a la salida aprendida por la RNA."
)

casos_extremos = [
    (13.5, 13.5, 5.0),
    (13.8, 13.8, 13.8),
    (20.0, 12.0, 12.0),
    (20.0, 13.5, 13.5),
    (20.0, 13.8, 13.8),
    (20.0, 13.9, 13.9),
    (14.0, 14.0, 2.0),
    (14.0, 14.0, 0.0),
    (14.1, 14.1, 4.0),
    (13.9, 14.0, 14.0),
    (13.9, 13.9, 14.0),
    (9.3, 9.3, 9.3),
]

filas_tabla = []
for c_u1, c_u2, c_u3 in casos_extremos:
    x_caso = np.array([[c_u1, c_u2, c_u3]], dtype=float)
    y_pred = int(modelo_rna.predict(x_caso)[0])
    p_pred = float(modelo_rna.predict_proba(x_caso)[0, 1])
    prom = float(np.mean(x_caso))
    unidades_ok = int(np.sum(x_caso >= umbral_unidad))
    estado_real = 1 if unidades_ok >= 2 else 0
    # Solo el promedio final tradicional redondea desde 13.5 a 14
    diria_prom = 1 if prom >= umbral_promedio_ref else 0

    filas_tabla.append({
        "U1": c_u1,
        "U2": c_u2,
        "U3": c_u3,
        "Promedio simple": round(prom, 2),
        "Criterio promedio": "Aprueba" if diria_prom == 1 else "Desaprueba",
        "Estado real (regla)": "Aprueba" if estado_real == 1 else "Desaprueba",
        "Prob. RNA": round(p_pred, 4),
        "Predicción RNA": "Aprueba" if y_pred == 1 else "Desaprueba",
        "Acierto": "Correcto" if y_pred == estado_real else "Error",
    })

st.dataframe(pd.DataFrame(filas_tabla), width="stretch", hide_index=True)

st.divider()
st.subheader("3. Rendimiento del entrenamiento y frontera aprendida")

m1, m2, m3, m4 = st.columns(4, gap="medium")
with m1:
    with st.container(border=True):
        st.metric("Accuracy en prueba", f"{metricas['accuracy_prueba'] * 100:.2f}%")
with m2:
    with st.container(border=True):
        st.metric("ROC-AUC en prueba", f"{metricas['roc_auc_prueba']:.4f}")
with m3:
    with st.container(border=True):
        st.metric("Enteros [0..20]^3 (9,261)", f"{9261 - metricas['errores_enteros']:,}/9,261")
with m4:
    with st.container(border=True):
        st.metric(
            "Permutaciones límite",
            f"{metricas['aciertos_permutaciones']}/{metricas['total_permutaciones']}",
        )

st.markdown('<div style="height: 1.2rem;"></div>', unsafe_allow_html=True)

# Fila 1 de gráficas: evolución de pérdida y comparación entre modelos
col_g1, col_g2 = st.columns(2, gap="large")

with col_g1:
    with st.container(border=True):
        st.markdown('<div class="titulo-grafica">Evolución de la función de pérdida</div>', unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(6.4, 4.2), dpi=150)
        epocas = np.arange(1, len(loss_curve) + 1)
        ax1.plot(epocas, loss_curve, color="#1E3A5F", linewidth=2.0)
        ax1.set_title("Curva de pérdida (Binary Cross-Entropy)", pad=15, fontsize=11)
        ax1.set_xlabel("Época", fontsize=10)
        ax1.set_ylabel("Pérdida (log_loss)", fontsize=10)
        ax1.grid(alpha=0.25)
        sns.despine(ax=ax1)
        fig1.tight_layout(pad=2.0)
        st.pyplot(fig1)
        plt.close(fig1)

with col_g2:
    with st.container(border=True):
        st.markdown('<div class="titulo-grafica">Capacidad frente a frontera lineal</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6.4, 4.2), dpi=150)
        nombres = list(comparacion_modelos.keys())
        accs_tr = [comparacion_modelos[k]["train"] for k in nombres]
        accs_te = [comparacion_modelos[k]["test"] for k in nombres]
        pos = np.arange(len(nombres))
        ancho = 0.34
        b_tr = ax2.bar(pos - ancho / 2, accs_tr, width=ancho, color="#64748B", label="Entrenamiento")
        b_te = ax2.bar(pos + ancho / 2, accs_te, width=ancho, color="#1E3A5F", label="Prueba")
        ax2.set_xticks(pos)
        ax2.set_xticklabels(nombres, fontsize=9.5)
        ax2.set_ylim(0.0, 1.14)
        ax2.set_ylabel("Accuracy", fontsize=10)
        ax2.set_title("Comparación de exactitud: Perceptrón vs. MLP", pad=15, fontsize=11)
        ax2.legend(frameon=False, fontsize=9.5, loc="upper left")
        ax2.grid(axis="y", alpha=0.25)
        sns.despine(ax=ax2)
        for barra in list(b_tr) + list(b_te):
            h = barra.get_height()
            ax2.text(
                barra.get_x() + barra.get_width() / 2,
                h + 0.018,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig2.tight_layout(pad=2.0)
        st.pyplot(fig2)
        plt.close(fig2)

st.markdown('<div style="height: 1.35rem;"></div>', unsafe_allow_html=True)

# Fila 2 de gráficas: matriz de confusión y corte 2D interactivo para U3
col_g3, col_g4 = st.columns(2, gap="large")

with col_g3:
    with st.container(border=True):
        st.markdown('<div class="titulo-grafica">Desempeño en el conjunto de prueba</div>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(6.4, 4.2), dpi=150)
        cm = np.array(metricas["matriz_confusion"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            annot_kws={"size": 12},
            xticklabels=["Desaprobado (0)", "Aprobado (1)"],
            yticklabels=["Desaprobado (0)", "Aprobado (1)"],
            ax=ax3,
        )
        ax3.set_title("Matriz de confusión (conjunto de prueba)", pad=15, fontsize=11)
        ax3.set_xlabel("Predicción RNA", fontsize=10)
        ax3.set_ylabel("Estado real", fontsize=10)
        fig3.tight_layout(pad=2.0)
        st.pyplot(fig3)
        plt.close(fig3)

with col_g4:
    with st.container(border=True):
        st.markdown('<div class="titulo-grafica">Superficie de decisión según U3 actual</div>', unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(6.4, 4.2), dpi=150)
        eje = np.linspace(0, 20, 101)
        g1, g2 = np.meshgrid(eje, eje)
        malla = np.column_stack([g1.ravel(), g2.ravel(), np.full(g1.size, u3)])
        prob_malla = modelo_rna.predict_proba(malla)[:, 1].reshape(g1.shape)

        cont = ax4.contourf(g1, g2, prob_malla, levels=np.linspace(0, 1, 11), cmap="RdBu", alpha=0.85)
        ax4.contour(g1, g2, prob_malla, levels=[0.5], colors=["#111827"], linewidths=1.6)
        ax4.axvline(umbral_unidad, color="#475569", linestyle="--", linewidth=1.0)
        ax4.axhline(umbral_unidad, color="#475569", linestyle="--", linewidth=1.0)
        ax4.scatter([u1], [u2], color="#111827", edgecolor="white", linewidth=1.3, s=78, zorder=5, label="Caso actual")
        ax4.set_xlim(-0.4, 20.4)
        ax4.set_ylim(-0.4, 20.4)
        ax4.set_title(f"Frontera aprendida para U3 = {u3:.1f}", pad=15, fontsize=11)
        ax4.set_xlabel("Nota U1", fontsize=10)
        ax4.set_ylabel("Nota U2", fontsize=10)
        ax4.legend(loc="lower left", fontsize=9, frameon=True, facecolor="white", framealpha=0.9)
        sns.despine(ax=ax4)
        fig4.colorbar(cont, ax=ax4, shrink=0.92, label="P(Aprobado)")
        fig4.tight_layout(pad=2.0)
        st.pyplot(fig4)
        plt.close(fig4)
