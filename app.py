import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CONFIGURACIÓN
st.set_page_config(page_title="Comparador Curvas H-Q", page_icon="🌊", layout="wide")

st.title("🌊 COMPARADOR DE CURVAS ALTURA-CAUDAL")
st.markdown("**Inserta tu curva teórica y compárala con curvas de ejemplo**")

# FUNCIONES MATEMÁTICAS
def funcion_potencial(H, a, b):
    return a * (H ** b)

def funcion_polinomica(H, a, b, c):
    return a * H**2 + b * H + c

def funcion_lineal(H, a, b):
    return a * H + b

# GENERAR CURVAS DE EJEMPLO (SIMULANDO IA)
def generar_curvas_ejemplo():
    """Genera curvas de ejemplo para comparar"""
    curvas = {
        'CURVA_IA_1': {
            'funcion': lambda H: funcion_potencial(H, 2.0, 1.8),
            'rango': (0.3, 2.0),
            'color': 'blue',
            'nombre': 'Potencial (IA)',
            'ecuacion': 'Q = 2.000 × H¹·⁸⁰⁰'
        },
        'CURVA_IA_2': {
            'funcion': lambda H: funcion_polinomica(H, 0.3, 1.2, 0.1),
            'rango': (1.8, 4.0),
            'color': 'red', 
            'nombre': 'Polinómica (IA)',
            'ecuacion': 'Q = 0.300H² + 1.200H + 0.100'
        }
    }
    return curvas

# INICIALIZAR
curvas_ia = generar_curvas_ejemplo()

# =============================================
# SECCIÓN 1: CONFIGURACIÓN DE CURVA PERSONALIZADA
# =============================================
st.header("🎯 CONFIGURA TU CURVA PERSONALIZADA")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📐 PARÁMETROS DE LA CURVA")
    
    tipo_curva = st.selectbox(
        "**Tipo de ecuación:**",
        ["Potencial", "Polinómica G2", "Lineal"],
        key="tipo_curva"
    )
    
    # PARÁMETROS SEGÚN TIPO DE CURVA
    if tipo_curva == "Potencial":
        col_a, col_b = st.columns(2)
        with col_a:
            a_personal = st.number_input("Coeficiente a:", value=1.8, min_value=0.1, step=0.1, format="%.3f")
        with col_b:
            b_personal = st.number_input("Exponente b:", value=2.2, min_value=0.1, step=0.1, format="%.3f")
        st.latex(f"Q = {a_personal:.3f} \\times H^{{{b_personal:.3f}}}")
        
    elif tipo_curva == "Polinómica G2":
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            a_personal = st.number_input("Coeficiente a (H²):", value=0.4, step=0.1, format="%.3f")
        with col_b:
            b_personal = st.number_input("Coeficiente b (H):", value=1.0, step=0.1, format="%.3f")
        with col_c:
            c_personal = st.number_input("Coeficiente c:", value=0.2, step=0.1, format="%.3f")
        st.latex(f"Q = {a_personal:.3f}H^2 + {b_personal:.3f}H + {c_personal:.3f}")
        
    else:  # Lineal
        col_a, col_b = st.columns(2)
        with col_a:
            a_personal = st.number_input("Pendiente a:", value=3.0, step=0.1, format="%.3f")
        with col_b:
            b_personal = st.number_input("Intercepto b:", value=-1.0, step=0.1, format="%.3f")
        st.latex(f"Q = {a_personal:.3f}H + {b_personal:.3f}")

with col2:
    st.subheader("📏 RANGO DE VALIDEZ")
    h_min = st.number_input("**Altura mínima H (m):**", value=0.2, min_value=0.0, step=0.1, format="%.2f")
    h_max = st.number_input("**Altura máxima H (m):**", value=5.0, min_value=0.1, step=0.1, format="%.2f")
    
    nombre_curva = st.text_input("**Nombre de tu curva:**", value="MI_CURVA_TEORICA")
    
    st.info(f"**Rango definido:** {h_min:.2f} ≤ H ≤ {h_max:.2f} m")

# DEFINIR FUNCIÓN PERSONALIZADA
if tipo_curva == "Potencial":
    def funcion_personalizada(H):
        return a_personal * (H ** b_personal)
elif tipo_curva == "Polinómica G2":
    def funcion_personalizada(H):
        return a_personal * H**2 + b_personal * H + c_personal
else:  # Lineal
    def funcion_personalizada(H):
        return a_personal * H + b_personal

# =============================================
# BOTÓN PRINCIPAL - GENERAR COMPARACIÓN
# =============================================
if st.button("🚀 GENERAR COMPARACIÓN COMPLETA", type="primary", use_container_width=True):
    
    # =============================================
    # 1. GRÁFICO COMPARATIVO VISUAL
    # =============================================
    st.header("📈 COMPARACIÓN VISUAL")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Graficar curvas IA
    for nombre, curva in curvas_ia.items():
        rango_min, rango_max = curva['rango']
        H_curve = np.linspace(rango_min, rango_max, 100)
        Q_curve = [curva['funcion'](h) for h in H_curve]
        
        ax.plot(H_curve, Q_curve, color=curva['color'], linewidth=3, 
               label=f"{curva['nombre']}\n{curva['ecuacion']}")
    
    # Graficar curva personalizada
    H_personal = np.linspace(h_min, h_max, 100)
    Q_personal = [funcion_personalizada(h) for h in H_personal]
    ax.plot(H_personal, Q_personal, color='purple', linewidth=4, linestyle='--',
           label=f'🔷 {nombre_curva}\nTu curva personalizada')
    
    # Configurar gráfico
    ax.set_xlabel('Nivel H (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Caudal Q (m³/s)', fontsize=14, fontweight='bold')
    ax.set_title('COMPARACIÓN: CURVAS DE EJEMPLO vs TU CURVA PERSONALIZADA', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 25)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # =============================================
    # 2. TABLA COMPARATIVA DE CAUDALES
    # =============================================
    st.header("📋 TABLA COMPARATIVA DE CAUDALES")
    
    # Generar alturas para comparación
    alturas_comparacion = np.linspace(0.2, 4.5, 15)
    datos_tabla = []
    
    for h in alturas_comparacion:
        # Calcular caudal IA (usar curva apropiada)
        q_ia = None
        ia_curve_name = ""
        for nombre, curva in curvas_ia.items():
            rango_min, rango_max = curva['rango']
            if rango_min <= h <= rango_max:
                q_ia = curva['funcion'](h)
                ia_curve_name = curva['nombre']
                break
        
        # Calcular caudal personalizado
        q_personal = funcion_personalizada(h) if h_min <= h <= h_max else None
        
        # Calcular diferencias
        if q_ia is not None and q_personal is not None:
            diferencia = q_personal - q_ia
            diferencia_porcentaje = (diferencia / q_ia) * 100
            status = "✅ Comparable"
        else:
            diferencia = None
            diferencia_porcentaje = None
            status = "❌ Fuera de rango"
        
        datos_tabla.append({
            'Altura (m)': h,
            'Curva IA': ia_curve_name,
            'Caudal IA (m³/s)': q_ia,
            'Caudal Personal (m³/s)': q_personal,
            'Diferencia (m³/s)': diferencia,
            'Diferencia (%)': diferencia_porcentaje,
            'Estado': status
        })
    
    df_comparativa = pd.DataFrame(datos_tabla)
    
    # Formatear para mostrar
    df_display = df_comparativa.copy()
    df_display['Altura (m)'] = df_display['Altura (m)'].apply(lambda x: f"{x:.2f}")
    
    for col in ['Caudal IA (m³/s)', 'Caudal Personal (m³/s)', 'Diferencia (m³/s)']:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}" if x is not None else "-")
    
    df_display['Diferencia (%)'] = df_display['Diferencia (%)'].apply(
        lambda x: f"{x:+.1f}%" if x is not None else "-")
    
    st.dataframe(df_display, use_container_width=True)
    
    # =============================================
    # 3. ESTADÍSTICAS DE COMPARACIÓN
    # =============================================
    st.header("📊 ESTADÍSTICAS DE COMPARACIÓN")
    
    # Filtrar datos donde ambas curvas están definidas
    df_comparable = df_comparativa[df_comparativa['Estado'] == "✅ Comparable"].copy()
    
    if len(df_comparable) > 0:
        diferencias = df_comparable['Diferencia (m³/s)'].values
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            diferencia_promedio = np.mean(diferencias)
            color = "red" if abs(diferencia_promedio) > 1.0 else "green"
            st.metric(
                "Diferencia Promedio", 
                f"{diferencia_promedio:+.3f} m³/s",
                delta=f"{diferencia_promedio:+.3f} m³/s",
                delta_color="inverse"
            )
        
        with col2:
            diferencia_maxima = np.max(np.abs(diferencias))
            st.metric("Diferencia Máxima", f"{diferencia_maxima:.3f} m³/s")
        
        with col3:
            rmsd = np.sqrt(np.mean(diferencias**2))
            st.metric("Error Cuadrático Medio", f"{rmsd:.3f} m³/s")
        
        with col4:
            correlacion = np.corrcoef(df_comparable['Caudal IA (m³/s)'], 
                                    df_comparable['Caudal Personal (m³/s)'])[0,1]
            st.metric("Coeficiente Correlación", f"{correlacion:.3f}")
        
        # Gráfico de diferencias
        st.subheader("📉 ANÁLISIS DE DIFERENCIAS")
        
        fig_diff, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Gráfico 1: Diferencias por altura
        ax1.plot(df_comparable['Altura (m)'], diferencias, 
                'red', linewidth=3, marker='o', markersize=6,
                label='Diferencia (Personal - IA)')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
        ax1.fill_between(df_comparable['Altura (m)'], diferencias, 0, 
                       where=(diferencias >= 0), color='red', alpha=0.3, label='Personal > IA')
        ax1.fill_between(df_comparable['Altura (m)'], diferencias, 0, 
                       where=(diferencias < 0), color='blue', alpha=0.3, label='Personal < IA')
        
        ax1.set_xlabel('Nivel H (m)', fontweight='bold')
        ax1.set_ylabel('Diferencia de Caudal (m³/s)', fontweight='bold')
        ax1.set_title('DIFERENCIA POR ALTURA', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Histograma de diferencias
        ax2.hist(diferencias, bins=10, color='purple', alpha=0.7, edgecolor='black')
        ax2.axvline(x=diferencia_promedio, color='red', linestyle='--', linewidth=2, 
                   label=f'Promedio: {diferencia_promedio:+.3f} m³/s')
        ax2.set_xlabel('Diferencia de Caudal (m³/s)', fontweight='bold')
        ax2.set_ylabel('Frecuencia', fontweight='bold')
        ax2.set_title('DISTRIBUCIÓN DE DIFERENCIAS', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_diff)
        
        # Resumen de comparación
        st.subheader("🎯 RESUMEN DE LA COMPARACIÓN")
        
        if abs(diferencia_promedio) < 0.5 and rmsd < 1.0:
            st.success("**✅ EXCELENTE CONCORDANCIA** - Tu curva se ajusta muy bien a las curvas de referencia")
        elif abs(diferencia_promedio) < 1.0 and rmsd < 2.0:
            st.warning("**⚠️ CONCORDANCIA MODERADA** - Existen algunas diferencias pero dentro de rangos aceptables")
        else:
            st.error("**❌ DIFERENCIAS SIGNIFICATIVAS** - Tu curva presenta discrepancias importantes con las referencias")
            
    else:
        st.warning("No hay superposición en los rangos para comparar las curvas")
    
    # =============================================
    # 4. DESCARGAR DATOS
    # =============================================
    st.header("💾 DESCARGAR RESULTADOS")
    
    # Crear datos completos para descarga
    H_descarga = np.linspace(0.1, 5.0, 50)
    datos_descarga = []
    
    for h in H_descarga:
        # Calcular todas las curvas
        q_ia1 = curvas_ia['CURVA_IA_1']['funcion'](h) if 0.3 <= h <= 2.0 else None
        q_ia2 = curvas_ia['CURVA_IA_2']['funcion'](h) if 1.8 <= h <= 4.0 else None
        q_personal = funcion_personalizada(h) if h_min <= h <= h_max else None
        
        datos_descarga.append({
            'Altura_m': h,
            'Caudal_IA1_Potencial_m3s': q_ia1,
            'Caudal_IA2_Polinomica_m3s': q_ia2,
            f'Caudal_{nombre_curva}_m3s': q_personal
        })
    
    df_descarga = pd.DataFrame(datos_descarga)
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_descarga.to_csv(index=False, float_format='%.4f')
        st.download_button(
            label="📥 DESCARGAR DATOS COMPLETOS (CSV)",
            data=csv,
            file_name=f"comparacion_curvas_{nombre_curva}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Resumen ejecutivo
        st.download_button(
            label="📄 DESCARGAR RESUMEN EJECUTIVO",
            data=f"""
COMPARACIÓN DE CURVAS H-Q
========================

CURVA PERSONALIZADA: {nombre_curva}
Tipo: {tipo_curva}
Rango: {h_min:.2f} ≤ H ≤ {h_max:.2f} m

RESULTADOS DE COMPARACIÓN:
- Diferencia promedio: {diferencia_promedio:+.3f} m³/s
- Diferencia máxima: {diferencia_maxima:.3f} m³/s  
- Error cuadrático medio: {rmsd:.3f} m³/s
- Coeficiente correlación: {correlacion:.3f}

Generado el: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
            """,
            file_name=f"resumen_comparacion_{nombre_curva}.txt",
            mime="text/plain",
            use_container_width=True
        )

# =============================================
# INFORMACIÓN ADICIONAL
# =============================================
st.markdown("---")
st.header("💡 INFORMACIÓN SOBRE LAS CURVAS DE REFERENCIA")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔵 Curva Potencial (IA)")
    st.write("**Rango:** 0.3 ≤ H ≤ 2.0 m")
    st.latex("Q = 2.000 \\times H^{1.800}")
    st.write("Apropiada para niveles bajos a medios")

with col2:
    st.subheader("🔴 Curva Polinómica (IA)")  
    st.write("**Rango:** 1.8 ≤ H ≤ 4.0 m")
    st.latex("Q = 0.300H^2 + 1.200H + 0.100")
    st.write("Apropiada para niveles medios a altos")

st.markdown("---")
st.markdown("**🌊 COMPARADOR INDEPENDIENTE DE CURVAS H-Q** - *Por: Sistema de Análisis Hidráulico*")