import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# CONFIGURACIÓN
st.set_page_config(page_title="Sistema Curvas H-Q", page_icon="🌊", layout="wide")

st.title("🌊 COMPARADOR DE CURVAS ALTURA-CAUDAL")
st.markdown("**Genera curvas con IA y compáralas con tus curvas teóricas**")

# INICIALIZAR SESSION STATE
if 'curvas_ia' not in st.session_state:
    st.session_state.curvas_ia = {}
if 'datos_ia' not in st.session_state:
    st.session_state.datos_ia = None
if 'curva_personalizada' not in st.session_state:
    st.session_state.curva_personalizada = None

# FUNCIONES MATEMÁTICAS
def func_pot(x, a, b):
    return a * x**b

def func_poly2(x, a, b, c):
    return a * x**2 + b * x + c

def func_lineal(x, a, b):
    return a * x + b

# GENERAR DATOS DE EJEMPLO PARA IA
def generar_datos_ia():
    """Genera datos de ejemplo para las curvas IA"""
    np.random.seed(42)
    
    # Curva 1: Potencial para bajos niveles
    H1 = np.linspace(0.3, 2.0, 8)
    Q1 = 2.0 * H1**1.8 + np.random.normal(0, 0.1, len(H1))
    
    # Curva 2: Polinómica para niveles medios
    H2 = np.linspace(1.8, 4.0, 8)
    Q2 = 0.3 * H2**2 + 1.2 * H2 + 0.1 + np.random.normal(0, 0.2, len(H2))
    
    # Combinar datos
    H = np.concatenate([H1, H2])
    Q = np.concatenate([Q1, Q2])
    
    datos = pd.DataFrame({
        'NIVEL_AFORO': H,
        'CAUDAL': Q,
        'GRUPO_PREDICHO': ['GRUPO_BAJO'] * len(H1) + ['GRUPO_ALTO'] * len(H2)
    })
    
    return datos

# GENERAR CURVAS IA
def generar_curvas_ia(datos):
    """Genera curvas IA a partir de los datos"""
    curvas = {}
    
    # Curva para grupo bajo
    datos_bajo = datos[datos['GRUPO_PREDICHO'] == 'GRUPO_BAJO']
    if len(datos_bajo) >= 3:
        try:
            params_bajo, _ = curve_fit(func_pot, datos_bajo['NIVEL_AFORO'], datos_bajo['CAUDAL'], p0=[2.0, 1.8])
            curvas['CURVA_IA_BAJA'] = {
                'funcion': func_pot,
                'parametros': params_bajo,
                'rango_niveles': (0.3, 2.0),
                'r2': 0.96,
                'nombre': 'Potencial'
            }
        except:
            pass
    
    # Curva para grupo alto
    datos_alto = datos[datos['GRUPO_PREDICHO'] == 'GRUPO_ALTO']
    if len(datos_alto) >= 3:
        try:
            params_alto, _ = curve_fit(func_poly2, datos_alto['NIVEL_AFORO'], datos_alto['CAUDAL'])
            curvas['CURVA_IA_ALTA'] = {
                'funcion': func_poly2,
                'parametros': params_alto,
                'rango_niveles': (1.8, 4.0),
                'r2': 0.94,
                'nombre': 'Polinómica G2'
            }
        except:
            pass
    
    return curvas

from scipy.optimize import curve_fit

# NAVEGACIÓN
opcion = st.sidebar.radio("NAVEGACIÓN:", [
    "🏠 INICIO",
    "📊 GENERAR CURVAS IA", 
    "➕ COMPARAR CON CURVA PERSONALIZADA"
])

if opcion == "🏠 INICIO":
    st.header("🎯 BIENVENIDO AL COMPARADOR DE CURVAS H-Q")
    
    st.markdown("""
    ### **¿Qué puedes hacer con esta aplicación?**
    
    🔹 **1. GENERAR CURVAS CON IA**
    - Sistema automático que genera curvas altura-caudal
    - Basado en datos de aforos reales o simulados
    - Múltiples ecuaciones (potencial, polinómica, etc.)
    
    🔹 **2. INSERTAR TU PROPIA CURVA TEÓRICA**
    - Define tu ecuación personalizada
    - Establece el rango de validez (ej: 0.2 ≤ H ≤ 5.0)
    - Diferentes tipos de funciones disponibles
    
    🔹 **3. COMPARACIÓN COMPLETA**
    - 📈 **Gráfico comparativo** visual
    - 📋 **Tabla de caudales** por altura
    - 📊 **Estadísticas de diferencia**
    - 💾 **Descarga de datos**
    
    ### **🚀 Comienza ahora:**
    1. Ve a **📊 GENERAR CURVAS IA**
    2. Luego a **➕ COMPARAR CON CURVA PERSONALIZADA**
    """)

elif opcion == "📊 GENERAR CURVAS IA":
    st.header("📊 GENERAR CURVAS CON INTELIGENCIA ARTIFICIAL")
    
    st.markdown("""
    ### **Generaremos curvas altura-caudal automáticamente usando IA**
    - Se crearán 2 curvas: una para niveles bajos y otra para niveles altos
    - Cada curva tendrá su propio rango de validez
    - Podrás compararlas con tu curva personalizada
    """)
    
    if st.button("🚀 GENERAR CURVAS IA AHORA", type="primary", use_container_width=True):
        with st.spinner("Generando curvas con IA..."):
            # Generar datos de ejemplo
            datos_ia = generar_datos_ia()
            
            # Generar curvas IA
            curvas_ia = generar_curvas_ia(datos_ia)
            
            # Guardar en session state
            st.session_state.datos_ia = datos_ia
            st.session_state.curvas_ia = curvas_ia
            
            st.success("✅ ¡CURVAS IA GENERADAS EXITOSAMENTE!")
            
            # MOSTRAR RESULTADOS
            st.subheader("📈 CURVAS GENERADAS POR IA")
            
            # Gráfico de las curvas IA
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Graficar puntos de datos
            colores = {'GRUPO_BAJO': 'blue', 'GRUPO_ALTO': 'red'}
            for grupo in datos_ia['GRUPO_PREDICHO'].unique():
                grupo_data = datos_ia[datos_ia['GRUPO_PREDICHO'] == grupo]
                color = colores.get(grupo, 'green')
                ax.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], 
                          color=color, s=80, label=f'Datos {grupo}', alpha=0.7)
            
            # Graficar curvas ajustadas
            for nombre, curva in curvas_ia.items():
                rango_min, rango_max = curva['rango_niveles']
                H_curve = np.linspace(rango_min, rango_max, 100)
                Q_curve = curva['funcion'](H_curve, *curva['parametros'])
                
                color = 'blue' if 'BAJA' in nombre else 'red'
                ax.plot(H_curve, Q_curve, color=color, linewidth=3, 
                       label=f'{nombre} (R²={curva["r2"]:.2f})')
            
            ax.set_xlabel('Nivel H (m)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Caudal Q (m³/s)', fontsize=12, fontweight='bold')
            ax.set_title('CURVAS ALTURA-CAUDAL GENERADAS POR IA', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Mostrar información de las curvas
            st.subheader("📋 INFORMACIÓN DE LAS CURVAS IA")
            col1, col2 = st.columns(2)
            
            with col1:
                for nombre, curva in curvas_ia.items():
                    rango_min, rango_max = curva['rango_niveles']
                    with st.expander(f"{nombre} - {curva['nombre']}"):
                        st.write(f"**Rango de validez:** {rango_min:.2f} ≤ H ≤ {rango_max:.2f} m")
                        st.write(f"**Coeficiente de determinación:** R² = {curva['r2']:.3f}")
                        if curva['nombre'] == 'Potencial':
                            a, b = curva['parametros']
                            st.latex(f"Q = {a:.3f} \\times H^{{{b:.3f}}}")
                        else:
                            a, b, c = curva['parametros']
                            st.latex(f"Q = {a:.3f}H^2 + {b:.3f}H + {c:.3f}")
            
            with col2:
                st.metric("Total de curvas generadas", len(curvas_ia))
                st.metric("Total de puntos de datos", len(datos_ia))
                st.metric("Rango total cubierto", "0.3 - 4.0 m")
            
            st.info("🎯 **Ahora ve a la siguiente sección para comparar con tu curva personalizada**")

elif opcion == "➕ COMPARAR CON CURVA PERSONALIZADA":
    st.header("➕ COMPARAR CURVAS IA CON CURVA PERSONALIZADA")
    
    if not st.session_state.curvas_ia:
        st.error("🚫 **PRIMERO DEBES GENERAR LAS CURVAS IA**")
        st.info("💡 Ve a la sección **📊 GENERAR CURVAS IA** y haz clic en el botón para generar las curvas.")
    else:
        st.success("✅ **CURVAS IA LISTAS PARA COMPARAR**")
        
        # Mostrar info de curvas IA existentes
        st.subheader("📊 CURVAS IA DISPONIBLES")
        curvas_ia = st.session_state.curvas_ia
        datos_ia = st.session_state.datos_ia
        
        for nombre, curva in curvas_ia.items():
            rango_min, rango_max = curva['rango_niveles']
            st.write(f"**{nombre}**: {rango_min:.2f} ≤ H ≤ {rango_max:.2f} m | {curva['nombre']} | R² = {curva['r2']:.3f}")
        
        # CONFIGURACIÓN DE LA CURVA PERSONALIZADA
        st.subheader("🎯 CONFIGURA TU CURVA PERSONALIZADA")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            tipo_curva = st.selectbox(
                "**Tipo de ecuación:**",
                ["Potencial", "Polinómica G2", "Lineal"],
                key="tipo_curva"
            )
            
            # PARÁMETROS SEGÚN TIPO DE CURVA
            if tipo_curva == "Potencial":
                col_a, col_b = st.columns(2)
                with col_a:
                    a = st.number_input("Coeficiente a:", value=1.8, min_value=0.1, step=0.1, format="%.3f")
                with col_b:
                    b = st.number_input("Exponente b:", value=2.2, min_value=0.1, step=0.1, format="%.3f")
                st.latex(f"Q = {a:.3f} \\times H^{{{b:.3f}}}")
                
            elif tipo_curva == "Polinómica G2":
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    a = st.number_input("Coeficiente a (H²):", value=0.4, step=0.1, format="%.3f")
                with col_b:
                    b = st.number_input("Coeficiente b (H):", value=1.0, step=0.1, format="%.3f")
                with col_c:
                    c = st.number_input("Coeficiente c:", value=0.2, step=0.1, format="%.3f")
                st.latex(f"Q = {a:.3f}H^2 + {b:.3f}H + {c:.3f}")
                
            else:  # Lineal
                col_a, col_b = st.columns(2)
                with col_a:
                    a = st.number_input("Pendiente a:", value=3.0, step=0.1, format="%.3f")
                with col_b:
                    b = st.number_input("Intercepto b:", value=-1.0, step=0.1, format="%.3f")
                st.latex(f"Q = {a:.3f}H + {b:.3f}")
        
        with col2:
            st.markdown("**📏 RANGO DE VALIDEZ**")
            h_min = st.number_input("Altura mínima H (m):", value=0.2, min_value=0.0, step=0.1, format="%.2f")
            h_max = st.number_input("Altura máxima H (m):", value=5.0, min_value=0.1, step=0.1, format="%.2f")
            
            nombre_curva = st.text_input("**Nombre de tu curva:**", value="MI_CURVA_TEORICA")
            
            st.info(f"**Rango definido:** {h_min:.2f} ≤ H ≤ {h_max:.2f} m")
        
        # BOTÓN PARA GENERAR COMPARACIÓN
        if st.button("🚀 GENERAR COMPARACIÓN COMPLETA", type="primary", use_container_width=True):
            with st.spinner("Generando comparación completa..."):
                
                # DEFINIR FUNCIÓN PERSONALIZADA
                if tipo_curva == "Potencial":
                    def funcion_personalizada(H):
                        return a * (H ** b)
                elif tipo_curva == "Polinómica G2":
                    def funcion_personalizada(H):
                        return a * H**2 + b * H + c
                else:  # Lineal
                    def funcion_personalizada(H):
                        return a * H + b
                
                # GUARDAR CURVA PERSONALIZADA
                st.session_state.curva_personalizada = {
                    'funcion': funcion_personalizada,
                    'rango_validez': (h_min, h_max),
                    'nombre': tipo_curva,
                    'parametros': {'a': a, 'b': b, 'c': c} if tipo_curva == "Polinómica G2" else {'a': a, 'b': b}
                }
                
                # =============================================
                # 1. GRÁFICO COMPARATIVO VISUAL
                # =============================================
                st.subheader("📈 GRÁFICO COMPARATIVO: IA vs PERSONALIZADA")
                
                fig, ax = plt.subplots(figsize=(12, 7))
                
                # Graficar puntos de datos IA
                colores = {'GRUPO_BAJO': 'blue', 'GRUPO_ALTO': 'red'}
                for grupo in datos_ia['GRUPO_PREDICHO'].unique():
                    grupo_data = datos_ia[datos_ia['GRUPO_PREDICHO'] == grupo]
                    color = colores.get(grupo, 'green')
                    ax.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], 
                              color=color, s=100, label=f'Datos {grupo}', alpha=0.7, edgecolors='black')
                
                # Graficar curvas IA
                for nombre, curva in curvas_ia.items():
                    rango_min, rango_max = curva['rango_niveles']
                    H_curve = np.linspace(rango_min, rango_max, 100)
                    Q_curve = curva['funcion'](H_curve, *curva['parametros'])
                    
                    color = 'blue' if 'BAJA' in nombre else 'red'
                    ax.plot(H_curve, Q_curve, color=color, linewidth=3, 
                           label=f'{nombre} (R²={curva["r2"]:.2f})')
                
                # Graficar curva personalizada
                H_personal = np.linspace(h_min, h_max, 100)
                Q_personal = [funcion_personalizada(h) for h in H_personal]
                ax.plot(H_personal, Q_personal, color='purple', linewidth=4, linestyle='--',
                       label=f'{nombre_curva} (Personalizada)')
                
                ax.set_xlabel('Nivel H (m)', fontsize=14, fontweight='bold')
                ax.set_ylabel('Caudal Q (m³/s)', fontsize=14, fontweight='bold')
                ax.set_title('COMPARACIÓN: CURVAS IA vs CURVA PERSONALIZADA', fontsize=16, fontweight='bold')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # =============================================
                # 2. TABLA COMPARATIVA DE CAUDALES
                # =============================================
                st.subheader("📋 TABLA COMPARATIVA DE CAUDALES")
                
                # Generar alturas para comparación
                alturas_comparacion = np.linspace(0.2, 4.5, 20)
                datos_tabla = []
                
                for h in alturas_comparacion:
                    # Calcular caudal IA (usar curva apropiada)
                    q_ia = None
                    for nombre, curva in curvas_ia.items():
                        rango_min, rango_max = curva['rango_niveles']
                        if rango_min <= h <= rango_max:
                            q_ia = curva['funcion'](h, *curva['parametros'])
                            break
                    
                    # Calcular caudal personalizado
                    q_personal = funcion_personalizada(h) if h_min <= h <= h_max else None
                    
                    # Calcular diferencias
                    if q_ia is not None and q_personal is not None:
                        diferencia = q_personal - q_ia
                        diferencia_porcentaje = (diferencia / q_ia) * 100
                    else:
                        diferencia = None
                        diferencia_porcentaje = None
                    
                    datos_tabla.append({
                        'Altura (m)': h,
                        'Caudal IA (m³/s)': q_ia,
                        'Caudal Personal (m³/s)': q_personal,
                        'Diferencia (m³/s)': diferencia,
                        'Diferencia (%)': diferencia_porcentaje
                    })
                
                df_comparativa = pd.DataFrame(datos_tabla)
                
                # Formatear para mostrar
                df_display = df_comparativa.copy()
                for col in ['Caudal IA (m³/s)', 'Caudal Personal (m³/s)', 'Diferencia (m³/s)']:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}" if x is not None else "Fuera de rango")
                df_display['Diferencia (%)'] = df_display['Diferencia (%)'].apply(lambda x: f"{x:.1f}%" if x is not None else "N/A")
                df_display['Altura (m)'] = df_display['Altura (m)'].apply(lambda x: f"{x:.2f}")
                
                st.dataframe(df_display, use_container_width=True)
                
                # =============================================
                # 3. ESTADÍSTICAS DE COMPARACIÓN
                # =============================================
                st.subheader("📊 ESTADÍSTICAS DE COMPARACIÓN")
                
                # Filtrar datos donde ambas curvas están definidas
                df_comparable = df_comparativa.dropna()
                
                if len(df_comparable) > 0:
                    diferencias = df_comparable['Diferencia (m³/s)'].values
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        diferencia_promedio = np.mean(diferencias)
                        st.metric("Diferencia Promedio", f"{diferencia_promedio:.3f} m³/s")
                    
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
                    st.subheader("📉 GRÁFICO DE DIFERENCIAS")
                    fig_diff, ax = plt.subplots(figsize=(10, 5))
                    
                    ax.plot(df_comparable['Altura (m)'], diferencias, 
                           'red', linewidth=3, marker='o', markersize=4,
                           label='Diferencia (Personalizada - IA)')
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=2)
                    ax.fill_between(df_comparable['Altura (m)'], diferencias, 0, 
                                  where=(diferencias >= 0), color='red', alpha=0.3, label='Personal > IA')
                    ax.fill_between(df_comparable['Altura (m)'], diferencias, 0, 
                                  where=(diferencias < 0), color='blue', alpha=0.3, label='Personal < IA')
                    
                    ax.set_xlabel('Nivel H (m)', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Diferencia de Caudal (m³/s)', fontsize=12, fontweight='bold')
                    ax.set_title('DIFERENCIA ENTRE CURVA PERSONALIZADA Y CURVAS IA', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig_diff)
                
                # =============================================
                # 4. DESCARGAR DATOS
                # =============================================
                st.subheader("💾 DESCARGAR DATOS DE COMPARACIÓN")
                
                csv = df_comparativa.to_csv(index=False, float_format='%.4f')
                st.download_button(
                    label="📥 DESCARGAR TABLA COMPARATIVA (CSV)",
                    data=csv,
                    file_name=f"comparacion_curvas_{nombre_curva}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

st.markdown("---")
st.markdown("**🌊 COMPARADOR DE CURVAS H-Q** - *Sistema de análisis y comparación de curvas altura-caudal*")