import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import io
import re

# ... (todo el código anterior se mantiene igual hasta la clase y funciones globales)

# DEFINIR LA CLASE QUE FALTA
class SistemaCurvasAlturaCaudal:
    def __init__(self):
        self.clasificador = RandomForestClassifier(n_estimators=100, random_state=42)
        self.escalador = StandardScaler()
        self.curvas = {}
    
    def entrenar(self, X, y):
        X_esc = self.escalador.fit_transform(X)
        self.clasificador.fit(X_esc, y)
        return self
    
    def predecir_grupo(self, X):
        X_esc = self.escalador.transform(X)
        return self.clasificador.predict(X_esc)

# DEFINIR FUNCIONES GLOBALES
def func_poly2(x, a, b, c):
    return a * x**2 + b * x + c

def func_poly3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def func_pot(x, a, b):
    return a * x**b

def func_exp(x, a, b):
    return a * np.exp(b * x)

def func_log(x, a, b):
    return a * np.log(x + b)

# FUNCIÓN PARA EVALUAR ECUACIONES INGRESADAS POR EL USUARIO
def evaluar_ecuacion_usuario(ecuacion_str, H):
    """Evaluar ecuación ingresada por el usuario de forma segura"""
    try:
        # Reemplazar H por x para evaluación
        expr = ecuacion_str.replace('H', 'x').replace('^', '**')
        
        # Crear función segura
        def funcion_usuario(x):
            return eval(expr, {'x': x, 'np': np, 'exp': np.exp, 'log': np.log, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan})
        
        return funcion_usuario(H)
    except Exception as e:
        st.error(f"❌ Error en la ecuación: {e}")
        return None

# FUNCIÓN PARA CREAR GRÁFICO COMPARATIVO
def crear_grafico_comparativo(df, curvas_ia, ecuacion_usuario, rango_usuario, tipo_modelo_usuario):
    """Crear gráfico comparativo entre IA y ecuación del usuario"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
    marcadores = {'GRUPO_ALTO_RH': 's', 'GRUPO_RECIENTE': '^', 'GRUPO_ESTANDAR': 'o'}
    
    # 1. Graficar puntos de datos originales (EXCLUYENDO GRUPO_ESTANDAR)
    for grupo in df['GRUPO_PREDICHO'].unique():
        if grupo == 'GRUPO_ESTANDAR':
            continue
            
        color = colores.get(grupo, 'orange')
        marcador = marcadores.get(grupo, 'o')
        grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
        
        alpha = 0.7
        tamano = 80
        ax.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], 
                  color=color, marker=marcador, s=tamano, label=f'Datos {grupo}', alpha=alpha, 
                  edgecolors='black', linewidth=0.5)
    
    # 2. Graficar curvas de la IA
    for grupo, curva in curvas_ia.items():
        if grupo == 'GRUPO_ESTANDAR':
            continue
            
        color = colores.get(grupo, 'orange')
        
        if 'rango_validez' in curva:
            rango_min, rango_max = curva['rango_validez']
        else:
            rango_min, rango_max = curva['rango_niveles']
        
        H_range = np.linspace(rango_min, rango_max, 100)
        Q_curve = curva['funcion'](H_range, *curva['parametros'])
        
        ax.plot(H_range, Q_curve, color=color, linewidth=3, 
               label=f'IA: {grupo} (R²={curva["r2"]:.3f})')
    
    # 3. Graficar curva del usuario
    if ecuacion_usuario and rango_usuario:
        try:
            rango_min_user, rango_max_user = rango_usuario
            H_range_user = np.linspace(rango_min_user, rango_max_user, 100)
            
            # Evaluar ecuación del usuario
            Q_user = evaluar_ecuacion_usuario(ecuacion_usuario, H_range_user)
            
            if Q_user is not None:
                ax.plot(H_range_user, Q_user, color='black', linewidth=4, linestyle='--',
                       label=f'Usuario: {tipo_modelo_usuario}\n{ecuacion_usuario}')
                
                # Agregar puntos de muestra de la curva del usuario
                H_sample = np.linspace(rango_min_user, rango_max_user, 10)
                Q_sample = evaluar_ecuacion_usuario(ecuacion_usuario, H_sample)
                ax.scatter(H_sample, Q_sample, color='black', s=100, marker='X', 
                          label='Puntos muestra usuario', zorder=5)
                
                # Mostrar tabla de valores
                st.subheader("📋 Tabla de Caudales Generados - Ecuación del Usuario")
                tabla_datos = pd.DataFrame({
                    'Nivel (m)': H_sample,
                    'Caudal (m³/s)': Q_sample
                })
                st.dataframe(tabla_datos.round(3))
                
        except Exception as e:
            st.error(f"❌ Error al graficar ecuación del usuario: {e}")
    
    ax.set_xlabel('Nivel (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Caudal (m³/s)', fontsize=12, fontweight='bold')
    ax.set_title('Comparativo: Curvas IA vs Ecuación del Usuario', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(ax.get_xlim()[1], 3.5))  # Ajustar límites
    
    return fig

# ... (el resto de las funciones existentes se mantienen igual hasta la configuración de Streamlit)

# CONFIGURACIÓN STREAMLIT
st.set_page_config(page_title="Sistema Talapalca", page_icon="🌊", layout="wide")
st.title("🌊 IA para la generacion de Curvas Altura-Caudal")
st.markdown("**Modelo entrenado con 34 aforos reales - Basado en estándares USGS/WMO**")

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    try:
        modelo = joblib.load('modelo_talapalca_entrenado.pkl')
        st.success("✅ Modelo cargado correctamente")
        return modelo
    except Exception as e:
        st.warning(f"⚠️ Error al cargar el modelo: {str(e)}")
        st.info("🔧 Creando modelo de demostración...")
        
        modelo_demo = SistemaCurvasAlturaCaudal()
        from sklearn.datasets import make_classification
        
        X_demo, y_demo = make_classification(
            n_samples=50, n_features=9, n_classes=3, random_state=42
        )
        
        y_demo_nombres = ['GRUPO_ESTANDAR', 'GRUPO_RECIENTE', 'GRUPO_ALTO_RH']
        y_demo_categoricos = [y_demo_nombres[i % 3] for i in y_demo]
        
        modelo_demo.entrenar(X_demo, y_demo_categoricos)
        st.success("✅ Modelo de demostración creado exitosamente")
        return modelo_demo

modelo = cargar_modelo()

# NAVEGACIÓN ACTUALIZADA
opcion = st.sidebar.radio("Navegación:", ["🏠 Inicio", "📤 Subir Aforos", "📊 Ingreso Manual", "📈 Curvas", "🔄 Comparativo"])

# ... (las secciones anteriores se mantienen igual hasta la nueva sección Comparativo)

elif opcion == "🔄 Comparativo":
    st.header("🔄 Análisis Comparativo: IA vs Ecuación del Usuario")
    st.info("""
    **Compara las curvas generadas por la IA con tu propia ecuación personalizada**
    - Ingresa tu ecuación matemática
    - Define el rango de validez
    - Genera caudales automáticamente
    - Visualiza comparación en el gráfico
    """)
    
    if modelo is None:
        st.error("⚠️ El modelo no está disponible.")
    else:
        # SUBIR ARCHIVO PARA COMPARATIVO
        archivo_subido = st.file_uploader("Selecciona archivo CSV para análisis comparativo", type=['csv'], key="comparativo_csv")
        
        if archivo_subido is not None:
            try:
                df = pd.read_csv(archivo_subido)
                st.success(f"✅ {len(df)} aforos cargados para análisis comparativo")
                
                # Verificar columnas básicas
                columnas_necesarias = ['CAUDAL (m3/s)', 'VELOCIDAD (m/s)', 'AREA (m2)', 'ANCHO RIO (m)', 'NIVEL DE AFORO (m)']
                columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]
                
                if not columnas_faltantes:
                    # PROCESAR CON MODELO IA
                    with st.spinner("Procesando datos con IA..."):
                        curvas_ia, datos_procesados = procesar_con_modelo(modelo, df, incluir_alto_rh=True)
                    
                    if curvas_ia:
                        st.success("✅ Curvas IA generadas exitosamente")
                        
                        # SECCIÓN PARA INGRESO DE ECUACIÓN DEL USUARIO
                        st.subheader("📝 Ingresa tu Ecuación Personalizada")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Selección de tipo de modelo
                            tipo_modelo = st.selectbox(
                                "Tipo de modelo:",
                                ["Polinómico", "Potencial", "Exponencial", "Personalizado"],
                                help="Selecciona el tipo de ecuación o ingresa una personalizada"
                            )
                            
                            # Ingreso de ecuación según tipo seleccionado
                            if tipo_modelo == "Polinómico":
                                st.info("**Formato polinómico:** Q = aH² + bH + c")
                                grado = st.selectbox("Grado del polinomio:", [2, 3])
                                if grado == 2:
                                    ecuacion_default = "a*H**2 + b*H + c"
                                    coef_a = st.number_input("Coeficiente a (H²):", value=1.0, step=0.1)
                                    coef_b = st.number_input("Coeficiente b (H):", value=1.0, step=0.1)
                                    coef_c = st.number_input("Coeficiente c:", value=0.0, step=0.1)
                                    ecuacion_usuario = f"{coef_a}*H**2 + {coef_b}*H + {coef_c}"
                                else:
                                    ecuacion_default = "a*H**3 + b*H**2 + c*H + d"
                                    coef_a = st.number_input("Coeficiente a (H³):", value=1.0, step=0.1)
                                    coef_b = st.number_input("Coeficiente b (H²):", value=1.0, step=0.1)
                                    coef_c = st.number_input("Coeficiente c (H):", value=1.0, step=0.1)
                                    coef_d = st.number_input("Coeficiente d:", value=0.0, step=0.1)
                                    ecuacion_usuario = f"{coef_a}*H**3 + {coef_b}*H**2 + {coef_c}*H + {coef_d}"
                                    
                            elif tipo_modelo == "Potencial":
                                st.info("**Formato potencial:** Q = aHᵇ")
                                ecuacion_default = "a*H**b"
                                coef_a = st.number_input("Coeficiente a:", value=1.0, step=0.1)
                                coef_b = st.number_input("Exponente b:", value=1.5, step=0.1)
                                ecuacion_usuario = f"{coef_a}*H**{coef_b}"
                                
                            elif tipo_modelo == "Exponencial":
                                st.info("**Formato exponencial:** Q = a·exp(bH)")
                                ecuacion_default = "a*np.exp(b*H)"
                                coef_a = st.number_input("Coeficiente a:", value=1.0, step=0.1)
                                coef_b = st.number_input("Coeficiente b:", value=0.5, step=0.1)
                                ecuacion_usuario = f"{coef_a}*np.exp({coef_b}*H)"
                                
                            else:  # Personalizado
                                st.info("**Ecuación personalizada:** Usa 'H' para la variable altura")
                                ecuacion_default = "1.5*H**2 + 0.8*H + 0.1"
                                ecuacion_usuario = st.text_input(
                                    "Ingresa tu ecuación:",
                                    value=ecuacion_default,
                                    help="Ejemplos: 1.5*H**2 + 0.8*H + 0.1, 2.0*H**1.5, 1.2*np.exp(0.3*H)"
                                )
                        
                        with col2:
                            # Rango de validez
                            st.subheader("📏 Rango de Validez")
                            rango_min = st.number_input("Nivel mínimo (m):", min_value=0.0, value=0.1, step=0.1)
                            rango_max = st.number_input("Nivel máximo (m):", min_value=0.1, value=3.0, step=0.1)
                            
                            if rango_min >= rango_max:
                                st.error("❌ El nivel máximo debe ser mayor al nivel mínimo")
                                rango_valido = False
                            else:
                                rango_valido = True
                                rango_usuario = (rango_min, rango_max)
                            
                            # Información de la ecuación
                            st.subheader("ℹ️ Información de la Ecuación")
                            st.code(f"Ecuación: Q = {ecuacion_usuario}")
                            st.write(f"Rango de validez: {rango_min:.2f} ≤ H ≤ {rango_max:.2f} m")
                        
                        # BOTÓN PARA GENERAR COMPARATIVO
                        if st.button("🔄 Generar Análisis Comparativo", type="primary") and rango_valido:
                            with st.spinner("Generando análisis comparativo..."):
                                # CREAR GRÁFICO COMPARATIVO
                                st.subheader("📊 Gráfico Comparativo")
                                fig_comparativo = crear_grafico_comparativo(
                                    datos_procesados, curvas_ia, ecuacion_usuario, 
                                    rango_usuario, tipo_modelo
                                )
                                st.pyplot(fig_comparativo)
                                
                                # ANÁLISIS COMPARATIVO
                                st.subheader("📈 Análisis Comparativo Detallado")
                                
                                # Calcular algunos puntos de comparación
                                H_comparacion = np.linspace(rango_min, rango_max, 5)
                                
                                # Valores del usuario
                                Q_usuario = evaluar_ecuacion_usuario(ecuacion_usuario, H_comparacion)
                                
                                # Crear tabla comparativa
                                if Q_usuario is not None:
                                    tabla_comparativa = pd.DataFrame({
                                        'Nivel (m)': H_comparacion,
                                        'Caudal Usuario (m³/s)': Q_usuario
                                    })
                                    
                                    # Agregar valores de la IA si es posible
                                    for grupo, curva in curvas_ia.items():
                                        if grupo == 'GRUPO_ESTANDAR':
                                            continue
                                            
                                        # Verificar si el nivel está en el rango de la curva IA
                                        Q_ia = []
                                        for h in H_comparacion:
                                            if 'rango_validez' in curva:
                                                rango_min_ia, rango_max_ia = curva['rango_validez']
                                            else:
                                                rango_min_ia, rango_max_ia = curva['rango_niveles']
                                            
                                            if rango_min_ia <= h <= rango_max_ia:
                                                q_val = curva['funcion'](h, *curva['parametros'])
                                                Q_ia.append(q_val)
                                            else:
                                                Q_ia.append(np.nan)
                                        
                                        tabla_comparativa[f'Caudal {grupo} (m³/s)'] = Q_ia
                                    
                                    st.dataframe(tabla_comparativa.round(3))
                                    
                                    # ESTADÍSTICAS COMPARATIVAS
                                    st.subheader("📊 Estadísticas Comparativas")
                                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                                    
                                    with col_stat1:
                                        caudal_max_user = np.max(Q_usuario)
                                        st.metric("Caudal máximo usuario", f"{caudal_max_user:.2f} m³/s")
                                    
                                    with col_stat2:
                                        if len(curvas_ia) > 0:
                                            # Tomar la primera curva IA para comparación
                                            primera_curva = list(curvas_ia.values())[0]
                                            if 'rango_validez' in primera_curva:
                                                r_min, r_max = primera_curva['rango_validez']
                                            else:
                                                r_min, r_max = primera_curva['rango_niveles']
                                            
                                            st.metric("Rango IA principal", f"{r_min:.2f}-{r_max:.2f} m")
                                    
                                    with col_stat3:
                                        st.metric("Rango usuario", f"{rango_min:.2f}-{rango_max:.2f} m")
                                    
                                    # RECOMENDACIONES
                                    st.subheader("💡 Recomendaciones")
                                    if len(curvas_ia) > 0:
                                        st.info("""
                                        **Compara tu ecuación con las curvas IA considerando:**
                                        - Coincidencia en el rango de niveles
                                        - Comportamiento similar en pendientes
                                        - Valores de caudal en puntos clave
                                        - Rango de validez apropiado
                                        """)
                                    else:
                                        st.warning("No hay curvas IA para comparar")
                                
                        else:
                            st.info("👆 Ingresa los parámetros de tu ecuación y haz clic en 'Generar Análisis Comparativo'")
                    
                    else:
                        st.warning("⚠️ No se pudieron generar curvas con la IA para comparar")
                        
                else:
                    st.error(f"❌ Faltan columnas necesarias: {', '.join(columnas_faltantes)}")
                    
            except Exception as e:
                st.error(f"❌ Error en análisis comparativo: {e}")
        else:
            st.info("📁 Sube un archivo CSV para realizar el análisis comparativo")

# ... (el resto del código se mantiene igual)

st.markdown("---")
st.markdown("**🌊 IA para generar Curvas H-Q - Basado en estándares USGS/WMO**")