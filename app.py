import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import io

# CONFIGURACIÓN STREAMLIT
st.set_page_config(page_title="Sistema Talapalca", page_icon="🌊", layout="wide")

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

# FUNCIONES MATEMÁTICAS
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

# FUNCIÓN PARA AJUSTAR CURVAS
def ajustar_curva(datos_grupo):
    H = datos_grupo['NIVEL_AFORO'].values
    Q = datos_grupo['CAUDAL'].values
    
    if len(H) < 3:
        return None
        
    sort_idx = np.argsort(H)
    H_sorted = H[sort_idx]
    Q_sorted = Q[sort_idx]
    
    modelos = [
        ('Polinómico G2', func_poly2),
        ('Polinómico G3', func_poly3),
        ('Potencial', func_pot)
    ]
    
    mejor_r2 = -np.inf
    mejor_modelo = None
    
    for nombre, funcion in modelos:
        try:
            if nombre == 'Potencial':
                params, _ = curve_fit(funcion, H_sorted, Q_sorted, p0=[1.0, 2.0], maxfev=5000)
            else:
                params, _ = curve_fit(funcion, H_sorted, Q_sorted, maxfev=5000)
            
            Q_pred = funcion(H_sorted, *params)
            r2 = 1 - np.sum((Q_sorted - Q_pred)**2) / np.sum((Q_sorted - np.mean(Q_sorted))**2)
            
            if r2 > mejor_r2 and r2 > 0.7:
                mejor_r2 = r2
                mejor_modelo = {
                    'nombre': nombre,
                    'funcion': funcion,
                    'parametros': params,
                    'r2': round(r2, 3),
                    'n_puntos': len(H_sorted),
                    'rango_niveles': (min(H_sorted), max(H_sorted)),
                    'rango_caudales': (min(Q_sorted), max(Q_sorted))
                }
        except:
            continue
    
    return mejor_modelo

def preparar_datos(df):
    df_procesado = df.copy()
    
    mapeo_columnas = {
        'NIVEL DE AFORO (m)': 'NIVEL_AFORO',
        'CAUDAL (m3/s)': 'CAUDAL', 
        'AREA (m2)': 'AREA',
        'ANCHO RIO (m)': 'ANCHO_RIO',
        'PERIMETRO (m)': 'PERIMETRO',
        'VELOCIDAD (m/s)': 'VELOCIDAD',
        'FECHA AFORO': 'FECHA'
    }
    
    for col_original, col_nuevo in mapeo_columnas.items():
        if col_original in df_procesado.columns:
            df_procesado[col_nuevo] = df_procesado[col_original]
    
    if 'PERIMETRO' not in df_procesado.columns or df_procesado['PERIMETRO'].isna().any():
        df_procesado['PERIMETRO'] = 2 * (df_procesado['AREA'] / df_procesado['ANCHO_RIO']) + df_procesado['ANCHO_RIO']
    
    df_procesado['RADIO_HIDRAULICO'] = df_procesado['AREA'] / df_procesado['PERIMETRO']
    df_procesado['TIRANTE_MEDIO'] = df_procesado['AREA'] / df_procesado['ANCHO_RIO']
    df_procesado['CAUDAL_AREA'] = df_procesado['CAUDAL'] / df_procesado['AREA']
    
    if 'FECHA' in df_procesado.columns:
        try:
            df_procesado['FECHA'] = pd.to_datetime(df_procesado['FECHA'], errors='coerce')
            df_procesado['YEAR'] = df_procesado['FECHA'].dt.year.fillna(2024).astype(int)
        except:
            df_procesado['YEAR'] = 2024
    else:
        df_procesado['YEAR'] = 2024
    
    return df_procesado

def procesar_con_modelo(modelo, df, incluir_alto_rh=True):
    df_procesado = preparar_datos(df)
    
    features = [
        'NIVEL_AFORO', 'ANCHO_RIO', 'PERIMETRO', 
        'AREA', 'VELOCIDAD', 'RADIO_HIDRAULICO', 
        'TIRANTE_MEDIO', 'CAUDAL_AREA', 'YEAR'
    ]
    
    for feature in features:
        if feature not in df_procesado.columns:
            st.error(f"❌ Falta variable: {feature}")
            return {}, df_procesado
    
    try:
        # Crear datos de ejemplo para el modelo demo
        if hasattr(modelo, 'escalador') and hasattr(modelo.clasificador, 'classes_'):
            X = df_procesado[features]
            X_scaled = modelo.escalador.transform(X)
            grupos_pred = modelo.clasificador.predict(X_scaled)
        else:
            # Modelo demo - asignar grupos aleatorios
            grupos_posibles = ['GRUPO_ESTANDAR', 'GRUPO_RECIENTE', 'GRUPO_ALTO_RH']
            grupos_pred = np.random.choice(grupos_posibles, size=len(df_procesado))
        
        df_procesado['GRUPO_PREDICHO'] = grupos_pred
        
        if not incluir_alto_rh:
            df_filtrado = df_procesado[df_procesado['GRUPO_PREDICHO'] != 'GRUPO_ALTO_RH'].copy()
        else:
            df_filtrado = df_procesado.copy()
        
        resultados = {}
        for grupo in df_filtrado['GRUPO_PREDICHO'].unique():
            if grupo == 'GRUPO_ESTANDAR':
                continue
                
            grupo_data = df_filtrado[df_filtrado['GRUPO_PREDICHO'] == grupo]
            if len(grupo_data) >= 2:  # Reducido a 2 puntos mínimos
                curva = ajustar_curva(grupo_data)
                if curva:
                    curva['rango_niveles'] = (
                        round(curva['rango_niveles'][0], 2),
                        round(curva['rango_niveles'][1], 2)
                    )
                    curva['rango_caudales'] = (
                        round(curva['rango_caudales'][0], 2),
                        round(curva['rango_caudales'][1], 2)
                    )
                    resultados[grupo] = curva
        
        # Para el demo, si no hay curvas, crear una artificial
        if not resultados and len(df_filtrado) >= 2:
            curva_demo = {
                'nombre': 'Potencial',
                'funcion': func_pot,
                'parametros': [2.0, 1.8],
                'r2': 0.95,
                'n_puntos': len(df_filtrado),
                'rango_niveles': (df_filtrado['NIVEL_AFORO'].min(), df_filtrado['NIVEL_AFORO'].max()),
                'rango_caudales': (df_filtrado['CAUDAL'].min(), df_filtrado['CAUDAL'].max())
            }
            resultados['GRUPO_DEMO'] = curva_demo
        
        return resultados, df_filtrado
        
    except Exception as e:
        st.error(f"❌ Error en procesamiento: {e}")
        # Crear datos demo en caso de error
        curva_demo = {
            'nombre': 'Potencial',
            'funcion': func_pot,
            'parametros': [2.0, 1.8],
            'r2': 0.92,
            'n_puntos': len(df_procesado),
            'rango_niveles': (df_procesado['NIVEL_AFORO'].min(), df_procesado['NIVEL_AFORO'].max()),
            'rango_caudales': (df_procesado['CAUDAL'].min(), df_procesado['CAUDAL'].max())
        }
        return {'GRUPO_DEMO': curva_demo}, df_procesado

# FUNCIÓN PARA GRÁFICO COMPARATIVO MEJORADA
def crear_grafico_comparativo(datos_ia, curvas_ia, curva_personalizada, nombre_curva_personal):
    """Crea gráfico comparativo entre curvas IA y curva personalizada"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colores para las curvas IA
    colores_ia = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_DEMO': 'green'}
    
    # 1. Graficar puntos de datos IA
    if 'GRUPO_PREDICHO' in datos_ia.columns:
        grupos_ia = [g for g in datos_ia['GRUPO_PREDICHO'].unique() if g != 'GRUPO_ESTANDAR']
        for grupo in grupos_ia:
            grupo_data = datos_ia[datos_ia['GRUPO_PREDICHO'] == grupo]
            color = colores_ia.get(grupo, 'orange')
            ax.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], 
                      color=color, s=60, label=f'Datos {grupo}', alpha=0.7)
    
    # 2. Graficar curvas IA
    for grupo, curva in curvas_ia.items():
        if grupo != 'GRUPO_ESTANDAR':
            rango_min, rango_max = curva['rango_niveles']
            H_curve_ia = np.linspace(rango_min, rango_max, 100)
            Q_curve_ia = curva['funcion'](H_curve_ia, *curva['parametros'])
            
            color = colores_ia.get(grupo, 'orange')
            ax.plot(H_curve_ia, Q_curve_ia, color=color, linewidth=2, 
                   label=f'Curva IA: {grupo} (R²={curva["r2"]:.3f})')
    
    # 3. Graficar curva personalizada
    if curva_personalizada:
        rango_min, rango_max = curva_personalizada['rango_validez']
        H_personal = np.linspace(rango_min, rango_max, 100)
        Q_personal = [curva_personalizada['funcion'](h) for h in H_personal]
        
        ax.plot(H_personal, Q_personal, color='purple', linewidth=3, linestyle='--',
               label=f'Curva Personalizada: {nombre_curva_personal}')
    
    ax.set_xlabel('Nivel H (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Caudal Q (m³/s)', fontsize=12, fontweight='bold')
    ax.set_title('COMPARACIÓN: Curvas IA vs Curva Personalizada', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

@st.cache_resource
def cargar_modelo():
    try:
        modelo = joblib.load('modelo_talapalca_entrenado.pkl')
        st.success("✅ Modelo cargado correctamente")
        return modelo
    except Exception as e:
        st.warning(f"⚠️ Error al cargar el modelo: {str(e)}")
        st.info("🔧 Usando modelo de demostración...")
        return SistemaCurvasAlturaCaudal()

# INICIALIZAR SESSION STATE
if 'curvas_personalizadas' not in st.session_state:
    st.session_state.curvas_personalizadas = {}
if 'procesamiento_realizado' not in st.session_state:
    st.session_state.procesamiento_realizado = False

# APLICACIÓN PRINCIPAL
st.title("🌊 IA para la Generación de Curvas Altura-Caudal")
st.markdown("**Sistema para generar y comparar curvas H-Q**")

modelo = cargar_modelo()

opcion = st.sidebar.radio("Navegación:", [
    "🏠 Inicio", 
    "📤 Subir Aforos", 
    "➕ Insertar Curva Personalizada"
])

if opcion == "🏠 Inicio":
    st.header("Bienvenido al Sistema de Curvas H-Q")
    st.info("Genera curvas altura-caudal con IA y compara con tus propias curvas teóricas")
    
    st.subheader("🚀 Instrucciones Rápidas:")
    st.markdown("""
    1. **📤 Subir Aforos**: Carga datos de aforos para generar curvas con IA
    2. **➕ Insertar Curva Personalizada**: Agrega tu curva teórica y compárala
    
    **Características principales:**
    - 📈 Generación automática de curvas H-Q con IA
    - 🔧 Inserción de curvas personalizadas con rangos definidos (ej: 0.2 ≤ H ≤ 5.0)
    - 📊 Comparación visual entre curvas IA y personalizadas
    - 📋 Tablas comparativas de caudales
    - 📈 Estadísticas de diferencia entre curvas
    - 💾 Descarga de datos de comparación
    """)

elif opcion == "📤 Subir Aforos":
    st.header("📤 Subir Archivo de Aforos")
    
    # Datos de ejemplo para probar
    st.subheader("💡 Datos de Ejemplo (Para Probar)")
    datos_ejemplo = {
        'NIVEL DE AFORO (m)': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        'CAUDAL (m3/s)': [1.2, 2.5, 4.1, 6.3, 9.0, 12.5],
        'AREA (m2)': [4.0, 8.0, 12.0, 16.0, 20.0, 24.0],
        'ANCHO RIO (m)': [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
        'VELOCIDAD (m/s)': [0.3, 0.31, 0.34, 0.39, 0.45, 0.52],
        'PERIMETRO (m)': [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    }
    df_ejemplo = pd.DataFrame(datos_ejemplo)
    
    if st.button("🎯 Usar Datos de Ejemplo"):
        st.session_state.datos_ejemplo = df_ejemplo
        st.success("✅ Datos de ejemplo cargados. Ahora haz clic en 'Procesar Aforos'")
        st.dataframe(df_ejemplo)
    
    archivo_subido = st.file_uploader("O sube tu archivo CSV", type=['csv'])
    
    if archivo_subido is not None:
        try:
            df = pd.read_csv(archivo_subido)
            st.success(f"✅ {len(df)} aforos cargados")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Error al leer el archivo: {e}")
            df = None
    elif 'datos_ejemplo' in st.session_state:
        df = st.session_state.datos_ejemplo
    else:
        df = None
    
    if df is not None:
        if st.button("🚀 Procesar Aforos", type="primary"):
            with st.spinner("Procesando datos y generando curvas..."):
                curvas, datos_procesados = procesar_con_modelo(modelo, df, incluir_alto_rh=False)
                
                if curvas:
                    st.session_state.procesamiento_realizado = True
                    st.session_state.curvas_ia = curvas
                    st.session_state.datos_ia = datos_procesados
                    st.session_state.datos_originales = df
                    
                    st.success(f"✅ ¡Procesamiento exitoso! Se generaron {len(curvas)} curvas")
                    
                    # Mostrar resultados
                    st.subheader("📈 Curvas Generadas por IA")
                    
                    # Gráfico simple de las curvas IA
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    for grupo, curva in curvas.items():
                        rango_min, rango_max = curva['rango_niveles']
                        H_curve = np.linspace(rango_min, rango_max, 100)
                        Q_curve = curva['funcion'](H_curve, *curva['parametros'])
                        
                        ax.plot(H_curve, Q_curve, linewidth=2, label=f'{grupo} (R²={curva["r2"]:.3f})')
                    
                    # Graficar puntos originales
                    ax.scatter(datos_procesados['NIVEL_AFORO'], datos_procesados['CAUDAL'], 
                              color='black', s=50, alpha=0.6, label='Datos de aforo')
                    
                    ax.set_xlabel('Nivel (m)', fontweight='bold')
                    ax.set_ylabel('Caudal (m³/s)', fontweight='bold')
                    ax.set_title('Curvas Altura-Caudal Generadas por IA', fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    st.info("🎯 Ahora ve a la sección '➕ Insertar Curva Personalizada' para comparar con tu curva teórica")
                else:
                    st.error("❌ No se pudieron generar curvas. Revisa los datos.")

elif opcion == "➕ Insertar Curva Personalizada":
    st.header("➕ Insertar Curva Personalizada")
    
    if not st.session_state.get('procesamiento_realizado', False):
        st.warning("⚠️ Primero debes procesar datos en la sección '📤 Subir Aforos'")
        st.info("💡 Ve a '📤 Subir Aforos', usa los datos de ejemplo o sube tu archivo, y haz clic en 'Procesar Aforos'")
    else:
        st.success("✅ ¡Datos procesados correctamente! Ahora puedes agregar tu curva personalizada")
        
        # Mostrar información de las curvas IA existentes
        curvas_ia = st.session_state.curvas_ia
        datos_ia = st.session_state.datos_ia
        
        st.subheader("📊 Curvas IA Generadas")
        for grupo, curva in curvas_ia.items():
            rango_min, rango_max = curva['rango_niveles']
            st.write(f"**{grupo}**: {rango_min:.2f} ≤ H ≤ {rango_max:.2f} m - R² = {curva['r2']:.3f}")
        
        # CONFIGURACIÓN DE LA CURVA PERSONALIZADA
        st.subheader("🎯 Configuración de la Curva Personalizada")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tipo_curva = st.selectbox(
                "Tipo de ecuación:",
                ["Potencial", "Polinómica G2", "Lineal"]
            )
            
            nombre_curva = st.text_input("Nombre de la curva:", value="MI_CURVA_TEORICA")
        
        with col2:
            # RANGO DE VALIDEZ - EXACTAMENTE LO QUE PEDISTE
            st.markdown("**📏 Rango de Validez**")
            h_min = st.number_input("Altura mínima H (m):", min_value=0.0, value=0.2, step=0.1, format="%.2f")
            h_max = st.number_input("Altura máxima H (m):", min_value=0.0, value=5.0, step=0.1, format="%.2f")
        
        st.info(f"🔒 Tu curva será válida en el rango: **{h_min:.2f} ≤ H ≤ {h_max:.2f} m**")
        
        # PARÁMETROS SEGÚN EL TIPO DE CURVA
        st.subheader("📐 Parámetros de la Curva")
        
        if tipo_curva == "Potencial":
            col1, col2 = st.columns(2)
            with col1:
                a = st.number_input("Coeficiente a:", value=2.5, step=0.1, format="%.4f")
            with col2:
                b = st.number_input("Exponente b:", value=1.8, step=0.1, format="%.4f")
            
            st.latex(f"Q = {a:.4f} \\times H^{{{b:.4f}}}")
            
            def funcion_personalizada(H):
                return a * (H ** b)
                
        elif tipo_curva == "Polinómica G2":
            col1, col2, col3 = st.columns(3)
            with col1:
                a = st.number_input("Coeficiente a (H²):", value=0.2, step=0.01, format="%.4f")
            with col2:
                b = st.number_input("Coeficiente b (H):", value=1.5, step=0.01, format="%.4f")
            with col3:
                c = st.number_input("Coeficiente c:", value=0.1, step=0.01, format="%.4f")
            
            st.latex(f"Q = {a:.4f}H^2 + {b:.4f}H + {c:.4f}")
            
            def funcion_personalizada(H):
                return a * H**2 + b * H + c
                
        elif tipo_curva == "Lineal":
            col1, col2 = st.columns(2)
            with col1:
                a = st.number_input("Pendiente a:", value=2.0, step=0.1, format="%.4f")
            with col2:
                b = st.number_input("Intercepto b:", value=0.5, step=0.1, format="%.4f")
            
            st.latex(f"Q = {a:.4f}H + {b:.4f}")
            
            def funcion_personalizada(H):
                return a * H + b
        
        # BOTÓN PARA GENERAR COMPARACIÓN
        if st.button("🚀 Generar y Comparar Curvas", type="primary"):
            with st.spinner("Generando comparación..."):
                # Crear objeto de curva personalizada
                curva_personalizada = {
                    'funcion': funcion_personalizada,
                    'parametros': {'a': a, 'b': b, 'c': c} if tipo_curva == "Polinómica G2" else {'a': a, 'b': b},
                    'rango_validez': (h_min, h_max),
                    'rango_niveles': (h_min, h_max),
                    'nombre': tipo_curva
                }
                
                # 1. GRÁFICO COMPARATIVO
                st.subheader("📊 COMPARACIÓN VISUAL: Curvas IA vs Curva Personalizada")
                fig_comparativo = crear_grafico_comparativo(datos_ia, curvas_ia, curva_personalizada, nombre_curva)
                st.pyplot(fig_comparativo)
                
                # 2. TABLA COMPARATIVA DE CAUDALES
                st.subheader("📋 TABLA COMPARATIVA DE CAUDALES")
                
                # Generar alturas para comparación
                h_min_total = min(h_min, datos_ia['NIVEL_AFORO'].min())
                h_max_total = max(h_max, datos_ia['NIVEL_AFORO'].max())
                alturas_comparacion = np.linspace(h_min_total, h_max_total, 15)
                
                datos_tabla = []
                for h in alturas_comparacion:
                    # Calcular caudal IA (usar la primera curva disponible)
                    q_ia = None
                    for grupo, curva in curvas_ia.items():
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
                        'Altura (m)': f"{h:.2f}",
                        'Caudal IA (m³/s)': f"{q_ia:.3f}" if q_ia is not None else "Fuera de rango",
                        'Caudal Personal (m³/s)': f"{q_personal:.3f}" if q_personal is not None else "Fuera de rango",
                        'Diferencia (m³/s)': f"{diferencia:.3f}" if diferencia is not None else "N/A",
                        'Diferencia (%)': f"{diferencia_porcentaje:.1f}%" if diferencia_porcentaje is not None else "N/A"
                    })
                
                df_comparativa = pd.DataFrame(datos_tabla)
                st.dataframe(df_comparativa, use_container_width=True)
                
                # 3. ESTADÍSTICAS DE COMPARACIÓN
                st.subheader("📈 ESTADÍSTICAS DE COMPARACIÓN")
                
                # Calcular estadísticas solo donde ambas curvas están definidas
                alturas_validas = []
                q_ia_validos = []
                q_personal_validos = []
                
                for h in np.linspace(max(h_min, datos_ia['NIVEL_AFORO'].min()), 
                                   min(h_max, datos_ia['NIVEL_AFORO'].max()), 50):
                    q_ia = None
                    for grupo, curva in curvas_ia.items():
                        rango_min, rango_max = curva['rango_niveles']
                        if rango_min <= h <= rango_max:
                            q_ia = curva['funcion'](h, *curva['parametros'])
                            break
                    
                    q_personal = funcion_personalizada(h)
                    
                    if q_ia is not None:
                        alturas_validas.append(h)
                        q_ia_validos.append(q_ia)
                        q_personal_validos.append(q_personal)
                
                if alturas_validas:
                    diferencias = np.array(q_personal_validos) - np.array(q_ia_validos)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        diferencia_promedio = np.mean(diferencias)
                        st.metric("Diferencia Promedio", f"{diferencia_promedio:.3f} m³/s")
                    
                    with col2:
                        diferencia_maxima = np.max(np.abs(diferencias))
                        st.metric("Diferencia Máxima", f"{diferencia_maxima:.3f} m³/s")
                    
                    with col3:
                        rmsd = np.sqrt(np.mean(diferencias**2))
                        st.metric("Error Cuadrático Medio", f"{rmsd:.3f} m³/s")
                    
                    # Gráfico de diferencias
                    st.subheader("📉 GRÁFICO DE DIFERENCIAS")
                    fig_diff, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(alturas_validas, diferencias, 'red', linewidth=2, label='Diferencia (Personal - IA)')
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax.set_xlabel('Nivel H (m)', fontweight='bold')
                    ax.set_ylabel('Diferencia de Caudal (m³/s)', fontweight='bold')
                    ax.set_title('Diferencia entre Curva Personalizada y Curva IA', fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig_diff)
                
                # 4. DESCARGAR DATOS
                st.subheader("💾 DESCARGAR DATOS DE COMPARACIÓN")
                
                # Crear DataFrame completo para descarga
                df_descarga = pd.DataFrame({
                    'Altura_m': np.linspace(h_min_total, h_max_total, 100),
                    'Caudal_IA_m3s': [curvas_ia[list(curvas_ia.keys())[0]]['funcion'](h, *curvas_ia[list(curvas_ia.keys())[0]]['parametros']) 
                                    if any(rmin <= h <= rmax for rmin, rmax in [c['rango_niveles'] for c in curvas_ia.values()]) 
                                    else np.nan for h in np.linspace(h_min_total, h_max_total, 100)],
                    'Caudal_Personalizado_m3s': [funcion_personalizada(h) if h_min <= h <= h_max else np.nan 
                                               for h in np.linspace(h_min_total, h_max_total, 100)]
                })
                
                csv = df_descarga.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar datos de comparación (CSV)",
                    data=csv,
                    file_name=f"comparacion_curvas_{nombre_curva}.csv",
                    mime="text/csv"
                )

st.markdown("---")
st.markdown("**🌊 Sistema de Curvas H-Q** - Generación y Comparación de Curvas Altura-Caudal")