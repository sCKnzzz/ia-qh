import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import io

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

# FUNCIÓN PARA PREPARAR DATOS
def preparar_datos(df):
    df_procesado = df.copy()
    
    # Mapear nombres de columnas
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
    
    # Estimar perímetro si falta
    if 'PERIMETRO' not in df_procesado.columns or df_procesado['PERIMETRO'].isna().any():
        tirante = df_procesado['AREA'] / df_procesado['ANCHO_RIO']
        df_procesado['PERIMETRO'] = 2 * tirante + df_procesado['ANCHO_RIO']
    
    # Calcular variables
    df_procesado['RADIO_HIDRAULICO'] = df_procesado['AREA'] / df_procesado['PERIMETRO']
    df_procesado['TIRANTE_MEDIO'] = df_procesado['AREA'] / df_procesado['ANCHO_RIO']
    df_procesado['CAUDAL_AREA'] = df_procesado['CAUDAL'] / df_procesado['AREA']
    
    # Año
    if 'FECHA' in df_procesado.columns:
        try:
            df_procesado['FECHA'] = pd.to_datetime(df_procesado['FECHA'], errors='coerce')
            df_procesado['YEAR'] = df_procesado['FECHA'].dt.year.fillna(2024).astype(int)
        except:
            df_procesado['YEAR'] = 2024
    else:
        df_procesado['YEAR'] = 2024
    
    return df_procesado

# FUNCIÓN PARA PROCESAR CON MODELO
def procesar_con_modelo(modelo, df, incluir_alto_rh=True):
    """Procesar datos con el modelo"""
    
    df_procesado = preparar_datos(df)
    
    features = [
        'NIVEL_AFORO', 'ANCHO_RIO', 'PERIMETRO', 
        'AREA', 'VELOCIDAD', 'RADIO_HIDRAULICO', 
        'TIRANTE_MEDIO', 'CAUDAL_AREA', 'YEAR'
    ]
    
    # Verificar features
    for feature in features:
        if feature not in df_procesado.columns:
            st.error(f"❌ Falta variable: {feature}")
            return {}, df_procesado
    
    try:
        X = df_procesado[features]
        X_scaled = modelo.escalador.transform(X)
        grupos_pred = modelo.clasificador.predict(X_scaled)
        df_procesado['GRUPO_PREDICHO'] = grupos_pred
        
        # Filtrar si no incluir GRUPO_ALTO_RH
        if not incluir_alto_rh:
            df_filtrado = df_procesado[df_procesado['GRUPO_PREDICHO'] != 'GRUPO_ALTO_RH'].copy()
        else:
            df_filtrado = df_procesado.copy()
        
        # Generar curvas
        resultados = {}
        for grupo in df_filtrado['GRUPO_PREDICHO'].unique():
            grupo_data = df_filtrado[df_filtrado['GRUPO_PREDICHO'] == grupo]
            if len(grupo_data) >= 3:
                curva = ajustar_curva(grupo_data)
                if curva:
                    resultados[grupo] = curva
        
        return resultados, df_filtrado
        
    except Exception as e:
        st.error(f"❌ Error en procesamiento: {e}")
        return {}, df_procesado

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
                    'r2': r2,
                    'n_puntos': len(H_sorted),
                    'rango_niveles': (min(H_sorted), max(H_sorted)),
                    'rango_caudales': (min(Q_sorted), max(Q_sorted))
                }
        except:
            continue
    
    return mejor_modelo

# FUNCIONES PARA GRÁFICOS
def crear_grafico_principal(df, curvas, titulo):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
    
    for grupo, curva in curvas.items():
        color = colores.get(grupo, 'orange')
        grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
        
        # Puntos
        ax.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], 
                  color=color, s=60, label=grupo, alpha=0.8)
        
        # Curva
        H_range = np.linspace(curva['rango_niveles'][0]*0.9, curva['rango_niveles'][1]*1.1, 100)
        Q_curve = curva['funcion'](H_range, *curva['parametros'])
        ax.plot(H_range, Q_curve, color=color, linewidth=2, 
               label=f"{grupo} (R²={curva['r2']:.3f})")
    
    ax.set_xlabel('Nivel (m)')
    ax.set_ylabel('Caudal (m³/s)')
    ax.set_title(titulo)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def crear_graficos_complementarios(df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Altura vs Velocidad
    axes[0,0].scatter(df['NIVEL_AFORO'], df['VELOCIDAD'], color='green', s=40)
    axes[0,0].set_xlabel('Nivel (m)')
    axes[0,0].set_ylabel('Velocidad (m/s)')
    axes[0,0].set_title('Altura vs Velocidad')
    axes[0,0].grid(True, alpha=0.3)
    
    # Altura vs Área
    axes[0,1].scatter(df['NIVEL_AFORO'], df['AREA'], color='red', s=40)
    axes[0,1].set_xlabel('Nivel (m)')
    axes[0,1].set_ylabel('Área (m²)')
    axes[0,1].set_title('Altura vs Área')
    axes[0,1].grid(True, alpha=0.3)
    
    # Altura vs Radio Hidráulico
    axes[1,0].scatter(df['NIVEL_AFORO'], df['RADIO_HIDRAULICO'], color='purple', s=40)
    axes[1,0].set_xlabel('Nivel (m)')
    axes[1,0].set_ylabel('Radio Hidráulico (m)')
    axes[1,0].set_title('Altura vs Radio Hidráulico')
    axes[1,0].grid(True, alpha=0.3)
    
    # Caudal vs Velocidad
    axes[1,1].scatter(df['CAUDAL'], df['VELOCIDAD'], color='orange', s=40)
    axes[1,1].set_xlabel('Caudal (m³/s)')
    axes[1,1].set_ylabel('Velocidad (m/s)')
    axes[1,1].set_title('Caudal vs Velocidad')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# NUEVA FUNCIÓN PARA GRÁFICOS DE ANÁLISIS HIDRÁULICO
def crear_graficos_analisis_hidraulico(df):
    """Crear gráficos específicos para análisis hidráulico"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Relación Altura vs Caudal (principal)
    axes[0,0].scatter(df['NIVEL_AFORO'], df['CAUDAL'], c='blue', alpha=0.7, s=50)
    axes[0,0].set_xlabel('Nivel de Aforo (m)')
    axes[0,0].set_ylabel('Caudal (m³/s)')
    axes[0,0].set_title('Relación Altura-Caudal')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Relación Velocidad vs Radio Hidráulico
    axes[0,1].scatter(df['VELOCIDAD'], df['RADIO_HIDRAULICO'], c='green', alpha=0.7, s=50)
    axes[0,1].set_xlabel('Velocidad (m/s)')
    axes[0,1].set_ylabel('Radio Hidráulico (m)')
    axes[0,1].set_title('Velocidad vs Radio Hidráulico')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Relación Área vs Caudal
    axes[1,0].scatter(df['AREA'], df['CAUDAL'], c='red', alpha=0.7, s=50)
    axes[1,0].set_xlabel('Área (m²)')
    axes[1,0].set_ylabel('Caudal (m³/s)')
    axes[1,0].set_title('Área vs Caudal')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Distribución de Tirante Medio
    axes[1,1].hist(df['TIRANTE_MEDIO'], bins=10, color='orange', alpha=0.7, edgecolor='black')
    axes[1,1].set_xlabel('Tirante Medio (m)')
    axes[1,1].set_ylabel('Frecuencia')
    axes[1,1].set_title('Distribución de Tirante Medio')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# CONFIGURACIÓN STREAMLIT
st.set_page_config(page_title="Sistema Talapalca", page_icon="🌊", layout="wide")
st.title("🌊 Sistema Inteligente de Curvas Altura-Caudal - TALAPALCA")
st.markdown("**Modelo entrenado con 34 aforos reales**")

# Cargar modelo con manejo mejorado de errores
@st.cache_resource
def cargar_modelo():
    try:
        # Intentar cargar normalmente
        modelo = joblib.load('modelo_talapalca_entrenado.pkl')
        st.success("✅ Modelo cargado correctamente")
        return modelo
    except Exception as e:
        st.warning(f"⚠️ Error al cargar el modelo: {str(e)}")
        st.info("🔧 Creando modelo de demostración...")
        
        # Crear un modelo básico para demostración
        modelo_demo = SistemaCurvasAlturaCaudal()
        
        # Para que funcione el procesamiento, necesitamos un clasificador y escalador básicos
        # Simulamos un modelo entrenado con datos mínimos
        from sklearn.datasets import make_classification
        
        # Crear datos de ejemplo para entrenar el modelo demo
        X_demo, y_demo = make_classification(
            n_samples=50, 
            n_features=9, 
            n_classes=3, 
            random_state=42
        )
        
        # Nombres de clases que espera la aplicación
        y_demo_nombres = ['GRUPO_ESTANDAR', 'GRUPO_RECIENTE', 'GRUPO_ALTO_RH']
        y_demo_categoricos = [y_demo_nombres[i % 3] for i in y_demo]
        
        modelo_demo.entrenar(X_demo, y_demo_categoricos)
        
        st.success("✅ Modelo de demostración creado exitosamente")
        st.info("💡 Nota: Este es un modelo de demostración. Para usar el modelo real, asegúrate de que el archivo 'modelo_talapalca_entrenado.pkl' esté disponible.")
        
        return modelo_demo

modelo = cargar_modelo()

# NAVEGACIÓN
opcion = st.sidebar.radio("Navegación:", ["🏠 Inicio", "📤 Subir Aforos", "📊 Ingreso Manual", "📈 Curvas", "🔍 Análisis Hidráulico"])

if opcion == "🏠 Inicio":
    st.header("Bienvenido al Sistema Talapalca")
    st.info("Sistema para generar curvas altura-caudal usando IA")
    
    st.subheader("Instrucciones de uso:")
    st.markdown("""
    1. **📤 Subir Aforos**: Carga un archivo CSV con datos de aforos
    2. **📊 Ingreso Manual**: Ingresa datos de aforos manualmente
    3. **📈 Curvas**: Visualiza las curvas generadas
    4. **🔍 Análisis Hidráulico**: Análisis detallado de variables hidráulicas
    
    **Columnas requeridas en CSV:**
    - NIVEL DE AFORO (m)
    - CAUDAL (m3/s)
    - AREA (m2)
    - ANCHO RIO (m)
    - VELOCIDAD (m/s)
    - PERIMETRO (m) [opcional]
    - FECHA AFORO [opcional]
    """)

elif opcion == "📤 Subir Aforos":
    st.header("📤 Subir Archivo de Aforos")
    
    if modelo is None:
        st.error("⚠️ El modelo no está disponible.")
    else:
        archivo_subido = st.file_uploader("Selecciona archivo CSV", type=['csv'])
        
        if archivo_subido is not None:
            try:
                df = pd.read_csv(archivo_subido)
                st.success(f"✅ {len(df)} aforos cargados")
                
                # Mostrar vista previa
                st.subheader("📋 Vista previa de datos")
                st.dataframe(df.head())
                
                # Verificar columnas básicas
                columnas_necesarias = ['CAUDAL (m3/s)', 'VELOCIDAD (m/s)', 'AREA (m2)', 'ANCHO RIO (m)', 'NIVEL DE AFORO (m)']
                columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]
                
                if not columnas_faltantes:
                    st.success("✅ Todas las columnas necesarias están presentes")
                    
                    # USAR STATE PARA CONTROLAR EL RECÁLCULO
                    if 'recalcular_con_alto_rh' not in st.session_state:
                        st.session_state.recalcular_con_alto_rh = False
                    
                    if st.button("🚀 Procesar Aforos", type="primary") or st.session_state.recalcular_con_alto_rh:
                        with st.spinner("Procesando datos..."):
                            if not st.session_state.recalcular_con_alto_rh:
                                # PROCESAMIENTO INICIAL - SIN GRUPO_ALTO_RH
                                curvas_sin, datos_sin = procesar_con_modelo(modelo, df, incluir_alto_rh=False)
                                
                                if curvas_sin:
                                    st.success(f"✅ Procesado exitoso: {len(datos_sin)} aforos (sin GRUPO_ALTO_RH)")
                                    
                                    # Mostrar resultados iniciales
                                    st.subheader("📊 Resultados Iniciales (sin GRUPO_ALTO_RH)")
                                    st.dataframe(datos_sin[['NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 'GRUPO_PREDICHO']].head())
                                    
                                    # Gráfico inicial
                                    st.subheader("📈 Curvas Altura-Caudal (sin GRUPO_ALTO_RH)")
                                    fig_sin = crear_grafico_principal(datos_sin, curvas_sin, "Curvas sin GRUPO_ALTO_RH")
                                    st.pyplot(fig_sin)
                                    
                                    # VERIFICAR SI HAY GRUPO_ALTO_RH PARA OFRECER RECÁLCULO
                                    _, datos_completos = procesar_con_modelo(modelo, df, incluir_alto_rh=True)
                                    tiene_alto_rh = 'GRUPO_ALTO_RH' in datos_completos['GRUPO_PREDICHO'].values
                                    
                                    if tiene_alto_rh:
                                        st.subheader("⚙️ Opción de Re-análisis")
                                        st.info("Se detectó GRUPO_ALTO_RH en los datos. ¿Deseas recalcular INCLUYÉNDOLO?")
                                        
                                        # BOTÓN DE RECÁLCULO - AHORA FUNCIONA CORRECTAMENTE
                                        if st.button("🔄 RECALCULAR con GRUPO_ALTO_RH", key="btn_recalcular"):
                                            st.session_state.recalcular_con_alto_rh = True
                                            st.rerun()
                                    else:
                                        st.info("✅ No se detectó GRUPO_ALTO_RH en los datos. Los resultados están completos.")
                                        
                                else:
                                    st.warning("⚠️ No se pudieron generar curvas con los datos proporcionados. Verifica que tengas suficientes puntos por grupo.")
                            
                            else:
                                # RECÁLCULO CON GRUPO_ALTO_RH
                                st.session_state.recalcular_con_alto_rh = False
                                
                                curvas_con, datos_con = procesar_con_modelo(modelo, df, incluir_alto_rh=True)
                                
                                st.success(f"✅ RECÁLCULO EXITOSO: {len(datos_con)} aforos (CON GRUPO_ALTO_RH)")
                                
                                # Mostrar NUEVOS resultados
                                st.subheader("📊 NUEVOS Resultados (CON GRUPO_ALTO_RH)")
                                st.dataframe(datos_con[['NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 'GRUPO_PREDICHO']].head())
                                
                                # NUEVO gráfico con GRUPO_ALTO_RH
                                st.subheader("📈 NUEVAS Curvas Altura-Caudal (CON GRUPO_ALTO_RH)")
                                fig_con = crear_grafico_principal(datos_con, curvas_con, "Curvas CON GRUPO_ALTO_RH")
                                st.pyplot(fig_con)
                                
                                # Gráficos complementarios
                                st.subheader("🔍 Análisis Complementario (CON GRUPO_ALTO_RH)")
                                fig_comp = crear_graficos_complementarios(datos_con)
                                st.pyplot(fig_comp)
                                
                                # Mostrar ecuaciones
                                st.subheader("📐 Ecuaciones (CON GRUPO_ALTO_RH)")
                                for grupo, curva in curvas_con.items():
                                    with st.expander(f"{grupo} - R² = {curva['r2']:.3f}"):
                                        if curva['nombre'] == 'Polinómico G2':
                                            a, b, c = curva['parametros']
                                            st.latex(f"Q = {a:.4f}H^2 + {b:.4f}H + {c:.4f}")
                                        elif curva['nombre'] == 'Polinómico G3':
                                            a, b, c, d = curva['parametros']
                                            st.latex(f"Q = {a:.4f}H^3 + {b:.4f}H^2 + {c:.4f}H + {d:.4f}")
                                        elif curva['nombre'] == 'Potencial':
                                            a, b = curva['parametros']
                                            st.latex(f"Q = {a:.4f}H^{{{b:.4f}}}")
                                
                else:
                    st.error(f"❌ Faltan las siguientes columnas necesarias: {', '.join(columnas_faltantes)}")
                    st.info("💡 Asegúrate de que tu archivo CSV tenga las columnas con los nombres exactos.")
                    
            except Exception as e:
                st.error(f"❌ Error al procesar el archivo: {e}")
                st.info("💡 Verifica que el archivo sea un CSV válido y tenga el formato correcto.")

elif opcion == "📊 Ingreso Manual":
    st.header("📊 Ingreso Manual de Aforos")
    
    if modelo is None:
        st.error("⚠️ El modelo no está disponible.")
    else:
        num_aforos = st.number_input("Número de aforos:", min_value=1, max_value=20, value=3)
        datos_manual = []
        
        for i in range(num_aforos):
            with st.expander(f"Aforo {i+1}", expanded=True if i == 0 else False):
                col1, col2 = st.columns(2)
                with col1:
                    nivel = st.number_input("Nivel (m)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key=f"n{i}")
                    caudal = st.number_input("Caudal (m³/s)", min_value=0.1, max_value=50.0, value=2.0, step=0.1, key=f"q{i}")
                    area = st.number_input("Área (m²)", min_value=0.1, max_value=50.0, value=3.0, step=0.1, key=f"a{i}")
                with col2:
                    ancho = st.number_input("Ancho (m)", min_value=0.1, max_value=20.0, value=8.0, step=0.1, key=f"w{i}")
                    perimetro = st.number_input("Perímetro (m)", min_value=0.1, max_value=30.0, value=8.5, step=0.1, key=f"p{i}")
                    velocidad = st.number_input("Velocidad (m/s)", min_value=0.1, max_value=5.0, value=0.7, step=0.1, key=f"v{i}")
                
                datos_manual.append({
                    'FECHA AFORO': '2024-01-01',
                    'NIVEL DE AFORO (m)': nivel,
                    'CAUDAL (m3/s)': caudal,
                    'AREA (m2)': area,
                    'ANCHO RIO (m)': ancho,
                    'PERIMETRO (m)': perimetro,
                    'VELOCIDAD (m/s)': velocidad
                })
        
        if st.button("🚀 Procesar Datos Manuales", type="primary") and datos_manual:
            with st.spinner("Procesando datos manuales..."):
                df_manual = pd.DataFrame(datos_manual)
                curvas, datos_procesados = procesar_con_modelo(modelo, df_manual, incluir_alto_rh=False)
                
                if curvas:
                    st.success("✅ Datos procesados exitosamente")
                    
                    st.subheader("📊 Datos Procesados")
                    st.dataframe(datos_procesados[['NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 'GRUPO_PREDICHO']])
                    
                    st.subheader("📈 Curvas Generadas")
                    fig = crear_grafico_principal(datos_procesados, curvas, "Curvas Altura-Caudal - Datos Manuales")
                    st.pyplot(fig)
                    
                    # Mostrar ecuaciones
                    st.subheader("📐 Ecuaciones de las Curvas")
                    for grupo, curva in curvas.items():
                        with st.expander(f"{grupo} - R² = {curva['r2']:.3f}"):
                            st.write(f"**Tipo:** {curva['nombre']}")
                            st.write(f"**Puntos usados:** {curva['n_puntos']}")
                            st.write(f"**Rango de niveles:** {curva['rango_niveles'][0]:.2f} - {curva['rango_niveles'][1]:.2f} m")
                            st.write(f"**Rango de caudales:** {curva['rango_caudales'][0]:.2f} - {curva['rango_caudales'][1]:.2f} m³/s")
                            
                            if curva['nombre'] == 'Polinómico G2':
                                a, b, c = curva['parametros']
                                st.latex(f"Q = {a:.4f}H^2 + {b:.4f}H + {c:.4f}")
                            elif curva['nombre'] == 'Polinómico G3':
                                a, b, c, d = curva['parametros']
                                st.latex(f"Q = {a:.4f}H^3 + {b:.4f}H^2 + {c:.4f}H + {d:.4f}")
                            elif curva['nombre'] == 'Potencial':
                                a, b = curva['parametros']
                                st.latex(f"Q = {a:.4f}H^{{{b:.4f}}}")
                else:
                    st.warning("⚠️ No se pudieron generar curvas con los datos ingresados. Intenta con más puntos o diferentes valores.")

elif opcion == "📈 Curvas":
    st.header("📈 Visualización de Curvas")
    st.info("Esta sección muestra información sobre las curvas del modelo")
    
    if modelo is None:
        st.error("⚠️ El modelo no está disponible.")
    else:
        st.success("✅ Modelo cargado y listo para generar curvas")
        
        # Información del modelo
        st.subheader("🔧 Información del Modelo")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Tipo de Modelo", "Random Forest")
            st.metric("Clases", "3 grupos")
        
        with col2:
            st.metric("Características", "9 variables")
            st.metric("Estado", "Activo")
        
        # Grupos del modelo
        st.subheader("🎯 Grupos de Clasificación")
        grupos_info = {
            "GRUPO_ESTANDAR": "Condiciones normales de flujo",
            "GRUPO_RECIENTE": "Datos recientes o condiciones específicas", 
            "GRUPO_ALTO_RH": "Alto radio hidráulico o condiciones extremas"
        }
        
        for grupo, descripcion in grupos_info.items():
            with st.expander(f"{grupo}"):
                st.write(descripcion)

elif opcion == "🔍 Análisis Hidráulico":
    st.header("🔍 Análisis Hidráulico Completo")
    
    if modelo is None:
        st.error("⚠️ El modelo no está disponible.")
    else:
        archivo_subido = st.file_uploader("Selecciona archivo CSV para análisis hidráulico", type=['csv'], key="analisis_csv")
        
        if archivo_subido is not None:
            try:
                df = pd.read_csv(archivo_subido)
                st.success(f"✅ {len(df)} aforos cargados para análisis")
                
                # Procesar datos
                df_procesado = preparar_datos(df)
                
                st.subheader("📊 Estadísticas Descriptivas")
                st.dataframe(df_procesado[['NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 'ANCHO_RIO', 'RADIO_HIDRAULICO', 'TIRANTE_MEDIO']].describe())
                
                st.subheader("📈 Gráficos de Análisis Hidráulico")
                fig_analisis = crear_graficos_analisis_hidraulico(df_procesado)
                st.pyplot(fig_analisis)
                
                st.subheader("🔍 Correlaciones entre Variables")
                # Calcular matriz de correlación
                variables_corr = ['NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 'ANCHO_RIO', 'RADIO_HIDRAULICO', 'TIRANTE_MEDIO']
                corr_matrix = df_procesado[variables_corr].corr()
                
                # Mostrar matriz de correlación
                fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                im = ax_corr.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                ax_corr.set_xticks(range(len(variables_corr)))
                ax_corr.set_yticks(range(len(variables_corr)))
                ax_corr.set_xticklabels(variables_corr, rotation=45)
                ax_corr.set_yticklabels(variables_corr)
                
                # Añadir valores de correlación
                for i in range(len(variables_corr)):
                    for j in range(len(variables_corr)):
                        text = ax_corr.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontsize=10)
                
                plt.colorbar(im, ax=ax_corr)
                ax_corr.set_title('Matriz de Correlación')
                plt.tight_layout()
                st.pyplot(fig_corr)
                
                st.subheader("📋 Resumen por Grupos")
                if 'GRUPO_PREDICHO' in df_procesado.columns:
                    resumen_grupos = df_procesado.groupby('GRUPO_PREDICHO').agg({
                        'NIVEL_AFORO': ['count', 'mean', 'std'],
                        'CAUDAL': ['mean', 'std', 'max'],
                        'VELOCIDAD': ['mean', 'std'],
                        'RADIO_HIDRAULICO': ['mean', 'std']
                    }).round(3)
                    st.dataframe(resumen_grupos)
                
            except Exception as e:
                st.error(f"❌ Error en el análisis: {e}")
        else:
            st.info("📁 Sube un archivo CSV con datos de aforos para realizar el análisis hidráulico")

st.markdown("---")
st.markdown("**🌊 IA para generar Curvas H-Q - Sistema Talapalca**")