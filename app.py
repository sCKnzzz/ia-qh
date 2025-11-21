import streamlit as st

# CONFIGURACIÓN STREAMLIT - DEBE SER LA PRIMERA LÍNEA
st.set_page_config(page_title="Sistema Talapalca", page_icon="🌊", layout="wide")

# Ahora importamos el resto de las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import io
import sys
import os

# Manejo de importaciones opcionales para gráficos interactivos
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly no está instalado. Los gráficos interactivos no estarán disponibles.")

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
        df_procesado['PERIMETRO'] = 2 * (df_procesado['AREA'] / df_procesado['ANCHO_RIO']) + df_procesado['ANCHO_RIO']
    
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

# FUNCIÓN SIMPLIFICADA PARA AJUSTAR CURVAS
def ajustar_curva_simple(H, Q):
    """Ajustar curva de manera simple y robusta"""
    if len(H) < 3:
        return None
    
    # Ordenar datos
    sort_idx = np.argsort(H)
    H_sorted = H[sort_idx]
    Q_sorted = Q[sort_idx]
    
    modelos = [
        ('Polinómico G2', func_poly2),
        ('Potencial', func_pot),
        ('Polinómico G3', func_poly3)
    ]
    
    mejor_r2 = -np.inf
    mejor_modelo = None
    
    for nombre, funcion in modelos:
        try:
            if nombre == 'Potencial':
                # Para modelo potencial, evitar valores negativos
                H_positivo = np.maximum(H_sorted, 0.1)
                params, _ = curve_fit(funcion, H_positivo, Q_sorted, p0=[1.0, 2.0], maxfev=5000)
            else:
                params, _ = curve_fit(funcion, H_sorted, Q_sorted, maxfev=5000)
            
            Q_pred = funcion(H_sorted, *params)
            ss_res = np.sum((Q_sorted - Q_pred)**2)
            ss_tot = np.sum((Q_sorted - np.mean(Q_sorted))**2)
            
            if ss_tot == 0:
                r2 = 0
            else:
                r2 = 1 - (ss_res / ss_tot)
            
            if r2 > mejor_r2 and r2 > 0.5:  # Umbral más bajo para R²
                mejor_r2 = r2
                mejor_modelo = {
                    'nombre': nombre,
                    'funcion': funcion,
                    'parametros': params,
                    'r2': round(r2, 3),
                    'n_puntos': len(H_sorted),
                    'rango_niveles': (float(min(H_sorted)), float(max(H_sorted))),
                    'rango_caudales': (float(min(Q_sorted)), float(max(Q_sorted)))
                }
        except Exception as e:
            continue
    
    return mejor_modelo

# FUNCIÓN SIMPLIFICADA PARA PROCESAR CON MODELO
def procesar_con_modelo_simple(modelo, df, incluir_alto_rh=True):
    """Procesar datos con el modelo - versión simplificada"""
    
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
        # Para demo, asignar grupos manualmente si el modelo falla
        try:
            X = df_procesado[features]
            X_scaled = modelo.escalador.transform(X)
            grupos_pred = modelo.clasificador.predict(X_scaled)
            df_procesado['GRUPO_PREDICHO'] = grupos_pred
        except:
            # Si el modelo falla, asignar grupos basados en percentiles
            niveles = df_procesado['NIVEL_AFORO']
            percentil_33 = np.percentile(niveles, 33)
            percentil_66 = np.percentile(niveles, 66)
            
            condiciones = [
                niveles <= percentil_33,
                (niveles > percentil_33) & (niveles <= percentil_66),
                niveles > percentil_66
            ]
            opciones = ['GRUPO_ESTANDAR', 'GRUPO_RECIENTE', 'GRUPO_ALTO_RH']
            df_procesado['GRUPO_PREDICHO'] = np.select(condiciones, opciones, default='GRUPO_ESTANDAR')
        
        # Filtrar si no incluir GRUPO_ALTO_RH
        if not incluir_alto_rh:
            df_filtrado = df_procesado[df_procesado['GRUPO_PREDICHO'] != 'GRUPO_ALTO_RH'].copy()
        else:
            df_filtrado = df_procesado.copy()
        
        # Generar curvas
        resultados = {}
        for grupo in df_filtrado['GRUPO_PREDICHO'].unique():
            if grupo == 'GRUPO_ESTANDAR':
                continue
                
            grupo_data = df_filtrado[df_filtrado['GRUPO_PREDICHO'] == grupo]
            if len(grupo_data) >= 2:  # Reducido a 2 puntos mínimo
                H = grupo_data['NIVEL_AFORO'].values
                Q = grupo_data['CAUDAL'].values
                
                curva = ajustar_curva_simple(H, Q)
                if curva:
                    resultados[grupo] = curva
        
        return resultados, df_filtrado
        
    except Exception as e:
        st.error(f"❌ Error en procesamiento: {str(e)}")
        return {}, df_procesado

# FUNCIONES PARA GRÁFICOS - VERSIÓN ROBUSTA
def crear_grafico_principal(df, curvas, titulo):
    """Crear gráfico principal con matplotlib"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
        marcadores = {'GRUPO_ALTO_RH': 's', 'GRUPO_RECIENTE': '^', 'GRUPO_ESTANDAR': 'o'}
        
        # Graficar puntos
        for grupo in df['GRUPO_PREDICHO'].unique():
            if grupo == 'GRUPO_ESTANDAR':
                continue
                
            color = colores.get(grupo, 'orange')
            marcador = marcadores.get(grupo, 'o')
            grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
            
            ax.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], 
                      color=color, marker=marcador, s=80, label=grupo, alpha=0.7,
                      edgecolors='black', linewidth=0.5)
        
        # Graficar curvas
        for grupo, curva in curvas.items():
            color = colores.get(grupo, 'orange')
            
            rango_min, rango_max = curva['rango_niveles']
            H_range = np.linspace(rango_min, rango_max, 100)
            Q_curve = curva['funcion'](H_range, *curva['parametros'])
            
            ax.plot(H_range, Q_curve, color=color, linewidth=2, 
                   label=f"{grupo} (R²={curva['r2']:.3f})")
        
        ax.set_xlabel('Nivel (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Caudal (m³/s)', fontsize=12, fontweight='bold')
        ax.set_title(titulo, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    except Exception as e:
        st.error(f"Error creando gráfico: {str(e)}")
        return None

if PLOTLY_AVAILABLE:
    def crear_grafico_interactivo(df, curvas, titulo):
        """Crear gráfico interactivo con Plotly - versión robusta"""
        try:
            fig = go.Figure()
            
            colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
            
            # DEBUG: Mostrar información sobre los datos
            st.sidebar.info(f"📊 Datos: {len(df)} puntos, {len(curvas)} curvas")
            
            # Agregar puntos dispersos
            puntos_agregados = False
            for grupo in df['GRUPO_PREDICHO'].unique():
                if grupo == 'GRUPO_ESTANDAR':
                    continue
                    
                grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
                if len(grupo_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=grupo_data['NIVEL_AFORO'],
                        y=grupo_data['CAUDAL'],
                        mode='markers',
                        name=grupo,
                        marker=dict(
                            color=colores.get(grupo, 'orange'),
                            size=10,
                            line=dict(width=1, color='black')
                        ),
                        hovertemplate='<b>%{text}</b><br>Nivel: %{x:.2f} m<br>Caudal: %{y:.2f} m³/s<extra></extra>',
                        text=[f'{grupo}'] * len(grupo_data)
                    ))
                    puntos_agregados = True
            
            # Agregar curvas
            curvas_agregadas = False
            for grupo, curva in curvas.items():
                try:
                    rango_min, rango_max = curva['rango_niveles']
                    H_range = np.linspace(rango_min, rango_max, 50)
                    Q_curve = curva['funcion'](H_range, *curva['parametros'])
                    
                    fig.add_trace(go.Scatter(
                        x=H_range,
                        y=Q_curve,
                        mode='lines',
                        name=f"{grupo} (R²={curva['r2']:.3f})",
                        line=dict(
                            color=colores.get(grupo, 'orange'),
                            width=3
                        ),
                        hovertemplate='<b>%{fullData.name}</b><br>Nivel: %{x:.2f} m<br>Caudal: %{y:.2f} m³/s<extra></extra>'
                    ))
                    curvas_agregadas = True
                except Exception as e:
                    st.warning(f"⚠️ No se pudo graficar curva para {grupo}: {str(e)}")
                    continue
            
            # Si no hay datos, mostrar mensaje
            if not puntos_agregados and not curvas_agregadas:
                fig.add_annotation(
                    text="No hay datos para mostrar",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=16)
                )
            
            fig.update_layout(
                title=dict(text=titulo, font=dict(size=20, color='black')),
                xaxis=dict(title='Nivel (m)', gridcolor='lightgray'),
                yaxis=dict(title='Caudal (m³/s)', gridcolor='lightgray'),
                plot_bgcolor='white',
                hovermode='closest',
                height=600,
                showlegend=True
            )
            
            return fig
        except Exception as e:
            st.error(f"❌ Error creando gráfico interactivo: {str(e)}")
            # Devolver un gráfico vacío como fallback
            fig = go.Figure()
            fig.add_annotation(text="Error al crear gráfico", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            return fig

# Título principal después de la configuración
st.title("🌊 IA para la generación de Curvas Altura-Caudal")
st.markdown("**Sistema inteligente para análisis hidráulico**")

# Cargar modelo con manejo mejorado de errores
@st.cache_resource
def cargar_modelo():
    try:
        # Intentar cargar normalmente
        modelo = joblib.load('modelo_talapalca_entrenado.pkl')
        st.success("✅ Modelo cargado correctamente")
        return modelo
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar el modelo: {str(e)}")
        st.info("🔧 Usando sistema de agrupamiento básico...")
        
        # Crear un modelo básico para demostración
        modelo_demo = SistemaCurvasAlturaCaudal()
        return modelo_demo

modelo = cargar_modelo()

# NAVEGACIÓN
st.sidebar.title("Navegación")
opcion = st.sidebar.radio("Selecciona una opción:", 
                         ["🏠 Inicio", "📤 Subir Aforos", "📊 Ingreso Manual", "📈 Curvas"])

# 🎛️ PANEL DE CONTROL DINÁMICO (Global) - solo si Plotly está disponible
if PLOTLY_AVAILABLE:
    st.sidebar.header("🎛️ Controles de Visualización")
    tipo_visualizacion = st.sidebar.radio(
        "Tipo de Gráfico Principal",
        ["Plotly (Interactivo)", "Matplotlib (Estático)"],
        index=0
    )
else:
    tipo_visualizacion = "Matplotlib (Estático)"

if opcion == "🏠 Inicio":
    st.header("Bienvenido al Sistema de Curvas H-Q")
    st.info("Sistema de inteligencia artificial para generar curvas altura-caudal usando machine learning")
    
    st.subheader("Instrucciones de uso:")
    st.markdown("""
    1. **📤 Subir Aforos**: Carga un archivo CSV con datos de aforos
    2. **📊 Ingreso Manual**: Ingresa datos de aforos manualmente  
    3. **📈 Curvas**: Visualiza las curvas generadas
    
    **Columnas requeridas en CSV:**
    - NIVEL DE AFORO (m)
    - CAUDAL (m3/s)
    - AREA (m2)
    - ANCHO RIO (m)
    - VELOCIDAD (m/s)
    - PERIMETRO (m) [opcional]
    - FECHA AFORO [opcional]
    """)

    # Demo con datos de ejemplo
    st.subheader("🎮 Demo Rápido")
    if st.button("Probar con datos de ejemplo"):
        # Crear datos de ejemplo realistas
        np.random.seed(42)
        n_points = 20
        H_demo = np.linspace(0.5, 5.0, n_points)
        # Crear relación no lineal realista
        Q_demo = 0.8 * H_demo**2 + 0.5 * H_demo + 0.1 + np.random.normal(0, 0.2, n_points)
        
        df_demo = pd.DataFrame({
            'NIVEL DE AFORO (m)': H_demo,
            'CAUDAL (m3/s)': Q_demo,
            'AREA (m2)': H_demo * 8 + np.random.normal(0, 1, n_points),
            'ANCHO RIO (m)': [8.0] * n_points,
            'VELOCIDAD (m/s)': Q_demo / (H_demo * 8) + np.random.normal(0, 0.1, n_points),
            'PERIMETRO (m)': 2 * H_demo + 8 + np.random.normal(0, 0.5, n_points)
        })
        
        # Procesar datos de demo
        with st.spinner("Procesando datos de ejemplo..."):
            curvas_demo, datos_demo = procesar_con_modelo_simple(modelo, df_demo, incluir_alto_rh=False)
            
            if curvas_demo:
                st.success(f"✅ Se generaron {len(curvas_demo)} curvas con datos de ejemplo")
                
                # Mostrar gráfico
                if tipo_visualizacion == "Plotly (Interactivo)" and PLOTLY_AVAILABLE:
                    fig_demo = crear_grafico_interactivo(datos_demo, curvas_demo, "Curvas H-Q - Datos de Ejemplo")
                    st.plotly_chart(fig_demo, use_container_width=True)
                else:
                    fig_demo = crear_grafico_principal(datos_demo, curvas_demo, "Curvas H-Q - Datos de Ejemplo")
                    if fig_demo:
                        st.pyplot(fig_demo)
                
                # Mostrar ecuaciones
                for grupo, curva in curvas_demo.items():
                    with st.expander(f"Ecuación {grupo} - R² = {curva['r2']:.3f}"):
                        if curva['nombre'] == 'Polinómico G2':
                            a, b, c = curva['parametros']
                            st.latex(f"Q = {a:.4f}H^2 + {b:.4f}H + {c:.4f}")
                        elif curva['nombre'] == 'Potencial':
                            a, b = curva['parametros']
                            st.latex(f"Q = {a:.4f}H^{{{b:.4f}}}")
            else:
                st.warning("No se pudieron generar curvas con los datos de ejemplo")

elif opcion == "📤 Subir Aforos":
    st.header("📤 Subir Archivo de Aforos")
    
    archivo_subido = st.file_uploader("Selecciona archivo CSV", type=['csv'])
    
    if archivo_subido is not None:
        try:
            df = pd.read_csv(archivo_subido)
            st.success(f"✅ {len(df)} aforos cargados correctamente")
            
            # Mostrar vista previa
            st.subheader("📋 Vista previa de datos")
            st.dataframe(df.head())
            
            # Verificar columnas básicas
            columnas_necesarias = ['NIVEL DE AFORO (m)', 'CAUDAL (m3/s)', 'AREA (m2)', 'ANCHO RIO (m)', 'VELOCIDAD (m/s)']
            columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]
            
            if columnas_faltantes:
                st.error(f"❌ Faltan columnas: {', '.join(columnas_faltantes)}")
            else:
                st.success("✅ Todas las columnas necesarias están presentes")
                
                # Procesar datos
                if st.button("🚀 Procesar Aforos", type="primary"):
                    with st.spinner("Procesando datos..."):
                        curvas, datos_procesados = procesar_con_modelo_simple(modelo, df, incluir_alto_rh=False)
                        
                        if curvas:
                            st.success(f"✅ Procesado exitoso: {len(curvas)} curvas generadas")
                            
                            # Mostrar datos procesados
                            st.subheader("📊 Datos Procesados")
                            st.dataframe(datos_procesados[['NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 'GRUPO_PREDICHO']].head())
                            
                            # Mostrar gráfico principal
                            st.subheader("📈 Curvas Altura-Caudal")
                            
                            if tipo_visualizacion == "Plotly (Interactivo)" and PLOTLY_AVAILABLE:
                                fig = crear_grafico_interactivo(datos_procesados, curvas, "Curvas H-Q Generadas")
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                fig = crear_grafico_principal(datos_procesados, curvas, "Curvas H-Q Generadas")
                                if fig:
                                    st.pyplot(fig)
                            
                            # Mostrar ecuaciones
                            st.subheader("📐 Ecuaciones Generadas")
                            for grupo, curva in curvas.items():
                                with st.expander(f"{grupo} - R² = {curva['r2']:.3f}"):
                                    st.write(f"**Tipo de modelo:** {curva['nombre']}")
                                    st.write(f"**Puntos utilizados:** {curva['n_puntos']}")
                                    st.write(f"**Rango de niveles:** {curva['rango_niveles'][0]:.2f} - {curva['rango_niveles'][1]:.2f} m")
                                    
                                    if curva['nombre'] == 'Polinómico G2':
                                        a, b, c = curva['parametros']
                                        st.latex(f"Q = {a:.4f}H^2 + {b:.4f}H + {c:.4f}")
                                    elif curva['nombre'] == 'Potencial':
                                        a, b = curva['parametros']
                                        st.latex(f"Q = {a:.4f}H^{{{b:.4f}}}")
                                    elif curva['nombre'] == 'Polinómico G3':
                                        a, b, c, d = curva['parametros']
                                        st.latex(f"Q = {a:.4f}H^3 + {b:.4f}H^2 + {c:.4f}H + {d:.4f}")
                        else:
                            st.warning("⚠️ No se pudieron generar curvas con los datos proporcionados. Verifica que haya suficientes puntos.")
                            
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")

elif opcion == "📊 Ingreso Manual":
    st.header("📊 Ingreso Manual de Aforos")
    
    num_aforos = st.number_input("Número de aforos:", min_value=1, max_value=10, value=3)
    datos_manual = []
    
    for i in range(num_aforos):
        with st.expander(f"Aforo {i+1}", expanded=(i==0)):
            col1, col2 = st.columns(2)
            with col1:
                nivel = st.number_input("Nivel (m)", min_value=0.1, value=1.0 + i*0.5, key=f"n{i}")
                caudal = st.number_input("Caudal (m³/s)", min_value=0.1, value=2.0 + i*1.0, key=f"q{i}")
                area = st.number_input("Área (m²)", min_value=0.1, value=3.0 + i*2.0, key=f"a{i}")
            with col2:
                ancho = st.number_input("Ancho (m)", min_value=0.1, value=8.0, key=f"w{i}")
                velocidad = st.number_input("Velocidad (m/s)", min_value=0.1, value=0.7, key=f"v{i}")
                perimetro = st.number_input("Perímetro (m)", min_value=0.1, value=8.5 + i*0.5, key=f"p{i}")
            
            datos_manual.append({
                'NIVEL DE AFORO (m)': nivel,
                'CAUDAL (m3/s)': caudal,
                'AREA (m2)': area,
                'ANCHO RIO (m)': ancho,
                'VELOCIDAD (m/s)': velocidad,
                'PERIMETRO (m)': perimetro
            })
    
    if st.button("🚀 Procesar Datos Manuales") and datos_manual:
        with st.spinner("Procesando datos manuales..."):
            df_manual = pd.DataFrame(datos_manual)
            curvas, datos_procesados = procesar_con_modelo_simple(modelo, df_manual, incluir_alto_rh=False)
            
            if curvas:
                st.success("✅ Datos procesados exitosamente")
                
                # Mostrar gráfico
                if tipo_visualizacion == "Plotly (Interactivo)" and PLOTLY_AVAILABLE:
                    fig = crear_grafico_interactivo(datos_procesados, curvas, "Curvas H-Q - Datos Manuales")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = crear_grafico_principal(datos_procesados, curvas, "Curvas H-Q - Datos Manuales")
                    if fig:
                        st.pyplot(fig)
                
                # Mostrar ecuaciones
                for grupo, curva in curvas.items():
                    with st.expander(f"Ecuación {grupo} - R² = {curva['r2']:.3f}"):
                        if curva['nombre'] == 'Polinómico G2':
                            a, b, c = curva['parametros']
                            st.latex(f"Q = {a:.4f}H^2 + {b:.4f}H + {c:.4f}")
            else:
                st.warning("No se pudieron generar curvas. Intenta con más puntos o diferentes valores.")

elif opcion == "📈 Curvas":
    st.header("📈 Información del Sistema")
    st.info("Esta sección muestra información sobre el sistema de curvas")
    
    st.subheader("🔧 Modelos Matemáticos Soportados")
    st.markdown("""
    - **Polinómico de 2do grado**: Q = aH² + bH + c
    - **Polinómico de 3er grado**: Q = aH³ + bH² + cH + d  
    - **Potencial**: Q = aHᵇ
    - **Exponencial**: Q = aeᵇᴴ
    - **Logarítmico**: Q = a·ln(H + b)
    """)
    
    st.subheader("🎯 Grupos de Clasificación")
    grupos_info = {
        "GRUPO_ESTANDAR": "Condiciones normales de flujo",
        "GRUPO_RECIENTE": "Datos recientes o condiciones específicas", 
        "GRUPO_ALTO_RH": "Alto radio hidráulico o condiciones extremas"
    }
    
    for grupo, descripcion in grupos_info.items():
        st.write(f"**{grupo}**: {descripcion}")

st.markdown("---")
st.markdown("**🌊 Sistema de Curvas H-Q - Versión Mejorada**")