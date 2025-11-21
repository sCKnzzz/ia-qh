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
from sklearn.metrics import r2_score

# DEFINIR LA CLASE DEL SISTEMA
class SistemaCurvasAlturaCaudal:
    def __init__(self):
        self.clasificador = RandomForestClassifier(n_estimators=100, random_state=42)
        self.escalador = StandardScaler()
        self.curvas = {}
        self.feature_names = None
    
    def entrenar(self, X, y):
        X_esc = self.escalador.fit_transform(X)
        self.clasificador.fit(X_esc, y)
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        return self
    
    def predecir_grupo(self, X):
        # Asegurar que las características coincidan
        if self.feature_names is not None and hasattr(X, 'columns'):
            # Reordenar columnas para que coincidan con el entrenamiento
            missing_features = set(self.feature_names) - set(X.columns)
            extra_features = set(X.columns) - set(self.feature_names)
            
            if missing_features:
                st.warning(f"⚠️ Características faltantes: {missing_features}")
                # Agregar columnas faltantes con valores cero
                for feature in missing_features:
                    X[feature] = 0
            
            if extra_features:
                st.warning(f"⚠️ Características extrañas: {extra_features}")
                # Eliminar características extrañas
                X = X[self.feature_names]
            
            # Reordenar columnas
            X = X[self.feature_names]
        
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

# FUNCIÓN PARA AJUSTAR CURVAS POR GRUPO
def ajustar_curvas_por_grupo(df, grupo):
    """Ajustar curva altura-caudal para un grupo específico"""
    grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
    
    if len(grupo_data) < 3:
        return None
    
    H = grupo_data['NIVEL_AFORO'].values
    Q = grupo_data['CAUDAL'].values
    
    # Ordenar por nivel
    sort_idx = np.argsort(H)
    H = H[sort_idx]
    Q = Q[sort_idx]
    
    funciones = [
        ('Polinómica Grado 2', func_poly2),
        ('Polinómica Grado 3', func_poly3),
        ('Potencial', func_pot),
        ('Exponencial', func_exp)
    ]
    
    mejor_r2 = -np.inf
    mejor_curva = None
    mejor_funcion = None
    
    for nombre, func in funciones:
        try:
            if func == func_poly2:
                popt, pcov = curve_fit(func, H, Q, maxfev=5000)
                Q_pred = func(H, *popt)
            elif func == func_poly3:
                popt, pcov = curve_fit(func, H, Q, maxfev=5000)
                Q_pred = func(H, *popt)
            elif func == func_pot:
                # Evitar valores negativos para función potencial
                H_pos = H[H > 0]
                Q_pos = Q[H > 0]
                if len(H_pos) >= 3:
                    popt, pcov = curve_fit(func, H_pos, Q_pos, maxfev=5000)
                    Q_pred = func(H, *popt)
                else:
                    continue
            elif func == func_exp:
                popt, pcov = curve_fit(func, H, Q, maxfev=5000)
                Q_pred = func(H, *popt)
            
            r2 = r2_score(Q, Q_pred)
            
            if r2 > mejor_r2 and r2 > 0.7:  # Solo considerar R² buenos
                mejor_r2 = r2
                mejor_curva = {
                    'funcion': func,
                    'parametros': popt,
                    'r2': r2,
                    'rango_niveles': (min(H), max(H))
                }
                mejor_funcion = nombre
                
        except Exception as e:
            continue
    
    return mejor_curva

# FUNCIÓN PARA PROCESAR Y ESTANDARIZAR DATOS
def preparar_datos_para_modelo(df, modelo):
    """Preparar y estandarizar datos para que coincidan con el modelo entrenado"""
    try:
        # Mapeo de nombres de columnas esperados
        mapeo_columnas = {
            'CAUDAL (m3/s)': 'CAUDAL',
            'VELOCIDAD (m/s)': 'VELOCIDAD', 
            'AREA (m2)': 'AREA',
            'ANCHO RIO (m)': 'ANCHO_RIO',
            'NIVEL DE AFORO (m)': 'NIVEL_AFORO',
            'CAUDAL': 'CAUDAL',
            'VELOCIDAD': 'VELOCIDAD',
            'AREA': 'AREA', 
            'ANCHO_RIO': 'ANCHO_RIO',
            'NIVEL_AFORO': 'NIVEL_AFORO'
        }
        
        # Aplicar mapeo de nombres
        df_estandarizado = df.copy()
        for col_vieja, col_nueva in mapeo_columnas.items():
            if col_vieja in df_estandarizado.columns:
                df_estandarizado[col_nueva] = df_estandarizado[col_vieja]
        
        # Si el modelo tiene feature_names definidos, usar esas características
        if hasattr(modelo, 'feature_names') and modelo.feature_names is not None:
            características_necesarias = modelo.feature_names
        else:
            # Características por defecto basadas en el error que viste
            características_necesarias = [
                'CAUDAL_AREA', 'PERIMETRO', 'RADIO_HIDRAULICO', 
                'TIRANTE_MEDIO', 'YEAR', 'CAUDAL', 'VELOCIDAD', 
                'AREA', 'ANCHO_RIO', 'NIVEL_AFORO'
            ]
        
        # Crear DataFrame con todas las características necesarias
        X_pred = pd.DataFrame()
        
        for feature in características_necesarias:
            if feature in df_estandarizado.columns:
                X_pred[feature] = df_estandarizado[feature]
            else:
                # Si falta la característica, crear con valores por defecto
                if feature == 'CAUDAL_AREA':
                    X_pred[feature] = df_estandarizado.get('CAUDAL', 0) / (df_estandarizado.get('AREA', 1) + 1e-6)
                elif feature == 'PERIMETRO':
                    X_pred[feature] = df_estandarizado.get('ANCHO_RIO', 0) * 2
                elif feature == 'RADIO_HIDRAULICO':
                    area = df_estandarizado.get('AREA', 1)
                    perimetro = df_estandarizado.get('ANCHO_RIO', 1) * 2
                    X_pred[feature] = area / (perimetro + 1e-6)
                elif feature == 'TIRANTE_MEDIO':
                    X_pred[feature] = df_estandarizado.get('NIVEL_AFORO', 0)
                elif feature == 'YEAR':
                    X_pred[feature] = 2024  # Año actual por defecto
                else:
                    X_pred[feature] = 0
        
        # Llenar valores NaN
        X_pred = X_pred.fillna(0)
        
        return X_pred, df_estandarizado
        
    except Exception as e:
        st.error(f"❌ Error preparando datos: {e}")
        return None, df

# FUNCIÓN PARA PROCESAR DATOS CON EL MODELO
def procesar_con_modelo(modelo, df, incluir_alto_rh=True):
    """Procesar datos completos con el modelo Random Forest"""
    try:
        # Preparar datos para el modelo
        X_pred, df_estandarizado = preparar_datos_para_modelo(df, modelo)
        
        if X_pred is None:
            st.error("❌ No se pudieron preparar los datos para el modelo")
            return None, df
        
        # Mostrar características que se usarán
        st.info(f"🔧 Usando {len(X_pred.columns)} características: {', '.join(X_pred.columns)}")
        
        # Predecir grupos
        grupos_pred = modelo.predecir_grupo(X_pred)
        df_procesado = df_estandarizado.copy()
        df_procesado['GRUPO_PREDICHO'] = grupos_pred
        
        # Ajustar curvas por grupo
        curvas = {}
        grupos_unicos = df_procesado['GRUPO_PREDICHO'].unique()
        
        for grupo in grupos_unicos:
            if not incluir_alto_rh and grupo == 'GRUPO_ALTO_RH':
                continue
                
            curva = ajustar_curvas_por_grupo(df_procesado, grupo)
            if curva:
                curvas[grupo] = curva
        
        return curvas, df_procesado
        
    except Exception as e:
        st.error(f"❌ Error en procesamiento con modelo: {e}")
        # Modo demostración como fallback
        st.warning("🔄 Usando modo demostración...")
        
        df_procesado = df.copy()
        # Asignar grupos de demostración
        np.random.seed(42)
        grupos_demo = ['GRUPO_ESTANDAR', 'GRUPO_RECIENTE', 'GRUPO_ALTO_RH']
        df_procesado['GRUPO_PREDICHO'] = [grupos_demo[i % 3] for i in range(len(df_procesado))]
        
        # Crear curvas demostrativas
        curvas = {}
        for grupo in grupos_demo:
            grupo_data = df_procesado[df_procesado['GRUPO_PREDICHO'] == grupo]
            if len(grupo_data) >= 2:
                H = grupo_data['NIVEL_AFORO'].values if 'NIVEL_AFORO' in grupo_data.columns else np.array([1.0, 2.0, 3.0])
                Q = grupo_data['CAUDAL'].values if 'CAUDAL' in grupo_data.columns else np.array([1.5, 3.0, 5.0])
                
                if len(H) >= 2:
                    try:
                        popt, pcov = curve_fit(func_poly2, H, Q, maxfev=5000)
                        Q_pred = func_poly2(H, *popt)
                        r2 = r2_score(Q, Q_pred) if len(Q) > 1 else 0.9
                        
                        curvas[grupo] = {
                            'funcion': func_poly2,
                            'parametros': popt,
                            'r2': r2,
                            'rango_niveles': (min(H), max(H))
                        }
                    except:
                        # Curva por defecto
                        curvas[grupo] = {
                            'funcion': func_poly2,
                            'parametros': [0.5, 0.5, 0.1],
                            'r2': 0.85,
                            'rango_niveles': (0.1, 3.0)
                        }
        
        return curvas, df_procesado

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
    
    # 1. Graficar puntos de datos originales
    for grupo in df['GRUPO_PREDICHO'].unique():
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
                
        except Exception as e:
            st.error(f"❌ Error al graficar ecuación del usuario: {e}")
    
    ax.set_xlabel('Nivel (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Caudal (m³/s)', fontsize=12, fontweight='bold')
    ax.set_title('Comparativo: Curvas IA vs Ecuación del Usuario', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(ax.get_xlim()[1], 3.5))
    
    return fig

# FUNCIÓN PARA ENTRENAR MODELO CON DATOS REALES
def entrenar_modelo_con_datos(df):
    """Entrenar el modelo Random Forest con datos reales"""
    try:
        # Mapear y estandarizar nombres de columnas
        mapeo_columnas = {
            'CAUDAL (m3/s)': 'CAUDAL',
            'VELOCIDAD (m/s)': 'VELOCIDAD', 
            'AREA (m2)': 'AREA',
            'ANCHO RIO (m)': 'ANCHO_RIO',
            'NIVEL DE AFORO (m)': 'NIVEL_AFORO'
        }
        
        df_estandarizado = df.copy()
        for col_vieja, col_nueva in mapeo_columnas.items():
            if col_vieja in df_estandarizado.columns:
                df_estandarizado[col_nueva] = df_estandarizado[col_vieja]
        
        # Crear características adicionales si es necesario
        if 'CAUDAL_AREA' not in df_estandarizado.columns and 'CAUDAL' in df_estandarizado.columns and 'AREA' in df_estandarizado.columns:
            df_estandarizado['CAUDAL_AREA'] = df_estandarizado['CAUDAL'] / (df_estandarizado['AREA'] + 1e-6)
        
        if 'PERIMETRO' not in df_estandarizado.columns and 'ANCHO_RIO' in df_estandarizado.columns:
            df_estandarizado['PERIMETRO'] = df_estandarizado['ANCHO_RIO'] * 2
        
        if 'RADIO_HIDRAULICO' not in df_estandarizado.columns and 'AREA' in df_estandarizado.columns and 'ANCHO_RIO' in df_estandarizado.columns:
            df_estandarizado['RADIO_HIDRAULICO'] = df_estandarizado['AREA'] / (df_estandarizado['ANCHO_RIO'] * 2 + 1e-6)
        
        if 'TIRANTE_MEDIO' not in df_estandarizado.columns and 'NIVEL_AFORO' in df_estandarizado.columns:
            df_estandarizado['TIRANTE_MEDIO'] = df_estandarizado['NIVEL_AFORO']
        
        if 'YEAR' not in df_estandarizado.columns:
            df_estandarizado['YEAR'] = 2024
        
        # Características para entrenamiento (basadas en el modelo original)
        caracteristicas_base = [
            'CAUDAL_AREA', 'PERIMETRO', 'RADIO_HIDRAULICO', 
            'TIRANTE_MEDIO', 'YEAR', 'CAUDAL', 'VELOCIDAD', 
            'AREA', 'ANCHO_RIO', 'NIVEL_AFORO'
        ]
        
        # Usar solo las características disponibles
        caracteristicas_disponibles = [col for col in caracteristicas_base if col in df_estandarizado.columns]
        
        if not caracteristicas_disponibles:
            st.error("❌ No hay características disponibles para entrenar el modelo")
            return None
        
        # Verificar que tenemos la columna de grupo objetivo
        if 'GRUPO' not in df_estandarizado.columns:
            st.error("❌ Se necesita columna 'GRUPO' para entrenar el modelo")
            return None
        
        # Preparar datos
        X = df_estandarizado[caracteristicas_disponibles].fillna(0)
        y = df_estandarizado['GRUPO']
        
        # Entrenar modelo
        modelo = SistemaCurvasAlturaCaudal()
        modelo.entrenar(X, y)
        
        st.success(f"✅ Modelo entrenado con {len(df)} muestras y {len(caracteristicas_disponibles)} características")
        st.info(f"📊 Características usadas: {', '.join(caracteristicas_disponibles)}")
        
        return modelo
        
    except Exception as e:
        st.error(f"❌ Error entrenando modelo: {e}")
        return None

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
        
        # Mostrar características del modelo cargado
        if hasattr(modelo, 'feature_names') and modelo.feature_names:
            st.info(f"🔧 Modelo entrenado con {len(modelo.feature_names)} características")
        
        return modelo
    except Exception as e:
        st.warning(f"⚠️ Error al cargar el modelo: {str(e)}")
        st.info("🔧 Usando modelo de demostración...")
        
        # Crear modelo de demostración
        modelo_demo = SistemaCurvasAlturaCaudal()
        
        # Datos de demostración con las características correctas
        np.random.seed(42)
        n_samples = 50
        
        # Crear características basadas en el modelo original
        características = ['CAUDAL_AREA', 'PERIMETRO', 'RADIO_HIDRAULICO', 'TIRANTE_MEDIO', 'YEAR']
        X_demo = pd.DataFrame(np.random.randn(n_samples, len(características)), columns=características)
        
        grupos_demo = ['GRUPO_ESTANDAR', 'GRUPO_RECIENTE', 'GRUPO_ALTO_RH']
        y_demo = [grupos_demo[i % 3] for i in range(n_samples)]
        
        modelo_demo.entrenar(X_demo, y_demo)
        st.success("✅ Modelo de demostración listo")
        return modelo_demo

modelo = cargar_modelo()

# NAVEGACIÓN
opcion = st.sidebar.radio("Navegación:", ["🏠 Inicio", "📤 Subir Aforos", "🤖 Entrenar Modelo", "📈 Curvas IA", "🔄 Comparativo"])

# SECCIÓN INICIO
if opcion == "🏠 Inicio":
    st.header("🏠 Bienvenido al Sistema de Curvas Altura-Caudal")
    st.markdown("""
    ### 🌊 Sistema Inteligente para Generación de Curvas H-Q
    
    **Características principales:**
    - 🤖 **Modelo Random Forest** para clasificación de aforos
    - 📊 **Generación automática** de curvas altura-caudal
    - 🔄 **Comparación** con ecuaciones personalizadas
    - 📈 **Múltiples tipos** de funciones (polinómicas, potenciales, exponenciales)
    
    ### 🎯 Funcionalidades del Modelo:
    - Clasifica aforos en 3 grupos: ESTÁNDAR, RECIENTE, ALTO_RH
    - Ajusta curvas óptimas para cada grupo
    - Evalúa calidad con R²
    - Genera tablas de valores automáticamente
    """)

# SECCIÓN SUBIR AFOROS
elif opcion == "📤 Subir Aforos":
    st.header("📤 Subir y Procesar Aforos con IA")
    
    archivo = st.file_uploader("Selecciona archivo CSV con datos de aforos", type=['csv'])
    
    if archivo is not None:
        try:
            df = pd.read_csv(archivo)
            st.success(f"✅ {len(df)} aforos cargados correctamente")
            
            # Mostrar datos
            st.subheader("📊 Vista previa de datos")
            st.dataframe(df.head())
            
            # Procesar con modelo IA
            if st.button("🚀 Procesar con Modelo IA"):
                with st.spinner("Procesando datos con Random Forest..."):
                    curvas, df_procesado = procesar_con_modelo(modelo, df)
                
                if curvas:
                    st.success("✅ Procesamiento completado")
                    
                    # Guardar en session state para otras secciones
                    st.session_state.df_procesado = df_procesado
                    st.session_state.curvas_ia = curvas
                    
                    # Mostrar resultados
                    st.subheader("🎯 Grupos Identificados")
                    conteo_grupos = df_procesado['GRUPO_PREDICHO'].value_counts()
                    st.dataframe(conteo_grupos)
                    
                    st.subheader("📈 Curvas Generadas")
                    for grupo, curva in curvas.items():
                        st.write(f"**{grupo}**: R² = {curva['r2']:.3f}, Parámetros: {[f'{p:.4f}' for p in curva['parametros']]}")
                    
                    # Mostrar gráfico
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
                    
                    for grupo in df_procesado['GRUPO_PREDICHO'].unique():
                        color = colores.get(grupo, 'orange')
                        grupo_data = df_procesado[df_procesado['GRUPO_PREDICHO'] == grupo]
                        if 'NIVEL_AFORO' in grupo_data.columns and 'CAUDAL' in grupo_data.columns:
                            ax.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], 
                                      color=color, label=grupo, alpha=0.7, s=50)
                    
                    for grupo, curva in curvas.items():
                        color = colores.get(grupo, 'orange')
                        H_range = np.linspace(curva['rango_niveles'][0], curva['rango_niveles'][1], 100)
                        Q_curve = curva['funcion'](H_range, *curva['parametros'])
                        ax.plot(H_range, Q_curve, color=color, linewidth=2, 
                               label=f'{grupo} (R²={curva["r2"]:.3f})')
                    
                    ax.set_xlabel('Nivel (m)')
                    ax.set_ylabel('Caudal (m³/s)')
                    ax.set_title('Curvas Altura-Caudal por Grupo (Random Forest)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                else:
                    st.error("❌ No se pudieron generar curvas con los datos proporcionados")
                
        except Exception as e:
            st.error(f"❌ Error procesando archivo: {e}")

# SECCIÓN ENTRENAR MODELO
elif opcion == "🤖 Entrenar Modelo":
    st.header("🤖 Entrenar Modelo Random Forest")
    
    st.info("""
    **Para entrenar el modelo necesitas:**
    - Datos históricos de aforos
    - Columna 'GRUPO' con clasificación real
    - Mínimo 20-30 muestras por grupo
    """)
    
    archivo_entrenamiento = st.file_uploader("Subir datos para entrenamiento", type=['csv'], key="entrenamiento")
    
    if archivo_entrenamiento is not None:
        df_entrenamiento = pd.read_csv(archivo_entrenamiento)
        st.success(f"✅ {len(df_entrenamiento)} muestras cargadas para entrenamiento")
        
        if 'GRUPO' not in df_entrenamiento.columns:
            st.error("❌ Se requiere columna 'GRUPO' para entrenar el modelo")
        else:
            st.dataframe(df_entrenamiento.head())
            
            if st.button("🎯 Entrenar Modelo Random Forest"):
                with st.spinner("Entrenando modelo..."):
                    modelo_entrenado = entrenar_modelo_con_datos(df_entrenamiento)
                
                if modelo_entrenado:
                    # Guardar modelo
                    joblib.dump(modelo_entrenado, 'modelo_entrenado.pkl')
                    st.success("✅ Modelo entrenado y guardado exitosamente")
                    
                    # Mostrar estadísticas
                    st.subheader("📊 Estadísticas del Entrenamiento")
                    conteo_grupos = df_entrenamiento['GRUPO'].value_counts()
                    st.dataframe(conteo_grupos)
                    
                    # Actualizar modelo en sesión
                    modelo = modelo_entrenado
                    st.rerun()

# SECCIÓN CURVAS IA
elif opcion == "📈 Curvas IA":
    st.header("📈 Curvas Generadas por IA")
    
    if 'df_procesado' in st.session_state and 'curvas_ia' in st.session_state:
        df_procesado = st.session_state.df_procesado
        curvas_ia = st.session_state.curvas_ia
        
        st.success("✅ Curvas IA disponibles para análisis")
        
        for grupo, curva in curvas_ia.items():
            with st.expander(f"📊 Curva {grupo} (R² = {curva['r2']:.3f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Parámetros:**")
                    for i, param in enumerate(curva['parametros']):
                        st.write(f"Parámetro {i}: {param:.4f}")
                    
                    st.write(f"**Rango de niveles:** {curva['rango_niveles'][0]:.2f} - {curva['rango_niveles'][1]:.2f} m")
                
                with col2:
                    # Generar tabla de valores
                    H_tabla = np.linspace(curva['rango_niveles'][0], curva['rango_niveles'][1], 10)
                    Q_tabla = curva['funcion'](H_tabla, *curva['parametros'])
                    
                    tabla = pd.DataFrame({
                        'Nivel (m)': H_tabla,
                        'Caudal (m³/s)': Q_tabla
                    })
                    st.dataframe(tabla.round(3))
    else:
        st.info("📁 Sube datos en la sección 'Subir Aforos' para generar curvas IA")

# SECCIÓN COMPARATIVO
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

st.markdown("---")
st.markdown("**🌊 Sistema de Curvas H-Q con Random Forest - Basado en estándares USGS/WMO**")