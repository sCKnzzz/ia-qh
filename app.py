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

# ... (las otras funciones se mantienen igual: evaluar_ecuacion_usuario, crear_grafico_comparativo, etc.)

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

# ... (el resto del código de navegación se mantiene igual)

# SECCIÓN SUBIR AFOROS (actualizada)
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

# ... (el resto de las secciones se mantienen igual)