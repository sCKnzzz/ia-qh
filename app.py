import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import io

# DEFINIR FUNCIONES GLOBALES (CRÍTICO para serialización)
def func_poly2(x, a, b, c):
    return a * x**2 + b * x + c

def func_poly3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def func_pot(x, a, b):
    return a * x**b

class SistemaCurvasAlturaCaudal:
    def __init__(self):
        self.clasificador = None
        self.escalador = None
        self.modelos_por_grupo = {}
        self.features = [
            'NIVEL_AFORO', 'ANCHO_RIO', 'PERIMETRO', 
            'AREA', 'VELOCIDAD', 'RADIO_HIDRAULICO', 
            'TIRANTE_MEDIO', 'CAUDAL_AREA', 'YEAR'
        ]
    
    def entrenar_con_datos_reales(self, df_real):
        """Entrenar el modelo con datos reales de Talapalca"""
        print("🏗️ Entrenando modelo con datos reales...")
        
        # Preparar datos reales
        df_procesado = self._preparar_datos_reales(df_real)
        
        # Crear variable objetivo
        df_procesado['GRUPO'] = self._asignar_grupos_reales(df_procesado)
        
        # Entrenar clasificador
        X = df_procesado[self.features]
        y = df_procesado['GRUPO']
        
        print(f"📊 Datos para entrenamiento: {X.shape}")
        print(f"🎯 Distribución de grupos: {y.value_counts().to_dict()}")
        
        self.escalador = StandardScaler()
        X_scaled = self.escalador.fit_transform(X)
        
        self.clasificador = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.clasificador.fit(X_scaled, y)
        
        # Entrenar modelos por grupo
        self._entrenar_modelos_por_grupo(df_procesado)
        
        print("✅ Modelo entrenado exitosamente con datos reales")
        
        return True
    
    def _preparar_datos_reales(self, df):
        """Preparar datos reales de Talapalca"""
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
        
        # Calcular variables derivadas
        df_procesado['RADIO_HIDRAULICO'] = df_procesado['AREA'] / df_procesado['PERIMETRO']
        df_procesado['TIRANTE_MEDIO'] = df_procesado['AREA'] / df_procesado['ANCHO_RIO']
        df_procesado['CAUDAL_AREA'] = df_procesado['CAUDAL'] / df_procesado['AREA']
        
        # Extraer año de la fecha
        if 'FECHA' in df_procesado.columns:
            try:
                df_procesado['FECHA'] = pd.to_datetime(df_procesado['FECHA'], errors='coerce')
                df_procesado['YEAR'] = df_procesado['FECHA'].dt.year.fillna(2024).astype(int)
            except:
                df_procesado['YEAR'] = 2024
        else:
            df_procesado['YEAR'] = 2024
        
        return df_procesado
    
    def _asignar_grupos_reales(self, df):
        """Asignar grupos basados en análisis de datos reales"""
        grupos = []
        for _, row in df.iterrows():
            if row['RADIO_HIDRAULICO'] > 0.6:
                grupos.append('GRUPO_ALTO_RH')
            elif row['YEAR'] >= 2024:
                grupos.append('GRUPO_RECIENTE')
            else:
                grupos.append('GRUPO_ESTANDAR')
        return grupos
    
    def predecir_curvas(self, nuevos_datos):
        """Predecir curvas para nuevos datos"""
        if self.clasificador is None:
            raise ValueError("El modelo debe ser entrenado primero")
        
        df_procesado = self._preparar_datos_reales(nuevos_datos)
        X = df_procesado[self.features]
        X_scaled = self.escalador.transform(X)
        
        # Predecir grupos
        grupos_pred = self.clasificador.predict(X_scaled)
        df_procesado['GRUPO_PREDICHO'] = grupos_pred
        
        # Generar curvas por grupo
        resultados = {}
        for grupo in df_procesado['GRUPO_PREDICHO'].unique():
            grupo_data = df_procesado[df_procesado['GRUPO_PREDICHO'] == grupo]
            if len(grupo_data) >= 3:
                curva = self._ajustar_curva(grupo_data)
                if curva:
                    resultados[grupo] = curva
        
        return resultados, df_procesado
    
    def _entrenar_modelos_por_grupo(self, df):
        """Entrenar modelos de regresión por grupo"""
        for grupo in df['GRUPO'].unique():
            grupo_data = df[df['GRUPO'] == grupo]
            if len(grupo_data) >= 3:
                curva = self._ajustar_curva(grupo_data)
                if curva:
                    self.modelos_por_grupo[grupo] = curva
    
    def _ajustar_curva(self, datos_grupo):
        """Ajustar curva altura-caudal para un grupo"""
        H = datos_grupo['NIVEL_AFORO'].values
        Q = datos_grupo['CAUDAL'].values
        
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
            except Exception as e:
                continue
        
        return mejor_modelo

# Configuración de la aplicación Streamlit
st.set_page_config(page_title="Sistema Talapalca", page_icon="🌊", layout="wide")
st.title("🌊 Sistema Inteligente de Curvas Altura-Caudal - TALAPALCA")
st.markdown("**Modelo entrenado con 34 aforos reales**")

# Cargar modelo con manejo de errores mejorado
@st.cache_resource
def cargar_modelo():
    try:
        modelo = joblib.load('modelo_talapalca_entrenado.pkl')
        st.success("✅ Modelo cargado correctamente")
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("💡 Asegúrate de que el archivo 'modelo_talapalca_entrenado.pkl' esté en el repositorio")
        return None

modelo = cargar_modelo()

# Navegación
opcion = st.sidebar.radio("Navegación:", ["🏠 Inicio", "📤 Subir Aforos", "📊 Ingreso Manual", "📈 Curvas"])

if opcion == "🏠 Inicio":
    st.header("Bienvenido al Sistema Talapalca")
    st.info("""
    **🎯 Funcionalidades:**
    - Clasificación automática de aforos
    - Generación de curvas altura-caudal
    - Modelo entrenado con datos reales
    - Interfaz fácil de usar
    - Soporte para carga de archivos CSV
    """)
    
    # Mostrar datos de ejemplo
    try:
        datos_demo = pd.read_csv('datos_talapalca_demo.csv')
        st.subheader("📋 Datos de Ejemplo")
        st.dataframe(datos_demo.head(6))
        st.write(f"**Total de aforos:** {len(datos_demo)}")
        
        # Mostrar formato esperado
        st.subheader("📝 Formato Esperado para Archivos CSV")
        formato_ejemplo = pd.DataFrame({
            'FECHA AFORO': ['2024-01-15', '2024-02-20', '2024-03-10'],
            'CAUDAL (m3/s)': [2.5, 3.1, 1.8],
            'VELOCIDAD (m/s)': [0.8, 0.9, 0.6],
            'AREA (m2)': [3.1, 3.4, 3.0],
            'PERIMETRO (m)': [8.5, 8.7, 8.3],
            'ANCHO RIO (m)': [8.2, 8.5, 8.0],
            'NIVEL DE AFORO (m)': [1.2, 1.3, 1.1]
        })
        st.dataframe(formato_ejemplo)
        
    except Exception as e:
        st.warning("No se encontraron datos de demo")

elif opcion == "📤 Subir Aforos":
    st.header("📤 Subir Archivo de Aforos")
    
    if modelo is None:
        st.error("⚠️ El modelo no está disponible. No se pueden procesar datos.")
    else:
        st.info("""
        **📋 Formato requerido para el archivo CSV:**
        - FECHA AFORO (ej: 2024-01-15)
        - CAUDAL (m3/s)
        - VELOCIDAD (m/s) 
        - AREA (m2)
        - PERIMETRO (m)
        - ANCHO RIO (m)
        - NIVEL DE AFORO (m)
        """)
        
        # Subir archivo
        archivo_subido = st.file_uploader(
            "Selecciona tu archivo CSV", 
            type=['csv'],
            help="El archivo debe tener las columnas especificadas arriba"
        )
        
        if archivo_subido is not None:
            try:
                # Leer el archivo
                df_subido = pd.read_csv(archivo_subido)
                
                # Verificar columnas requeridas
                columnas_requeridas = [
                    'FECHA AFORO', 'CAUDAL (m3/s)', 'VELOCIDAD (m/s)', 
                    'AREA (m2)', 'PERIMETRO (m)', 'ANCHO RIO (m)', 'NIVEL DE AFORO (m)'
                ]
                
                columnas_faltantes = [col for col in columnas_requeridas if col not in df_subido.columns]
                
                if columnas_faltantes:
                    st.error(f"❌ Faltan las siguientes columnas: {', '.join(columnas_faltantes)}")
                    st.info("💡 Asegúrate de que los nombres de las columnas coincidan exactamente")
                else:
                    st.success(f"✅ Archivo cargado correctamente - {len(df_subido)} aforos encontrados")
                    
                    # Mostrar vista previa
                    st.subheader("👀 Vista Previa de los Datos")
                    st.dataframe(df_subido.head())
                    
                    # Mostrar estadísticas básicas
                    st.subheader("📊 Estadísticas Básicas")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Número de aforos", len(df_subido))
                        st.metric("Nivel mínimo", f"{df_subido['NIVEL DE AFORO (m)'].min():.2f} m")
                    with col2:
                        st.metric("Caudal mínimo", f"{df_subido['CAUDAL (m3/s)'].min():.2f} m³/s")
                        st.metric("Caudal máximo", f"{df_subido['CAUDAL (m3/s)'].max():.2f} m³/s")
                    with col3:
                        st.metric("Nivel máximo", f"{df_subido['NIVEL DE AFORO (m)'].max():.2f} m")
                        st.metric("Área promedio", f"{df_subido['AREA (m2)'].mean():.2f} m²")
                    
                    # Procesar datos
                    if st.button("🚀 Procesar Aforos", type="primary"):
                        with st.spinner("Procesando aforos y generando curvas..."):
                            try:
                                curvas, clasificados = modelo.predecir_curvas(df_subido)
                                
                                st.success(f"✅ {len(df_subido)} aforos procesados exitosamente")
                                
                                # Mostrar resultados de clasificación
                                st.subheader("📋 Resultados de Clasificación")
                                st.dataframe(clasificados)
                                
                                # Mostrar distribución de grupos
                                distribucion = clasificados['GRUPO_PREDICHO'].value_counts()
                                st.subheader("📈 Distribución de Grupos")
                                col1, col2, col3 = st.columns(3)
                                for i, (grupo, count) in enumerate(distribucion.items()):
                                    with [col1, col2, col3][i % 3]:
                                        st.metric(f"Grupo {grupo}", count)
                                
                                if curvas:
                                    # Gráfico
                                    st.subheader("📈 Curvas Altura-Caudal Generadas")
                                    fig, ax = plt.subplots(figsize=(12, 8))
                                    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
                                    
                                    for grupo, curva in curvas.items():
                                        color = colores.get(grupo, 'orange')
                                        grupo_data = clasificados[clasificados['GRUPO_PREDICHO'] == grupo]
                                        ax.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], 
                                                  color=color, s=100, label=grupo, alpha=0.7)
                                        
                                        H_range = np.linspace(curva['rango_niveles'][0]*0.9, curva['rango_niveles'][1]*1.1, 100)
                                        Q_curve = curva['funcion'](H_range, *curva['parametros'])
                                        ax.plot(H_range, Q_curve, color=color, linewidth=3, 
                                                label=f"{grupo} (R²={curva['r2']:.3f})")
                                    
                                    ax.set_xlabel('Nivel (m)', fontsize=12)
                                    ax.set_ylabel('Caudal (m³/s)', fontsize=12)
                                    ax.set_title('Curvas Altura-Caudal por Grupo', fontsize=14)
                                    ax.legend(fontsize=10)
                                    ax.grid(True, alpha=0.3)
                                    st.pyplot(fig)
                                    
                                    # Ecuaciones
                                    st.subheader("📐 Ecuaciones de las Curvas Generadas")
                                    for grupo, curva in curvas.items():
                                        with st.expander(f"📊 {grupo} - {curva['nombre']} (R² = {curva['r2']:.3f})"):
                                            if curva['nombre'] == 'Polinómico G2':
                                                a, b, c = curva['parametros']
                                                st.latex(f"Q = {a:.4f}H^2 + {b:.4f}H + {c:.4f}")
                                            elif curva['nombre'] == 'Polinómico G3':
                                                a, b, c, d = curva['parametros']
                                                st.latex(f"Q = {a:.4f}H^3 + {b:.4f}H^2 + {c:.4f}H + {d:.4f}")
                                            elif curva['nombre'] == 'Potencial':
                                                a, b = curva['parametros']
                                                st.latex(f"Q = {a:.4f}H^{{{b:.4f}}}")
                                            
                                            st.write(f"**Puntos utilizados:** {curva['n_puntos']}")
                                            st.write(f"**Rango de niveles:** {curva['rango_niveles'][0]:.3f} - {curva['rango_niveles'][1]:.3f} m")
                                            st.write(f"**Rango de caudales:** {curva['rango_caudales'][0]:.3f} - {curva['rango_caudales'][1]:.3f} m³/s")
                                    
                                    # Opción para descargar resultados
                                    st.subheader("💾 Descargar Resultados")
                                    resultado_csv = clasificados.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Descargar Resultados en CSV",
                                        data=resultado_csv,
                                        file_name="resultados_aforos_talapalca.csv",
                                        mime="text/csv"
                                    )
                                    
                                else:
                                    st.warning("⚠️ No se pudieron generar curvas con los datos proporcionados")
                                    st.info("💡 Se necesitan al menos 3 aforos por grupo para generar curvas")
                                    
                            except Exception as e:
                                st.error(f"❌ Error al procesar los datos: {str(e)}")
                                st.info("💡 Verifica que los datos estén en el formato correcto y tengan valores válidos")
            
            except Exception as e:
                st.error(f"❌ Error al leer el archivo: {str(e)}")
                st.info("💡 Asegúrate de que el archivo sea un CSV válido")

elif opcion == "📊 Ingreso Manual":
    st.header("📊 Ingreso Manual de Aforos")
    
    if modelo is None:
        st.error("⚠️ El modelo no está disponible. No se pueden procesar datos.")
    else:
        # Ingreso manual (código anterior que ya tenías)
        num_aforos = st.number_input("Número de aforos:", 1, 20, 3)
        nuevos_datos = []
        
        for i in range(num_aforos):
            with st.expander(f"Aforo {i+1}"):
                cols = st.columns(3)
                with cols[0]:
                    nivel = st.number_input("Nivel (m)", 0.1, 10.0, 1.0, key=f"n{i}")
                    caudal = st.number_input("Caudal (m³/s)", 0.1, 50.0, 2.0, key=f"q{i}")
                with cols[1]:
                    area = st.number_input("Área (m²)", 0.1, 50.0, 3.0, key=f"a{i}")
                    ancho = st.number_input("Ancho (m)", 0.1, 20.0, 8.0, key=f"w{i}")
                with cols[2]:
                    perimetro = st.number_input("Perímetro (m)", 0.1, 30.0, 8.5, key=f"p{i}")
                    velocidad = st.number_input("Velocidad (m/s)", 0.1, 5.0, 0.7, key=f"v{i}")
                    year = st.number_input("Año", 2000, 2030, 2024, key=f"y{i}")
                
                nuevos_datos.append({
                    'FECHA AFORO': f'{year}-01-01',  # Fecha por defecto
                    'NIVEL DE AFORO (m)': nivel, 
                    'CAUDAL (m3/s)': caudal, 
                    'AREA (m2)': area,
                    'ANCHO RIO (m)': ancho, 
                    'PERIMETRO (m)': perimetro, 
                    'VELOCIDAD (m/s)': velocidad
                })
        
        if st.button("🚀 Procesar") and nuevos_datos:
            try:
                df_nuevos = pd.DataFrame(nuevos_datos)
                curvas, clasificados = modelo.predecir_curvas(df_nuevos)
                
                st.success(f"✅ {len(df_nuevos)} aforos procesados")
                st.subheader("📊 Resultados de Clasificación")
                st.dataframe(clasificados)
                
                if curvas:
                    # Gráfico y ecuaciones (código anterior)
                    st.subheader("📈 Curvas Altura-Caudal")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
                    
                    for grupo, curva in curvas.items():
                        color = colores.get(grupo, 'orange')
                        grupo_data = clasificados[clasificados['GRUPO_PREDICHO'] == grupo]
                        ax.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], color=color, s=80, label=grupo)
                        
                        H_range = np.linspace(curva['rango_niveles'][0]*0.9, curva['rango_niveles'][1]*1.1, 100)
                        Q_curve = curva['funcion'](H_range, *curva['parametros'])
                        ax.plot(H_range, Q_curve, color=color, linewidth=2, label=f"{grupo} (R²={curva['r2']:.3f})")
                    
                    ax.set_xlabel('Nivel (m)')
                    ax.set_ylabel('Caudal (m³/s)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Ecuaciones
                    st.subheader("📐 Ecuaciones de las Curvas")
                    for grupo, curva in curvas.items():
                        with st.expander(f"Ecuación {grupo}"):
                            if curva['nombre'] == 'Polinómico G2':
                                a, b, c = curva['parametros']
                                st.latex(f"Q = {a:.4f}H^2 + {b:.4f}H + {c:.4f}")
                            elif curva['nombre'] == 'Polinómico G3':
                                a, b, c, d = curva['parametros']
                                st.latex(f"Q = {a:.4f}H^3 + {b:.4f}H^2 + {c:.4f}H + {d:.4f}")
                            elif curva['nombre'] == 'Potencial':
                                a, b = curva['parametros']
                                st.latex(f"Q = {a:.4f}H^{{{b:.4f}}}")
                            st.write(f"**R²:** {curva['r2']:.3f}")
                            st.write(f"**Puntos utilizados:** {curva['n_puntos']}")
                else:
                    st.warning("⚠️ No se pudieron generar curvas con los datos proporcionados")
                    
            except Exception as e:
                st.error(f"❌ Error al procesar datos: {str(e)}")

elif opcion == "📈 Curvas":
    st.header("Curvas del Modelo Actual")
    if modelo and hasattr(modelo, 'modelos_por_grupo') and modelo.modelos_por_grupo:
        for grupo, curva in modelo.modelos_por_grupo.items():
            with st.expander(f"{grupo} (R² = {curva['r2']:.3f})"):
                st.write(f"**Modelo:** {curva['nombre']}")
                st.write(f"**Puntos de entrenamiento:** {curva['n_puntos']}")
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
        st.info("ℹ️ No hay curvas disponibles o el modelo no está cargado")

st.markdown("---")
st.markdown("**Sistema Hidráulico Inteligente - Estación Talapalca**")