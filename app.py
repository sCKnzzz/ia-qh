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
        
        # ESTIMAR PERÍMETRO SI NO ESTÁ PRESENTE
        if 'PERIMETRO' not in df_procesado.columns or df_procesado['PERIMETRO'].isna().any():
            df_procesado['PERIMETRO'] = self._estimar_perimetro(df_procesado['AREA'], df_procesado['ANCHO_RIO'])
        
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
    
    def _estimar_perimetro(self, area, ancho_rio):
        """Estimar perímetro en función del área y ancho del río"""
        # Fórmula empírica: P ≈ 2*tirante + ancho
        # donde tirante = area / ancho
        tirante = area / ancho_rio
        return 2 * tirante + ancho_rio
    
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
    
    def predecir_curvas(self, nuevos_datos, incluir_alto_rh=True):
        """Predecir curvas para nuevos datos"""
        if self.clasificador is None:
            raise ValueError("El modelo debe ser entrenado primero")
        
        df_procesado = self._preparar_datos_reales(nuevos_datos)
        X = df_procesado[self.features]
        X_scaled = self.escalador.transform(X)
        
        # Predecir grupos
        grupos_pred = self.clasificador.predict(X_scaled)
        df_procesado['GRUPO_PREDICHO'] = grupos_pred
        
        # Filtrar grupos si no se incluye GRUPO_ALTO_RH
        if not incluir_alto_rh:
            df_procesado = df_procesado[df_procesado['GRUPO_PREDICHO'] != 'GRUPO_ALTO_RH']
            st.info(f"⚠️ Se excluyó GRUPO_ALTO_RH. Aforos restantes: {len(df_procesado)}")
        
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

# FUNCIONES PARA GRÁFICOS PROFESIONALES (TAMAÑO REDUCIDO A LA MITAD)
def crear_grafico_profesional(ax, x, y, xlabel, ylabel, titulo, color='blue', marca='o', tamano_marca=40):
    """Crear gráfico profesional con estilo consistente - TAMAÑO REDUCIDO"""
    ax.scatter(x, y, color=color, marker=marca, s=tamano_marca, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.set_title(titulo, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

def agregar_ecuacion_grafico(ax, x_pos, y_pos, ecuacion, fontsize=8):
    """Agregar ecuación al gráfico"""
    ax.text(x_pos, y_pos, ecuacion, fontsize=fontsize, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
            verticalalignment='top')

def crear_graficos_complementarios(df_procesado, curvas):
    """Crear gráficos complementarios de relaciones hidráulicas - TAMAÑO REDUCIDO"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Relaciones Hidráulicas - Análisis Complementario', fontsize=12, fontweight='bold')
    
    # Gráfico 1: Altura vs Velocidad
    ax1 = axes[0, 0]
    crear_grafico_profesional(ax1, df_procesado['NIVEL_AFORO'], df_procesado['VELOCIDAD'],
                             'Nivel (m)', 'Velocidad (m/s)', 'Altura vs Velocidad', color='green')
    
    # Ajustar curva para altura-velocidad
    try:
        H_vel = df_procesado['NIVEL_AFORO'].values
        V_vel = df_procesado['VELOCIDAD'].values
        sort_idx = np.argsort(H_vel)
        params_vel, _ = curve_fit(func_poly2, H_vel[sort_idx], V_vel[sort_idx])
        H_range_vel = np.linspace(min(H_vel), max(H_vel), 100)
        V_curve = func_poly2(H_range_vel, *params_vel)
        ax1.plot(H_range_vel, V_curve, 'green', linewidth=2, alpha=0.8)
        ecuacion_vel = f'V = {params_vel[0]:.3f}H² + {params_vel[1]:.3f}H + {params_vel[2]:.3f}'
        agregar_ecuacion_grafico(ax1, 0.05, 0.95, ecuacion_vel)
    except:
        pass
    
    # Gráfico 2: Altura vs Área
    ax2 = axes[0, 1]
    crear_grafico_profesional(ax2, df_procesado['NIVEL_AFORO'], df_procesado['AREA'],
                             'Nivel (m)', 'Área (m²)', 'Altura vs Área', color='red')
    
    # Ajustar curva para altura-área
    try:
        H_area = df_procesado['NIVEL_AFORO'].values
        A_area = df_procesado['AREA'].values
        sort_idx = np.argsort(H_area)
        params_area, _ = curve_fit(func_poly2, H_area[sort_idx], A_area[sort_idx])
        H_range_area = np.linspace(min(H_area), max(H_area), 100)
        A_curve = func_poly2(H_range_area, *params_area)
        ax2.plot(H_range_area, A_curve, 'red', linewidth=2, alpha=0.8)
        ecuacion_area = f'A = {params_area[0]:.3f}H² + {params_area[1]:.3f}H + {params_area[2]:.3f}'
        agregar_ecuacion_grafico(ax2, 0.05, 0.95, ecuacion_area)
    except:
        pass
    
    # Gráfico 3: Altura vs Radio Hidráulico
    ax3 = axes[1, 0]
    crear_grafico_profesional(ax3, df_procesado['NIVEL_AFORO'], df_procesado['RADIO_HIDRAULICO'],
                             'Nivel (m)', 'Radio Hidráulico (m)', 'Altura vs Radio Hidráulico', color='purple')
    
    # Gráfico 4: Caudal vs Velocidad
    ax4 = axes[1, 1]
    crear_grafico_profesional(ax4, df_procesado['CAUDAL'], df_procesado['VELOCIDAD'],
                             'Caudal (m³/s)', 'Velocidad (m/s)', 'Caudal vs Velocidad', color='orange')
    
    plt.tight_layout()
    return fig

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

# INFORMACIÓN SOBRE LOS GRUPOS
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Significado de los Grupos")
st.sidebar.info("""
**🔵 GRUPO_ESTANDAR:** 
- Condiciones hidráulicas normales
- Radio hidráulico medio
- Comportamiento típico del río

**🔴 GRUPO_ALTO_RH:**
- Alto radio hidráulico (> 0.6)
- Mayor eficiencia hidráulica
- Menor resistencia al flujo
- *Puede excluirse del análisis*

**🟢 GRUPO_RECIENTE:**
- Aforos más recientes (≥ 2024)
- Condiciones actuales del río
- Posibles cambios morfológicos
""")

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
    - Análisis de relaciones hidráulicas completas
    - Estimación automática de parámetros faltantes
    - **Opción para excluir GRUPO_ALTO_RH del análisis**
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
        st.info("💡 **Nota:** Si no tienes datos de PERIMETRO, el sistema los estimará automáticamente")
        
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
        - PERIMETRO (m) *opcional - se estima si falta*
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
                    'AREA (m2)', 'ANCHO RIO (m)', 'NIVEL DE AFORO (m)'
                ]
                
                columnas_faltantes = [col for col in columnas_requeridas if col not in df_subido.columns]
                
                if columnas_faltantes:
                    st.error(f"❌ Faltan las siguientes columnas: {', '.join(columnas_faltantes)}")
                    st.info("💡 Asegúrate de que los nombres de las columnas coincidan exactamente")
                else:
                    st.success(f"✅ Archivo cargado correctamente - {len(df_subido)} aforos encontrados")
                    
                    # Verificar si falta perímetro
                    if 'PERIMETRO (m)' not in df_subido.columns or df_subido['PERIMETRO (m)'].isna().any():
                        st.warning("⚠️ No se encontraron datos de PERIMETRO. Se estimarán automáticamente.")
                    
                    # Mostrar vista previa
                    st.subheader("👀 Vista Previa de los Datos")
                    st.dataframe(df_subido.head())
                    
                    # Mostrar estadísticas básicas
                    st.subheader("📊 Estadísticas Básicas")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Número de aforos", len(df_subido))
                        st.metric("Nivel mínimo", f"{df_subido['NIVEL DE AFORO (m)'].min():.2f} m")
                    with col2:
                        st.metric("Caudal mínimo", f"{df_subido['CAUDAL (m3/s)'].min():.2f} m³/s")
                        st.metric("Caudal máximo", f"{df_subido['CAUDAL (m3/s)'].max():.2f} m³/s")
                    with col3:
                        st.metric("Nivel máximo", f"{df_subido['NIVEL DE AFORO (m)'].max():.2f} m")
                        st.metric("Área promedio", f"{df_subido['AREA (m2)'].mean():.2f} m²")
                    with col4:
                        st.metric("Velocidad promedio", f"{df_subido['VELOCIDAD (m/s)'].mean():.2f} m/s")
                        st.metric("Ancho promedio", f"{df_subido['ANCHO RIO (m)'].mean():.2f} m")
                    
                    # OPCIÓN PARA EXCLUIR GRUPO_ALTO_RH
                    st.subheader("⚙️ Opciones de Análisis")
                    incluir_alto_rh = st.radio(
                        "¿Incluir GRUPO_ALTO_RH en el análisis?",
                        ["Sí", "No"],
                        help="GRUPO_ALTO_RH representa aforos con alto radio hidráulico. Puede excluirse si se consideran atípicos."
                    )
                    
                    incluir_alto_rh_bool = (incluir_alto_rh == "Sí")
                    
                    if not incluir_alto_rh_bool:
                        st.warning("⚠️ Se excluirá GRUPO_ALTO_RH del análisis. Las curvas se recalcularán sin estos aforos.")
                    
                    # Procesar datos
                    if st.button("🚀 Procesar Aforos", type="primary"):
                        with st.spinner("Procesando aforos y generando curvas..."):
                            try:
                                curvas, clasificados = modelo.predecir_curvas(df_subido, incluir_alto_rh_bool)
                                
                                st.success(f"✅ {len(df_subido)} aforos procesados exitosamente")
                                if not incluir_alto_rh_bool:
                                    st.info(f"📊 Aforos utilizados después de excluir GRUPO_ALTO_RH: {len(clasificados)}")
                                
                                # Mostrar resultados de clasificación (COLUMNAS REDUCIDAS)
                                st.subheader("📋 Resultados de Clasificación")
                                # Eliminar columnas no deseadas
                                columnas_a_mostrar = [
                                    'NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 
                                    'ANCHO_RIO', 'PERIMETRO', 'RADIO_HIDRAULICO', 'GRUPO_PREDICHO'
                                ]
                                df_resultados = clasificados[columnas_a_mostrar].copy()
                                # Renombrar columnas para mejor presentación
                                df_resultados.columns = [
                                    'Nivel (m)', 'Caudal (m³/s)', 'Velocidad (m/s)', 'Área (m²)',
                                    'Ancho (m)', 'Perímetro (m)', 'Radio Hidráulico (m)', 'Grupo'
                                ]
                                st.dataframe(df_resultados)
                                
                                # Mostrar distribución de grupos
                                distribucion = clasificados['GRUPO_PREDICHO'].value_counts()
                                st.subheader("📈 Distribución de Grupos")
                                cols = st.columns(4)
                                grupos_mostrados = 0
                                for i, (grupo, count) in enumerate(distribucion.items()):
                                    with cols[i % 4]:
                                        st.metric(f"Grupo {grupo}", count)
                                        grupos_mostrados += 1
                                
                                if curvas:
                                    # GRÁFICO PRINCIPAL MEJORADO - TAMAÑO REDUCIDO
                                    st.subheader("📈 Curvas Altura-Caudal Generadas")
                                    fig_principal, ax_principal = plt.subplots(figsize=(8, 5))
                                    
                                    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
                                    marcadores = {'GRUPO_ALTO_RH': 's', 'GRUPO_RECIENTE': '^', 'GRUPO_ESTANDAR': 'o'}
                                    
                                    for grupo, curva in curvas.items():
                                        color = colores.get(grupo, 'orange')
                                        marcador = marcadores.get(grupo, 'o')
                                        grupo_data = clasificados[clasificados['GRUPO_PREDICHO'] == grupo]
                                        
                                        # Puntos de datos
                                        ax_principal.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], 
                                                          color=color, marker=marcador, s=60, label=grupo,
                                                          alpha=0.8, edgecolors='black', linewidth=0.5)
                                        
                                        # Curva ajustada
                                        H_range = np.linspace(curva['rango_niveles'][0]*0.9, curva['rango_niveles'][1]*1.1, 100)
                                        Q_curve = curva['funcion'](H_range, *curva['parametros'])
                                        ax_principal.plot(H_range, Q_curve, color=color, linewidth=2, 
                                                       label=f"{grupo} (R²={curva['r2']:.3f})")
                                        
                                        # Agregar ecuación al gráfico
                                        if curva['nombre'] == 'Polinómico G2':
                                            a, b, c = curva['parametros']
                                            ecuacion = f'Q = {a:.3f}H² + {b:.3f}H + {c:.3f}'
                                        elif curva['nombre'] == 'Polinómico G3':
                                            a, b, c, d = curva['parametros']
                                            ecuacion = f'Q = {a:.3f}H³ + {b:.3f}H² + {c:.3f}H + {d:.3f}'
                                        elif curva['nombre'] == 'Potencial':
                                            a, b = curva['parametros']
                                            ecuacion = f'Q = {a:.3f}H^{{{b:.3f}}}'
                                        
                                        # Posicionar ecuación en el gráfico
                                        x_pos = curva['rango_niveles'][0] + 0.1
                                        y_pos = curva['rango_caudales'][1] * 0.7
                                        ax_principal.text(x_pos, y_pos, ecuacion, fontsize=8,
                                                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2))
                                    
                                    ax_principal.set_xlabel('Nivel (m)', fontsize=10, fontweight='bold')
                                    ax_principal.set_ylabel('Caudal (m³/s)', fontsize=10, fontweight='bold')
                                    
                                    titulo_grafico = 'Curvas Altura-Caudal por Grupo'
                                    if not incluir_alto_rh_bool:
                                        titulo_grafico += ' (sin GRUPO_ALTO_RH)'
                                    ax_principal.set_title(titulo_grafico, fontsize=11, fontweight='bold')
                                    
                                    ax_principal.legend(fontsize=8)
                                    ax_principal.grid(True, alpha=0.3, linestyle='--')
                                    ax_principal.spines['top'].set_visible(False)
                                    ax_principal.spines['right'].set_visible(False)
                                    st.pyplot(fig_principal)
                                    
                                    # GRÁFICOS COMPLEMENTARIOS
                                    st.subheader("🔍 Análisis de Relaciones Hidráulicas")
                                    fig_complementarios = crear_graficos_complementarios(clasificados, curvas)
                                    st.pyplot(fig_complementarios)
                                    
                                    # Ecuaciones detalladas
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
                                    
                                    # Resumen del análisis
                                    st.subheader("📊 Resumen del Análisis")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Total de aforos procesados", len(clasificados))
                                        st.metric("Número de grupos identificados", len(curvas))
                                    with col2:
                                        if not incluir_alto_rh_bool:
                                            st.metric("GRUPO_ALTO_RH excluido", "Sí")
                                        else:
                                            st.metric("GRUPO_ALTO_RH excluido", "No")
                                    
                                    # Opción para descargar resultados
                                    st.subheader("💾 Descargar Resultados")
                                    resultado_csv = clasificados[columnas_a_mostrar].to_csv(index=False)
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
        # OPCIÓN PARA EXCLUIR GRUPO_ALTO_RH (también en ingreso manual)
        st.subheader("⚙️ Opciones de Análisis")
        incluir_alto_rh = st.radio(
            "¿Incluir GRUPO_ALTO_RH en el análisis?",
            ["Sí", "No"],
            help="GRUPO_ALTO_RH representa aforos con alto radio hidráulico. Puede excluirse si se consideran atípicos."
        )
        
        incluir_alto_rh_bool = (incluir_alto_rh == "Sí")
        
        if not incluir_alto_rh_bool:
            st.warning("⚠️ Se excluirá GRUPO_ALTO_RH del análisis. Las curvas se recalcularán sin estos aforos.")
        
        # Ingreso manual
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
                    'FECHA AFORO': f'{year}-01-01',
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
                curvas, clasificados = modelo.predecir_curvas(df_nuevos, incluir_alto_rh_bool)
                
                st.success(f"✅ {len(df_nuevos)} aforos procesados")
                if not incluir_alto_rh_bool:
                    st.info(f"📊 Aforos utilizados después de excluir GRUPO_ALTO_RH: {len(clasificados)}")
                
                # Mostrar resultados de clasificación (COLUMNAS REDUCIDAS)
                st.subheader("📋 Resultados de Clasificación")
                columnas_a_mostrar = [
                    'NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 
                    'ANCHO_RIO', 'PERIMETRO', 'RADIO_HIDRAULICO', 'GRUPO_PREDICHO'
                ]
                df_resultados = clasificados[columnas_a_mostrar].copy()
                df_resultados.columns = [
                    'Nivel (m)', 'Caudal (m³/s)', 'Velocidad (m/s)', 'Área (m²)',
                    'Ancho (m)', 'Perímetro (m)', 'Radio Hidráulico (m)', 'Grupo'
                ]
                st.dataframe(df_resultados)
                
                if curvas:
                    # Gráfico principal mejorado - TAMAÑO REDUCIDO
                    st.subheader("📈 Curvas Altura-Caudal")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
                    
                    for grupo, curva in curvas.items():
                        color = colores.get(grupo, 'orange')
                        grupo_data = clasificados[clasificados['GRUPO_PREDICHO'] == grupo]
                        ax.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], color=color, s=60, label=grupo)
                        
                        H_range = np.linspace(curva['rango_niveles'][0]*0.9, curva['rango_niveles'][1]*1.1, 100)
                        Q_curve = curva['funcion'](H_range, *curva['parametros'])
                        ax.plot(H_range, Q_curve, color=color, linewidth=2, label=f"{grupo} (R²={curva['r2']:.3f})")
                    
                    ax.set_xlabel('Nivel (m)', fontsize=10)
                    ax.set_ylabel('Caudal (m³/s)', fontsize=10)
                    
                    titulo_grafico = 'Curvas Altura-Caudal por Grupo'
                    if not incluir_alto_rh_bool:
                        titulo_grafico += ' (sin GRUPO_ALTO_RH)'
                    ax.set_title(titulo_grafico, fontsize=11)
                    
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Gráficos complementarios
                    st.subheader("🔍 Análisis de Relaciones Hidráulicas")
                    fig_comp = crear_graficos_complementarios(clasificados, curvas)
                    st.pyplot(fig_comp)
                    
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
st.markdown("**IA para generar Curvas H-Q**")