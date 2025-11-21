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

# FUNCIÓN PARA FILTRAR DATOS Y GENERAR CURVAS (SEPARADA DEL MODELO)
def predecir_curvas_con_filtro(modelo, nuevos_datos, incluir_alto_rh=True):
    """Función separada para predecir curvas con opción de filtrado"""
    if modelo.clasificador is None:
        raise ValueError("El modelo debe ser entrenado primero")
    
    # Preparar datos reales (usando el método del modelo)
    df_procesado = modelo._preparar_datos_reales(nuevos_datos)
    X = df_procesado[modelo.features]
    X_scaled = modelo.escalador.transform(X)
    
    # Predecir grupos
    grupos_pred = modelo.clasificador.predict(X_scaled)
    df_procesado['GRUPO_PREDICHO'] = grupos_pred
    
    # Filtrar grupos si no se incluye GRUPO_ALTO_RH
    if not incluir_alto_rh:
        df_filtrado = df_procesado[df_procesado['GRUPO_PREDICHO'] != 'GRUPO_ALTO_RH'].copy()
        st.info(f"⚠️ Se excluyó GRUPO_ALTO_RH. Aforos restantes: {len(df_filtrado)}")
    else:
        df_filtrado = df_procesado.copy()
    
    # Generar curvas por grupo
    resultados = {}
    for grupo in df_filtrado['GRUPO_PREDICHO'].unique():
        grupo_data = df_filtrado[df_filtrado['GRUPO_PREDICHO'] == grupo]
        if len(grupo_data) >= 3:
            curva = ajustar_curva(grupo_data)
            if curva:
                resultados[grupo] = curva
    
    return resultados, df_filtrado

def ajustar_curva(datos_grupo):
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
                    
                    # Procesar datos inicialmente (siempre incluyendo todos los grupos)
                    if st.button("🚀 Procesar Aforos", type="primary"):
                        with st.spinner("Procesando aforos y generando curvas..."):
                            try:
                                # Procesamiento inicial incluyendo todos los grupos
                                curvas, clasificados = predecir_curvas_con_filtro(modelo, df_subido, incluir_alto_rh=True)
                                
                                st.success(f"✅ {len(df_subido)} aforos procesados exitosamente")
                                
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
                                
                                # VERIFICAR SI EXISTE GRUPO_ALTO_RH Y OFRECER OPCIÓN DE EXCLUSIÓN
                                if 'GRUPO_ALTO_RH' in clasificados['GRUPO_PREDICHO'].values:
                                    st.subheader("⚙️ Opción de Re-análisis")
                                    st.info("""
                                    **Se detectó GRUPO_ALTO_RH en los datos.** 
                                    Este grupo representa aforos con alto radio hidráulico que pueden considerarse atípicos.
                                    ¿Deseas recalcular las curvas excluyendo este grupo?
                                    """)
                                    
                                    if st.button("🔄 Recalcular excluyendo GRUPO_ALTO_RH"):
                                        with st.spinner("Recalculando curvas sin GRUPO_ALTO_RH..."):
                                            try:
                                                # Recalcular excluyendo GRUPO_ALTO_RH
                                                curvas_sin_alto_rh, clasificados_sin_alto_rh = predecir_curvas_con_filtro(modelo, df_subido, incluir_alto_rh=False)
                                                
                                                st.success(f"✅ Re-cálculo exitoso. Aforos utilizados: {len(clasificados_sin_alto_rh)}")
                                                
                                                # Mostrar nueva distribución
                                                st.subheader("📈 Nueva Distribución de Grupos (sin GRUPO_ALTO_RH)")
                                                distribucion_nueva = clasificados_sin_alto_rh['GRUPO_PREDICHO'].value_counts()
                                                cols_nuevos = st.columns(4)
                                                for i, (grupo, count) in enumerate(distribucion_nueva.items()):
                                                    with cols_nuevos[i % 4]:
                                                        st.metric(f"Grupo {grupo}", count)
                                                
                                                if curvas_sin_alto_rh:
                                                    # GRÁFICO PRINCIPAL SIN GRUPO_ALTO_RH
                                                    st.subheader("📈 Curvas Altura-Caudal (sin GRUPO_ALTO_RH)")
                                                    fig_principal, ax_principal = plt.subplots(figsize=(8, 5))
                                                    
                                                    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
                                                    marcadores = {'GRUPO_ALTO_RH': 's', 'GRUPO_RECIENTE': '^', 'GRUPO_ESTANDAR': 'o'}
                                                    
                                                    for grupo, curva in curvas_sin_alto_rh.items():
                                                        color = colores.get(grupo, 'orange')
                                                        marcador = marcadores.get(grupo, 'o')
                                                        grupo_data = clasificados_sin_alto_rh[clasificados_sin_alto_rh['GRUPO_PREDICHO'] == grupo]
                                                        
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
                                                    ax_principal.set_title('Curvas Altura-Caudal (sin GRUPO_ALTO_RH)', fontsize=11, fontweight='bold')
                                                    ax_principal.legend(fontsize=8)
                                                    ax_principal.grid(True, alpha=0.3, linestyle='--')
                                                    ax_principal.spines['top'].set_visible(False)
                                                    ax_principal.spines['right'].set_visible(False)
                                                    st.pyplot(fig_principal)
                                                    
                                                    # Ecuaciones detalladas sin GRUPO_ALTO_RH
                                                    st.subheader("📐 Ecuaciones de las Curvas (sin GRUPO_ALTO_RH)")
                                                    for grupo, curva in curvas_sin_alto_rh.items():
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
                                                
                                                else:
                                                    st.warning("⚠️ No se pudieron generar curvas después de excluir GRUPO_ALTO_RH")
                                                    
                                            except Exception as e:
                                                st.error(f"❌ Error al recalcular los datos: {str(e)}")
                                
                                # MOSTRAR RESULTADOS INICIALES (con todos los grupos)
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
                                    ax_principal.set_title('Curvas Altura-Caudal por Grupo', fontsize=11, fontweight='bold')
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

# ... (el resto del código para las otras secciones se mantiene similar)

st.markdown("---")
st.markdown("**IA para generar Curvas H-Q**")