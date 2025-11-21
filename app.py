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

def func_exp(x, a, b):
    return a * np.exp(b * x)

def func_log(x, a, b):
    return a * np.log(x + b)

# FUNCIÓN PARA DETECTAR INTERSECCIONES ENTRE CURVAS
def encontrar_interseccion(curva1, curva2, rango_min, rango_max, tolerancia=0.01):
    """Encontrar punto de intersección entre dos curvas"""
    def diferencia(H):
        return curva1['funcion'](H, *curva1['parametros']) - curva2['funcion'](H, *curva2['parametros'])
    
    # Buscar intersección en el rango común
    H_test = np.linspace(rango_min, rango_max, 1000)
    diferencias = diferencia(H_test)
    
    # Encontrar cambios de signo
    intersecciones = []
    for i in range(len(diferencias)-1):
        if diferencias[i] * diferencias[i+1] <= 0:  # Cambio de signo
            # Refinar búsqueda alrededor del cambio de signo
            H_interseccion = H_test[i]
            try:
                from scipy.optimize import brentq
                interseccion = brentq(diferencia, H_test[i], H_test[i+1], xtol=tolerancia)
                intersecciones.append(interseccion)
            except:
                continue
    
    return intersecciones

# FUNCIÓN PARA DEFINIR RANGOS DE VALIDEZ SIN SUPERPOSICIÓN
def definir_rangos_validez(curvas):
    """Definir rangos de validez para curvas que se intersectan"""
    if len(curvas) < 2:
        return curvas
    
    grupos = list(curvas.keys())
    rangos_definidos = {}
    
    # Ordenar grupos por rango mínimo de nivel
    grupos_ordenados = sorted(grupos, key=lambda g: curvas[g]['rango_niveles'][0])
    
    for i, grupo in enumerate(grupos_ordenados):
        curva_actual = curvas[grupo]
        rango_min_actual = curva_actual['rango_niveles'][0]
        rango_max_actual = curva_actual['rango_niveles'][1]
        
        # Buscar intersecciones con curvas posteriores
        punto_quiebre = None
        for j in range(i+1, len(grupos_ordenados)):
            grupo_siguiente = grupos_ordenados[j]
            curva_siguiente = curvas[grupo_siguiente]
            
            intersecciones = encontrar_interseccion(
                curva_actual, curva_siguiente, 
                rango_min_actual, min(rango_max_actual, curva_siguiente['rango_niveles'][1])
            )
            
            if intersecciones:
                punto_quiebre = intersecciones[0]
                break
        
        # Definir rango de validez
        if punto_quiebre:
            # Esta curva es válida hasta el punto de quiebre
            rango_validez = (rango_min_actual, punto_quiebre)
            # La siguiente curva empezará desde el punto de quiebre
            if grupo_siguiente in curvas:
                curvas[grupo_siguiente]['rango_validez'] = (punto_quiebre, curvas[grupo_siguiente]['rango_niveles'][1])
        else:
            # No hay intersección, usar rango completo
            rango_validez = (rango_min_actual, rango_max_actual)
        
        rangos_definidos[grupo] = rango_validez
    
    # Actualizar curvas con rangos definidos
    for grupo, rango in rangos_definidos.items():
        curvas[grupo]['rango_validez'] = rango
    
    return curvas

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
        
        # Generar curvas - EXCLUIR GRUPO_ESTANDAR
        resultados = {}
        for grupo in df_filtrado['GRUPO_PREDICHO'].unique():
            # EXCLUIR GRUPO_ESTANDAR
            if grupo == 'GRUPO_ESTANDAR':
                continue
                
            grupo_data = df_filtrado[df_filtrado['GRUPO_PREDICHO'] == grupo]
            if len(grupo_data) >= 3:
                curva = ajustar_curva(grupo_data)
                if curva:
                    resultados[grupo] = curva
        
        # Definir rangos de validez sin superposición
        if len(resultados) >= 2:
            resultados = definir_rangos_validez(resultados)
        else:
            # Para una sola curva, usar rango completo
            for grupo in resultados:
                resultados[grupo]['rango_validez'] = resultados[grupo]['rango_niveles']
        
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

# FUNCIÓN MEJORADA PARA AJUSTAR MODELOS A RELACIONES HIDRÁULICAS
def ajustar_modelo_relacion(x, y, nombre_relacion):
    """Ajustar diferentes modelos y seleccionar el mejor según R²"""
    
    modelos = [
        ('Lineal', lambda x, a, b: a * x + b),
        ('Polinómico G2', func_poly2),
        ('Exponencial', func_exp),
        ('Logarítmico', func_log),
        ('Potencial', func_pot)
    ]
    
    mejor_r2 = -np.inf
    mejor_modelo = None
    mejor_params = None
    
    for nombre, funcion in modelos:
        try:
            if nombre == 'Exponencial':
                params, _ = curve_fit(funcion, x, y, p0=[1.0, 0.1], maxfev=5000)
            elif nombre == 'Logarítmico':
                # Asegurar que x sea positivo para logaritmo
                x_positivo = x + 0.001  # Evitar log(0)
                params, _ = curve_fit(funcion, x_positivo, y, p0=[1.0, 1.0], maxfev=5000)
            elif nombre == 'Potencial':
                params, _ = curve_fit(funcion, x, y, p0=[1.0, 1.0], maxfev=5000)
            else:
                params, _ = curve_fit(funcion, x, y, maxfev=5000)
            
            y_pred = funcion(x, *params)
            r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
            
            if r2 > mejor_r2:
                mejor_r2 = r2
                mejor_modelo = nombre
                mejor_params = params
                mejor_funcion = funcion
                
        except Exception as e:
            continue
    
    return mejor_modelo, mejor_params, mejor_r2, mejor_funcion

# FUNCIÓN PARA FORMATEAR RANGOS CON SÍMBOLOS MATEMÁTICOS
def formatear_rango(rango_min, rango_max):
    """Formatear rango con símbolos matemáticos correctos"""
    return f"{rango_min:.2f} ≤ H ≤ {rango_max:.2f}"

# FUNCIONES PARA GRÁFICOS - MEJORADAS CON RANGOS DE VALIDEZ
def crear_grafico_principal(df, curvas, titulo):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
    marcadores = {'GRUPO_ALTO_RH': 's', 'GRUPO_RECIENTE': '^', 'GRUPO_ESTANDAR': 'o'}
    tamanos = {'GRUPO_ALTO_RH': 100, 'GRUPO_RECIENTE': 80, 'GRUPO_ESTANDAR': 80}
    
    # Primero graficar todos los puntos (EXCLUYENDO GRUPO_ESTANDAR)
    for grupo in df['GRUPO_PREDICHO'].unique():
        # EXCLUIR GRUPO_ESTANDAR
        if grupo == 'GRUPO_ESTANDAR':
            continue
            
        color = colores.get(grupo, 'orange')
        marcador = marcadores.get(grupo, 'o')
        tamano = tamanos.get(grupo, 80)
        grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
        
        # Puntos - hacer GRUPO_ALTO_RH más visible
        alpha = 0.9 if grupo == 'GRUPO_ALTO_RH' else 0.7
        ax.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], 
                  color=color, marker=marcador, s=tamano, label=grupo, alpha=alpha, 
                  edgecolors='black', linewidth=1 if grupo == 'GRUPO_ALTO_RH' else 0.5)
    
    # Luego graficar las curvas con sus rangos de validez
    for grupo, curva in curvas.items():
        # EXCLUIR GRUPO_ESTANDAR
        if grupo == 'GRUPO_ESTANDAR':
            continue
            
        color = colores.get(grupo, 'orange')
        
        # Usar rango de validez si está definido, sino usar rango de niveles
        if 'rango_validez' in curva:
            rango_min, rango_max = curva['rango_validez']
        else:
            rango_min, rango_max = curva['rango_niveles']
        
        # Curva solo en su rango de validez
        H_range = np.linspace(rango_min, rango_max, 100)
        Q_curve = curva['funcion'](H_range, *curva['parametros'])
        
        # Hacer la línea más gruesa para GRUPO_ALTO_RH
        linewidth = 3 if grupo == 'GRUPO_ALTO_RH' else 2
        
        # Agregar información del rango de validez en la etiqueta
        rango_formateado = formatear_rango(rango_min, rango_max)
        label = f"{grupo} (R²={curva['r2']:.3f})\n{rango_formateado} m"
        
        ax.plot(H_range, Q_curve, color=color, linewidth=linewidth, label=label)
    
    ax.set_xlabel('Nivel (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Caudal (m³/s)', fontsize=12, fontweight='bold')
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# FUNCIÓN MEJORADA PARA GRÁFICOS COMPLEMENTARIOS EXCLUYENDO GRUPO_ESTANDAR
def crear_graficos_complementarios(df, titulo_sufijo=""):
    """Crear gráficos complementarios excluyendo GRUPO_ESTANDAR"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Análisis de Relaciones Hidráulicas {titulo_sufijo}', fontsize=16, fontweight='bold')
    
    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
    marcadores = {'GRUPO_ALTO_RH': 's', 'GRUPO_RECIENTE': '^', 'GRUPO_ESTANDAR': 'o'}
    
    # 1. Altura vs Velocidad
    ax1 = axes[0, 0]
    
    # Graficar puntos por grupo (EXCLUYENDO GRUPO_ESTANDAR)
    for grupo in df['GRUPO_PREDICHO'].unique():
        # EXCLUIR GRUPO_ESTANDAR
        if grupo == 'GRUPO_ESTANDAR':
            continue
            
        color = colores.get(grupo, 'orange')
        marcador = marcadores.get(grupo, 'o')
        grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
        
        alpha = 0.9 if grupo == 'GRUPO_ALTO_RH' else 0.7
        tamano = 100 if grupo == 'GRUPO_ALTO_RH' else 60
        
        ax1.scatter(grupo_data['NIVEL_AFORO'], grupo_data['VELOCIDAD'], 
                   color=color, marker=marcador, s=tamano, label=grupo, alpha=alpha,
                   edgecolors='black', linewidth=1 if grupo == 'GRUPO_ALTO_RH' else 0.5)
    
    # Ajustar mejor modelo para todos los datos (sin GRUPO_ESTANDAR)
    df_filtrado = df[df['GRUPO_PREDICHO'] != 'GRUPO_ESTANDAR']
    x_vel = df_filtrado['NIVEL_AFORO'].values
    y_vel = df_filtrado['VELOCIDAD'].values
    modelo_vel, params_vel, r2_vel, funcion_vel = ajustar_modelo_relacion(x_vel, y_vel, "Altura-Velocidad")
    
    # Graficar curva del mejor modelo
    if modelo_vel and r2_vel > 0:
        x_range_vel = np.linspace(min(x_vel), max(x_vel), 100)
        if modelo_vel == 'Logarítmico':
            y_pred_vel = funcion_vel(x_range_vel + 0.001, *params_vel)
        else:
            y_pred_vel = funcion_vel(x_range_vel, *params_vel)
        ax1.plot(x_range_vel, y_pred_vel, 'black', linewidth=2, linestyle='--',
                label=f'{modelo_vel} (R²={r2_vel:.3f})')
    
    ax1.set_xlabel('Nivel (m)', fontweight='bold')
    ax1.set_ylabel('Velocidad (m/s)', fontweight='bold')
    ax1.set_title(f'Altura vs Velocidad\nMejor modelo: {modelo_vel}', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Altura vs Área
    ax2 = axes[0, 1]
    
    for grupo in df['GRUPO_PREDICHO'].unique():
        # EXCLUIR GRUPO_ESTANDAR
        if grupo == 'GRUPO_ESTANDAR':
            continue
            
        color = colores.get(grupo, 'orange')
        marcador = marcadores.get(grupo, 'o')
        grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
        
        alpha = 0.9 if grupo == 'GRUPO_ALTO_RH' else 0.7
        tamano = 100 if grupo == 'GRUPO_ALTO_RH' else 60
        
        ax2.scatter(grupo_data['NIVEL_AFORO'], grupo_data['AREA'], 
                   color=color, marker=marcador, s=tamano, label=grupo, alpha=alpha,
                   edgecolors='black', linewidth=1 if grupo == 'GRUPO_ALTO_RH' else 0.5)
    
    x_area = df_filtrado['NIVEL_AFORO'].values
    y_area = df_filtrado['AREA'].values
    modelo_area, params_area, r2_area, funcion_area = ajustar_modelo_relacion(x_area, y_area, "Altura-Área")
    
    if modelo_area and r2_area > 0:
        x_range_area = np.linspace(min(x_area), max(x_area), 100)
        if modelo_area == 'Logarítmico':
            y_pred_area = funcion_area(x_range_area + 0.001, *params_area)
        else:
            y_pred_area = funcion_area(x_range_area, *params_area)
        ax2.plot(x_range_area, y_pred_area, 'black', linewidth=2, linestyle='--',
                label=f'{modelo_area} (R²={r2_area:.3f})')
    
    ax2.set_xlabel('Nivel (m)', fontweight='bold')
    ax2.set_ylabel('Área (m²)', fontweight='bold')
    ax2.set_title(f'Altura vs Área\nMejor modelo: {modelo_area}', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Altura vs Radio Hidráulico
    ax3 = axes[1, 0]
    
    for grupo in df['GRUPO_PREDICHO'].unique():
        # EXCLUIR GRUPO_ESTANDAR
        if grupo == 'GRUPO_ESTANDAR':
            continue
            
        color = colores.get(grupo, 'orange')
        marcador = marcadores.get(grupo, 'o')
        grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
        
        alpha = 0.9 if grupo == 'GRUPO_ALTO_RH' else 0.7
        tamano = 100 if grupo == 'GRUPO_ALTO_RH' else 60
        
        ax3.scatter(grupo_data['NIVEL_AFORO'], grupo_data['RADIO_HIDRAULICO'], 
                   color=color, marker=marcador, s=tamano, label=grupo, alpha=alpha,
                   edgecolors='black', linewidth=1 if grupo == 'GRUPO_ALTO_RH' else 0.5)
    
    x_rh = df_filtrado['NIVEL_AFORO'].values
    y_rh = df_filtrado['RADIO_HIDRAULICO'].values
    modelo_rh, params_rh, r2_rh, funcion_rh = ajustar_modelo_relacion(x_rh, y_rh, "Altura-Radio Hidráulico")
    
    if modelo_rh and r2_rh > 0:
        x_range_rh = np.linspace(min(x_rh), max(x_rh), 100)
        if modelo_rh == 'Logarítmico':
            y_pred_rh = funcion_rh(x_range_rh + 0.001, *params_rh)
        else:
            y_pred_rh = funcion_rh(x_range_rh, *params_rh)
        ax3.plot(x_range_rh, y_pred_rh, 'black', linewidth=2, linestyle='--',
                label=f'{modelo_rh} (R²={r2_rh:.3f})')
    
    ax3.set_xlabel('Nivel (m)', fontweight='bold')
    ax3.set_ylabel('Radio Hidráulico (m)', fontweight='bold')
    ax3.set_title(f'Altura vs Radio Hidráulico\nMejor modelo: {modelo_rh}', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Caudal vs Velocidad
    ax4 = axes[1, 1]
    
    for grupo in df['GRUPO_PREDICHO'].unique():
        # EXCLUIR GRUPO_ESTANDAR
        if grupo == 'GRUPO_ESTANDAR':
            continue
            
        color = colores.get(grupo, 'orange')
        marcador = marcadores.get(grupo, 'o')
        grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
        
        alpha = 0.9 if grupo == 'GRUPO_ALTO_RH' else 0.7
        tamano = 100 if grupo == 'GRUPO_ALTO_RH' else 60
        
        ax4.scatter(grupo_data['CAUDAL'], grupo_data['VELOCIDAD'], 
                   color=color, marker=marcador, s=tamano, label=grupo, alpha=alpha,
                   edgecolors='black', linewidth=1 if grupo == 'GRUPO_ALTO_RH' else 0.5)
    
    x_caudal = df_filtrado['CAUDAL'].values
    y_caudal = df_filtrado['VELOCIDAD'].values
    modelo_caudal, params_caudal, r2_caudal, funcion_caudal = ajustar_modelo_relacion(x_caudal, y_caudal, "Caudal-Velocidad")
    
    if modelo_caudal and r2_caudal > 0:
        x_range_caudal = np.linspace(min(x_caudal), max(x_caudal), 100)
        if modelo_caudal == 'Logarítmico':
            y_pred_caudal = funcion_caudal(x_range_caudal + 0.001, *params_caudal)
        else:
            y_pred_caudal = funcion_caudal(x_range_caudal, *params_caudal)
        ax4.plot(x_range_caudal, y_pred_caudal, 'black', linewidth=2, linestyle='--',
                label=f'{modelo_caudal} (R²={r2_caudal:.3f})')
    
    ax4.set_xlabel('Caudal (m³/s)', fontweight='bold')
    ax4.set_ylabel('Velocidad (m/s)', fontweight='bold')
    ax4.set_title(f'Caudal vs Velocidad\nMejor modelo: {modelo_caudal}', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# CONFIGURACIÓN STREAMLIT
st.set_page_config(page_title="Sistema Talapalca", page_icon="🌊", layout="wide")
st.title("🌊 IA para la generacion de Curvas Altura-Caudal")
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
opcion = st.sidebar.radio("Navegación:", ["🏠 Inicio", "📤 Subir Aforos", "📊 Ingreso Manual", "📈 Curvas"])

if opcion == "🏠 Inicio":
    st.header("Bienvenido a la IA para curvas H-Q")
    st.info("Aplicacion IA para generar curvas altura-caudal usando IA")
    
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
                    if 'procesamiento_realizado' not in st.session_state:
                        st.session_state.procesamiento_realizado = False
                    if 'curvas_sin_alto_rh' not in st.session_state:
                        st.session_state.curvas_sin_alto_rh = None
                    if 'datos_sin_alto_rh' not in st.session_state:
                        st.session_state.datos_sin_alto_rh = None
                    if 'tiene_alto_rh' not in st.session_state:
                        st.session_state.tiene_alto_rh = False
                    if 'datos_completos' not in st.session_state:
                        st.session_state.datos_completos = None
                    
                    # BOTÓN PRINCIPAL DE PROCESAMIENTO
                    if st.button("🚀 Procesar Aforos", type="primary"):
                        with st.spinner("Procesando datos..."):
                            # PROCESAMIENTO INICIAL - SIN GRUPO_ALTO_RH
                            curvas_sin, datos_sin = procesar_con_modelo(modelo, df, incluir_alto_rh=False)
                            
                            if curvas_sin:
                                st.session_state.procesamiento_realizado = True
                                st.session_state.curvas_sin_alto_rh = curvas_sin
                                st.session_state.datos_sin_alto_rh = datos_sin
                                
                                # Verificar si hay GRUPO_ALTO_RH y guardar datos completos
                                _, datos_completos = procesar_con_modelo(modelo, df, incluir_alto_rh=True)
                                st.session_state.tiene_alto_rh = 'GRUPO_ALTO_RH' in datos_completos['GRUPO_PREDICHO'].values
                                st.session_state.datos_completos = datos_completos
                    
                    # MOSTRAR RESULTADOS SI EL PROCESAMIENTO SE REALIZÓ
                    if st.session_state.procesamiento_realizado and st.session_state.curvas_sin_alto_rh is not None:
                        curvas_sin = st.session_state.curvas_sin_alto_rh
                        datos_sin = st.session_state.datos_sin_alto_rh
                        
                        st.success(f"✅ Procesado exitoso: {len(datos_sin)} aforos (sin GRUPO_ALTO_RH)")
                        
                        # Mostrar resultados iniciales (EXCLUYENDO GRUPO_ESTANDAR)
                        st.subheader("📊 Resultados Iniciales (sin GRUPO_ALTO_RH)")
                        datos_sin_filtrados = datos_sin[datos_sin['GRUPO_PREDICHO'] != 'GRUPO_ESTANDAR']
                        st.dataframe(datos_sin_filtrados[['NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 'GRUPO_PREDICHO']].head())
                        
                        # Gráfico inicial
                        st.subheader("📈 Curvas Altura-Caudal (sin GRUPO_ALTO_RH)")
                        fig_sin = crear_grafico_principal(datos_sin, curvas_sin, "Curvas sin GRUPO_ALTO_RH")
                        st.pyplot(fig_sin)
                        
                        # Mostrar ecuaciones con RANGOS DE VALIDEZ
                        st.subheader("📐 Ecuaciones y Rangos de Validez (sin GRUPO_ALTO_RH)")
                        for grupo, curva in curvas_sin.items():
                            # EXCLUIR GRUPO_ESTANDAR
                            if grupo == 'GRUPO_ESTANDAR':
                                continue
                                
                            # Usar rango de validez si está definido
                            if 'rango_validez' in curva:
                                rango_min, rango_max = curva['rango_validez']
                            else:
                                rango_min, rango_max = curva['rango_niveles']
                            
                            rango_formateado = formatear_rango(rango_min, rango_max)
                            
                            with st.expander(f"{grupo} - R² = {curva['r2']:.3f} - {rango_formateado} m"):
                                st.write(f"**Tipo de modelo:** {curva['nombre']}")
                                st.write(f"**Puntos utilizados:** {curva['n_puntos']}")
                                st.write(f"**Rango de validez:** {rango_formateado} m")
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
                        
                        # Gráficos complementarios para SIN GRUPO_ALTO_RH
                        st.subheader("🔍 Análisis Complementario (sin GRUPO_ALTO_RH)")
                        fig_comp_sin = crear_graficos_complementarios(datos_sin, "(sin GRUPO_ALTO_RH)")
                        st.pyplot(fig_comp_sin)
                        
                        # VERIFICAR SI HAY GRUPO_ALTO_RH PARA OFRECER RECÁLCULO
                        if st.session_state.tiene_alto_rh:
                            st.subheader("⚙️ Opción de Re-análisis")
                            
                            # Mostrar información específica sobre GRUPO_ALTO_RH
                            datos_completos = st.session_state.datos_completos
                            alto_rh_data = datos_completos[datos_completos['GRUPO_PREDICHO'] == 'GRUPO_ALTO_RH']
                            
                            st.warning(f"🔴 Se detectaron {len(alto_rh_data)} aforos del GRUPO_ALTO_RH:")
                            st.dataframe(alto_rh_data[['NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 'RADIO_HIDRAULICO']])
                            
                            st.info("¿Deseas recalcular INCLUYENDO el GRUPO_ALTO_RH?")
                            
                            # BOTÓN DE RECÁLCULO
                            if st.button("🔄 RECALCULAR con GRUPO_ALTO_RH", key="btn_recalcular"):
                                with st.spinner("Recalculando con GRUPO_ALTO_RH..."):
                                    # RECÁLCULO REAL INCLUYENDO GRUPO_ALTO_RH
                                    curvas_con, datos_con = procesar_con_modelo(modelo, df, incluir_alto_rh=True)
                                    
                                    st.success(f"✅ RECÁLCULO EXITOSO: {len(datos_con)} aforos (CON GRUPO_ALTO_RH)")
                                    
                                    # Mostrar comparación
                                    st.subheader("📊 COMPARACIÓN: Con vs Sin GRUPO_ALTO_RH")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.metric("Aforos SIN GRUPO_ALTO_RH", len(datos_sin))
                                        st.metric("Curvas generadas", len(curvas_sin))
                                    
                                    with col2:
                                        st.metric("Aforos CON GRUPO_ALTO_RH", len(datos_con))
                                        st.metric("Curvas generadas", len(curvas_con))
                                    
                                    # Mostrar NUEVOS resultados (EXCLUYENDO GRUPO_ESTANDAR)
                                    st.subheader("📊 NUEVOS Resultados (CON GRUPO_ALTO_RH)")
                                    datos_con_filtrados = datos_con[datos_con['GRUPO_PREDICHO'] != 'GRUPO_ESTANDAR']
                                    st.dataframe(datos_con_filtrados[['NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 'GRUPO_PREDICHO']].head())
                                    
                                    # NUEVO gráfico con GRUPO_ALTO_RH
                                    st.subheader("📈 NUEVAS Curvas Altura-Caudal (CON GRUPO_ALTO_RH)")
                                    fig_con = crear_grafico_principal(datos_con, curvas_con, "Curvas CON GRUPO_ALTO_RH")
                                    st.pyplot(fig_con)
                                    
                                    # Mostrar ecuaciones con RANGOS DE VALIDEZ
                                    st.subheader("📐 Ecuaciones y Rangos de Validez (CON GRUPO_ALTO_RH)")
                                    for grupo, curva in curvas_con.items():
                                        # EXCLUIR GRUPO_ESTANDAR
                                        if grupo == 'GRUPO_ESTANDAR':
                                            continue
                                            
                                        # Usar rango de validez si está definido
                                        if 'rango_validez' in curva:
                                            rango_min, rango_max = curva['rango_validez']
                                        else:
                                            rango_min, rango_max = curva['rango_niveles']
                                        
                                        rango_formateado = formatear_rango(rango_min, rango_max)
                                        
                                        with st.expander(f"{grupo} - R² = {curva['r2']:.3f} - {rango_formateado} m"):
                                            st.write(f"**Tipo de modelo:** {curva['nombre']}")
                                            st.write(f"**Puntos utilizados:** {curva['n_puntos']}")
                                            st.write(f"**Rango de validez:** {rango_formateado} m")
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
                                    
                                    # Gráficos complementarios para CON GRUPO_ALTO_RH
                                    st.subheader("🔍 Análisis Complementario (CON GRUPO_ALTO_RH)")
                                    fig_comp_con = crear_graficos_complementarios(datos_con, "(CON GRUPO_ALTO_RH)")
                                    st.pyplot(fig_comp_con)
                        else:
                            st.info("✅ No se detectó GRUPO_ALTO_RH en los datos. Los resultados están completos.")
                            
                else:
                    st.error(f"❌ Faltan las siguientes columnas necesarias: {', '.join(columnas_faltantes)}")
                    st.info("💡 Asegúrate de que tu archivo CSV tenga las columnas con los nombres exactos.")
                    
            except Exception as e:
                st.error(f"❌ Error al procesar el archivo: {e}")
                st.info("💡 Verifica que el archivo sea un CSV válido y tenga el formato correcto.")

# ... (el resto del código para Ingreso Manual y Curvas se mantiene similar)

st.markdown("---")
st.markdown("**🌊 IA para generar Curvas H-Q**")