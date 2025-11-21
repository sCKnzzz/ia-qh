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
    st.info("💡 Para instalar Plotly: `pip install plotly`")

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
            try:
                from scipy.optimize import brentq
                interseccion = brentq(diferencia, H_test[i], H_test[i+1], xtol=tolerancia)
                # Verificar que la intersección esté dentro del rango válido
                if rango_min <= interseccion <= rango_max:
                    intersecciones.append(interseccion)
            except:
                continue
    
    return intersecciones

# FUNCIÓN PARA DEFINIR RANGOS DE VALIDEZ SIN ESPACIOS VACÍOS
def definir_rangos_validez(curvas):
    """Definir rangos de validez para curvas que se intersectan, sin espacios vacíos"""
    if len(curvas) < 2:
        # Para una sola curva, usar rango completo
        for grupo in curvas:
            curvas[grupo]['rango_validez'] = (
                round(curvas[grupo]['rango_niveles'][0], 2),
                round(curvas[grupo]['rango_niveles'][1], 2)
            )
        return curvas
    
    grupos = list(curvas.keys())
    
    # Ordenar grupos por rango mínimo de nivel
    grupos_ordenados = sorted(grupos, key=lambda g: curvas[g]['rango_niveles'][0])
    
    # Encontrar todas las intersecciones
    puntos_quiebre = []
    
    for i in range(len(grupos_ordenados)):
        for j in range(i+1, len(grupos_ordenados)):
            grupo1 = grupos_ordenados[i]
            grupo2 = grupos_ordenados[j]
            
            curva1 = curvas[grupo1]
            curva2 = curvas[grupo2]
            
            # Rango común donde buscar intersección
            rango_min = max(curva1['rango_niveles'][0], curva2['rango_niveles'][0])
            rango_max = min(curva1['rango_niveles'][1], curva2['rango_niveles'][1])
            
            if rango_min < rango_max:  # Solo si hay superposición
                intersecciones = encontrar_interseccion(curva1, curva2, rango_min, rango_max)
                if intersecciones:
                    punto_quiebre = round(intersecciones[0], 2)
                    # Verificar que el punto de quiebre esté dentro de ambos rangos
                    if (punto_quiebre >= curva1['rango_niveles'][0] and 
                        punto_quiebre <= curva1['rango_niveles'][1] and
                        punto_quiebre >= curva2['rango_niveles'][0] and 
                        punto_quiebre <= curva2['rango_niveles'][1]):
                        puntos_quiebre.append((punto_quiebre, grupo1, grupo2))
    
    # Ordenar puntos de quiebre
    puntos_quiebre.sort(key=lambda x: x[0])
    
    # Definir rangos continuos
    if puntos_quiebre:
        rangos_por_grupo = {}
        
        # Primer rango
        primer_punto = puntos_quiebre[0][0]
        rangos_por_grupo[grupos_ordenados[0]] = (
            round(curvas[grupos_ordenados[0]]['rango_niveles'][0], 2), 
            primer_punto
        )
        
        # Rangos intermedios
        for k in range(len(puntos_quiebre)):
            punto_actual = puntos_quiebre[k][0]
            grupo_actual = puntos_quiebre[k][2]
            
            if k < len(puntos_quiebre) - 1:
                punto_siguiente = puntos_quiebre[k+1][0]
                rangos_por_grupo[grupo_actual] = (punto_actual, punto_siguiente)
            else:
                # Último rango
                rangos_por_grupo[grupo_actual] = (
                    punto_actual, 
                    round(curvas[grupo_actual]['rango_niveles'][1], 2)
                )
        
        # Asignar rangos
        for grupo in grupos_ordenados:
            if grupo in rangos_por_grupo:
                curvas[grupo]['rango_validez'] = rangos_por_grupo[grupo]
            else:
                curvas[grupo]['rango_validez'] = (
                    round(curvas[grupo]['rango_niveles'][0], 2),
                    round(curvas[grupo]['rango_niveles'][1], 2)
                )
    else:
        # Si no hay intersecciones, crear rangos continuos
        for i, grupo in enumerate(grupos_ordenados):
            rango_original = curvas[grupo]['rango_niveles']
            
            if i == 0:
                # Primer grupo
                if len(grupos_ordenados) > 1:
                    siguiente_min = curvas[grupos_ordenados[1]]['rango_niveles'][0]
                    nuevo_max = round(min(rango_original[1], siguiente_min), 2)
                    curvas[grupo]['rango_validez'] = (
                        round(rango_original[0], 2), 
                        nuevo_max
                    )
                else:
                    curvas[grupo]['rango_validez'] = (
                        round(rango_original[0], 2),
                        round(rango_original[1], 2)
                    )
            elif i == len(grupos_ordenados) - 1:
                # Último grupo
                anterior_max = curvas[grupos_ordenados[i-1]]['rango_validez'][1]
                nuevo_min = round(max(rango_original[0], anterior_max), 2)
                curvas[grupo]['rango_validez'] = (
                    nuevo_min, 
                    round(rango_original[1], 2)
                )
            else:
                # Grupos intermedios
                anterior_max = curvas[grupos_ordenados[i-1]]['rango_validez'][1]
                siguiente_min = curvas[grupos_ordenados[i+1]]['rango_niveles'][0]
                
                nuevo_min = round(max(rango_original[0], anterior_max), 2)
                nuevo_max = round(min(rango_original[1], siguiente_min), 2)
                
                curvas[grupo]['rango_validez'] = (nuevo_min, nuevo_max)
    
    return curvas

# FUNCIÓN PARA FORMATEAR RANGOS CON SÍMBOLOS MATEMÁTICOS
def formatear_rango(rango_min, rango_max):
    """Formatear rango con símbolos matemáticos correctos"""
    return f"{rango_min:.2f} ≤ H ≤ {rango_max:.2f}"

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
                    # Redondear rangos a 2 decimales
                    curva['rango_niveles'] = (
                        round(curva['rango_niveles'][0], 2),
                        round(curva['rango_niveles'][1], 2)
                    )
                    curva['rango_caudales'] = (
                        round(curva['rango_caudales'][0], 2),
                        round(curva['rango_caudales'][1], 2)
                    )
                    resultados[grupo] = curva
        
        # Definir rangos de validez sin superposición y sin espacios vacíos
        if len(resultados) >= 2:
            resultados = definir_rangos_validez(resultados)
            
            # Verificar continuidad y ajustar si es necesario
            grupos_ordenados = sorted(resultados.keys(), 
                                    key=lambda g: resultados[g]['rango_validez'][0])
            
            # Asegurar continuidad perfecta
            for i in range(len(grupos_ordenados)-1):
                grupo_actual = grupos_ordenados[i]
                grupo_siguiente = grupos_ordenados[i+1]
                
                fin_actual = resultados[grupo_actual]['rango_validez'][1]
                inicio_siguiente = resultados[grupo_siguiente]['rango_validez'][0]
                
                # Si hay espacio vacío, ajustar
                if abs(fin_actual - inicio_siguiente) > 0.01:
                    # Punto medio para transición suave
                    punto_transicion = round((fin_actual + inicio_siguiente) / 2, 2)
                    resultados[grupo_actual]['rango_validez'] = (
                        resultados[grupo_actual]['rango_validez'][0], 
                        punto_transicion
                    )
                    resultados[grupo_siguiente]['rango_validez'] = (
                        punto_transicion, 
                        resultados[grupo_siguiente]['rango_validez'][1]
                    )
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
                    'r2': round(r2, 3),
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
    
    return mejor_modelo, mejor_params, round(mejor_r2, 3), mejor_funcion

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

# 🎯 NUEVAS FUNCIONES PARA GRÁFICOS INTERACTIVOS (solo si Plotly está disponible)
if PLOTLY_AVAILABLE:
    def crear_grafico_interactivo(df, curvas, titulo):
        """Crear gráfico interactivo con Plotly"""
        fig = go.Figure()
        
        colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
        
        # Agregar puntos dispersos
        for grupo in df['GRUPO_PREDICHO'].unique():
            if grupo == 'GRUPO_ESTANDAR':
                continue
                
            grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
            fig.add_trace(go.Scatter(
                x=grupo_data['NIVEL_AFORO'],
                y=grupo_data['CAUDAL'],
                mode='markers',
                name=grupo,
                marker=dict(
                    color=colores.get(grupo, 'orange'),
                    size=12 if grupo == 'GRUPO_ALTO_RH' else 8,
                    line=dict(width=1, color='black')
                ),
                hovertemplate='<b>%{text}</b><br>Nivel: %{x:.2f} m<br>Caudal: %{y:.2f} m³/s<extra></extra>',
                text=[f'{grupo}'] * len(grupo_data)
            ))
        
        # Agregar curvas
        for grupo, curva in curvas.items():
            if grupo == 'GRUPO_ESTANDAR':
                continue
                
            if 'rango_validez' in curva:
                rango_min, rango_max = curva['rango_validez']
            else:
                rango_min, rango_max = curva['rango_niveles']
            
            H_range = np.linspace(rango_min, rango_max, 100)
            Q_curve = curva['funcion'](H_range, *curva['parametros'])
            
            fig.add_trace(go.Scatter(
                x=H_range,
                y=Q_curve,
                mode='lines',
                name=f"{grupo} (R²={curva['r2']:.3f})",
                line=dict(
                    color=colores.get(grupo, 'orange'),
                    width=4 if grupo == 'GRUPO_ALTO_RH' else 2
                ),
                hovertemplate='<b>%{fullData.name}</b><br>Nivel: %{x:.2f} m<br>Caudal: %{y:.2f} m³/s<extra></extra>'
            ))
        
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

    def crear_graficos_complementarios_interactivos(df, titulo_sufijo=""):
        """Crear gráficos complementarios interactivos"""
        # Filtrar datos excluyendo GRUPO_ESTANDAR
        df_filtrado = df[df['GRUPO_PREDICHO'] != 'GRUPO_ESTANDAR']
        
        if len(df_filtrado) == 0:
            return None
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Altura vs Velocidad', 
                'Altura vs Área',
                'Altura vs Radio Hidráulico', 
                'Caudal vs Velocidad'
            )
        )
        
        colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue'}
        
        # 1. Altura vs Velocidad
        for grupo in df_filtrado['GRUPO_PREDICHO'].unique():
            grupo_data = df_filtrado[df_filtrado['GRUPO_PREDICHO'] == grupo]
            fig.add_trace(
                go.Scatter(
                    x=grupo_data['NIVEL_AFORO'],
                    y=grupo_data['VELOCIDAD'],
                    mode='markers',
                    name=f'{grupo} - Velocidad',
                    marker=dict(
                        color=colores.get(grupo, 'orange'),
                        size=10 if grupo == 'GRUPO_ALTO_RH' else 6
                    ),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 2. Altura vs Área
        for grupo in df_filtrado['GRUPO_PREDICHO'].unique():
            grupo_data = df_filtrado[df_filtrado['GRUPO_PREDICHO'] == grupo]
            fig.add_trace(
                go.Scatter(
                    x=grupo_data['NIVEL_AFORO'],
                    y=grupo_data['AREA'],
                    mode='markers',
                    name=f'{grupo} - Área',
                    marker=dict(
                        color=colores.get(grupo, 'orange'),
                        size=10 if grupo == 'GRUPO_ALTO_RH' else 6
                    ),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Altura vs Radio Hidráulico
        for grupo in df_filtrado['GRUPO_PREDICHO'].unique():
            grupo_data = df_filtrado[df_filtrado['GRUPO_PREDICHO'] == grupo]
            fig.add_trace(
                go.Scatter(
                    x=grupo_data['NIVEL_AFORO'],
                    y=grupo_data['RADIO_HIDRAULICO'],
                    mode='markers',
                    name=f'{grupo} - Radio Hidráulico',
                    marker=dict(
                        color=colores.get(grupo, 'orange'),
                        size=10 if grupo == 'GRUPO_ALTO_RH' else 6
                    ),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Caudal vs Velocidad
        for grupo in df_filtrado['GRUPO_PREDICHO'].unique():
            grupo_data = df_filtrado[df_filtrado['GRUPO_PREDICHO'] == grupo]
            fig.add_trace(
                go.Scatter(
                    x=grupo_data['CAUDAL'],
                    y=grupo_data['VELOCIDAD'],
                    mode='markers',
                    name=f'{grupo} - Caudal-Vel',
                    marker=dict(
                        color=colores.get(grupo, 'orange'),
                        size=10 if grupo == 'GRUPO_ALTO_RH' else 6
                    ),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text=f'Análisis de Relaciones Hidráulicas {titulo_sufijo}',
            height=700,
            showlegend=True
        )
        
        # Actualizar ejes
        fig.update_xaxes(title_text="Nivel (m)", row=1, col=1)
        fig.update_yaxes(title_text="Velocidad (m/s)", row=1, col=1)
        fig.update_xaxes(title_text="Nivel (m)", row=1, col=2)
        fig.update_yaxes(title_text="Área (m²)", row=1, col=2)
        fig.update_xaxes(title_text="Nivel (m)", row=2, col=1)
        fig.update_yaxes(title_text="Radio Hidráulico (m)", row=2, col=1)
        fig.update_xaxes(title_text="Caudal (m³/s)", row=2, col=2)
        fig.update_yaxes(title_text="Velocidad (m/s)", row=2, col=2)
        
        return fig

    def crear_analisis_residuos_interactivo(df, curvas):
        """Crear análisis de residuos interactivo"""
        datos_con_residuos = df.copy()
        
        # Calcular residuos
        for idx, row in datos_con_residuos.iterrows():
            grupo = row['GRUPO_PREDICHO']
            if grupo in curvas and grupo != 'GRUPO_ESTANDAR':
                curva = curvas[grupo]
                Q_pred = curva['funcion'](row['NIVEL_AFORO'], *curva['parametros'])
                datos_con_residuos.loc[idx, 'RESIDUO'] = row['CAUDAL'] - Q_pred
                datos_con_residuos.loc[idx, 'CAUDAL_PRED'] = Q_pred
            else:
                datos_con_residuos.loc[idx, 'RESIDUO'] = np.nan
                datos_con_residuos.loc[idx, 'CAUDAL_PRED'] = np.nan
        
        # Filtrar datos válidos
        datos_validos = datos_con_residuos.dropna(subset=['RESIDUO'])
        
        if len(datos_validos) == 0:
            return None
        
        # Crear gráfico de residuos
        fig = px.scatter(
            datos_validos,
            x='NIVEL_AFORO',
            y='RESIDUO',
            color='GRUPO_PREDICHO',
            title='Análisis de Residuos - Residuos vs Nivel',
            hover_data=['CAUDAL', 'CAUDAL_PRED'],
            labels={'RESIDUO': 'Residuo (m³/s)', 'NIVEL_AFORO': 'Nivel (m)'}
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Línea Cero")
        
        # Calcular estadísticas de residuos
        rmse = np.sqrt(np.mean(datos_validos['RESIDUO']**2))
        bias = np.mean(datos_validos['RESIDUO'])
        
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=f"RMSE: {rmse:.3f} m³/s<br>Bias: {bias:.3f} m³/s",
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        return fig

# Título principal después de la configuración
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

# Resto del código de navegación (igual que antes)...
# [Aquí iría el resto del código de las secciones de navegación que ya tienes]

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

    # Demo interactivo
    if PLOTLY_AVAILABLE:
        st.subheader("🎮 Demo Interactivo")
        if st.checkbox("Mostrar demo de gráficos interactivos"):
            # Crear datos de demo
            np.random.seed(42)
            H_demo = np.linspace(0.5, 8, 50)
            Q_demo = 2.5 * H_demo**1.8 + np.random.normal(0, 0.5, 50)
            
            df_demo = pd.DataFrame({
                'NIVEL_AFORO': H_demo,
                'CAUDAL': Q_demo,
                'GRUPO_PREDICHO': ['GRUPO_RECIENTE'] * 50
            })
            
            # Crear curva demo
            curva_demo = {
                'GRUPO_RECIENTE': {
                    'nombre': 'Potencial',
                    'funcion': func_pot,
                    'parametros': [2.3, 1.75],
                    'r2': 0.95,
                    'rango_validez': (0.5, 8.0)
                }
            }
            
            fig_demo = crear_grafico_interactivo(df_demo, curva_demo, "Demo - Curva H-Q Interactiva")
            st.plotly_chart(fig_demo, use_container_width=True)

# [Aquí continúan las demás secciones de tu código...]

st.markdown("---")
st.markdown("**🌊 IA para generar Curvas H-Q - Versión Mejorada con Gráficos Interactivos**")