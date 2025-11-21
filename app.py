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

# FUNCIÓN PARA INTERSECCIONES
def encontrar_interseccion(curva1, curva2, rango_min, rango_max, tolerancia=0.01):
    def diferencia(H):
        return curva1['funcion'](H, *curva1['parametros']) - curva2['funcion'](H, *curva2['parametros'])
    
    H_test = np.linspace(rango_min, rango_max, 1000)
    diferencias = diferencia(H_test)
    
    intersecciones = []
    for i in range(len(diferencias)-1):
        if diferencias[i] * diferencias[i+1] <= 0:
            try:
                from scipy.optimize import brentq
                interseccion = brentq(diferencia, H_test[i], H_test[i+1], xtol=tolerancia)
                if rango_min <= interseccion <= rango_max:
                    intersecciones.append(interseccion)
            except:
                continue
    return intersecciones

# FUNCIÓN PARA RANGOS DE VALIDEZ
def definir_rangos_validez(curvas):
    if len(curvas) < 2:
        for grupo in curvas:
            curvas[grupo]['rango_validez'] = (
                round(curvas[grupo]['rango_niveles'][0], 2),
                round(curvas[grupo]['rango_niveles'][1], 2)
            )
        return curvas
    
    grupos = list(curvas.keys())
    grupos_ordenados = sorted(grupos, key=lambda g: curvas[g]['rango_niveles'][0])
    
    puntos_quiebre = []
    for i in range(len(grupos_ordenados)):
        for j in range(i+1, len(grupos_ordenados)):
            grupo1 = grupos_ordenados[i]
            grupo2 = grupos_ordenados[j]
            curva1 = curvas[grupo1]
            curva2 = curvas[grupo2]
            
            rango_min = max(curva1['rango_niveles'][0], curva2['rango_niveles'][0])
            rango_max = min(curva1['rango_niveles'][1], curva2['rango_niveles'][1])
            
            if rango_min < rango_max:
                intersecciones = encontrar_interseccion(curva1, curva2, rango_min, rango_max)
                if intersecciones:
                    punto_quiebre = round(intersecciones[0], 2)
                    if (punto_quiebre >= curva1['rango_niveles'][0] and 
                        punto_quiebre <= curva1['rango_niveles'][1] and
                        punto_quiebre >= curva2['rango_niveles'][0] and 
                        punto_quiebre <= curva2['rango_niveles'][1]):
                        puntos_quiebre.append((punto_quiebre, grupo1, grupo2))
    
    puntos_quiebre.sort(key=lambda x: x[0])
    
    if puntos_quiebre:
        rangos_por_grupo = {}
        primer_punto = puntos_quiebre[0][0]
        rangos_por_grupo[grupos_ordenados[0]] = (
            round(curvas[grupos_ordenados[0]]['rango_niveles'][0], 2), 
            primer_punto
        )
        
        for k in range(len(puntos_quiebre)):
            punto_actual = puntos_quiebre[k][0]
            grupo_actual = puntos_quiebre[k][2]
            
            if k < len(puntos_quiebre) - 1:
                punto_siguiente = puntos_quiebre[k+1][0]
                rangos_por_grupo[grupo_actual] = (punto_actual, punto_siguiente)
            else:
                rangos_por_grupo[grupo_actual] = (
                    punto_actual, 
                    round(curvas[grupo_actual]['rango_niveles'][1], 2)
                )
        
        for grupo in grupos_ordenados:
            if grupo in rangos_por_grupo:
                curvas[grupo]['rango_validez'] = rangos_por_grupo[grupo]
            else:
                curvas[grupo]['rango_validez'] = (
                    round(curvas[grupo]['rango_niveles'][0], 2),
                    round(curvas[grupo]['rango_niveles'][1], 2)
                )
    else:
        for i, grupo in enumerate(grupos_ordenados):
            rango_original = curvas[grupo]['rango_niveles']
            
            if i == 0:
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
                anterior_max = curvas[grupos_ordenados[i-1]]['rango_validez'][1]
                nuevo_min = round(max(rango_original[0], anterior_max), 2)
                curvas[grupo]['rango_validez'] = (
                    nuevo_min, 
                    round(rango_original[1], 2)
                )
            else:
                anterior_max = curvas[grupos_ordenados[i-1]]['rango_validez'][1]
                siguiente_min = curvas[grupos_ordenados[i+1]]['rango_niveles'][0]
                
                nuevo_min = round(max(rango_original[0], anterior_max), 2)
                nuevo_max = round(min(rango_original[1], siguiente_min), 2)
                
                curvas[grupo]['rango_validez'] = (nuevo_min, nuevo_max)
    
    return curvas

def formatear_rango(rango_min, rango_max):
    return f"{rango_min:.2f} ≤ H ≤ {rango_max:.2f}"

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
        tirante = df_procesado['AREA'] / df_procesado['ANCHO_RIO']
        df_procesado['PERIMETRO'] = 2 * tirante + df_procesado['ANCHO_RIO']
    
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
        X = df_procesado[features]
        X_scaled = modelo.escalador.transform(X)
        grupos_pred = modelo.clasificador.predict(X_scaled)
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
            if len(grupo_data) >= 3:
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
        
        if len(resultados) >= 2:
            resultados = definir_rangos_validez(resultados)
            
            grupos_ordenados = sorted(resultados.keys(), 
                                    key=lambda g: resultados[g]['rango_validez'][0])
            
            for i in range(len(grupos_ordenados)-1):
                grupo_actual = grupos_ordenados[i]
                grupo_siguiente = grupos_ordenados[i+1]
                
                fin_actual = resultados[grupo_actual]['rango_validez'][1]
                inicio_siguiente = resultados[grupo_siguiente]['rango_validez'][0]
                
                if abs(fin_actual - inicio_siguiente) > 0.01:
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

def ajustar_modelo_relacion(x, y, nombre_relacion):
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
                x_positivo = x + 0.001
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

def crear_grafico_principal_con_personalizadas(df, curvas, titulo):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
    marcadores = {'GRUPO_ALTO_RH': 's', 'GRUPO_RECIENTE': '^', 'GRUPO_ESTANDAR': 'o'}
    tamanos = {'GRUPO_ALTO_RH': 100, 'GRUPO_RECIENTE': 80, 'GRUPO_ESTANDAR': 80}
    
    grupos_unicos = df['GRUPO_PREDICHO'].unique() if 'GRUPO_PREDICHO' in df.columns else []
    
    for grupo in grupos_unicos:
        if grupo == 'GRUPO_ESTANDAR':
            continue
            
        color = colores.get(grupo, 'orange')
        marcador = marcadores.get(grupo, 'o')
        tamano = tamanos.get(grupo, 80)
        grupo_data = df[df['GRUPO_PREDICHO'] == grupo]
        
        alpha = 0.9 if grupo == 'GRUPO_ALTO_RH' else 0.7
        ax.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], 
                  color=color, marker=marcador, s=tamano, label=grupo, alpha=alpha, 
                  edgecolors='black', linewidth=1 if grupo == 'GRUPO_ALTO_RH' else 0.5)
    
    todas_las_curvas = curvas.copy()
    
    if 'curvas_personalizadas' in st.session_state:
        for nombre_curva, curva_personalizada in st.session_state.curvas_personalizadas.items():
            todas_las_curvas[nombre_curva] = curva_personalizada
    
    for grupo, curva in todas_las_curvas.items():
        if grupo == 'GRUPO_ESTANDAR':
            continue
            
        if 'PERSONALIZADA' in grupo.upper() or 'CURVA_PERSONALIZADA' in grupo.upper():
            color = 'purple'
        else:
            color = colores.get(grupo, 'orange')
        
        if 'rango_validez' in curva:
            rango_min, rango_max = curva['rango_validez']
        else:
            rango_min, rango_max = curva['rango_niveles']
        
        H_range = np.linspace(rango_min, rango_max, 100)
        
        try:
            if 'parametros' in curva and isinstance(curva['parametros'], dict):
                if curva['nombre'] == 'Potencial':
                    a = curva['parametros'].get('a', 1.0)
                    b = curva['parametros'].get('b', 2.0)
                    Q_curve = a * (H_range ** b)
                elif curva['nombre'] == 'Polinómica G2':
                    a = curva['parametros'].get('a', 0.1)
                    b = curva['parametros'].get('b', 0.5)
                    c = curva['parametros'].get('c', 0.1)
                    Q_curve = a * H_range**2 + b * H_range + c
                elif curva['nombre'] == 'Polinómica G3':
                    a = curva['parametros'].get('a', 0.01)
                    b = curva['parametros'].get('b', 0.1)
                    c = curva['parametros'].get('c', 0.5)
                    d = curva['parametros'].get('d', 0.1)
                    Q_curve = a * H_range**3 + b * H_range**2 + c * H_range + d
                elif curva['nombre'] == 'Exponencial':
                    a = curva['parametros'].get('a', 1.0)
                    b = curva['parametros'].get('b', 0.5)
                    Q_curve = a * np.exp(b * H_range)
                else:
                    Q_curve = curva['funcion'](H_range)
            else:
                Q_curve = curva['funcion'](H_range, *curva['parametros'])
        except:
            try:
                Q_curve = curva['funcion'](H_range)
            except:
                continue
        
        linewidth = 3 if grupo == 'GRUPO_ALTO_RH' or 'PERSONALIZADA' in grupo.upper() else 2
        linestyle = '--' if 'PERSONALIZADA' in grupo.upper() or 'CURVA_PERSONALIZADA' in grupo.upper() else '-'
        
        rango_formateado = formatear_rango(rango_min, rango_max)
        
        if 'PERSONALIZADA' in grupo.upper() or 'CURVA_PERSONALIZADA' in grupo.upper():
            label = f"{grupo} (Teórica)\n{rango_formateado} m"
        else:
            r2_value = curva.get('r2', 'N/A')
            if isinstance(r2_value, (int, float)):
                label = f"{grupo} (R²={r2_value:.3f})\n{rango_formateado} m"
            else:
                label = f"{grupo}\n{rango_formateado} m"
        
        ax.plot(H_range, Q_curve, color=color, linewidth=linewidth, label=label, linestyle=linestyle)
    
    ax.set_xlabel('Nivel (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Caudal (m³/s)', fontsize=12, fontweight='bold')
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def crear_graficos_complementarios(df, titulo_sufijo=""):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Análisis de Relaciones Hidráulicas {titulo_sufijo}', fontsize=16, fontweight='bold')
    
    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
    marcadores = {'GRUPO_ALTO_RH': 's', 'GRUPO_RECIENTE': '^', 'GRUPO_ESTANDAR': 'o'}
    
    ax1 = axes[0, 0]
    for grupo in df['GRUPO_PREDICHO'].unique():
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
    
    df_filtrado = df[df['GRUPO_PREDICHO'] != 'GRUPO_ESTANDAR']
    x_vel = df_filtrado['NIVEL_AFORO'].values
    y_vel = df_filtrado['VELOCIDAD'].values
    modelo_vel, params_vel, r2_vel, funcion_vel = ajustar_modelo_relacion(x_vel, y_vel, "Altura-Velocidad")
    
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
    
    ax2 = axes[0, 1]
    for grupo in df['GRUPO_PREDICHO'].unique():
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
    
    ax3 = axes[1, 0]
    for grupo in df['GRUPO_PREDICHO'].unique():
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
    
    ax4 = axes[1, 1]
    for grupo in df['GRUPO_PREDICHO'].unique():
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

@st.cache_resource
def cargar_modelo():
    try:
        modelo = joblib.load('modelo_talapalca_entrenado.pkl')
        st.success("✅ Modelo cargado correctamente")
        return modelo
    except Exception as e:
        st.warning(f"⚠️ Error al cargar el modelo: {str(e)}")
        st.info("🔧 Creando modelo de demostración...")
        
        modelo_demo = SistemaCurvasAlturaCaudal()
        from sklearn.datasets import make_classification
        
        X_demo, y_demo = make_classification(
            n_samples=50, 
            n_features=9, 
            n_classes=3, 
            random_state=42
        )
        
        y_demo_nombres = ['GRUPO_ESTANDAR', 'GRUPO_RECIENTE', 'GRUPO_ALTO_RH']
        y_demo_categoricos = [y_demo_nombres[i % 3] for i in y_demo]
        
        modelo_demo.entrenar(X_demo, y_demo_categoricos)
        
        st.success("✅ Modelo de demostración creado exitosamente")
        st.info("💡 Nota: Este es un modelo de demostración. Para usar el modelo real, asegúrate de que el archivo 'modelo_talapalca_entrenado.pkl' esté disponible.")
        
        return modelo_demo

# APLICACIÓN PRINCIPAL
st.title("🌊 IA para la generacion de Curvas Altura-Caudal")
st.markdown("**Modelo entrenado con 34 aforos reales**")

modelo = cargar_modelo()

opcion = st.sidebar.radio("Navegación:", [
    "🏠 Inicio", 
    "📤 Subir Aforos", 
    "📊 Ingreso Manual", 
    "📈 Curvas",
    "➕ Insertar Curva Personalizada"
])

if opcion == "🏠 Inicio":
    st.header("Bienvenido a la IA para curvas H-Q")
    st.info("Aplicacion IA para generar curvas altura-caudal usando IA")
    
    st.subheader("Instrucciones de uso:")
    st.markdown("""
    1. **📤 Subir Aforos**: Carga un archivo CSV con datos de aforos
    2. **📊 Ingreso Manual**: Ingresa datos de aforos manualmente
    3. **📈 Curvas**: Visualiza las curvas generadas
    4. **➕ Insertar Curva Personalizada**: Agrega tu propia curva teórica
    
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
                
                st.subheader("📋 Vista previa de datos")
                st.dataframe(df.head())
                
                columnas_necesarias = ['CAUDAL (m3/s)', 'VELOCIDAD (m/s)', 'AREA (m2)', 'ANCHO RIO (m)', 'NIVEL DE AFORO (m)']
                columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]
                
                if not columnas_faltantes:
                    st.success("✅ Todas las columnas necesarias están presentes")
                    
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
                    
                    if st.button("🚀 Procesar Aforos", type="primary"):
                        with st.spinner("Procesando datos..."):
                            curvas_sin, datos_sin = procesar_con_modelo(modelo, df, incluir_alto_rh=False)
                            
                            if curvas_sin:
                                st.session_state.procesamiento_realizado = True
                                st.session_state.curvas_sin_alto_rh = curvas_sin
                                st.session_state.datos_sin_alto_rh = datos_sin
                                
                                _, datos_completos = procesar_con_modelo(modelo, df, incluir_alto_rh=True)
                                st.session_state.tiene_alto_rh = 'GRUPO_ALTO_RH' in datos_completos['GRUPO_PREDICHO'].values
                                st.session_state.datos_completos = datos_completos
                    
                    if st.session_state.procesamiento_realizado and st.session_state.curvas_sin_alto_rh is not None:
                        curvas_sin = st.session_state.curvas_sin_alto_rh
                        datos_sin = st.session_state.datos_sin_alto_rh
                        
                        st.success(f"✅ Procesado exitoso: {len(datos_sin)} aforos (sin GRUPO_ALTO_RH)")
                        
                        st.subheader("📊 Resultados Iniciales (sin GRUPO_ALTO_RH)")
                        datos_sin_filtrados = datos_sin[datos_sin['GRUPO_PREDICHO'] != 'GRUPO_ESTANDAR']
                        st.dataframe(datos_sin_filtrados[['NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 'GRUPO_PREDICHO']].head())
                        
                        st.subheader("📈 Curvas Altura-Caudal (sin GRUPO_ALTO_RH)")
                        fig_sin = crear_grafico_principal_con_personalizadas(datos_sin, curvas_sin, "Curvas sin GRUPO_ALTO_RH")
                        st.pyplot(fig_sin)
                        
                        st.subheader("📐 Ecuaciones y Rangos de Validez (sin GRUPO_ALTO_RH)")
                        for grupo, curva in curvas_sin.items():
                            if grupo == 'GRUPO_ESTANDAR':
                                continue
                                
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
                        
                        st.subheader("🔍 Análisis Complementario (sin GRUPO_ALTO_RH)")
                        fig_comp_sin = crear_graficos_complementarios(datos_sin, "(sin GRUPO_ALTO_RH)")
                        st.pyplot(fig_comp_sin)
                        
                        if st.session_state.tiene_alto_rh:
                            st.subheader("⚙️ Opción de Re-análisis")
                            
                            datos_completos = st.session_state.datos_completos
                            alto_rh_data = datos_completos[datos_completos['GRUPO_PREDICHO'] == 'GRUPO_ALTO_RH']
                            
                            st.warning(f"🔴 Se detectaron {len(alto_rh_data)} aforos del GRUPO_ALTO_RH:")
                            st.dataframe(alto_rh_data[['NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 'RADIO_HIDRAULICO']])
                            
                            st.info("¿Deseas recalcular INCLUYENDO el GRUPO_ALTO_RH?")
                            
                            if st.button("🔄 RECALCULAR con GRUPO_ALTO_RH", key="btn_recalcular"):
                                with st.spinner("Recalculando con GRUPO_ALTO_RH..."):
                                    curvas_con, datos_con = procesar_con_modelo(modelo, df, incluir_alto_rh=True)
                                    
                                    st.success(f"✅ RECÁLCULO EXITOSO: {len(datos_con)} aforos (CON GRUPO_ALTO_RH)")
                                    
                                    st.subheader("📊 COMPARACIÓN: Con vs Sin GRUPO_ALTO_RH")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.metric("Aforos SIN GRUPO_ALTO_RH", len(datos_sin))
                                        st.metric("Curvas generadas", len(curvas_sin))
                                    
                                    with col2:
                                        st.metric("Aforos CON GRUPO_ALTO_RH", len(datos_con))
                                        st.metric("Curvas generadas", len(curvas_con))
                                    
                                    st.subheader("📊 NUEVOS Resultados (CON GRUPO_ALTO_RH)")
                                    datos_con_filtrados = datos_con[datos_con['GRUPO_PREDICHO'] != 'GRUPO_ESTANDAR']
                                    st.dataframe(datos_con_filtrados[['NIVEL_AFORO', 'CAUDAL', 'VELOCIDAD', 'AREA', 'GRUPO_PREDICHO']].head())
                                    
                                    st.subheader("📈 NUEVAS Curvas Altura-Caudal (CON GRUPO_ALTO_RH)")
                                    fig_con = crear_grafico_principal_con_personalizadas(datos_con, curvas_con, "Curvas CON GRUPO_ALTO_RH")
                                    st.pyplot(fig_con)
                                    
                                    st.subheader("📐 Ecuaciones y Rangos de Validez (CON GRUPO_ALTO_RH)")
                                    for grupo, curva in curvas_con.items():
                                        if grupo == 'GRUPO_ESTANDAR':
                                            continue
                                            
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
                    fig = crear_grafico_principal_con_personalizadas(datos_procesados, curvas, "Curvas Altura-Caudal - Datos Manuales")
                    st.pyplot(fig)
                    
                    st.subheader("📐 Ecuaciones y Rangos de Validez")
                    for grupo, curva in curvas.items():
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
                    
                    st.subheader("🔍 Análisis Complementario")
                    fig_comp = crear_graficos_complementarios(datos_procesados, "(Datos Manuales)")
                    st.pyplot(fig_comp)
                else:
                    st.warning("⚠️ No se pudieron generar curvas con los datos ingresados. Intenta con más puntos o diferentes valores.")

elif opcion == "📈 Curvas":
    st.header("📈 Visualización de Curvas")
    st.info("Esta sección muestra información sobre las curvas del modelo")
    
    if modelo is None:
        st.error("⚠️ El modelo no está disponible.")
    else:
        st.success("✅ Modelo cargado y listo para generar curvas")
        
        st.subheader("🔧 Información del Modelo")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Tipo de Modelo", "Random Forest")
            st.metric("Clases", "3 grupos")
        
        with col2:
            st.metric("Características", "9 variables")
            st.metric("Estado", "Activo")
        
        st.subheader("🎯 Grupos de Clasificación")
        grupos_info = {
            "GRUPO_ESTANDAR": "Condiciones normales de flujo",
            "GRUPO_RECIENTE": "Datos recientes o condiciones específicas", 
            "GRUPO_ALTO_RH": "Alto radio hidráulico o condiciones extremas"
        }
        
        for grupo, descripcion in grupos_info.items():
            with st.expander(f"{grupo}"):
                st.write(descripcion)

elif opcion == "➕ Insertar Curva Personalizada":
    st.header("➕ Insertar Curva Altura-Caudal Personalizada")
    st.info("Agrega tu propia curva altura-caudal teórica y compárala con las curvas generadas por IA")
    
    if 'curvas_sin_alto_rh' not in st.session_state or not st.session_state.curvas_sin_alto_rh:
        st.warning("⚠️ Primero necesitas generar curvas con IA en la sección '📤 Subir Aforos'")
        st.info("💡 Ve a la sección '📤 Subir Aforos', carga tus datos y genera las curvas antes de agregar una curva personalizada.")
    else:
        st.subheader("📊 Curvas IA Existentes")
        curvas_ia = st.session_state.curvas_sin_alto_rh
        
        for grupo, curva in curvas_ia.items():
            if grupo != 'GRUPO_ESTANDAR':
                rango_min, rango_max = curva.get('rango_validez', curva['rango_niveles'])
                st.write(f"**{grupo}**: {formatear_rango(rango_min, rango_max)} - R² = {curva['r2']:.3f}")
        
        st.subheader("🎯 Configuración de la Curva Personalizada")
        
        tipo_curva = st.selectbox(
            "Tipo de ecuación:",
            ["Potencial", "Polinómica G2", "Polinómica G3", "Exponencial", "Lineal"]
        )
        
        st.subheader("📐 Parámetros de la Curva")
        
        if tipo_curva == "Potencial":
            col1, col2 = st.columns(2)
            with col1:
                a = st.number_input("Coeficiente a:", value=2.5, step=0.1, format="%.4f")
            with col2:
                b = st.number_input("Exponente b:", value=1.8, step=0.1, format="%.4f")
            
            st.latex(f"Q = {a:.4f} \\times H^{{{b:.4f}}}")
            
        elif tipo_curva == "Polinómica G2":
            col1, col2, col3 = st.columns(3)
            with col1:
                a = st.number_input("Coeficiente a (H²):", value=0.2, step=0.01, format="%.4f")
            with col2:
                b = st.number_input("Coeficiente b (H):", value=1.5, step=0.01, format="%.4f")
            with col3:
                c = st.number_input("Coeficiente c:", value=0.1, step=0.01, format="%.4f")
            
            st.latex(f"Q = {a:.4f}H^2 + {b:.4f}H + {c:.4f}")
            
        elif tipo_curva == "Polinómica G3":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                a = st.number_input("Coeficiente a (H³):", value=0.05, step=0.001, format="%.4f")
            with col2:
                b = st.number_input("Coeficiente b (H²):", value=0.1, step=0.01, format="%.4f")
            with col3:
                c = st.number_input("Coeficiente c (H):", value=1.2, step=0.01, format="%.4f")
            with col4:
                d = st.number_input("Coeficiente d:", value=0.1, step=0.01, format="%.4f")
            
            st.latex(f"Q = {a:.4f}H^3 + {b:.4f}H^2 + {c:.4f}H + {d:.4f}")
            
        elif tipo_curva == "Exponencial":
            col1, col2 = st.columns(2)
            with col1:
                a = st.number_input("Coeficiente a:", value=1.5, step=0.1, format="%.4f")
            with col2:
                b = st.number_input("Exponente b:", value=0.8, step=0.01, format="%.4f")
            
            st.latex(f"Q = {a:.4f} \\times e^{{{b:.4f}H}}")
            
        elif tipo_curva == "Lineal":
            col1, col2 = st.columns(2)
            with col1:
                a = st.number_input("Pendiente a:", value=2.0, step=0.1, format="%.4f")
            with col2:
                b = st.number_input("Intercepto b:", value=0.5, step=0.1, format="%.4f")
            
            st.latex(f"Q = {a:.4f}H + {b:.4f}")
        
        st.subheader("📏 Rango de Validez de la Curva Personalizada")
        col_min, col_max = st.columns(2)
        with col_min:
            h_min_personalizada = st.number_input("Altura mínima H (m):", 
                                                min_value=0.0, value=0.5, step=0.1, format="%.2f",
                                                key="h_min_personal")
        with col_max:
            h_max_personalizada = st.number_input("Altura máxima H (m):", 
                                                min_value=0.0, value=3.5, step=0.1, format="%.2f",
                                                key="h_max_personal")
        
        if h_min_personalizada >= h_max_personalizada:
            st.error("❌ La altura mínima debe ser menor que la altura máxima")
        
        nombre_curva_personal = st.text_input("Nombre para identificar la curva:", 
                                            value="CURVA_PERSONALIZADA_TEORICA")
        
        if st.button("🚀 Generar y Comparar Curvas", type="primary") and h_min_personalizada < h_max_personalizada:
            with st.spinner("Generando curva personalizada y comparando..."):
                datos_ia = st.session_state.datos_sin_alto_rh
                curvas_ia = st.session_state.curvas_sin_alto_rh
                
                if tipo_curva == "Potencial":
                    def funcion_personalizada(H):
                        return a * (H ** b)
                elif tipo_curva == "Polinómica G2":
                    def funcion_personalizada(H):
                        return a * H**2 + b * H + c
                elif tipo_curva == "Polinómica G3":
                    def funcion_personalizada(H):
                        return a * H**3 + b * H**2 + c * H + d
                elif tipo_curva == "Exponencial":
                    def funcion_personalizada(H):
                        return a * np.exp(b * H)
                elif tipo_curva == "Lineal":
                    def funcion_personalizada(H):
                        return a * H + b
                
                alturas_unicas_ia = sorted(datos_ia['NIVEL_AFORO'].unique())
                h_min_total = min(h_min_personalizada, min(alturas_unicas_ia))
                h_max_total = max(h_max_personalizada, max(alturas_unicas_ia))
                
                H_comparacion = np.linspace(h_min_total, h_max_total, 200)
                
                Q_personalizada = []
                for h in H_comparacion:
                    if h_min_personalizada <= h <= h_max_personalizada:
                        Q_personalizada.append(funcion_personalizada(h))
                    else:
                        Q_personalizada.append(np.nan)
                
                Q_ia_combinado = np.zeros(len(H_comparacion))
                Q_ia_combinado[:] = np.nan
                
                for i, h in enumerate(H_comparacion):
                    for grupo, curva in curvas_ia.items():
                        if grupo != 'GRUPO_ESTANDAR':
                            if 'rango_validez' in curva:
                                rango_min, rango_max = curva['rango_validez']
                            else:
                                rango_min, rango_max = curva['rango_niveles']
                            
                            if rango_min <= h <= rango_max:
                                try:
                                    Q_ia_combinado[i] = curva['funcion'](h, *curva['parametros'])
                                    break
                                except:
                                    continue
                
                df_comparativo = pd.DataFrame({
                    'ALTURA': H_comparacion,
                    'CAUDAL_IA': Q_ia_combinado,
                    'CAUDAL_PERSONALIZADO': Q_personalizada
                })
                
                if 'curvas_personalizadas' not in st.session_state:
                    st.session_state.curvas_personalizadas = {}
                
                st.session_state.curvas_personalizadas[nombre_curva_personal] = {
                    'funcion': funcion_personalizada,
                    'parametros': {'a': a, 'b': b, 'c': c, 'd': d} if tipo_curva in ["Polinómica G2", "Polinómica G3"] else {'a': a, 'b': b},
                    'rango_validez': (h_min_personalizada, h_max_personalizada),
                    'rango_niveles': (h_min_personalizada, h_max_personalizada),
                    'rango_caudales': (min([q for q in Q_personalizada if not np.isnan(q)]), 
                                     max([q for q in Q_personalizada if not np.isnan(q)])),
                    'nombre': tipo_curva,
                    'r2': None,
                    'n_puntos': len(H_comparacion),
                    'datos': df_comparativo
                }
                
                st.success(f"✅ Curva '{nombre_curva_personal}' generada exitosamente!")
                
                st.subheader("📊 COMPARACIÓN: Curvas IA vs Curva Personalizada")
                
                fig_comparativo, ax = plt.subplots(figsize=(12, 8))
                
                grupos_ia = [g for g in datos_ia['GRUPO_PREDICHO'].unique() if g != 'GRUPO_ESTANDAR']
                colores_ia = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue'}
                marcadores_ia = {'GRUPO_ALTO_RH': 's', 'GRUPO_RECIENTE': '^'}
                
                for grupo in grupos_ia:
                    grupo_data = datos_ia[datos_ia['GRUPO_PREDICHO'] == grupo]
                    color = colores_ia.get(grupo, 'orange')
                    marcador = marcadores_ia.get(grupo, 'o')
                    
                    ax.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], 
                              color=color, marker=marcador, s=80, label=f'Datos {grupo}', 
                              alpha=0.7, edgecolors='black', linewidth=0.5)
                
                for grupo, curva in curvas_ia.items():
                    if grupo != 'GRUPO_ESTANDAR':
                        if 'rango_validez' in curva:
                            rango_min, rango_max = curva['rango_validez']
                        else:
                            rango_min, rango_max = curva['rango_niveles']
                        
                        H_curve_ia = np.linspace(rango_min, rango_max, 100)
                        Q_curve_ia = curva['funcion'](H_curve_ia, *curva['parametros'])
                        
                        color = colores_ia.get(grupo, 'orange')
                        ax.plot(H_curve_ia, Q_curve_ia, color=color, linewidth=2, 
                               label=f'Curva IA: {grupo} (R²={curva["r2"]:.3f})')
                
                mask_valido = ~np.isnan(Q_personalizada)
                if np.any(mask_valido):
                    ax.plot(H_comparacion[mask_valido], np.array(Q_personalizada)[mask_valido], 
                           color='purple', linewidth=3, linestyle='--',
                           label=f'Curva Personalizada: {nombre_curva_personal}')
                
                ax.set_xlabel('Nivel H (m)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Caudal Q (m³/s)', fontsize=12, fontweight='bold')
                ax.set_title('Comparación: Curvas IA vs Curva Personalizada', fontsize=14, fontweight='bold')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig_comparativo)
                
                st.subheader("📋 Tabla Comparativa de Caudales")
                
                alturas_comparacion = np.linspace(h_min_total, h_max_total, 20)
                datos_tabla = []
                
                for h in alturas_comparacion:
                    q_ia = np.nan
                    for grupo, curva in curvas_ia.items():
                        if grupo != 'GRUPO_ESTANDAR':
                            if 'rango_validez' in curva:
                                rango_min, rango_max = curva['rango_validez']
                            else:
                                rango_min, rango_max = curva['rango_niveles']
                            
                            if rango_min <= h <= rango_max:
                                try:
                                    q_ia = curva['funcion'](h, *curva['parametros'])
                                    break
                                except:
                                    continue
                    
                    q_personal = funcion_personalizada(h) if h_min_personalizada <= h <= h_max_personalizada else np.nan
                    
                    diferencia = q_personal - q_ia if not np.isnan(q_ia) and not np.isnan(q_personal) else np.nan
                    diferencia_porcentaje = (diferencia / q_ia * 100) if not np.isnan(diferencia) and q_ia != 0 else np.nan
                    
                    datos_tabla.append({
                        'Altura (m)': f"{h:.2f}",
                        'Caudal IA (m³/s)': f"{q_ia:.3f}" if not np.isnan(q_ia) else "Fuera de rango",
                        'Caudal Personalizado (m³/s)': f"{q_personal:.3f}" if not np.isnan(q_personal) else "Fuera de rango",
                        'Diferencia (m³/s)': f"{diferencia:.3f}" if not np.isnan(diferencia) else "N/A",
                        'Diferencia (%)': f"{diferencia_porcentaje:.1f}%" if not np.isnan(diferencia_porcentaje) else "N/A"
                    })
                
                df_tabla_comparativa = pd.DataFrame(datos_tabla)
                st.dataframe(df_tabla_comparativa)
                
                st.subheader("📈 Estadísticas de Comparación")
                
                datos_comparables = df_comparativo.dropna()
                
                if len(datos_comparables) > 0:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        diferencia_promedio = (datos_comparables['CAUDAL_PERSONALIZADO'] - datos_comparables['CAUDAL_IA']).mean()
                        st.metric("Diferencia Promedio", f"{diferencia_promedio:.3f} m³/s")
                    
                    with col2:
                        diferencia_maxima = (datos_comparables['CAUDAL_PERSONALIZADO'] - datos_comparables['CAUDAL_IA']).max()
                        st.metric("Diferencia Máxima", f"{diferencia_maxima:.3f} m³/s")
                    
                    with col3:
                        rmsd = np.sqrt(((datos_comparables['CAUDAL_PERSONALIZADO'] - datos_comparables['CAUDAL_IA'])**2).mean())
                        st.metric("Raíz Error Cuadrático", f"{rmsd:.3f} m³/s")
                    
                    fig_diferencias, ax2 = plt.subplots(figsize=(10, 5))
                    
                    alturas_validas = datos_comparables['ALTURA']
                    diferencias = datos_comparables['CAUDAL_PERSONALIZADO'] - datos_comparables['CAUDAL_IA']
                    
                    ax2.plot(alturas_validas, diferencias, 'red', linewidth=2, label='Diferencia (Personalizado - IA)')
                    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax2.set_xlabel('Nivel H (m)', fontweight='bold')
                    ax2.set_ylabel('Diferencia de Caudal (m³/s)', fontweight='bold')
                    ax2.set_title('Diferencia entre Curva Personalizada y Curvas IA', fontweight='bold')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    st.pyplot(fig_diferencias)
                else:
                    st.warning("⚠️ No hay superposición en los rangos de validez para comparar las curvas")
                
                st.subheader("💾 Descargar Datos de Comparación")
                
                csv_comparativo = df_comparativo.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar datos completos de comparación (CSV)",
                    data=csv_comparativo,
                    file_name=f"comparacion_curvas_{nombre_curva_personal}.csv",
                    mime="text/csv"
                )

st.markdown("---")
st.markdown("**🌊 IA para generar Curvas H-Q**")