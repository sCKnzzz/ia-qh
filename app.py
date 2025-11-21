
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Configuración
st.set_page_config(page_title="Sistema Talapalca", page_icon="🌊", layout="wide")
st.title("🌊 Sistema Inteligente de Curvas Altura-Caudal - TALAPALCA")
st.markdown("**Modelo entrenado con 34 aforos reales**")

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    try:
        return joblib.load('modelo_talapalca_entrenado.pkl')
    except:
        st.error("❌ Error al cargar el modelo")
        return None

modelo = cargar_modelo()

# Navegación
opcion = st.sidebar.radio("Navegación:", ["🏠 Inicio", "📊 Procesar Datos", "📈 Curvas"])

if opcion == "🏠 Inicio":
    st.header("Bienvenido al Sistema Talapalca")
    st.info("""
    **🎯 Funcionalidades:**
    - Clasificación automática de aforos
    - Generación de curvas altura-caudal
    - Modelo entrenado con datos reales
    - Interfaz fácil de usar
    """)
    
    # Mostrar datos de ejemplo
    try:
        datos_demo = pd.read_csv('datos_talapalca_demo.csv')
        st.subheader("📋 Datos de Ejemplo")
        st.dataframe(datos_demo.head(6))
    except:
        st.warning("No se encontraron datos de demo")

elif opcion == "📊 Procesar Datos":
    st.header("Procesar Nuevos Aforos")
    
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
                'NIVEL DE AFORO (m)': nivel, 'CAUDAL (m3/s)': caudal, 'AREA (m2)': area,
                'ANCHO RIO (m)': ancho, 'PERIMETRO (m)': perimetro, 'VELOCIDAD (m/s)': velocidad, 'YEAR': year
            })
    
    if st.button("🚀 Procesar") and nuevos_datos and modelo:
        try:
            df_nuevos = pd.DataFrame(nuevos_datos)
            curvas, clasificados = modelo.predecir_curvas(df_nuevos)
            
            st.success(f"✅ {len(df_nuevos)} aforos procesados")
            st.dataframe(clasificados)
            
            if curvas:
                # Gráfico
                fig, ax = plt.subplots(figsize=(10, 6))
                colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
                
                for grupo, curva in curvas.items():
                    color = colores.get(grupo, 'orange')
                    grupo_data = clasificados[clasificados['GRUPO_PREDICHO'] == grupo]
                    ax.scatter(grupo_data['NIVEL_AFORO'], grupo_data['CAUDAL'], color=color, s=80, label=grupo)
                    
                    H_range = np.linspace(curva['rango_niveles'][0]*0.9, curva['rango_niveles'][1]*1.1, 100)
                    Q_curve = curva['funcion'](H_range, *curva['parametros'])
                    ax.plot(H_range, Q_curve, color=color, linewidth=2, label=f"{grupo} (R²={curva['r2']:.3f})")
                
                ax.set_xlabel('Nivel (m)'); ax.set_ylabel('Caudal (m³/s)')
                ax.legend(); ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Ecuaciones
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
        except Exception as e:
            st.error(f"Error: {e}")

elif opcion == "📈 Curvas":
    st.header("Curvas del Modelo Actual")
    if modelo and modelo.modelos_por_grupo:
        for grupo, curva in modelo.modelos_por_grupo.items():
            with st.expander(f"{grupo} (R² = {curva['r2']:.3f})"):
                st.write(f"**Modelo:** {curva['nombre']}")
                st.write(f"**Puntos:** {curva['n_puntos']}")
                if curva['nombre'] == 'Polinómico G2':
                    a, b, c = curva['parametros']
                    st.latex(f"Q = {a:.4f}H^2 + {b:.4f}H + {c:.4f}")
                elif curva['nombre'] == 'Polinómico G3':
                    a, b, c, d = curva['parametros']
                    st.latex(f"Q = {a:.4f}H^3 + {b:.4f}H^2 + {c:.4f}H + {d:.4f}")
                elif curva['nombre'] == 'Potencial':
                    a, b = curva['parametros']
                    st.latex(f"Q = {a:.4f}H^{{{b:.4f}}}")

st.markdown("---")
st.markdown("**Sistema Hidráulico Inteligente - Estación Talapalca**")
