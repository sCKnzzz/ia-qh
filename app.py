# AGREGAR ESTA SECCIÓN DESPUÉS DE LA OPCIÓN "📈 Curvas" Y ANTES DEL FOOTER

elif opcion == "➕ Insertar Curva Personalizada":
    st.header("➕ Insertar Curva Altura-Caudal Personalizada")
    st.info("Agrega tu propia curva altura-caudal para visualizarla en los gráficos")
    
    # Selección del tipo de curva
    tipo_curva = st.selectbox(
        "Tipo de curva:",
        ["Potencial", "Polinómica G2", "Polinómica G3", "Exponencial"]
    )
    
    # Parámetros según el tipo de curva seleccionado
    st.subheader("📐 Parámetros de la Curva")
    
    if tipo_curva == "Potencial":
        col1, col2 = st.columns(2)
        with col1:
            a = st.number_input("Coeficiente a:", value=1.0, step=0.1, format="%.4f")
        with col2:
            b = st.number_input("Exponente b:", value=2.0, step=0.1, format="%.4f")
        
        # Mostrar ecuación
        st.latex(f"Q = {a:.4f} \\times H^{{{b:.4f}}}")
        
        # Definir función
        def curva_personalizada(H, a=a, b=b):
            return a * (H ** b)
            
    elif tipo_curva == "Polinómica G2":
        col1, col2, col3 = st.columns(3)
        with col1:
            a = st.number_input("Coeficiente a (H²):", value=0.1, step=0.01, format="%.4f")
        with col2:
            b = st.number_input("Coeficiente b (H):", value=0.5, step=0.01, format="%.4f")
        with col3:
            c = st.number_input("Coeficiente c:", value=0.1, step=0.01, format="%.4f")
        
        # Mostrar ecuación
        st.latex(f"Q = {a:.4f}H^2 + {b:.4f}H + {c:.4f}")
        
        # Definir función
        def curva_personalizada(H, a=a, b=b, c=c):
            return a * H**2 + b * H + c
            
    elif tipo_curva == "Polinómica G3":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            a = st.number_input("Coeficiente a (H³):", value=0.01, step=0.001, format="%.4f")
        with col2:
            b = st.number_input("Coeficiente b (H²):", value=0.1, step=0.01, format="%.4f")
        with col3:
            c = st.number_input("Coeficiente c (H):", value=0.5, step=0.01, format="%.4f")
        with col4:
            d = st.number_input("Coeficiente d:", value=0.1, step=0.01, format="%.4f")
        
        # Mostrar ecuación
        st.latex(f"Q = {a:.4f}H^3 + {b:.4f}H^2 + {c:.4f}H + {d:.4f}")
        
        # Definir función
        def curva_personalizada(H, a=a, b=b, c=c, d=d):
            return a * H**3 + b * H**2 + c * H + d
            
    elif tipo_curva == "Exponencial":
        col1, col2 = st.columns(2)
        with col1:
            a = st.number_input("Coeficiente a:", value=1.0, step=0.1, format="%.4f")
        with col2:
            b = st.number_input("Exponente b:", value=0.5, step=0.01, format="%.4f")
        
        # Mostrar ecuación
        st.latex(f"Q = {a:.4f} \\times e^{{{b:.4f}H}}")
        
        # Definir función
        def curva_personalizada(H, a=a, b=b):
            return a * np.exp(b * H)
    
    # Rango de validez
    st.subheader("📏 Rango de Validez")
    col_min, col_max = st.columns(2)
    with col_min:
        h_min = st.number_input("Altura mínima H (m):", min_value=0.0, value=0.5, step=0.1, format="%.2f")
    with col_max:
        h_max = st.number_input("Altura máxima H (m):", min_value=0.0, value=3.6, step=0.1, format="%.2f")
    
    # Validar rango
    if h_min >= h_max:
        st.error("❌ La altura mínima debe ser menor que la altura máxima")
    
    # Nombre de la curva personalizada
    nombre_curva = st.text_input("Nombre de la curva personalizada:", value="CURVA_PERSONALIZADA")
    
    # Generar datos de la curva
    if st.button("🚀 Generar y Visualizar Curva", type="primary") and h_min < h_max:
        with st.spinner("Generando curva..."):
            # Generar puntos de la curva
            H_curve = np.linspace(h_min, h_max, 100)
            Q_curve = curva_personalizada(H_curve)
            
            # Crear DataFrame con los datos generados
            datos_curva = pd.DataFrame({
                'NIVEL_AFORO': H_curve,
                'CAUDAL': Q_curve,
                'GRUPO_PREDICHO': nombre_curva,
                'VELOCIDAD': Q_curve / 5.0,  # Valor estimado para visualización
                'AREA': H_curve * 8.0,       # Valor estimado para visualización
                'ANCHO_RIO': 8.0,            # Valor por defecto
                'PERIMETRO': H_curve * 2 + 8.0,  # Valor estimado
                'RADIO_HIDRAULICO': (H_curve * 8.0) / (H_curve * 2 + 8.0),  # Valor estimado
                'TIRANTE_MEDIO': H_curve,
                'CAUDAL_AREA': Q_curve / (H_curve * 8.0)
            })
            
            # Almacenar en session state para usar en otras secciones
            if 'curvas_personalizadas' not in st.session_state:
                st.session_state.curvas_personalizadas = {}
            
            # Guardar información de la curva
            st.session_state.curvas_personalizadas[nombre_curva] = {
                'funcion': curva_personalizada,
                'parametros': locals(),  # Guardar parámetros actuales
                'rango_validez': (h_min, h_max),
                'rango_niveles': (h_min, h_max),
                'rango_caudales': (min(Q_curve), max(Q_curve)),
                'nombre': tipo_curva,
                'r2': 1.0,  # R² perfecto para curva teórica
                'n_puntos': 100,
                'datos': datos_curva
            }
            
            st.success(f"✅ Curva '{nombre_curva}' generada exitosamente!")
            
            # Mostrar información de la curva
            st.subheader("📊 Información de la Curva Generada")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Rango de alturas", f"{h_min:.2f} - {h_max:.2f} m")
                st.metric("Rango de caudales", f"{min(Q_curve):.2f} - {max(Q_curve):.2f} m³/s")
            
            with col2:
                st.metric("Tipo de curva", tipo_curva)
                st.metric("Puntos generados", 100)
            
            # Visualizar la curva
            st.subheader("📈 Gráfico de la Curva Personalizada")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Graficar curva
            ax.plot(H_curve, Q_curve, 'purple', linewidth=3, label=f"{nombre_curva} ({tipo_curva})")
            
            # Configurar gráfico
            ax.set_xlabel('Nivel (m)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Caudal (m³/s)', fontsize=12, fontweight='bold')
            ax.set_title(f'Curva Personalizada: {nombre_curva}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Mostrar tabla de valores
            st.subheader("📋 Tabla de Valores (primeros 10 puntos)")
            st.dataframe(datos_curva[['NIVEL_AFORO', 'CAUDAL']].head(10))
            
            # Opción para descargar datos
            csv = datos_curva[['NIVEL_AFORO', 'CAUDAL']].to_csv(index=False)
            st.download_button(
                label="📥 Descargar datos de la curva (CSV)",
                data=csv,
                file_name=f"curva_personalizada_{nombre_curva}.csv",
                mime="text/csv"
            )

# MODIFICAR LAS FUNCIONES DE GRÁFICOS PARA INCLUIR CURVAS PERSONALIZADAS

def crear_grafico_principal_con_personalizadas(df, curvas, titulo):
    """Versión mejorada que incluye curvas personalizadas"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colores = {
        'GRUPO_ALTO_RH': 'red', 
        'GRUPO_RECIENTE': 'blue', 
        'GRUPO_ESTANDAR': 'green',
        'CURVA_PERSONALIZADA': 'purple'
    }
    
    marcadores = {
        'GRUPO_ALTO_RH': 's', 
        'GRUPO_RECIENTE': '^', 
        'GRUPO_ESTANDAR': 'o',
        'CURVA_PERSONALIZADA': 'D'
    }
    
    tamanos = {
        'GRUPO_ALTO_RH': 100, 
        'GRUPO_RECIENTE': 80, 
        'GRUPO_ESTANDAR': 80,
        'CURVA_PERSONALIZADA': 60
    }
    
    # Primero graficar todos los puntos (EXCLUYENDO GRUPO_ESTANDAR)
    grupos_unicos = df['GRUPO_PREDICHO'].unique() if 'GRUPO_PREDICHO' in df.columns else []
    
    for grupo in grupos_unicos:
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
    
    # Luego graficar las curvas con sus rangos de validez (incluyendo personalizadas)
    todas_las_curvas = curvas.copy()
    
    # Agregar curvas personalizadas del session state
    if 'curvas_personalizadas' in st.session_state:
        todas_las_curvas.update(st.session_state.curvas_personalizadas)
    
    for grupo, curva in todas_las_curvas.items():
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
        Q_curve = curva['funcion'](H_range, *[curva['parametros'][key] for key in ['a', 'b', 'c', 'd'] if key in curva['parametros']])
        
        # Hacer la línea más gruesa para GRUPO_ALTO_RH y curvas personalizadas
        linewidth = 3 if grupo == 'GRUPO_ALTO_RH' or 'PERSONALIZADA' in grupo.upper() else 2
        
        # Estilo diferente para curvas personalizadas
        linestyle = '--' if 'PERSONALIZADA' in grupo.upper() else '-'
        
        # Agregar información del rango de validez en la etiqueta
        rango_formateado = formatear_rango(rango_min, rango_max)
        
        if 'PERSONALIZADA' in grupo.upper():
            label = f"{grupo} (Teórica)\n{rango_formateado} m"
        else:
            label = f"{grupo} (R²={curva.get('r2', 'N/A'):.3f})\n{rango_formateado} m"
        
        ax.plot(H_range, Q_curve, color=color, linewidth=linewidth, label=label, linestyle=linestyle)
    
    ax.set_xlabel('Nivel (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Caudal (m³/s)', fontsize=12, fontweight='bold')
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# ACTUALIZAR LA NAVEGACIÓN PARA INCLUIR LA NUEVA OPCIÓN
# (Reemplaza la línea existente de navegación con esta)

opcion = st.sidebar.radio("Navegación:", [
    "🏠 Inicio", 
    "📤 Subir Aforos", 
    "📊 Ingreso Manual", 
    "📈 Curvas",
    "➕ Insertar Curva Personalizada"  # NUEVA OPCIÓN
])

# MODIFICAR LAS SECCIONES EXISTENTES PARA USAR LA NUEVA FUNCIÓN DE GRÁFICOS

# En la sección "📤 Subir Aforos", reemplaza:
# fig_sin = crear_grafico_principal(datos_sin, curvas_sin, "Curvas sin GRUPO_ALTO_RH")
# con:
# fig_sin = crear_grafico_principal_con_personalizadas(datos_sin, curvas_sin, "Curvas sin GRUPO_ALTO_RH")

# Y:
# fig_con = crear_grafico_principal(datos_con, curvas_con, "Curvas CON GRUPO_ALTO_RH")  
# con:
# fig_con = crear_grafico_principal_con_personalizadas(datos_con, curvas_con, "Curvas CON GRUPO_ALTO_RH")

# En la sección "📊 Ingreso Manual", reemplaza:
# fig = crear_grafico_principal(datos_procesados, curvas, "Curvas Altura-Caudal - Datos Manuales")
# con:
# fig = crear_grafico_principal_con_personalizadas(datos_procesados, curvas, "Curvas Altura-Caudal - Datos Manuales")

st.markdown("---")
st.markdown("**🌊 IA para generar Curvas H-Q**")