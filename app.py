# ... (código anterior se mantiene igual)

                                # VERIFICAR SI EXISTE GRUPO_ALTO_RH EN LOS DATOS ORIGINALES Y OFRECER OPCIÓN DE INCLUSIÓN
                                # Primero procesamos para ver si existe GRUPO_ALTO_RH en los datos originales
                                _, clasificados_completos = predecir_curvas_con_filtro(modelo, df_subido, incluir_alto_rh=True)
                                
                                if 'GRUPO_ALTO_RH' in clasificados_completos['GRUPO_PREDICHO'].values:
                                    st.subheader("⚙️ Opción de Re-análisis")
                                    st.info("""
                                    **Se detectó GRUPO_ALTO_RH en los datos.** 
                                    Este grupo representa aforos con alto radio hidráulico que pueden mejorar el análisis.
                                    ¿Deseas recalcular las curvas INCLUYENDO este grupo?
                                    """)
                                    
                                    if st.button("🔄 Recalcular INCLUYENDO GRUPO_ALTO_RH"):
                                        with st.spinner("Recalculando curvas con GRUPO_ALTO_RH..."):
                                            try:
                                                # Recalcular INCLUYENDO GRUPO_ALTO_RH
                                                curvas_con_alto_rh, clasificados_con_alto_rh = predecir_curvas_con_filtro(modelo, df_subido, incluir_alto_rh=True)
                                                
                                                st.success(f"✅ Re-cálculo exitoso. Aforos utilizados: {len(clasificados_con_alto_rh)}")
                                                
                                                # Mostrar nueva distribución
                                                st.subheader("📈 Nueva Distribución de Grupos (con GRUPO_ALTO_RH)")
                                                distribucion_nueva = clasificados_con_alto_rh['GRUPO_PREDICHO'].value_counts()
                                                cols_nuevos = st.columns(4)
                                                for i, (grupo, count) in enumerate(distribucion_nueva.items()):
                                                    with cols_nuevos[i % 4]:
                                                        st.metric(f"Grupo {grupo}", count)
                                                
                                                if curvas_con_alto_rh:
                                                    # GRÁFICO PRINCIPAL CON GRUPO_ALTO_RH
                                                    st.subheader("📈 Curvas Altura-Caudal (con GRUPO_ALTO_RH)")
                                                    fig_principal, ax_principal = plt.subplots(figsize=(8, 5))
                                                    
                                                    colores = {'GRUPO_ALTO_RH': 'red', 'GRUPO_RECIENTE': 'blue', 'GRUPO_ESTANDAR': 'green'}
                                                    marcadores = {'GRUPO_ALTO_RH': 's', 'GRUPO_RECIENTE': '^', 'GRUPO_ESTANDAR': 'o'}
                                                    
                                                    for grupo, curva in curvas_con_alto_rh.items():
                                                        color = colores.get(grupo, 'orange')
                                                        marcador = marcadores.get(grupo, 'o')
                                                        grupo_data = clasificados_con_alto_rh[clasificados_con_alto_rh['GRUPO_PREDICHO'] == grupo]
                                                        
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
                                                    ax_principal.set_title('Curvas Altura-Caudal (con GRUPO_ALTO_RH)', fontsize=11, fontweight='bold')
                                                    ax_principal.legend(fontsize=8)
                                                    ax_principal.grid(True, alpha=0.3, linestyle='--')
                                                    ax_principal.spines['top'].set_visible(False)
                                                    ax_principal.spines['right'].set_visible(False)
                                                    st.pyplot(fig_principal)
                                                    
                                                    # GRÁFICOS COMPLEMENTARIOS CON GRUPO_ALTO_RH
                                                    st.subheader("🔍 Análisis de Relaciones Hidráulicas (con GRUPO_ALTO_RH)")
                                                    fig_complementarios = crear_graficos_complementarios(clasificados_con_alto_rh, curvas_con_alto_rh)
                                                    st.pyplot(fig_complementarios)
                                                    
                                                    # Ecuaciones detalladas con GRUPO_ALTO_RH
                                                    st.subheader("📐 Ecuaciones de las Curvas (con GRUPO_ALTO_RH)")
                                                    for grupo, curva in curvas_con_alto_rh.items():
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
                                                    
                                                    # Resumen del análisis con GRUPO_ALTO_RH
                                                    st.subheader("📊 Resumen del Análisis (con GRUPO_ALTO_RH)")
                                                    col1, col2 = st.columns(2)
                                                    with col1:
                                                        st.metric("Total de aforos utilizados", len(clasificados_con_alto_rh))
                                                        st.metric("Número de grupos identificados", len(curvas_con_alto_rh))
                                                    with col2:
                                                        st.metric("GRUPO_ALTO_RH incluido", "Sí")
                                                        # Calcular R² promedio
                                                        r2_promedio = np.mean([curva['r2'] for curva in curvas_con_alto_rh.values()])
                                                        st.metric("R² promedio", f"{r2_promedio:.3f}")
                                                    
                                                    # Opción para descargar resultados con GRUPO_ALTO_RH
                                                    st.subheader("💾 Descargar Resultados (con GRUPO_ALTO_RH)")
                                                    resultado_csv_con_alto_rh = clasificados_con_alto_rh[columnas_a_mostrar].to_csv(index=False)
                                                    st.download_button(
                                                        label="📥 Descargar Resultados con GRUPO_ALTO_RH",
                                                        data=resultado_csv_con_alto_rh,
                                                        file_name="resultados_aforos_talapalca_con_alto_rh.csv",
                                                        mime="text/csv"
                                                    )
                                                
                                                else:
                                                    st.warning("⚠️ No se pudieron generar curvas después de incluir GRUPO_ALTO_RH")
                                                    st.info("💡 Se necesitan al menos 3 aforos por grupo para generar curvas")
                                                    
                                            except Exception as e:
                                                st.error(f"❌ Error al recalcular los datos: {str(e)}")

                                # MOSTRAR RESULTADOS INICIALES (sin GRUPO_ALTO_RH)
                                if curvas:
                                    # GRÁFICO PRINCIPAL MEJORADO - TAMAÑO REDUCIDO
                                    st.subheader("📈 Curvas Altura-Caudal Generadas (sin GRUPO_ALTO_RH)")
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
                                    ax_principal.set_title('Curvas Altura-Caudal (sin GRUPO_ALTO_RH)', fontsize=11, fontweight='bold')
                                    ax_principal.legend(fontsize=8)
                                    ax_principal.grid(True, alpha=0.3, linestyle='--')
                                    ax_principal.spines['top'].set_visible(False)
                                    ax_principal.spines['right'].set_visible(False)
                                    st.pyplot(fig_principal)
                                    
                                    # GRÁFICOS COMPLEMENTARIOS
                                    st.subheader("🔍 Análisis de Relaciones Hidráulicas (sin GRUPO_ALTO_RH)")
                                    fig_complementarios = crear_graficos_complementarios(clasificados, curvas)
                                    st.pyplot(fig_complementarios)
                                    
                                    # Ecuaciones detalladas
                                    st.subheader("📐 Ecuaciones de las Curvas Generadas (sin GRUPO_ALTO_RH)")
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
                                    
                                    # Resumen del análisis sin GRUPO_ALTO_RH
                                    st.subheader("📊 Resumen del Análisis (sin GRUPO_ALTO_RH)")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Total de aforos utilizados", len(clasificados))
                                        st.metric("Número de grupos identificados", len(curvas))
                                    with col2:
                                        st.metric("GRUPO_ALTO_RH incluido", "No")
                                        # Calcular R² promedio
                                        r2_promedio = np.mean([curva['r2'] for curva in curvas.values()])
                                        st.metric("R² promedio", f"{r2_promedio:.3f}")
                                    
                                    # Opción para descargar resultados
                                    st.subheader("💾 Descargar Resultados (sin GRUPO_ALTO_RH)")
                                    resultado_csv = clasificados[columnas_a_mostrar].to_csv(index=False)
                                    st.download_button(
                                        label="📥 Descargar Resultados sin GRUPO_ALTO_RH",
                                        data=resultado_csv,
                                        file_name="resultados_aforos_talapalca_sin_alto_rh.csv",
                                        mime="text/csv"
                                    )
                                    
                                else:
                                    st.warning("⚠️ No se pudieron generar curvas con los datos proporcionados")
                                    st.info("💡 Se necesitan al menos 3 aforos por grupo para generar curvas")

# ... (el resto del código se mantiene igual)