import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
#instalar esto: pip install streamlit tensorflow numpy
#ejecutar: streamlit run app.py
# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="LSTM Cantuña", page_icon="📝", layout="centered")

# --- 2. CARGAR MODELO Y DICCIONARIO (Con Caché) ---
# El caché evita que la computadora colapse cargando la red neuronal en cada clic
@st.cache_resource
def cargar_cerebro():
    try:
        modelo = load_model('modelo_autocompletado_profundo.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return modelo, tokenizer
    except Exception as e:
        st.error(f"⚠️ Error al cargar los archivos: {e}")
        return None, None

modelo, tokenizer = cargar_cerebro()

# --- 3. LÓGICA DE PREDICCIÓN (Fase 3 adaptada) ---
def obtener_sugerencias(texto_semilla, temperatura):
    if not modelo or not tokenizer:
        return []
        
    secuencias = tokenizer.texts_to_sequences([texto_semilla])
    if len(secuencias[0]) == 0:
        return [] # Palabra fuera del vocabulario

    # Relleno exacto para la ventana del LSTM
    tokens = pad_sequences([secuencias[0]], maxlen=5, padding='pre')
    
    # Predicción
    probas = modelo.predict(tokens, verbose=0)[0]
    probas[0] = 0.0 # Ignorar el padding
    
    # Aplicar Temperatura para Inferencia Estocástica
    probas = np.log(probas + 1e-7) / float(temperatura)
    exp_probas = np.exp(probas)
    probas = exp_probas / np.sum(exp_probas)
    
    # Extraer el Top 3
    mejores_indices = np.argsort(probas)[-3:][::-1]
    
    sugerencias = []
    for indice in mejores_indices:
        for palabra, i in tokenizer.word_index.items():
            if i == indice:
                sugerencias.append(palabra)
                break
    return sugerencias

# --- 4. INTERFAZ VISUAL ---
st.title("📝 LSTM Autocomplete: Cantuña")
st.caption("Red Neuronal Recurrente para Procesamiento de Lenguaje Natural.")

# Inicializar la memoria de la sesión para el texto
if 'texto_usuario' not in st.session_state:
    st.session_state.texto_usuario = ""

# Área de texto vinculada a la memoria de la sesión
texto_actual = st.text_area(
    "Escribe el inicio de la leyenda:", 
    value=st.session_state.texto_usuario, 
    height=150
)

# Controladores
temperatura = st.slider("Temperatura (Creatividad vs Precisión):", min_value=0.1, max_value=1.5, value=0.7, step=0.1)

# Botón para accionar la red
if st.button("🧠 Generar Predicciones", use_container_width=True):
    if texto_actual:
        with st.spinner('Consultando la red neuronal...'):
            sugerencias = obtener_sugerencias(texto_actual, temperatura)
            
            if sugerencias:
                st.subheader("Sugerencias:")
                # Crear 3 columnas para que los botones se vean como un teclado predictivo
                cols = st.columns(3)
                
                for idx, palabra in enumerate(sugerencias):
                    # Usamos un callback para actualizar el texto si se hace clic
                    def actualizar_texto(palabra_elegida=palabra):
                        st.session_state.texto_usuario = texto_actual + " " + palabra_elegida
                        
                    cols[idx].button(
                        palabra, 
                        key=f"btn_{idx}_{palabra}", 
                        use_container_width=True, 
                        on_click=actualizar_texto
                    )
            else:
                st.warning("Escribe una palabra válida de la leyenda.")
    else:
        st.info("Por favor, escribe algo primero.")