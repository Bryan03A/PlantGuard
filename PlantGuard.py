import os
import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import joblib
import pandas as pd
import requests
import cohere
import tempfile
import numpy as np

# Configura tu clave de API de Cohere
api_key = os.getenv('COHERE_API_KEY')
co = cohere.Client(api_key)

# Función para obtener respuesta de Cohere
def get_response(prompt):
    response = co.generate(
        model='command-r-plus',
        prompt=prompt,
        max_tokens=300
    )
    return response.generations[0].text.strip()

# Función para guardar resultados en un archivo temporal
def save_results_to_tempfile(predicted_class, grado, porcentaje, temperature, humidity, precipitation, region_message):
    content = (
        f"Enfermedad: {predicted_class}\n"
        f"Grado de mortalidad: {grado}\n"
        f"Porcentaje de mortalidad: {porcentaje}%\n"
        f"Temperatura: {temperature}°C\n"
        f"Humedad: {humidity}%\n"
        f"Precipitación: {precipitation} mm\n"
        f"{region_message}"
    )
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8')
    with open(temp_file.name, 'w', encoding='utf-8') as file:
        file.write(content)
    return temp_file.name

def get_analysis_from_cohere(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    context = (
        "Eres un agricultor experto y quiero que analices la enfermedad y los factores que rodean a la planta. "
        "Considera los siguientes aspectos en tu análisis:\n"
        "- Análisis General (Título):\n"
        "Aquí me dirás tu análisis general de la planta (cuerpo)\n"
        "- Grado y Porcentaje de Mortalidad (Título):\n"
        "Aquí harás un análisis del grado y porcentaje de mortalidad (cuerpo)\n"
        "- Factores Climáticos (Título):\n"
        "Aquí me dirás un análisis de los factores del clima y la región que pueden afectar a la planta (cuerpo)\n"
        "- Recomendaciones (Título):\n"
        "Aquí me dirás un listado de recomendaciones que pueden ayudarnos a mantener un buen estado de la planta o, "
        "si la planta tiene altas probabilidades de morir, cómo evitarlas para futuros cultivos (solo 3)\n"
        "Realiza tu análisis dentro del límite de 150 tokens."
    )
    prompt = context + content
    analysis = get_response(prompt)
    
    # Procesar el análisis para una mejor presentación
    formatted_analysis = analysis.replace('\n', '\n\n')  # Agregar doble salto de línea entre párrafos
    formatted_analysis = formatted_analysis.replace('- ', '\n- ')  # Asegurar que las listas sean legibles
    
    return formatted_analysis

# Cargar el modelo entrenado de predicción de mortalidad
mortality_model = joblib.load('./model/plant_disease_combined_model.pkl')

# Diccionarios necesarios (cargar desde archivos)
def load_data(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            key, value = line.strip().split(',')
            data[key] = value
    return data

provincias = load_data('./data/archivos/provincias_regiones.txt')
class_labels = load_data('./data/archivos/class_labels.txt')
enfermedades_favorecidas = load_data('./data/archivos/enfermedades_favorecidas.txt')

# Lista de provincias
provincias_list = [
    "Azuay", "Bolívar", "Cañar", "Carchi", "Chimborazo", "Cotopaxi", 
    "El Oro", "Esmeraldas", "Galápagos", "Guayas", "Imbabura", 
    "Loja", "Los Ríos", "Manabí", "Morona Santiago", "Napo", 
    "Orellana", "Pastaza", "Pichincha", "Sucumbíos", "Santa Elena", 
    "Santo Domingo de los Tsáchilas", "Tungurahua", "Zamora-Chinchipe"
]

# Función para predecir la clase de la imagen
def predict_image(model, image):
    try:
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, axis=0)

        logits = model(image_tensor)
        probabilities = tf.nn.softmax(logits, axis=-1).numpy().flatten()
        predicted_index = np.argmax(probabilities)
        predicted_label = class_labels[str(predicted_index)]

        # st.write(f"Procesando...")

        # Verificar el índice de precisión
        precision_threshold = 0.06  # Ajustar el umbral según sea necesario
        if probabilities[predicted_index] < precision_threshold:
            st.error("La imagen cargada no parece ser una planta o es una hoja que no se encuentra registrada.")
            return None
        return predicted_label
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
        return None

# Función para obtener datos meteorológicos de la API de OpenWeather
def get_weather_data(province):
    cities = { "Esmeraldas": "Esmeraldas", "Guayas": "Guayaquil", "Los Ríos": "Babahoyo", 
                "Manabí": "Portoviejo", "Santa Elena": "La Libertad", "Santo Domingo de los Tsáchilas": "Santo Domingo", 
                "El Oro": "Machala", "Galápagos": "Puerto Ayora", "Azuay": "Cuenca", "Bolívar": "Guaranda", 
                "Cañar": "Azogues", "Carchi": "Tulcán", "Chimborazo": "Riobamba", "Cotopaxi": "Latacunga", 
                "Imbabura": "Ibarra", "Loja": "Loja", "Pichincha": "Quito", "Tungurahua": "Ambato", 
                "Sucumbíos": "Nueva Loja", "Napo": "Tena", "Pastaza": "Puyo", "Morona Santiago": "Macas", 
                "Zamora-Chinchipe": "Zamora", "Orellana": "Coca" }
    
    city = cities.get(province)
    if not city:
        raise ValueError(f"Provincia no encontrada: {province}")

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},EC&appid=f1408d640a5e38ddbdb8b8123568e8a5&units=metric"
    response = requests.get(url)
    data = response.json()

    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    precipitation = data.get('rain', {}).get('1h', 0)

    return temperature, humidity, precipitation

# Función para predecir la mortalidad
def predict_mortality(disease, province):
    try:
        temperature, humidity, precipitation = get_weather_data(province)

        region_actual = provincias.get(province)
        region_favorecida = enfermedades_favorecidas.get(disease)

        if not region_favorecida:
            raise ValueError(f"Enfermedad no encontrada: {disease}")

        input_data = pd.DataFrame([[disease, temperature, humidity, precipitation, region_actual]], 
                                  columns=['Enfermedad', 'Temperatura', 'Humedad', 'Precipitacion', 'Region'])

        predictions = mortality_model.predict(input_data)

        grado_mortalidad = round(predictions[0][0])
        porcentaje_mortalidad = round(predictions[0][1] * 10, 2)
        
        region_message = f"La región {region_actual} favorece a la planta." if region_actual == region_favorecida else \
                         f"Esta planta es de la región {region_favorecida}, en la región {region_actual} disminuye su esperanza de vida."

        return grado_mortalidad, porcentaje_mortalidad, temperature, humidity, precipitation, region_message
    except Exception as e:
        st.error(f"Error al predecir la mortalidad: {e}")
        return None, None, None, None, None, None

def predict_disease_and_mortality(image, province):
    saved_model_url = "./model/saved_model"  # Aqui la ruta de el modelo
    try:
        disease_model = hub.load(saved_model_url)
    except Exception as e:
        st.error(f"Error al cargar el modelo de enfermedad: {e}")
        return None, None, None, None, None, None, None

    # Asegúrate de que la imagen es un objeto PIL.Image
    if isinstance(image, Image.Image):
        predicted_class = predict_image(disease_model, image)
        if predicted_class:
            grado, porcentaje, temperature, humidity, precipitation, region_message = predict_mortality(predicted_class, province)
            return predicted_class, grado, porcentaje, temperature, humidity, precipitation, region_message
        else:
            return None, None, None, None, None, None, None
    else:
        st.error("Por favor, carga una imagen válida.")
        return None, None, None, None, None, None, None

def format_text(text):
    # Añadir encabezados para cada sección
    text = text.replace("Análisis General:", "\n**Análisis General**:\n")
    text = text.replace("Grado y Porcentaje de Mortalidad:", "\n**Grado y Porcentaje de Mortalidad**:\n")
    text = text.replace("Factores Climáticos:", "\n**Factores Climáticos**:\n")
    text = text.replace("Recomendaciones:", "\n**Recomendaciones**:\n")
    
    # Reemplazar saltos de línea simples por dobles para separar párrafos
    text = text.replace('\n', '\n\n')

    # Añadir un espacio antes de cada lista de recomendaciones
    text = text.replace('- ', '\n- ')
    
    return text.strip()

# Interfaz de usuario
# Logo ASCII de planta
# Contenido HTML centrado
content_html = """
<div style="text-align: center;">
    <h1>PlantGuard</h1>
    <div>
        <pre>
                                     ▄▀▀▀▀▀▀▀▄                                        
                                    ▐─▄█▀▀▀█▄─▌                                       
                                    ▐─▀█▄▄▄█▀─▌                                       
                                     ▀▄▄▄▄▄▄▄▀                                        
                                    ▐▀▄▄▐█▌▄▄▀▌                                       
                                     ▀▄▄███▄▄▀                                        
        </pre>
    </div>
    <h3>Sistema de Predicción de Enfermedades y Mortalidad</h3>
</div>
"""

# Configura el título, el logo y el subtítulo
st.markdown(content_html, unsafe_allow_html=True)


# Opciones de carga o captura de imagen
option = st.selectbox("Selecciona una opción para cargar una imagen", ("Cargar imagen", "Capturar con cámara"))

if option == "Cargar imagen":
    uploaded_file = st.file_uploader("Elige una imagen de la planta", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.image = image
        except Exception as e:
            st.error(f"Error al cargar la imagen: {e}")
elif option == "Capturar con cámara":
    captured_image = st.camera_input("Captura una imagen de la planta")

    if captured_image:
        try:
            image = Image.open(captured_image).convert("RGB")
            st.session_state.image = image
        except Exception as e:
            st.error(f"Error al capturar la imagen: {e}")

if 'image' in st.session_state and st.session_state.image:
    st.image(st.session_state.image, caption='Imagen cargada', use_column_width=True)

    province = st.selectbox("Selecciona una provincia", provincias_list)
    
    if st.button("Predecir"):
        if province:
            predicted_class, grado, porcentaje, temperature, humidity, precipitation, region_message = \
                predict_disease_and_mortality(st.session_state.image, province)

            if predicted_class:
                st.session_state.predicted_class = predicted_class
                st.session_state.grado = grado
                st.session_state.porcentaje = porcentaje
                st.session_state.temperature = temperature
                st.session_state.humidity = humidity
                st.session_state.precipitation = precipitation
                st.session_state.region_message = region_message

                file_path = save_results_to_tempfile(predicted_class, grado, porcentaje, temperature, humidity, precipitation, region_message)

                st.session_state.analysis = get_analysis_from_cohere(file_path)

                # Mostrar resultados en una tabla
                st.write("**Resultados de la predicción:**")
                
                results_df = pd.DataFrame({
                    'Descripción': [
                        'Detectado', 
                        'Grado de Mortalidad', 
                        'Porcentaje de Mortalidad', 
                        'Temperatura', 
                        'Humedad', 
                        'Precipitación', 
                        'Mensaje Regional'
                    ],
                    'Valor': [
                        st.session_state.predicted_class,
                        st.session_state.grado,
                        f"{st.session_state.porcentaje}%",
                        f"{st.session_state.temperature}°C",
                        f"{st.session_state.humidity}%",
                        f"{st.session_state.precipitation} mm",
                        st.session_state.region_message
                    ]
                })

                st.table(results_df)

                st.write("**Análisis del Dr. PlantON**")
                formatted_analysis = format_text(st.session_state.analysis)
                st.write(formatted_analysis)
else:
    st.write("Por favor, carga o captura una imagen.")
