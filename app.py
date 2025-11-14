import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os

# ================================
# CONFIGURACIÓN DE LA PÁGINA
# ================================
st.set_page_config(
    page_title="Detección de Objetos Urbanos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# ESTILOS
# ================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    h1, h2, h3 {
        color: #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# TÍTULO
# ================================
st.title("🚗 Detección de Objetos Urbanos")
st.markdown("Carga una imagen o video para detectar carros, peatones, autobuses, camiones, motocicletas y semáforos.")

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.header("⚙️ Configuración del Modelo")

    model_choice = st.selectbox(
        "Selecciona la versión de YOLO11",
        ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
        index=2
    )

    confidence_threshold = st.slider(
        "Confianza mínima",
        0.1, 1.0, 0.35, 0.05
    )

    st.markdown("---")
    st.markdown("""
    ### Objetos Detectados:
    - 🚗 Carros  
    - 🚌 Autobuses  
    - 🚚 Camiones  
    - 🚶 Personas  
    - 🚦 Semáforos  
    - 🏍️ Motocicletas  
    """)

# ================================
# CARGA DEL MODELO (CACHE)
# ================================
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

model = load_model(model_choice)

# ================================
# CONFIGURACIÓN DE CLASES
# ================================
ALLOWED_CLASSES = [0, 2, 3, 5, 7, 9]

class_mapping = {
    2: {"name": "Carro", "color": (255, 0, 0)},
    5: {"name": "Autobús", "color": (0, 255, 255)},
    7: {"name": "Camión", "color": (255, 165, 0)},
    0: {"name": "Persona", "color": (0, 255, 0)},
    9: {"name": "Semáforo", "color": (255, 255, 0)},
    3: {"name": "Motocicleta", "color": (255, 0, 255)},
}

# ================================
# DETECTOR
# ================================
def run_detection(image_array):
    """Ejecuta YOLO con parámetros optimizados."""
    return model(
        image_array,
        conf=confidence_threshold,
        iou=0.6,
        imgsz=960,
        classes=ALLOWED_CLASSES,
        agnostic_nms=False,
        verbose=False
    )

# ================================
# SUBIDA DE ARCHIVOS
# ================================
uploaded_file = st.file_uploader(
    "Carga una imagen o video",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"]
)

# ================================
# PROCESAMIENTO
# ================================
if uploaded_file is None:
    st.info("👆 Carga una imagen o video para comenzar.")
    st.stop()

file_type = uploaded_file.type

# ----------------------------------------
# 🖼️ IMÁGENES
# ----------------------------------------
if "image" in file_type:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    with st.spinner("Detectando objetos..."):
        results = run_detection(image_array)

    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    detection_counts = {}

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls)
            conf = float(box.conf)

            if class_id in class_mapping:
                name = class_mapping[class_id]["name"]
                color = class_mapping[class_id]["color"]

                detection_counts[name] = detection_counts.get(name, 0) + 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1, y1 - 10), f"{name} {conf:.2%}", fill=color)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagen Original")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detecciones")
        st.image(image_with_boxes, use_container_width=True)

    # Resumen
    st.markdown("### 📊 Resumen")
    if detection_counts:
        cols = st.columns(len(detection_counts))
        for (name, count), col in zip(detection_counts.items(), cols):
            col.metric(label=name, value=count)
    else:
        st.warning("No se detectaron objetos.")

# ----------------------------------------
# 🎥 VIDEO
# ----------------------------------------
else:
    st.subheader("Procesamiento de Video")
    temp_video = "temp_video.mp4"

    with open(temp_video, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_video)
    placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = run_detection(rgb)

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls)
                conf = float(box.conf)

                if class_id in class_mapping:
                    name = class_mapping[class_id]["name"]
                    color = class_mapping[class_id]["color"]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(rgb, f"{name} {conf:.2%}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        placeholder.image(rgb, channels="RGB", use_container_width=True)

    cap.release()
    os.remove(temp_video)
    st.success("Video procesado correctamente 🎉")
