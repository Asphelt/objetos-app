import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import cv2
import io
import os

# Configuración de la página
st.set_page_config(
    page_title="Detección de Objetos Urbanos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #00d4ff;
            color: white;
        }
        h1, h2, h3 {
            color: #00d4ff;
        }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.title("Detección de Objetos Urbanos")
st.markdown("Carga una imagen o video y detecta carros, peatones, autobuses, camiones y más con una precisión mejorada.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración del Modelo")

    model_choice = st.selectbox(
        "Selecciona la versión del modelo YOLO11",
        ["yolo11m.pt", "yolo11l.pt", "yolo11x.pt", "yolo11s.pt", "yolo11n.pt"],
        index=0,
        help="Modelos más grandes = mejor precisión, pero más lentos."
    )

    confidence_threshold = st.slider(
        "Confianza mínima",
        min_value=0.1,
        max_value=1.0,
        value=0.35,
        step=0.05
    )

    st.markdown("---")
    st.markdown("""
    ### Objetos detectados:
    - Carros
    - Autobuses
    - Camiones
    - Personas
    - Semáforos
    - Motocicletas
    """)

# Cargar modelo con caché
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

model = load_model(model_choice)

# SOLO clases urbanas
ALLOWED_CLASSES = [0, 2, 3, 5, 7, 9]

# Colores y nombres
class_mapping = {
    2: {"name": "Carro", "color": (255, 0, 0)},
    5: {"name": "Autobús", "color": (0, 255, 255)},
    7: {"name": "Camión", "color": (255, 165, 0)},
    0: {"name": "Persona", "color": (0, 255, 0)},
    9: {"name": "Semáforo", "color": (255, 255, 0)},
    3: {"name": "Motocicleta", "color": (255, 0, 255)},
}

# Área de carga
st.markdown("### Cargar Imagen o Video")
uploaded_file = st.file_uploader(
    "Selecciona una imagen o video",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"]
)

def run_detection(image_array):
    """Ejecuta YOLO con parámetros mejorados."""
    return model(
        image_array,
        conf=confidence_threshold,
        iou=0.6,                 # mejora la separación de cajas
        imgsz=960,               # más detalle → mejor precisión
        classes=ALLOWED_CLASSES, # reduce falsos positivos
        agnostic_nms=False,
        verbose=False
    )

if uploaded_file is not None:
    file_type = uploaded_file.type

    # ============================
    # 🖼️ IMÁGENES
    # ============================
    if "image" in file_type:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        with st.spinner("Detectando objetos con precisión mejorada..."):
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
            st.subheader("Detecciones Mejoradas")
            st.image(image_with_boxes, use_container_width=True)

        # Resumen
        st.markdown("### Detecciones")
        if detection_counts:
            cols = st.columns(len(detection_counts))
            for (name, count), col in zip(detection_counts.items(), cols):
                with col:
                    st.metric(label=name, value=count)
        else:
            st.warning("No se detectaron objetos urbanísticos.")

    # ============================
    # 🎥 VIDEO
    # ============================
    elif "video" in file_type:
        st.subheader("Análisis de Video")

        temp_video = "temp_video.mp4"
        with open(temp_video, "wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_video)
        video_placeholder = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = run_detection(rgb)

            # Dibujar
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls)
                    conf = float(box.conf)

                    if class_id in class_mapping:
                        name = class_mapping[class_id]["name"]
                        color = class_mapping[class_id]["color"]

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(rgb, f"{name} {conf:.2%}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            video_placeholder.image(rgb, channels="RGB", use_container_width=True)

        cap.release()
        os.remove(temp_video)
        st.success("Video procesado completamente")

else:
    st.info(" Ingresa una imagen o video para comenzar.")

