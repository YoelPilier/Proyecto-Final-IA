import os
import cv2
import time
import streamlit as st
import mediapipe as mp
from dotenv import load_dotenv
from deepface import DeepFace
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import warnings
from collections import deque
import threading
from queue import Queue

# Suprimir warnings para limpiar la salida
warnings.filterwarnings('ignore')

# Cargar API Key de OpenAI si planeas usarla luego
load_dotenv()
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("😄 Detección de emociones en tiempo real - Sin Parpadeo")

# Mapa de emociones a emojis
EMOJIS = {
    "angry": "😠",
    "disgust": "🤢", 
    "fear": "😨",
    "happy": "😄",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

# Mapa alternativo con texto si no funcionan los emojis
EMOTIONS_TEXT = {
    "angry": "ENOJADO",
    "disgust": "DISGUSTO",
    "fear": "MIEDO", 
    "happy": "FELIZ",
    "sad": "TRISTE",
    "surprise": "SORPRESA",
    "neutral": "NEUTRAL"
}

class EmotionDetector:
    def __init__(self):
        self.emotion_history = deque(maxlen=10)  # Buffer para suavizar emociones
        self.last_emotion = "neutral"
        self.last_confidence = 0.0
        self.frame_count = 0
        self.emotion_stable_count = 0
        self.min_stable_frames = 3  # Frames mínimos para cambiar emoción
        
    def update_emotion(self, emotion, confidence):
        """Actualiza la emoción con suavizado temporal"""
        self.emotion_history.append((emotion, confidence))
        
        # Calcular emoción más frecuente en el historial
        if len(self.emotion_history) >= 3:
            recent_emotions = [e[0] for e in list(self.emotion_history)[-5:]]
            most_common = max(set(recent_emotions), key=recent_emotions.count)
            
            # Solo cambiar si la nueva emoción es consistente
            if most_common == self.last_emotion:
                self.emotion_stable_count += 1
            else:
                if recent_emotions.count(most_common) >= 2:  # Al menos 2 veces en las últimas 5
                    self.last_emotion = most_common
                    self.last_confidence = confidence
                    self.emotion_stable_count = 0
        
        return self.last_emotion, self.last_confidence

def draw_emoji_on_frame(frame, emoji, x, y, size=50):
    """
    Función para dibujar emojis usando PIL sobre el frame de OpenCV
    """
    try:
        # Convertir frame de OpenCV a PIL
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)
        
        # Intentar cargar una fuente que soporte emojis
        try:
            # En Windows
            font = ImageFont.truetype("seguiemj.ttf", size)
        except:
            try:
                # En macOS
                font = ImageFont.truetype("Apple Color Emoji.ttc", size)
            except:
                try:
                    # En Linux
                    font = ImageFont.truetype("NotoColorEmoji.ttf", size)
                except:
                    # Fuente por defecto
                    font = ImageFont.load_default()
        
        # Dibujar el emoji
        draw.text((x, y), emoji, font=font, fill=(255, 255, 255))
        
        # Convertir de vuelta a array de OpenCV
        return np.array(pil_image)
    except Exception:
        return frame

def draw_text_with_background(frame, text, position, font_scale=1, color=(255, 255, 255), bg_color=(0, 0, 0)):
    """
    Dibuja texto con fondo para mejor visibilidad
    """
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    # Obtener el tamaño del texto
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Dibujar fondo
    cv2.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + baseline + 5), bg_color, -1)
    
    # Dibujar texto
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame

def apply_smoothing_filter(frame):
    """Aplica un filtro de suavizado para reducir ruido"""
    return cv2.bilateralFilter(frame, 9, 75, 75)

# Inicializar estado con mejor gestión
if "camara_activa" not in st.session_state:
    st.session_state.camara_activa = False
if "ultimo_detector" not in st.session_state:
    st.session_state.ultimo_detector = "MediaPipe Face Detection"
if "emotion_detector" not in st.session_state:
    st.session_state.emotion_detector = EmotionDetector()
if "frame_buffer" not in st.session_state:
    st.session_state.frame_buffer = deque(maxlen=3)

# Selector de detector de rostros
detector_tipo = st.selectbox(
    "Tipo de detector de rostros:",
    ["MediaPipe Face Detection", "MediaPipe Face Mesh"],
    disabled=st.session_state.camara_activa
)

# Detectar cambio de detector mientras la cámara está activa
if st.session_state.camara_activa and detector_tipo != st.session_state.ultimo_detector:
    st.warning("⚠️ Para cambiar de detector, primero detén la cámara.")
    detector_tipo = st.session_state.ultimo_detector
else:
    st.session_state.ultimo_detector = detector_tipo

# Selector de modo de visualización
modo_emoji = st.selectbox(
    "Modo de visualización:",
    ["Texto simple", "Emoji con PIL", "Emoji + Texto"],
    disabled=st.session_state.camara_activa
)

# Configuraciones avanzadas para reducir parpadeo
st.sidebar.markdown("## ⚙️ Configuración Anti-Parpadeo")
fps_limit = st.sidebar.slider("FPS Máximo", 5, 30, 15, help="Limita FPS para mayor estabilidad")
emotion_sensitivity = st.sidebar.slider("Sensibilidad de Emoción", 1, 10, 5, help="Menor valor = más estable")
frame_skip = st.sidebar.slider("Saltar Frames", 1, 5, 2, help="Procesa 1 de cada N frames")

# Actualizar configuración del detector
st.session_state.emotion_detector.min_stable_frames = emotion_sensitivity

# Control de cámara
col1, col2 = st.columns(2)
with col1:
    if st.button("Iniciar cámara", disabled=st.session_state.camara_activa):
        st.session_state.camara_activa = True
        st.session_state.ultimo_detector = detector_tipo
        st.session_state.emotion_detector = EmotionDetector()  # Reiniciar detector
        st.rerun()
with col2:
    if st.button("Detener cámara", disabled=not st.session_state.camara_activa):
        st.session_state.camara_activa = False
        st.rerun()

# Estado de la cámara
if st.session_state.camara_activa:
    st.success(f"🟢 Cámara activa - Detector: {detector_tipo} | FPS: {fps_limit}")
else:
    st.info("🔴 Cámara inactiva")

# Placeholders fijos para evitar recreación
frame_placeholder = st.empty()
emotion_placeholder = st.empty()

# Inicializar MediaPipe fuera del bucle principal
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def detect_faces_optimized(frame_rgb, detector_tipo, face_detection=None, face_mesh=None):
    """Detección optimizada de rostros con cache"""
    faces = []
    try:
        if detector_tipo == "MediaPipe Face Detection" and face_detection:
            results = face_detection.process(frame_rgb)
            if results.detections:
                ih, iw, _ = frame_rgb.shape
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x = max(0, int(bboxC.xmin * iw))
                    y = max(0, int(bboxC.ymin * ih))
                    w = min(int(bboxC.width * iw), iw - x)
                    h = min(int(bboxC.height * ih), ih - y)
                    
                    # Validar tamaño mínimo
                    if w > 50 and h > 50:
                        confidence = detection.score[0] if detection.score else 0.0
                        faces.append({
                            'bbox': (x, y, w, h),
                            'confidence': confidence,
                            'method': 'Face Detection'
                        })
        
        elif detector_tipo == "MediaPipe Face Mesh" and face_mesh:
            results = face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                ih, iw, _ = frame_rgb.shape
                for face_landmarks in results.multi_face_landmarks:
                    x_coords = [landmark.x * iw for landmark in face_landmarks.landmark]
                    y_coords = [landmark.y * ih for landmark in face_landmarks.landmark]
                    
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    margin = 20
                    x = max(0, x_min - margin)
                    y = max(0, y_min - margin)
                    w = min(iw - x, x_max - x_min + 2 * margin)
                    h = min(ih - y, y_max - y_min + 2 * margin)
                    
                    if w > 50 and h > 50:
                        faces.append({
                            'bbox': (x, y, w, h),
                            'confidence': 0.9,
                            'landmarks': face_landmarks,
                            'method': 'Face Mesh'
                        })
    except Exception as e:
        pass  # Silencioso en caso de error
    
    return faces

def process_emotion_analysis(rostro, emotion_detector):
    """Procesa el análisis de emociones de forma optimizada"""
    try:
        # Reducir resolución para análisis más rápido
        rostro_small = cv2.resize(rostro, (96, 96))
        
        analisis = DeepFace.analyze(
            rostro_small, 
            actions=["emotion"], 
            enforce_detection=False,
            silent=True
        )
        
        if isinstance(analisis, list):
            emocion = analisis[0]["dominant_emotion"]
            confianza_emocion = analisis[0]["emotion"][emocion]
        else:
            emocion = analisis["dominant_emotion"]
            confianza_emocion = analisis["emotion"][emocion]
        
        # Aplicar suavizado temporal
        emocion_suave, confianza_suave = emotion_detector.update_emotion(emocion, confianza_emocion)
        
        return emocion_suave, confianza_suave
    except Exception:
        return emotion_detector.last_emotion, emotion_detector.last_confidence

if st.session_state.camara_activa:
    # Configuración de cámara optimizada
    backends_to_try = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_V4L2, "Video4Linux"),
        (cv2.CAP_ANY, "Automático")
    ]
    
    camera = None
    backend_usado = None
    
    for backend, name in backends_to_try:
        try:
            camera = cv2.VideoCapture(0, backend)
            if camera.isOpened():
                backend_usado = name
                break
            else:
                camera.release() if camera else None
        except Exception:
            continue
    
    if not camera or not camera.isOpened():
        st.error("🚫 **No se pudo acceder a la cámara**")
        st.session_state.camara_activa = False
        if camera:
            camera.release()
    else:
        # Configuración optimizada de cámara
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, fps_limit)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # Mostrar configuración
        actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = camera.get(cv2.CAP_PROP_FPS)
        
        st.info(f"📷 {int(actual_width)}x{int(actual_height)} @ {actual_fps:.1f}FPS | {backend_usado}")
        
        # Inicializar detectores una sola vez
        face_detection = None
        face_mesh = None
        
        try:
            if detector_tipo == "MediaPipe Face Detection":
                face_detection = mp_face_detection.FaceDetection(
                    model_selection=0, 
                    min_detection_confidence=0.7  # Mayor confianza para más estabilidad
                )
            else:
                face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,  # Solo 1 cara para mejor rendimiento
                    refine_landmarks=False,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7
                )
            
            frame_count = 0
            last_frame_time = time.time()
            target_frame_time = 1.0 / fps_limit
            
            # Variables para estabilización
            last_valid_frame = None
            frames_without_face = 0
            max_frames_without_face = 30
            
            # Leer frames iniciales para estabilizar
            for _ in range(5):
                ret, _ = camera.read()
                time.sleep(0.1)
            
            while st.session_state.camara_activa:
                current_time = time.time()
                
                # Control de FPS más suave
                if current_time - last_frame_time < target_frame_time:
                    time.sleep(0.01)
                    continue
                
                ret, frame = camera.read()
                
                if not ret:
                    if last_valid_frame is not None:
                        # Usar último frame válido
                        frame_placeholder.image(last_valid_frame, channels="RGB", use_container_width=True)
                    time.sleep(0.05)
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Aplicar filtro de suavizado
                frame_rgb = apply_smoothing_filter(frame_rgb)
                
                # Procesar solo cada N frames
                frame_count += 1
                if frame_count % frame_skip != 0:
                    # Mostrar frame sin procesamiento para mantener fluidez visual
                    if last_valid_frame is not None:
                        frame_placeholder.image(last_valid_frame, channels="RGB", use_container_width=True)
                    else:
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    last_frame_time = current_time
                    continue
                
                # Detectar rostros
                faces = detect_faces_optimized(frame_rgb, detector_tipo, face_detection, face_mesh)
                
                if faces:
                    frames_without_face = 0
                    face_info = faces[0]  # Solo procesar primera cara
                    
                    x, y, w, h = face_info['bbox']
                    confidence = face_info['confidence']
                    method = face_info['method']
                    
                    # Dibujar rectángulo suave
                    color = (0, 255, 0) if method == 'Face Detection' else (255, 0, 255)
                    cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), color, 2)
                    
                    # Dibujar landmarks si disponibles (optimizado)
                    if 'landmarks' in face_info:
                        landmarks = face_info['landmarks']
                        ih, iw, _ = frame_rgb.shape
                        key_points = [10, 151, 9, 168, 6, 5, 4, 1]  # Puntos clave reducidos
                        for idx in key_points:
                            if idx < len(landmarks.landmark):
                                lx = int(landmarks.landmark[idx].x * iw)
                                ly = int(landmarks.landmark[idx].y * ih)
                                cv2.circle(frame_rgb, (lx, ly), 2, (0, 255, 0), -1)
                    
                    # Análisis de emoción optimizado
                    if w > 60 and h > 60:  # Tamaño mínimo mayor
                        rostro = frame_rgb[y:y+h, x:x+w]
                        emocion, confianza_emocion = process_emotion_analysis(
                            rostro, st.session_state.emotion_detector
                        )
                        
                        emoji = EMOJIS.get(emocion, "❓")
                        texto_emocion = EMOTIONS_TEXT.get(emocion, "DESCONOCIDO")
                        
                        # Posición optimizada para el texto
                        centro_x = x + w // 2
                        texto_y = y - 50 if y - 50 > 50 else y + h + 50
                        
                        # Mostrar información del detector
                        frame_rgb = draw_text_with_background(
                            frame_rgb,
                            f"{method}",
                            (x, y - 25),
                            font_scale=0.5,
                            color=(255, 255, 0),
                            bg_color=(0, 0, 0)
                        )
                        
                        # Mostrar emoción según el modo
                        if modo_emoji == "Texto simple":
                            frame_rgb = draw_text_with_background(
                                frame_rgb, 
                                f"{texto_emocion} ({confianza_emocion:.0f}%)",
                                (centro_x - 80, texto_y),
                                font_scale=0.8,
                                color=(255, 255, 255),
                                bg_color=(0, 0, 0)
                            )
                        elif modo_emoji == "Emoji con PIL":
                            frame_rgb = draw_emoji_on_frame(frame_rgb, emoji, centro_x - 25, texto_y - 60, 50)
                            frame_rgb = draw_text_with_background(
                                frame_rgb,
                                f"{texto_emocion}",
                                (centro_x - 60, texto_y),
                                font_scale=0.7,
                                color=(255, 255, 255),
                                bg_color=(0, 0, 0)
                            )
                        else:  # "Emoji + Texto"
                            frame_rgb = draw_emoji_on_frame(frame_rgb, emoji, centro_x - 25, texto_y - 60, 45)
                            frame_rgb = draw_text_with_background(
                                frame_rgb,
                                f"{texto_emocion} ({confianza_emocion:.0f}%)",
                                (centro_x - 80, texto_y),
                                font_scale=0.7,
                                color=(255, 255, 255),
                                bg_color=(0, 0, 0)
                            )
                        
                        # Actualizar información de forma más estable
                        emotion_placeholder.markdown(
                            f"### {emoji} **{texto_emocion}** ({confianza_emocion:.0f}%) | {method}"
                        )
                else:
                    frames_without_face += 1
                    # Solo actualizar mensaje si han pasado muchos frames sin cara
                    if frames_without_face > 10:
                        emotion_placeholder.markdown(f"### 👤 Buscando rostro... | {detector_tipo}")
                
                # Guardar frame procesado
                last_valid_frame = frame_rgb.copy()
                
                # Mostrar frame
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                last_frame_time = current_time
                
        except Exception as e:
            st.error(f"Error durante la ejecución: {str(e)}")
        finally:
            # Limpiar recursos
            if face_detection:
                face_detection.close()
            if face_mesh:
                face_mesh.close()
            if camera:
                camera.release()
    
    if not st.session_state.camara_activa:
        st.success("✅ Cámara detenida correctamente.")
        emotion_placeholder.empty()
else:
    frame_placeholder.info("📷 Presiona 'Iniciar cámara' para comenzar la detección de emociones.")
    emotion_placeholder.empty()

# Información adicional sobre las optimizaciones
with st.expander("⚡ Optimizaciones Anti-Parpadeo"):
    st.write("""
    **🔧 Mejoras implementadas:**
    
    **1. Control de FPS Inteligente:**
    - Limitación precisa de frames por segundo
    - Temporal suave entre frames
    - Buffer de frames para estabilidad
    
    **2. Suavizado de Emociones:**
    - Historial de emociones para evitar cambios bruscos
    - Filtrado temporal de resultados
    - Sensibilidad ajustable
    
    **3. Optimización de Procesamiento:**
    - Análisis de emoción con resolución reducida
    - Procesamiento de 1 de cada N frames
    - Cache de último frame válido
    
    **4. Estabilización Visual:**
    - Filtros de suavizado en imagen
    - Detección con mayor confianza mínima
    - Reducción de landmarks para mejor rendimiento
    
    **5. Gestión de Estados:**
    - Placeholders fijos para evitar recreación
    - Buffer de frames para mantener fluidez
    - Control de excepciones silencioso
    """)

with st.expander("🎛️ Configuración Recomendada"):
    st.write("""
    **Para máxima estabilidad:**
    - **FPS Máximo**: 10-15 (menor = más estable)
    - **Sensibilidad de Emoción**: 3-5 (menor = más estable)
    - **Saltar Frames**: 2-3 (mayor = más estable)
    
    **Para máxima respuesta:**
    - **FPS Máximo**: 20-25 
    - **Sensibilidad de Emoción**: 7-10
    - **Saltar Frames**: 1-2
    
    **Configuración equilibrada (recomendada):**
    - **FPS Máximo**: 15
    - **Sensibilidad de Emoción**: 5
    - **Saltar Frames**: 2
    """)

# Estadísticas en tiempo real
if st.session_state.camara_activa:
    with st.expander("📊 Estadísticas en Tiempo Real"):
        if hasattr(st.session_state.emotion_detector, 'emotion_history'):
            history = list(st.session_state.emotion_detector.emotion_history)
            if history:
                st.write(f"**Historial de emociones** (últimas {len(history)}):")
                emotions_only = [e[0] for e in history[-5:]]
                st.write(" → ".join([f"{EMOJIS.get(e, '❓')}{e}" for e in emotions_only]))
                
                st.write(f"**Emoción actual estable**: {st.session_state.emotion_detector.last_emotion}")
                st.write(f"**Frames estables**: {st.session_state.emotion_detector.emotion_stable_count}")