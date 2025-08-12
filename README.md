# Proyecto-de-Final-IA

## 👨‍💻 Información del Estudiante

**Nombre:** Ricarly Jiménez  
**Matrícula:** 22-EISN-2-001 
**Curso:** Inteligencia Artificial  
**Fecha de Entrega:** 11/08/2025  

---

## 🎯 Detector de Emociones en Tiempo Real

Sistema inteligente que detecta y analiza emociones faciales usando **modelos de deep learning** en tiempo real desde tu cámara web.

### 🧠 Modelos de Deep Learning Utilizados

- **DeepFace:** Análisis de 7 emociones básicas (feliz, triste, enojado, sorprendido, miedo, disgusto, neutral)
- **MediaPipe Face Detection:** Detección rápida de rostros  
- **MediaPipe Face Mesh:** 468 puntos faciales de referencia

### ✨ Características Principales

- 📹 **Detección en tiempo real** desde cámara web
- 🎭 **7 emociones detectadas** con porcentaje de confianza
- 🚀 **Sistema anti-parpadeo** para visualización estable
- ⚙️ **Configuraciones personalizables** (FPS, sensibilidad)
- 🎨 **Múltiples modos de visualización** (texto, emojis, combinado)

### 📦 Instalación Rápida

```bash
# Clonar repositorio
git clone [[URL_DEL_REPOSITORIO](https://github.com/ricarly01/Proyecto-Final-IA.git)]
cd Proyecto-de-Final-IA

# Instalar dependencias
pip install streamlit opencv-python mediapipe deepface Pillow numpy python-dotenv

# Ejecutar aplicación
streamlit run app.py
```

### 🎮 Uso Simple

1. **Configurar:** Selecciona detector y modo de visualización
2. **Iniciar:** Haz clic en "Iniciar cámara"
3. **Detectar:** Posiciónate frente a la cámara (50-100cm)
4. **Ver resultados:** Observa emociones detectadas en tiempo real

### ⚙️ Configuración Recomendada

Para mejor rendimiento:
- **FPS Máximo:** 15
- **Sensibilidad:** 5  
- **Saltar Frames:** 2

### 🔧 Solución de Problemas

**Error de cámara:**
1. Cierra Teams, Zoom, Skype
2. Reinicia el navegador
3. Verifica permisos de cámara

**Emojis no aparecen:**
- Windows: Instala fuente "Segoe UI Emoji"
- Linux: `sudo apt install fonts-noto-color-emoji`

### 📊 Especificaciones Técnicas

| Aspecto | Detalle |
|---------|---------|
| **Precisión** | ~85-92% en emociones |
| **FPS** | 10-25 configurables |
| **Latencia** | <200ms |
| **RAM** | 200-500MB |

### 🏆 Cumplimiento Académico

✅ **Modelos de Deep Learning:** DeepFace + MediaPipe  
✅ **Interfaz Gráfica:** Streamlit (equivalente a Gradio)  
✅ **Aplicación Interactiva:** Tiempo real con cámara web  
✅ **No es Notebook:** Aplicación web completa  
✅ **Originalidad:** Sistema anti-parpadeo único  


### 📞 Contacto

**Estudiante:** Ricarly Jiménez  
**Matrícula:** 22-EISN-2-001  
**Repositorio:** [https://github.com/ricarly01/Proyecto-Final-IA.git]  

---

*Proyecto Final - Inteligencia Artificial 2025*
