"""
error_detection.py - Detección de errores combinada (visión tradicional + IA)
Integra detección por líneas y clasificación con IA preentrenada
"""

import cv2
import numpy as np
from inference import InferenceModel

class ErrorDetector:
    """
    Detector de errores que combina visión tradicional con IA
    """
    
    def __init__(self, use_ai=True, model_path="modelo_impresion.pth"):
        """
        Inicializa el detector con capacidad tradicional y IA
        
        Args:
            use_ai: si es True, activa la detección con IA
            model_path: ruta al modelo de IA entrenado
        """
        self.use_ai = use_ai
        self.ai_model_available = False
        
        # Umbrales configurables para detección tradicional
        self.max_lines_threshold = 30
        self.min_lines_threshold = 1
        self.vertical_threshold = 0.6
        self.horizontal_threshold = 0.6
        
        # Inicializar IA si está habilitada
        if use_ai:
            try:
                self.ai_model = InferenceModel(model_path)
                self.ai_model_available = self.ai_model.model_loaded
                if self.ai_model_available:
                    print("✅ IA cargada correctamente")
                else:
                    print("⚠ IA no disponible, usando solo detección tradicional")
                    self.use_ai = False
            except Exception as e:
                print(f"❌ Error cargando IA: {e}")
                print("⚠ Usando solo detección tradicional")
                self.use_ai = False
        else:
            self.ai_model = None
        
        # Contadores para estadísticas
        self.total_predictions = 0
        self.error_counts = {
            "desplazamiento": 0,
            "spaghetti": 0,
            "warping": 0,
            "normal": 0
        }
    
    def detect_traditional_errors(self, edges, lines):
        """
        Detección de errores mediante procesamiento clásico de visión artificial
        
        Args:
            edges: imagen con bordes detectados (Canny)
            lines: líneas detectadas por Hough Transform
        
        Returns:
            list: lista de advertencias detectadas
        """
        warnings = []
        
        # Caso 1: No se ven líneas -> posible fallo de extrusión
        if lines is None or len(lines) == 0:
            warnings.append("⚠ No se detectan trazos. Posible falta de extrusión o cama fría.")
            return warnings
        
        # Caso 2: Demasiadas líneas -> ruido o warping
        if len(lines) > self.max_lines_threshold:
            warnings.append(f"⚠ Demasiados trazos ({len(lines)}). Posible ruido o impresión irregular.")
        
        # Caso 3: Muy pocas líneas -> posible subextrusión
        if len(lines) < self.min_lines_threshold:
            warnings.append(f"⚠ Muy pocos trazos ({len(lines)}). Posible subextrusión.")
        
        # Caso 4: Análisis de orientación de líneas
        if lines is not None and len(lines) > 0:
            vertical_count = 0
            horizontal_count = 0
            diagonal_count = 0
            
            for line in lines:
                rho, theta = line[0]
                angle = np.degrees(theta)
                
                # Clasificar según orientación
                if angle < 10 or angle > 170:
                    vertical_count += 1
                elif 80 < angle < 100:
                    horizontal_count += 1
                else:
                    diagonal_count += 1
            
            total = len(lines)
            
            # Detectar predominancia anómala
            if vertical_count / total > self.vertical_threshold:
                warnings.append("⚠ Predominan líneas verticales. Posible desalineación en eje X.")
            
            if horizontal_count / total > self.horizontal_threshold:
                warnings.append("⚠ Predominan líneas horizontales. Posible desalineación en eje Y.")
            
            # Detector de spaghetti (líneas caóticas en todas direcciones)
            if (vertical_count > 0 and horizontal_count > 0 and 
                diagonal_count > total * 0.3 and total > 15):
                warnings.append("🍝 Patrón caótico detectado - Posible fallo tipo spaghetti")
        
        return warnings
    
    def detect_ai_errors(self, frame):
        """
        Detección de errores mediante IA preentrenada
        
        Args:
            frame: imagen en formato BGR (OpenCV)
        
        Returns:
            str: mensaje de error o None si es normal
        """
        if not self.ai_model_available or not self.use_ai:
            return None
        
        try:
            # Obtener predicción con confianza
            result = self.ai_model.predict_with_confidence(frame)
            
            # Actualizar contadores
            self.total_predictions += 1
            pred = result["prediction"]
            if pred in self.error_counts:
                self.error_counts[pred] += 1
            
            # Mapear clases a mensajes descriptivos
            class_messages = {
                "normal": None,  # No mostrar advertencia si es normal
                "desplazamiento": "🔄 Desplazamiento de capa detectado - Revisar calibración",
                "spaghetti": "🍝 FALLO CRÍTICO: Spaghetti detectado - ¡DETENER IMPRESIÓN!",
                "warping": "📈 Warping detectado - Revisar adhesión a cama caliente"
            }
            
            confidence = result["confidence"]
            
            # Mostrar advertencia solo si confianza es aceptable
            if result["prediction"] in class_messages and class_messages[result["prediction"]]:
                if confidence > 0.6:  # Umbral de confianza
                    return f"{class_messages[result['prediction']]} (confianza: {confidence:.1%})"
                elif confidence > 0.4:
                    return f"⚠ Posible {result['prediction']} (confianza: {confidence:.1%})"
            
            return None
                
        except Exception as e:
            print(f"❌ Error en inferencia IA: {e}")
            return None
    
    def detect_all(self, frame, edges, lines):
        """
        Detección combinada: tradicional + IA
        
        Args:
            frame: imagen original en BGR
            edges: imagen con bordes detectados
            lines: líneas detectadas
        
        Returns:
            dict: resultados completos de detección
        """
        warnings = []
        
        # 1. Detección tradicional (siempre activa)
        trad_warnings = self.detect_traditional_errors(edges, lines)
        warnings.extend(trad_warnings)
        
        # 2. Detección con IA
        ai_error = None
        if self.use_ai:
            ai_error = self.detect_ai_errors(frame)
            if ai_error:
                warnings.append(ai_error)
        
        # 3. Determinar severidad
        if len(warnings) == 0:
            severity = "🟢 NORMAL"
            severity_code = "normal"
        elif len(warnings) == 1:
            severity = "🟡 ATENCIÓN"
            severity_code = "warning"
        else:
            severity = "🔴 CRÍTICO"
            severity_code = "critical"
        
        # 4. Detectar si hay spaghetti (emergencia)
        emergency_stop = False
        if ai_error and "spaghetti" in ai_error.lower():
            emergency_stop = True
            severity = "🔴 EMERGENCIA - DETENER IMPRESIÓN"
        
        return {
            "warnings": warnings,
            "count": len(warnings),
            "severity": severity,
            "severity_code": severity_code,
            "traditional_count": len(trad_warnings),
            "ai_detected": ai_error is not None,
            "emergency_stop": emergency_stop,
            "ai_prediction": ai_error.split(" - ")[0] if ai_error and " - " in ai_error else None
        }
    
    def get_statistics(self):
        """
        Obtiene estadísticas de detección
        
        Returns:
            dict: estadísticas de errores detectados
        """
        total = self.total_predictions
        if total == 0:
            return {
                "total_predictions": 0,
                "error_percentages": {k: 0 for k in self.error_counts}
            }
        
        percentages = {
            k: (v / total) * 100 for k, v in self.error_counts.items()
        }
        
        return {
            "total_predictions": total,
            "error_counts": self.error_counts.copy(),
            "error_percentages": percentages
        }
    
    def reset_statistics(self):
        """Reinicia las estadísticas de detección"""
        self.total_predictions = 0
        self.error_counts = {
            "desplazamiento": 0,
            "spaghetti": 0,
            "warping": 0,
            "normal": 0
        }
    
    def set_thresholds(self, max_lines=None, vertical=None, horizontal=None):
        """
        Configura los umbrales de detección tradicional
        
        Args:
            max_lines: máximo número de líneas considerado normal
            vertical: porcentaje de líneas verticales para alertar
            horizontal: porcentaje de líneas horizontales para alertar
        """
        if max_lines is not None:
            self.max_lines_threshold = max_lines
        if vertical is not None:
            self.vertical_threshold = vertical
        if horizontal is not None:
            self.horizontal_threshold = horizontal

# Función legacy para compatibilidad con código existente
def detect_trace_errors(*args):
    """
    Función legacy para mantener compatibilidad con código antiguo
    
    Args:
        Puede recibir (edges, lines) o solo (lines)
    
    Returns:
        dict: diccionario con advertencias
    """
    if len(args) == 2:
        edges, lines = args
    elif len(args) == 1:
        lines = args[0]
        edges = None
    else:
        return {"warnings": []}
    
    detector = ErrorDetector(use_ai=False)
    return {"warnings": detector.detect_traditional_errors(edges, lines)}

# Función rápida para uso simple
def detectar_error(frame):
    """
    Función rápida para detección de errores con IA
    
    Args:
        frame: imagen en BGR
    
    Returns:
        str: mensaje de error o "Impresión correcta"
    """
    detector = ErrorDetector(use_ai=True)
    result = detector.detect_ai_errors(frame)
    
    if result:
        return result
    return "Impresión correcta"