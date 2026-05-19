"""
inference.py - Módulo de inferencia en tiempo real para detección de fallos en impresión 3D
Clases: desplazamiento, normal, spaghetti, warping
"""

import torch
import cv2
import numpy as np
import time
from torchvision import transforms
from model import create_model

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🤖 InferenceModel usando dispositivo: {device}")

class InferenceModel:
    """
    Clase para inferencia en tiempo real del modelo de clasificación de fallos
    """
    
    def __init__(self, model_path="modelo_impresion.pth", num_classes=4):
        """
        Inicializa el modelo de inferencia para monitoreo en tiempo real
        
        Args:
            model_path: ruta al archivo de pesos del modelo entrenado
            num_classes: número de clases (por defecto 4)
        """
        self.device = device
        self.num_classes = num_classes
        self.class_names = ["desplazamiento", "normal", "spaghetti", "warping"]
        
        print(f"🚀 Inicializando InferenceModel...")
        
        try:
            # Crear modelo y cargar pesos
            self.model = create_model(num_classes=num_classes)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Modelo cargado exitosamente desde: {model_path}")
            self.model_loaded = True
            
        except FileNotFoundError:
            print(f"❌ Archivo de modelo no encontrado: {model_path}")
            print(f"⚠️ Ejecuta train.py primero para entrenar el modelo")
            self.model_loaded = False
            self.model = None
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            self.model_loaded = False
            self.model = None
        
        # Transform optimizada para inferencia
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Transform rápida (sin ToPILImage)
        self.transform_fast = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Métricas de rendimiento
        self.inference_times = []
        self.last_inference_time = 0
        self.total_inferences = 0
        
        print(f"📊 Clases disponibles: {self.class_names}")
    
    def preprocess_frame(self, frame, fast=True):
        """
        Preprocesa el frame para entrada al modelo
        
        Args:
            frame: imagen en formato BGR (OpenCV)
            fast: si es True usa preprocesamiento rápido
        
        Returns:
            tensor: tensor listo para el modelo
        """
        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if fast:
            # Método rápido: redimensionar con OpenCV
            resized = cv2.resize(frame_rgb, (224, 224))
            # Convertir a tensor y normalizar
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            # Normalizar con valores ImageNet
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std
            return tensor.unsqueeze(0)
        else:
            # Método tradicional (más preciso pero más lento)
            tensor = self.transform(frame_rgb)
            return tensor.unsqueeze(0)
    
    def predict(self, frame, fast=True):
        """
        Predice la clase del frame
        
        Args:
            frame: imagen en formato BGR (OpenCV)
            fast: True para inferencia rápida, False para máxima precisión
        
        Returns:
            str: nombre de la clase detectada
        """
        if not self.model_loaded:
            return "normal"  # Fallback seguro si no hay modelo
        
        try:
            start_time = time.time()
            
            # Preprocesar frame
            input_tensor = self.preprocess_frame(frame, fast=fast)
            input_tensor = input_tensor.to(self.device)
            
            # Inferencia
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Registrar tiempo de inferencia
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 30:
                self.inference_times.pop(0)
            self.last_inference_time = inference_time
            self.total_inferences += 1
            
            class_name = self.class_names[predicted.item()]
            confidence_value = confidence.item()
            
            # Si la confianza es muy baja y no es normal, considerar como normal
            if confidence_value < 0.5 and class_name != "normal":
                print(f"⚠ Baja confianza ({confidence_value:.2%}) para {class_name}, considerando normal")
                return "normal"
            
            # Para depuración (opcional)
            # if self.total_inferences % 50 == 0:
            #     print(f"🔍 Predicción: {class_name} ({confidence_value:.1%}) - {inference_time*1000:.1f}ms")
            
            return class_name
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return "normal"
    
    def predict_with_confidence(self, frame, fast=True):
        """
        Versión detallada que retorna confianza y todas las probabilidades
        
        Args:
            frame: imagen en formato BGR (OpenCV)
            fast: True para inferencia rápida
        
        Returns:
            dict: diccionario con predicción detallada
        """
        if not self.model_loaded:
            return {
                "prediction": "normal",
                "confidence": 1.0,
                "probabilities": {name: 0.0 for name in self.class_names},
                "inference_time": 0
            }
        
        try:
            start_time = time.time()
            
            input_tensor = self.preprocess_frame(frame, fast=fast)
            input_tensor = input_tensor.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                probs = probabilities.cpu().numpy()[0]
            
            inference_time = time.time() - start_time
            
            # Crear diccionario de probabilidades
            results = {
                self.class_names[i]: float(probs[i])
                for i in range(len(self.class_names))
            }
            
            predicted_class = max(results, key=results.get)
            confidence = results[predicted_class]
            
            return {
                "prediction": predicted_class,
                "confidence": confidence,
                "probabilities": results,
                "inference_time": inference_time,
                "fast_inference": fast
            }
            
        except Exception as e:
            print(f"❌ Error en predicción detallada: {e}")
            return {
                "prediction": "error",
                "confidence": 0.0,
                "probabilities": {},
                "inference_time": 0
            }
    
    def predict_batch(self, frames, fast=True):
        """
        Predice múltiples frames en lote (más eficiente)
        
        Args:
            frames: lista de imágenes en formato BGR
            fast: True para inferencia rápida
        
        Returns:
            list: lista de predicciones
        """
        if not self.model_loaded or not frames:
            return ["normal"] * len(frames)
        
        try:
            batch_tensors = []
            for frame in frames:
                tensor = self.preprocess_frame(frame, fast=fast)
                batch_tensors.append(tensor)
            
            batch = torch.cat(batch_tensors, dim=0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)
            
            predictions = [self.class_names[idx.item()] for idx in predicted]
            return predictions
            
        except Exception as e:
            print(f"❌ Error en predicción por lotes: {e}")
            return ["normal"] * len(frames)
    
    def get_performance_stats(self):
        """
        Retorna estadísticas de rendimiento
        
        Returns:
            dict: estadísticas de inferencia
        """
        if not self.inference_times:
            return {
                "avg_time": 0,
                "fps": 0,
                "min_time": 0,
                "max_time": 0,
                "total_inferences": self.total_inferences
            }
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        
        return {
            "avg_time": avg_time,
            "fps": 1.0 / avg_time if avg_time > 0 else 0,
            "min_time": min(self.inference_times),
            "max_time": max(self.inference_times),
            "total_inferences": self.total_inferences
        }
    
    def reset_stats(self):
        """Reinicia las estadísticas de rendimiento"""
        self.inference_times = []
        self.last_inference_time = 0
        self.total_inferences = 0
    
    def get_class_info(self):
        """
        Retorna información sobre las clases
        
        Returns:
            dict: información de clases
        """
        class_info = {
            "desplazamiento": {
                "nombre": "Desplazamiento de capa",
                "descripcion": "Capas impresas desplazadas horizontalmente",
                "gravedad": "alta",
                "accion": "Revisar tensión de correas y calibración"
            },
            "normal": {
                "nombre": "Impresión normal",
                "descripcion": "Impresión correcta sin defectos",
                "gravedad": "ninguna",
                "accion": "Continuar monitoreo"
            },
            "spaghetti": {
                "nombre": "Fallo spaghetti",
                "descripcion": "Filamento enmarañado, impresión fallida",
                "gravedad": "critica",
                "accion": "DETENER IMPRESIÓN INMEDIATAMENTE"
            },
            "warping": {
                "nombre": "Deformación (Warping)",
                "descripcion": "Esquinas levantadas, mala adhesión a cama",
                "gravedad": "media",
                "accion": "Revisar temperatura de cama y adhesión"
            }
        }
        return class_info

# Función de utilidad para probar el modelo
def test_inference():
    """
    Función de prueba para verificar que el modelo funciona correctamente
    """
    print("\n🧪 Probando InferenceModel...")
    print("=" * 50)
    
    # Inicializar modelo
    model = InferenceModel()
    
    if not model.model_loaded:
        print("❌ Modelo no cargado. Ejecuta train.py primero.")
        return
    
    # Crear frame dummy para prueba
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Probar predicción
    print("\n📸 Probando con frame dummy...")
    result = model.predict(dummy_frame)
    print(f"   Predicción: {result}")
    
    # Probar predicción con confianza
    print("\n📊 Probando predicción detallada...")
    detailed = model.predict_with_confidence(dummy_frame)
    print(f"   Predicción: {detailed['prediction']}")
    print(f"   Confianza: {detailed['confidence']:.2%}")
    print(f"   Tiempo: {detailed['inference_time']*1000:.1f}ms")
    
    # Estadísticas
    stats = model.get_performance_stats()
    print(f"\n📈 Estadísticas:")
    print(f"   Tiempo promedio: {stats['avg_time']*1000:.1f}ms")
    print(f"   FPS teórico: {stats['fps']:.1f}")
    
    print("\n✅ Prueba completada!")

if __name__ == "__main__":
    test_inference()