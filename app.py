"""
app.py - Aplicación CLI para monitoreo de impresión 3D con visión artificial e IA
Versión simple para terminal, sin interfaz gráfica
"""

import cv2
import time
import numpy as np
import camera_module
import image_processing
import trace_analysis
from error_detection import ErrorDetector

def main():
    """
    Función principal de la aplicación CLI
    """
    print("=" * 60)
    print("🖨️  MONITOREO DE IMPRESIÓN 3D - VISIÓN ARTIFICIAL + IA")
    print("=" * 60)
    
    # Inicializar cámara
    cap = camera_module.start_camera()
    if cap is None:
        print("❌ Error: No se pudo acceder a la cámara.")
        print("   Verifica que la cámara esté conectada y no esté siendo usada por otra aplicación.")
        return
    
    # Configurar resolución para mejor rendimiento
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n🔄 Inicializando sistema de detección...")
    
    # Inicializar detector de errores con IA
    detector = ErrorDetector(use_ai=True, model_path="modelo_impresion.pth")
    
    print("\n📷 Sistema listo. Controles:")
    print("   'q' - Salir")
    print("   's' - Guardar captura de pantalla")
    print("   'r' - Reiniciar estadísticas")
    print("   'i' - Mostrar estadísticas de IA")
    print("   't' - Mostrar umbrales actuales")
    print("=" * 60)
    
    # Variables para estadísticas
    frame_count = 0
    error_count = 0
    last_time = time.time()
    fps = 0
    total_frames = 0
    
    # Variables para captura
    capture_requested = False
    
    try:
        while True:
            # Capturar frame
            ret, frame = cap.read()
            if not ret:
                print("❌ No se pudo leer el frame. Verifica la conexión de la cámara.")
                break
            
            frame_count += 1
            total_frames += 1
            
            # --- PROCESAMIENTO TRADICIONAL ---
            edges = image_processing.preprocess_frame(frame)
            lines = image_processing.detect_lines(edges)
            
            # Análisis de trazos
            analysis = trace_analysis.analyze_lines(lines)
            line_count = analysis['count']
            
            # --- DETECCIÓN DE ERRORES (TRADICIONAL + IA) ---
            result = detector.detect_all(frame, edges, lines)
            
            if result['count'] > 0:
                error_count += 1
            
            # --- CÁLCULO DE FPS ---
            current_time = time.time()
            if current_time - last_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_time = current_time
            
            # --- MOSTRAR INFORMACIÓN EN CONSOLA (cada 10 frames) ---
            if total_frames % 10 == 0:
                # Limpiar línea actual
                print("\r" + " " * 80, end="")
                
                # Mostrar estado
                status_icon = "✅" if result['count'] == 0 else "⚠️" if result['count'] == 1 else "🔴"
                print(f"\r{status_icon} FPS: {fps} | Líneas: {line_count} | Alertas: {result['count']} | {result['severity']}", end="")
                
                # Mostrar detección IA si hay
                if result['ai_detected']:
                    print(f" | 🤖 IA: {result['ai_prediction']}", end="")
            
            # --- VISUALIZACIÓN EN VENTANA ---
            # Dibujar líneas en el frame
            frame_with_lines = image_processing.draw_lines(frame, lines)
            
            # Agregar overlay de información
            h, w = frame_with_lines.shape[:2]
            
            # Fondo semitransparente para texto
            overlay = frame_with_lines.copy()
            cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame_with_lines, 0.4, 0, frame_with_lines)
            
            # Mostrar estadísticas en la imagen
            cv2.putText(frame_with_lines, f"FPS: {fps}", (w - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame_with_lines, f"Lineas: {line_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_with_lines, f"Alertas: {result['count']}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 0, 255) if result['count'] > 0 else (0, 255, 0), 2)
            cv2.putText(frame_with_lines, result['severity'], (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 0, 255) if result['count'] > 0 else (0, 255, 0), 2)
            
            # Mostrar estado de IA
            if detector.ai_model_available:
                stats = detector.ai_model.get_performance_stats()
                ia_text = f"IA: {stats['fps']:.1f} fps" if stats['fps'] > 0 else "IA: Activa"
                cv2.putText(frame_with_lines, ia_text, (w - 150, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            # Mostrar advertencias en pantalla (máximo 2)
            if result['warnings']:
                y_offset = 130
                for i, warning in enumerate(result['warnings'][:2]):
                    color = (0, 0, 255) if "spaghetti" in warning.lower() else (0, 165, 255)
                    cv2.putText(frame_with_lines, warning[:45], (10, y_offset + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Mostrar frame
            cv2.imshow("Monitoreo Impresora 3D - IA + Vision", frame_with_lines)
            
            # --- MANEJO DE TECLAS ---
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n\n👋 Cerrando aplicación...")
                break
                
            elif key == ord('s'):
                # Guardar captura
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"captura_{timestamp}.jpg"
                cv2.imwrite(filename, frame_with_lines)
                print(f"\n📸 Captura guardada: {filename}")
                
            elif key == ord('r'):
                # Reiniciar estadísticas
                detector.reset_statistics()
                error_count = 0
                total_frames = 0
                print("\n🔄 Estadísticas reiniciadas")
                
            elif key == ord('i'):
                # Mostrar estadísticas de IA
                print("\n" + "=" * 60)
                print("📊 ESTADÍSTICAS DEL SISTEMA")
                print("=" * 60)
                
                if detector.ai_model_available:
                    stats = detector.ai_model.get_performance_stats()
                    print(f"\n🤖 IA - Rendimiento:")
                    print(f"   Tiempo promedio inferencia: {stats['avg_time']*1000:.2f} ms")
                    print(f"   FPS IA: {stats['fps']:.1f}")
                    print(f"   Total inferencias: {stats['total_inferences']}")
                
                detector_stats = detector.get_statistics()
                if detector_stats['total_predictions'] > 0:
                    print(f"\n📈 Detección de errores:")
                    print(f"   Total predicciones: {detector_stats['total_predictions']}")
                    for clase, porcentaje in detector_stats['error_percentages'].items():
                        if porcentaje > 0:
                            print(f"   {clase}: {porcentaje:.1f}%")
                
                print(f"\n🎯 Estado actual:")
                print(f"   Frames procesados: {total_frames}")
                print(f"   Alertas totales: {error_count}")
                print("=" * 60)
                
            elif key == ord('t'):
                # Mostrar umbrales
                print("\n⚙️ CONFIGURACIÓN ACTUAL")
                print("=" * 60)
                print(f"   Máximo líneas normales: {detector.max_lines_threshold}")
                print(f"   Mínimo líneas para alerta: {detector.min_lines_threshold}")
                print(f"   Umbral líneas verticales: {detector.vertical_threshold:.0%}")
                print(f"   Umbral líneas horizontales: {detector.horizontal_threshold:.0%}")
                print(f"   IA activada: {'✅' if detector.use_ai else '❌'}")
                print("=" * 60)
            
            # Alerta de emergencia por spaghetti
            if result['emergency_stop']:
                print("\n" + "=" * 60)
                print("🚨 ¡ALERTA DE EMERGENCIA! 🚨")
                print("   SPAGHETTI DETECTADO - DETENER IMPRESIÓN INMEDIATAMENTE")
                print("=" * 60)
                # Beep de alerta (si está disponible)
                try:
                    import platform
                    if platform.system() == "Windows":
                        import winsound
                        winsound.Beep(1000, 500)
                        winsound.Beep(1500, 500)
                    else:
                        print("\a", end="", flush=True)
                except:
                    pass
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupción por teclado")
    
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
    
    finally:
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Aplicación cerrada correctamente")

def test_camera():
    """
    Función de prueba para verificar que la cámara funciona
    """
    print("🔧 Probando cámara...")
    cap = camera_module.start_camera()
    
    if cap is None:
        print("❌ No se detectó cámara")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo capturar frame")
        cap.release()
        return False
    
    print(f"✅ Cámara funcionando correctamente")
    print(f"   Resolución: {frame.shape[1]}x{frame.shape[0]}")
    
    cap.release()
    return True

if __name__ == "__main__":
    # Probar cámara primero
    if test_camera():
        main()
    else:
        print("\n❌ No se puede iniciar la aplicación sin cámara.")