"""
app_video.py - Procesar video grabado en lugar de cámara en vivo
"""

import cv2
import time
import numpy as np
import image_processing
import trace_analysis

def main():
    print("=" * 60)
    print("🖨️ MONITOREO DE IMPRESIÓN 3D - MODO VIDEO")
    print("   Procesando video grabado")
    print("=" * 60)
    
    # CAMBIA ESTA RUTA por tu video de impresora 3D
    video_path = "impresora_3d.mp4"  # Ruta de tu video
    
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ No se pudo abrir el video: {video_path}")
        print("   Coloca un video en la misma carpeta o cambia la ruta")
        return
    
    print(f"✅ Video cargado correctamente")
    
    # Obtener información del video
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   FPS: {fps_video}")
    print(f"   Total frames: {total_frames}")
    
    print("\n📷 Controles:")
    print("   'q' - Salir")
    print("   'p' - Pausa/Reanudar")
    print("   's' - Guardar captura")
    print("=" * 60)
    
    frame_count = 0
    last_time = time.time()
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n📹 Fin del video. Repitiendo...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar video
                continue
        
        if not paused:
            # Procesamiento
            edges = image_processing.preprocess_frame(frame)
            lines = image_processing.detect_lines(edges)
            analysis = trace_analysis.analyze_lines(lines)
            
            # Detección de errores básica
            warnings = []
            if lines is None or len(lines) == 0:
                warnings.append("⚠ NO HAY TRAZOS")
            elif len(lines) > 30:
                warnings.append(f"⚠ DEMASIADOS TRAZOS ({len(lines)})")
            
            # Calcular FPS de procesamiento
            frame_count += 1
            current_time = time.time()
            if current_time - last_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_time = current_time
                print(f"\r📊 FPS: {fps} | Líneas: {analysis['count']} | Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{total_frames}", end="")
            
            # Dibujar líneas
            frame_with_lines = image_processing.draw_lines(frame, lines)
            
            # Agregar información
            cv2.putText(frame_with_lines, f"Lineas: {analysis['count']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_with_lines, f"Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{total_frames}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Mostrar
            cv2.imshow("Monitoreo Impresora 3D - Video", frame_with_lines)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("\n" + ("⏸ PAUSA" if paused else "▶ REANUDANDO"))
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"captura_{timestamp}.jpg"
            cv2.imwrite(filename, frame_with_lines)
            print(f"\n📸 Captura guardada: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Aplicación cerrada")

if __name__ == "__main__":
    main()