"""
app_optimized.py - Versión optimizada del monitor de impresión 3D
Optimizaciones aplicadas:
1. Reducción de resolución (640x480 → 320x240)
2. Frame skipping (procesa 1 de cada 3 frames)
3. Threshold de Hough optimizado (120 → 150)
4. Modo rápido de inferencia
"""

import cv2
import time
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))

import camera_module
import image_processing
import trace_analysis
from error_detection import ErrorDetector


def main():
    print("=" * 60)
    print("🖨️ MONITOREO 3D - VERSIÓN OPTIMIZADA")
    print("   Modo: Alta velocidad (frame skipping + baja resolución)")
    print("=" * 60)

    # ========== 1. INICIAR CÁMARA CON RESOLUCIÓN REDUCIDA ==========
    cap = camera_module.start_camera(0)
    if cap is None:
        print("❌ Error: No se pudo acceder a la cámara.")
        return

    # 🔥 OPTIMIZACIÓN 1: Reducir resolución (640x480 → 320x240)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"📷 Cámara configurada: 320x240 píxeles")

    # ========== 2. INICIALIZAR DETECTOR ==========
    detector = ErrorDetector(use_ai=True)

    print("\n📋 CONFIGURACIÓN DE OPTIMIZACIÓN:")
    print("   - Resolución: 320x240 (reducida)")
    print("   - Frame skipping: 1 de cada 3 frames")
    print("   - Threshold Hough: 150 (optimizado)")
    print("   - IA: Activada (modo rápido)")
    print("\n📷 Controles:")
    print("   'q' - Salir")
    print("   's' - Guardar captura")
    print("=" * 50)

    # ========== 3. VARIABLES DE OPTIMIZACIÓN ==========
    frame_counter = 0
    SKIP_FRAMES = 3  # 🔥 OPTIMIZACIÓN 2: Procesar 1 de cada 3 frames
    last_time = time.time()
    fps = 0

    # ========== 4. BUCLE PRINCIPAL OPTIMIZADO ==========
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # 🔥 OPTIMIZACIÓN 2: Frame skipping
        if frame_counter % SKIP_FRAMES == 0:
            # ----- PROCESAMIENTO COMPLETO (solo 1 de cada 3 frames) -----
            edges = image_processing.preprocess_frame(frame)

            # 🔥 OPTIMIZACIÓN 3: Hough con threshold más alto
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)

            analysis = trace_analysis.analyze_lines(lines)
            result = detector.detect_all(frame, edges, lines)

            # Dibujar líneas
            frame_with_lines = image_processing.draw_lines(frame, lines)

            # Mostrar información en pantalla
            h, w = frame_with_lines.shape[:2]

            # FPS
            cv2.putText(frame_with_lines, f"FPS: {fps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Contador de líneas
            cv2.putText(frame_with_lines, f"Lineas: {analysis['count']}", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Estado del sistema
            cv2.putText(frame_with_lines, result['severity'], (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 0, 255) if result['count'] > 0 else (0, 255, 0), 1)

            # Mostrar que estamos en modo optimizado
            cv2.putText(frame_with_lines, "MODO OPTIMIZADO", (w - 120, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Mostrar alertas
            if result['warnings']:
                for i, warning in enumerate(result['warnings'][:2]):
                    cv2.putText(frame_with_lines, warning[:35], (10, 105 + i * 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            cv2.imshow("Monitoreo 3D - Optimizado", frame_with_lines)

        else:
            # ----- SOLO MOSTRAR FRAME SIN PROCESAR (mantiene fluidez) -----
            cv2.imshow("Monitoreo 3D - Optimizado", frame)

        # ========== 5. CÁLCULO DE FPS ==========
        current_time = time.time()
        if current_time - last_time >= 1.0:
            fps = frame_counter
            frame_counter = 0
            last_time = current_time
            print(f"\r📊 FPS: {fps} | Procesando 1/{SKIP_FRAMES} frames", end="")

        # ========== 6. MANEJO DE TECLAS ==========
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n\n👋 Cerrando aplicación...")
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"captura_optimizada_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"\n📸 Captura guardada: {filename}")

    # ========== 7. LIBERAR RECURSOS ==========
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Aplicación cerrada correctamente")


if __name__ == "__main__":
    main()