"""
test_rendimiento_pr01.py - Prueba de rendimiento PR-01
Objetivo: Comparar FPS entre versión normal y versión optimizada
"""

import cv2
import time
import sys
import os
import threading

sys.path.append(os.path.dirname(__file__))

def test_fps_normal(duracion=5):
    """
    PR-01A: Medir FPS con configuración normal (640x480, todos los frames)
    """
    print("\n" + "=" * 60)
    print("PR-01A: FPS - Configuración NORMAL")
    print("Resolución: 640x480 | Frame skipping: NO")
    print("=" * 60)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < duracion:
        ret, frame = cap.read()
        if ret:
            frame_count += 1

    cap.release()

    fps = frame_count / duracion
    print(f"📊 Resultados:")
    print(f"   Frames capturados: {frame_count}")
    print(f"   Duración: {duracion} segundos")
    print(f"   FPS promedio: {fps:.1f}")
    print(f"   Tiempo por frame: {(1/fps)*1000:.1f} ms" if fps > 0 else "   Tiempo por frame: N/A")

    return fps


def test_fps_optimizada(duracion=5):
    """
    PR-01B: Medir FPS con configuración optimizada (320x240, todos los frames)
    """
    print("\n" + "=" * 60)
    print("PR-01B: FPS - Configuración OPTIMIZADA")
    print("Resolución: 320x240 | Frame skipping: NO")
    print("=" * 60)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < duracion:
        ret, frame = cap.read()
        if ret:
            frame_count += 1

    cap.release()

    fps = frame_count / duracion
    print(f"📊 Resultados:")
    print(f"   Frames capturados: {frame_count}")
    print(f"   Duración: {duracion} segundos")
    print(f"   FPS promedio: {fps:.1f}")
    print(f"   Tiempo por frame: {(1/fps)*1000:.1f} ms" if fps > 0 else "   Tiempo por frame: N/A")

    return fps


def test_fps_con_procesamiento(duracion=5):
    """
    PR-01C: Medir FPS con procesamiento completo (Canny + Hough)
    """
    print("\n" + "=" * 60)
    print("PR-01C: FPS - Con procesamiento (Canny + Hough)")
    print("Resolución: 320x240 | Frame skipping: NO")
    print("=" * 60)

    import image_processing

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < duracion:
        ret, frame = cap.read()
        if ret:
            edges = image_processing.preprocess_frame(frame)
            lines = image_processing.detect_lines(edges)
            frame_count += 1

    cap.release()

    fps = frame_count / duracion
    print(f"📊 Resultados:")
    print(f"   Frames procesados: {frame_count}")
    print(f"   Duración: {duracion} segundos")
    print(f"   FPS promedio: {fps:.1f}")
    print(f"   Tiempo por frame: {(1/fps)*1000:.1f} ms" if fps > 0 else "   Tiempo por frame: N/A")

    return fps


def run_pruebas_rendimiento():
    """Ejecutar todas las pruebas de rendimiento"""
    print("\n" + "⚡" * 30)
    print("PRUEBAS DE RENDIMIENTO - MÉTRICAS DE FPS")
    print("⚡" * 30)

    resultados = {}

    # PR-01A: Normal
    fps_normal = test_fps_normal(duracion=5)
    resultados['PR-01A: Normal (640x480)'] = fps_normal

    # PR-01B: Optimizada
    fps_optimizada = test_fps_optimizada(duracion=5)
    resultados['PR-01B: Optimizada (320x240)'] = fps_optimizada

    # PR-01C: Con procesamiento
    fps_procesamiento = test_fps_con_procesamiento(duracion=5)
    resultados['PR-01C: Con procesamiento'] = fps_procesamiento

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE PRUEBAS DE RENDIMIENTO")
    print("=" * 60)

    for nombre, fps in resultados.items():
        estado = "✅" if fps >= 15 else "⚠️"
        print(f"{estado} {nombre}: {fps:.1f} FPS")

    print("\n📊 Comparativa de mejora:")
    if resultados.get('PR-01A: Normal (640x480)', 0) > 0:
        mejora = (resultados['PR-01B: Optimizada (320x240)'] / resultados['PR-01A: Normal (640x480)']) * 100
        print(f"   Mejora al reducir resolución: +{mejora:.0f}%")


if __name__ == "__main__":
    run_pruebas_rendimiento()