"""
test_rendimiento_pr02_simple.py - Prueba de rendimiento PR-02 (SIN IA)
Solo mide procesamiento tradicional (Canny, Hough, análisis)
"""

import cv2
import time
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(__file__))

# Importar solo módulos necesarios (sin inference.py)
import image_processing
import trace_analysis


def test_tiempos_procesamiento():
    """
    PR-02: Medir tiempos de procesamiento tradicional
    """
    print("\n" + "=" * 60)
    print("PR-02: Tiempos de procesamiento (Visión Tradicional)")
    print("=" * 60)

    # Capturar frame de prueba
    print("\n📷 Capturando frame de prueba...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ No se pudo capturar frame")
        return False

    print("✅ Frame capturado correctamente")

    detecciones = 30
    tiempos = {
        'preprocess': [],
        'detect_lines': [],
        'trace_analysis': []
    }

    print(f"\n📊 Midiendo {detecciones} iteraciones...")

    for i in range(detecciones):
        # Tiempo de preprocesamiento (Canny)
        start = time.time()
        edges = image_processing.preprocess_frame(frame)
        tiempos['preprocess'].append((time.time() - start) * 1000)

        # Tiempo de detección de líneas (Hough)
        start = time.time()
        lines = image_processing.detect_lines(edges)
        tiempos['detect_lines'].append((time.time() - start) * 1000)

        # Tiempo de análisis de líneas
        start = time.time()
        analysis = trace_analysis.analyze_lines(lines)
        tiempos['trace_analysis'].append((time.time() - start) * 1000)

    # Calcular promedios
    print("\n📈 Resultados:")
    print("-" * 40)

    tiempo_total = 0
    for etapa, mediciones in tiempos.items():
        promedio = sum(mediciones) / len(mediciones)
        minimo = min(mediciones)
        maximo = max(mediciones)
        tiempo_total += promedio
        print(f"   {etapa}:")
        print(f"      Promedio: {promedio:.2f} ms")
        print(f"      Mínimo:   {minimo:.2f} ms")
        print(f"      Máximo:   {maximo:.2f} ms")

    print("-" * 40)
    print(f"   TIEMPO TOTAL POR FRAME: {tiempo_total:.2f} ms")
    print(f"   FPS TEÓRICO: {1000/tiempo_total:.1f} FPS")

    print("\n" + "=" * 60)
    print("RESULTADO PR-02")
    print("=" * 60)

    if tiempo_total < 150:
        print("✅ Tiempo total dentro del límite (<150 ms)")
    else:
        print(f"⚠ Tiempo total excede el límite: {tiempo_total:.1f} ms > 150 ms")

    return True


if __name__ == "__main__":
    test_tiempos_procesamiento()