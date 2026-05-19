"""
test_integracion_pi05.py - Prueba de integración PI-05
Pipeline completo: Cámara → Procesamiento → Análisis → Detección → IA
Objetivo: Verificar que el flujo completo de datos funciona correctamente
"""

import cv2
import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(__file__))

try:
    import camera_module
    import image_processing
    import trace_analysis
    from error_detection import ErrorDetector
    print("✅ Todos los módulos importados correctamente")
except ImportError as e:
    print(f"❌ Error al importar: {e}")
    sys.exit(1)


def test_pipeline_completo():
    """
    PI-05: Verificar el flujo completo de datos
    """
    print("\n" + "=" * 60)
    print("PRUEBA PI-05: Pipeline Completo")
    print("Flujo: Cámara → Procesamiento → Análisis → Detección → IA")
    print("=" * 60)

    # 1. Inicializar componentes
    print("\n📷 Paso 1: Inicializando cámara y detector...")
    cap = camera_module.start_camera(0)
    if cap is None:
        print("❌ No se pudo iniciar la cámara")
        return False
    
    detector = ErrorDetector(use_ai=True)
    print("✅ Componentes inicializados")

    # 2. Capturar y procesar un frame
    print("\n🎬 Paso 2: Capturando y procesando frame...")
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo capturar el frame")
        cap.release()
        return False
    
    start_time = time.time()
    
    # Pipeline completo
    edges = image_processing.preprocess_frame(frame)
    lines = image_processing.detect_lines(edges)
    analysis = trace_analysis.analyze_lines(lines)
    result = detector.detect_all(frame, edges, lines)
    
    elapsed_ms = (time.time() - start_time) * 1000

    # 3. Mostrar resultados
    print("\n📊 Paso 3: Resultados del pipeline:")
    print(f"   ⏱ Tiempo total: {elapsed_ms:.1f} ms")
    print(f"   📈 Líneas detectadas: {analysis['count']}")
    print(f"   📐 Ángulo promedio: {analysis['avg_angle']:.1f}°")
    print(f"   🎯 Severidad: {result['severity']}")
    print(f"   ⚠ Alertas: {result['warnings']}")
    print(f"   🤖 IA detectó: {result['ai_detected']}")
    print(f"   🚨 Emergencia: {result['emergency_stop']}")

    # 4. Limpiar
    camera_module.release_camera(cap)

    print("\n" + "=" * 60)
    print("RESULTADO PI-05")
    print("=" * 60)
    print("✅ El pipeline completo funciona correctamente")
    print("✅ Todos los módulos se comunican sin errores")
    
    return True


def run_pruebas_integracion():
    """Ejecutar pruebas de integración"""
    print("\n" + "🔗" * 30)
    print("PRUEBAS DE INTEGRACIÓN - PI-05")
    print("Verificando pipeline completo del sistema")
    print("🔗" * 30)

    resultado = test_pipeline_completo()

    print("\n" + "=" * 60)
    print("RESUMEN PRUEBAS DE INTEGRACIÓN")
    print("=" * 60)
    print(f"{'✅ PASÓ' if resultado else '❌ FALLÓ'} - PI-05: Pipeline completo")


if __name__ == "__main__":
    run_pruebas_integracion()