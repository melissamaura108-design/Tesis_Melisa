"""
camera_module.py - Módulo para manejo de cámara en impresión 3D
"""

import cv2

def start_camera(index=0):
    """
    Inicia la cámara con el índice especificado
    
    Args:
        index: índice de la cámara (0, 1, 2, etc.)
    
    Returns:
        cap: objeto VideoCapture o None si hay error
    """
    print("🎥 Iniciando cámara...")

    # Intentar abrir cámara con diferentes backends según SO
    import platform
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return None
    
    # Configurar resolución para mejor rendimiento
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✔ Cámara iniciada correctamente.")
    return cap

def release_camera(cap):
    """Libera la cámara"""
    if cap is not None:
        cap.release()
        print("📷 Cámara liberada")