"""
app_no_torch.py - Monitoreo de impresión 3d SIN PyTorch
Solo visión artificial tradicional - No requiere IA
"""

import cv2
import numpy as np
import time

def start_camera(index=0):
    print("🎥 Iniciando cámara...")
    
    # Intentar abrir cámara
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        # Probar otros índices
        for i in range(0, 4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✔ Cámara encontrada en índice {i}")
                break
        else:
            print("❌ No se pudo abrir la cámara.")
            return None
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✔ Cámara iniciada correctamente.")
    return cap

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def detect_lines(edges):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
    return lines

def draw_lines(frame, lines):
    if lines is None:
        return frame
    
    frame_copy = frame.copy()
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frame_copy

def analyze_lines(lines):
    if lines is None:
        return {"count": 0, "angles": []}
    
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle_deg = np.degrees(theta)
        angles.append(angle_deg)
    
    return {
        "count": len(lines),
        "angles": angles
    }

def main():
    print("=" * 60)
    print("🖨️ MONITOREO DE IMPRESIÓN 3D")
    print("   Modo: Visión Artificial (Sin IA)")
    print("=" * 60)
    
    cap = start_camera()
    if cap is None:
        print("❌ Error: No se pudo acceder a la cámara.")
        return
    
    print("\n📷 Controles:")
    print("   'q' - Salir")
    print("   's' - Guardar captura")
    print("=" * 60)
    
    frame_count = 0
    last_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo leer el frame.")
            break
        
        # Procesamiento
        edges = preprocess_frame(frame)
        lines = detect_lines(edges)
        analysis = analyze_lines(lines)
        
        # Detección de errores básica
        warnings = []
        
        if lines is None or len(lines) == 0:
            warnings.append("⚠ NO HAY TRAZOS - Posible falta de extrusión")
        elif len(lines) > 30:
            warnings.append(f"⚠ DEMASIADOS TRAZOS ({len(lines)})")
        elif len(lines) < 5:
            warnings.append(f"⚠ POCOS TRAZOS ({len(lines)})")
        
        # Calcular FPS
        frame_count += 1
        current_time = time.time()
        if current_time - last_time >= 1.0:
            fps = frame_count
            frame_count = 0
            last_time = current_time
        
        # Mostrar información en consola
        print(f"\r📊 FPS: {fps} | Líneas: {analysis['count']} | Alertas: {len(warnings)}", end="")
        
        # Dibujar en imagen
        frame_with_lines = draw_lines(frame, lines)
        
        # Agregar información en pantalla
        cv2.putText(frame_with_lines, f"FPS: {fps}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_with_lines, f"Lineas: {analysis['count']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if warnings:
            cv2.putText(frame_with_lines, warnings[0][:40], (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Mostrar
        cv2.imshow("Monitoreo Impresora 3D", frame_with_lines)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n\n👋 Cerrando...")
            break
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