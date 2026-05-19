"""
trace_analysis.py - Análisis estadístico de trazos detectados en impresión 3D
"""

import numpy as np

def analyze_lines(lines):
    """
    Analiza las líneas detectadas por Hough Transform
    
    Args:
        lines: líneas en formato polares (rho, theta) de cv2.HoughLines
    
    Returns:
        dict con estadísticas:
            - count: número de líneas detectadas
            - angles: lista de ángulos en grados
            - avg_angle: ángulo promedio (para gráficos)
            - std_angle: desviación estándar (para detección de anomalías)
    """
    if lines is None or len(lines) == 0:
        return {
            "count": 0,
            "angles": [],
            "avg_angle": 0.0,
            "std_angle": 0.0
        }
    
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle_deg = np.degrees(theta)
        angles.append(angle_deg)
    
    # Calcular estadísticas para gráficos
    angles_array = np.array(angles)
    
    return {
        "count": len(lines),
        "angles": angles,
        "avg_angle": float(np.mean(angles_array)),
        "std_angle": float(np.std(angles_array))
    }

def analyze_lines_probabilistic(lines):
    """
    Analiza líneas en formato probabilístico (x1, y1, x2, y2)
    
    Args:
        lines: líneas en formato (x1, y1, x2, y2)
    
    Returns:
        dict con estadísticas de líneas
    """
    if lines is None or len(lines) == 0:
        return {
            "count": 0,
            "lengths": [],
            "angles": [],
            "avg_length": 0.0,
            "avg_angle": 0.0
        }
    
    lengths = []
    angles = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calcular longitud
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        lengths.append(length)
        
        # Calcular ángulo
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    
    return {
        "count": len(lines),
        "lengths": lengths,
        "angles": angles,
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        "avg_angle": float(np.mean(angles)) if angles else 0.0,
        "std_angle": float(np.std(angles)) if angles else 0.0
    }