"""
image_processing.py - Procesamiento de imágenes para detección de trazos en impresión 3D
"""

import cv2
import numpy as np

def preprocess_frame(frame):
    """
    Preprocesa el frame para detección de bordes
    
    Args:
        frame: imagen en formato BGR (OpenCV)
    
    Returns:
        edges: imagen con bordes detectados (Canny)
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar blur para reducir ruido
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    
    # Detectar bordes con Canny
    edges = cv2.Canny(blur, 50, 150)
    
    return edges

def detect_lines(edges):
    """
    Detecta líneas usando Transformada de Hough Probabilística
    
    Args:
        edges: imagen con bordes detectados
    
    Returns:
        lines: líneas detectadas en formato polar (rho, theta)
    """
    # Parámetros ajustables para mejor detección
    threshold = 120
    min_line_length = 50
    max_line_gap = 10
    
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    
    return lines

def draw_lines(frame, lines):
    """
    Dibuja las líneas detectadas sobre el frame original
    
    Args:
        frame: imagen original
        lines: líneas detectadas por HoughLines
    
    Returns:
        frame_copy: imagen con líneas dibujadas
    """
    if lines is None:
        return frame

    frame_copy = frame.copy()
    
    for line in lines:
        rho, theta = line[0]
        
        # Convertir coordenadas polares a cartesianas
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        # Dibujar línea en verde
        cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame_copy

def detect_lines_probabilistic(edges):
    """
    Versión probabilística de detección de líneas (alternativa más rápida)
    
    Args:
        edges: imagen con bordes detectados
    
    Returns:
        lines: líneas en formato (x1, y1, x2, y2)
    """
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    return lines

def draw_lines_p(frame, lines):
    """
    Dibuja líneas en formato probabilístico
    
    Args:
        frame: imagen original
        lines: líneas en formato (x1, y1, x2, y2)
    
    Returns:
        frame_copy: imagen con líneas dibujadas
    """
    if lines is None:
        return frame
    
    frame_copy = frame.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frame_copy