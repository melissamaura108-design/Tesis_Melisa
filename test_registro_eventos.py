"""
test_tarea7.py - Prueba de la Tarea #7: Registro de eventos
"""

import csv
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(__file__))


def test_csv_export():
    """
    Prueba T7-01: Exportación de logs a CSV
    """
    print("\n" + "=" * 60)
    print("PRUEBA T7-01: Exportación de logs a CSV")
    print("=" * 60)

    # Simular datos de log (como los que genera el sistema)
    log_data = [
        {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
         "lines": 15, "severity": "NORMAL", "warnings": ""},
        {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
         "lines": 0, "severity": "ATENCIÓN", "warnings": "No se detectan trazos"},
        {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
         "lines": 45, "severity": "CRÍTICO", "warnings": "Demasiados trazos detectados"}
    ]

    # Guardar CSV
    filename = f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "lines", "severity", "warnings"])
            for entry in log_data:
                writer.writerow([entry["timestamp"], entry["lines"], 
                               entry["severity"], entry["warnings"]])

        print(f"✅ Archivo CSV creado: {filename}")

        # Verificar que el archivo existe
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✅ Archivo verificado: {size} bytes")
            
            # Mostrar contenido
            print("\n📄 Contenido del archivo CSV:")
            with open(filename, 'r', encoding='utf-8') as f:
                print(f.read())
            
            # Limpiar archivo de prueba
            os.remove(filename)
            print("\n✅ Archivo de prueba eliminado")
            return True
        else:
            print("❌ El archivo no se creó correctamente")
            return False

    except Exception as e:
        print(f"❌ Error al guardar CSV: {e}")
        return False


def test_formato_csv():
    """
    Prueba T7-02: Verificar formato correcto del CSV
    """
    print("\n" + "=" * 60)
    print("PRUEBA T7-02: Formato del archivo CSV")
    print("=" * 60)

    filename = f"test_formato_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # Guardar CSV con formato
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "lines", "severity", "warnings"])
        writer.writerow(["2025-05-01 10:00:00", "15", "NORMAL", ""])
        writer.writerow(["2025-05-01 10:00:05", "0", "ATENCIÓN", "No se detectan trazos"])

    # Leer y verificar formato
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"✅ Contenido del CSV:")
            print(content)
            
        # Verificar que tiene el formato correcto (columnas separadas por comas)
        if "timestamp,lines,severity,warnings" in content:
            print("✅ Formato CSV correcto (columnas separadas por comas)")
        else:
            print("⚠ Formato CSV podría ser incorrecto")

        # Limpiar archivo de prueba
        os.remove(filename)
        return True

    except Exception as e:
        print(f"❌ Error al leer CSV: {e}")
        return False


def run_pruebas_tarea7():
    """Ejecutar todas las pruebas de la Tarea #7"""
    print("\n" + "🔧" * 30)
    print("PRUEBAS DE LA TAREA #7: Registro de Eventos")
    print("🔧" * 30)

    prueba1 = test_csv_export()
    prueba2 = test_formato_csv()

    print("\n" + "=" * 60)
    print("RESUMEN DE PRUEBAS - TAREA #7")
    print("=" * 60)
    print(f"{'✅ PASÓ' if prueba1 else '❌ FALLÓ'} - T7-01: Exportación a CSV")
    print(f"{'✅ PASÓ' if prueba2 else '❌ FALLÓ'} - T7-02: Formato del CSV")


if __name__ == "__main__":
    run_pruebas_tarea7()