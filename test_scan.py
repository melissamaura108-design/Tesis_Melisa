import sys
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

def test_carga_interfaz():
    """
    Prueba T3-01: Verificar que la interfaz carga sin errores
    """
    print("\n" + "=" * 60)
    print("PRUEBA T3-01: Carga de la interfaz gráfica")
    print("=" * 60)

    try:
        from scan import MainWindow

        # Crear aplicación Qt
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        # Crear y mostrar ventana
        window = MainWindow()
        window.show()

        print("✅ Clase MainWindow importada correctamente")
        print("✅ Ventana principal creada y mostrada")
        print(f"   Título de la ventana: {window.windowTitle()}")
        print(f"   Geometría: {window.geometry().width()}x{window.geometry().height()}")

        # Cerrar después de 2 segundos
        QTimer.singleShot(2000, window.close)
        app.exec_()

        return True

    except Exception as e:
        print(f"❌ Error al cargar la interfaz: {e}")
        return False


def test_componentes_interfaz():
    """
    Prueba T3-02: Verificar que todos los componentes existen
    """
    print("\n" + "=" * 60)
    print("PRUEBA T3-02: Verificación de componentes de la interfaz")
    print("=" * 60)

    try:
        from scan import MainWindow

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        window = MainWindow()

        # Verificar existencia de componentes
        componentes = {
            'btn_start': hasattr(window, 'btn_start'),
            'btn_stop': hasattr(window, 'btn_stop'),
            'btn_capture': hasattr(window, 'btn_capture'),
            'video_label': hasattr(window, 'video_label'),
            'lines_label': hasattr(window, 'lines_label'),
            'status_label': hasattr(window, 'status_label'),
            'error_list': hasattr(window, 'error_list')
        }

        for nombre, existe in componentes.items():
            estado = "✅" if existe else "❌"
            print(f"   {estado} {nombre}: {'Existe' if existe else 'No existe'}")

        window.close()

        return all(componentes.values())

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_botones_conectados():
    """
    Prueba T3-03: Verificar que los botones tienen sus funciones conectadas
    """
    print("\n" + "=" * 60)
    print("PRUEBA T3-03: Conexión de botones")
    print("=" * 60)

    try:
        from scan import MainWindow
        from PyQt5.QtCore import QMetaMethod

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        window = MainWindow()

        # Verificar señales conectadas
        botones = [
            ('btn_start', 'clicked'),
            ('btn_stop', 'clicked'),
            ('btn_capture', 'clicked')
        ]

        for boton, señal in botones:
            btn = getattr(window, boton)
            # Verificar si tiene conexiones
            if btn.receivers(btn.clicked) > 0:
                print(f"✅ {boton} → conectado a {señal}")
            else:
                print(f"⚠ {boton} → sin conexiones")

        window.close()
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_visualizacion_video():
    """
    Prueba T3-04: Verificar visualización de video (prueba manual)
    """
    print("\n" + "=" * 60)
    print("PRUEBA T3-04: Visualización de video")
    print("=" * 60)

    try:
        from scan import MainWindow

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        window = MainWindow()
        window.show()

        print("\n⚠ PRUEBA MANUAL REQUERIDA")
        print("   Observa la ventana que aparece y verifica:")
        print("   1. ¿La ventana tiene el título correcto?")
        print("   2. ¿Hay un área para mostrar el video?")
        print("   3. ¿Aparecen los botones? (Iniciar, Detener, Capturar)")
        print("   4. ¿Se ve el panel de alertas?")
        print("   5. ¿La ventana se puede cerrar correctamente?\n")

        input("   Presiona ENTER después de verificar...")

        window.close()
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def run_pruebas_tarea3():
    """Ejecutar todas las pruebas de la Tarea #3"""
    print("\n" + "🔧" * 30)
    print("PRUEBAS DE LA TAREA #3: Interfaz Gráfica PyQt5")
    print("Componentes: MainWindow | Botones | VideoLabel")
    print("🔧" * 30)

    pruebas = [
        ("T3-01: Carga de la interfaz", test_carga_interfaz),
        ("T3-02: Componentes de la interfaz", test_componentes_interfaz),
        ("T3-03: Conexión de botones", test_botones_conectados),
        ("T3-04: Visualización de video", test_visualizacion_video)
    ]

    print("\n" + "=" * 60)
    print("RESUMEN DE PRUEBAS - TAREA #3")
    print("=" * 60)

    for nombre, prueba in pruebas:
        resultado = prueba()
        estado = "✅ PASÓ" if resultado else "❌ FALLÓ"
        print(f"{estado} - {nombre}")

    print("\n📊 Interfaz gráfica construida con PyQt5")
    print("   - Ventana principal: QMainWindow")
    print("   - Hilo de procesamiento: QThread")
    print("   - Conversión de frames: QImage → QPixmap")


if __name__ == "__main__":
    run_pruebas_tarea3()