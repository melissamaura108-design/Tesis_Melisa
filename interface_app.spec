# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import os

cv2_path = r"E:\Tesis_Proyecto\tesis-vision3d\venv\Lib\site-packages\cv2"
numpy_path = r"E:\Tesis_Proyecto\tesis-vision3d\venv\Lib\site-packages\numpy"

a = Analysis(
    ['interface_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        (cv2_path, 'cv2'),
        (numpy_path, 'numpy'),
        ('image_processing.py', '.'),
        ('camera_module.py', '.'),
        ('trace_analysis.py', '.'),
        ('error_detection.py', '.'),
        ('app.py', '.'),
    ],
    hiddenimports=['cv2', 'numpy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='interface_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
)
