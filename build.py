#!/usr/bin/env python3
"""
TKGM Parsel Veri Cekme - Build Script
PyInstaller ile EXE/APP olusturur
"""

import subprocess
import sys
import platform

def build():
    """PyInstaller ile standalone uygulama olusturur."""
    system = platform.system()

    # PyInstaller komutu
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "TKGM-Parsel",
        "--windowed",  # Console penceresi gosterme
        "--onefile",   # Tek dosya
        "--clean",
        # Icon eklemek isterseniz:
        # "--icon", "icon.ico",  # Windows
        # "--icon", "icon.icns", # macOS
    ]

    # Hidden imports (gerekli olabilir)
    hidden_imports = [
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "qfluentwidgets",
        "requests",
    ]

    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])

    # Veri dosyalari
    # cmd.extend(["--add-data", "icon.png:."]) # Ornek

    # Ana dosya
    cmd.append("app.py")

    print(f"Building for {system}...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print("\n" + "="*50)
        print("BUILD BASARILI!")
        print("="*50)
        if system == "Darwin":
            print("Cikti: dist/TKGM-Parsel.app")
        elif system == "Windows":
            print("Cikti: dist/TKGM-Parsel.exe")
        else:
            print("Cikti: dist/TKGM-Parsel")
    else:
        print("\nBuild basarisiz!")
        sys.exit(1)

if __name__ == "__main__":
    build()
