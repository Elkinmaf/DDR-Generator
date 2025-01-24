import os
import sys
print("Python version:", sys.version)
print("Working directory:", os.getcwd())

try:
    print("\nProbando importación de moviepy...")
    from moviepy.editor import VideoFileClip
    print("✓ moviepy importado correctamente")
except Exception as e:
    print("✗ Error importando moviepy:", str(e))

try:
    print("\nProbando importación de librosa...")
    import librosa
    print("✓ librosa importado correctamente")
except Exception as e:
    print("✗ Error importando librosa:", str(e))

try:
    print("\nProbando importación de OpenCV...")
    import cv2
    print("✓ OpenCV importado correctamente")
except Exception as e:
    print("✗ Error importando OpenCV:", str(e))

try:
    print("\nProbando importación de numpy...")
    import numpy as np
    print("✓ numpy importado correctamente")
    print("Versión de numpy:", np.__version__)
except Exception as e:
    print("✗ Error importando numpy:", str(e))