from cx_Freeze import setup, Executable
import os

import sys

sys.path.insert(0, os.path.abspath("app"))

# Specify the path to the .onnx file
onnx_file = os.path.join('app', 'model', 'ShapeDetection.onnx')

# Modify setup to include the .onnx file
setup(
    name="Camvas",
    version="0.1",
    description="Camvas",
    options={
        'build_exe': {
            'includes': ['VirtualCam', 'loading_screen', 'status_app', 'frame_processor'],
            'include_files': [
                (onnx_file, 'model\\ShapeDetection.onnx'),
                "Camvas.ico"
            ]
        }
    },
    executables=[Executable("app/main.py", base="Win32GUI", target_name="Camvas.exe", icon="Camvas.ico")],
)
