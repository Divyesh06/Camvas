from cx_Freeze import setup, Executable
import os

import sys

sys.path.insert(0, os.path.abspath("app"))

# Specify the path to the .onnx file
onnx_file = os.path.join('ml', 'models', 'ShapeDetection.onnx')

icon_path = os.path.join('assets', 'Camvas.ico')
main_file = os.path.join('app', 'main.py')
hand_model = os.path.join('ml', 'models', 'hand_landmarker.task')

# Modify setup to include the .onnx file
setup(
    name="Camvas",
    version="0.1",
    description="Camvas",
    options={
        'build_exe': {
            'includes': ['VirtualCam', 'loading_screen', 'status_app', 'frame_processor'],
            'include_files': [
                ("models/ShapeDetection.onnx", onnx_file),
                ("models/hand_landmarker.task", hand_model),
                (icon_path, os.path.basename(icon_path)),
            ]
        }
    },
    executables=[Executable(main_file, base="gui", target_name="Camvas.exe", icon=icon_path)],
)
