from cx_Freeze import setup, Executable
import os

import sys

sys.path.insert(0, os.path.abspath("app"))

# Specify the path to the .onnx file
onnx_file = os.path.join('model', 'ShapeDetection.onnx')
hand_model_path = os.path.join('model', 'hand_landmarker.task')
icon_path = os.path.join('assets', 'Camvas.ico')
main_file = os.path.join('app', 'main.py')

# Modify setup to include the .onnx file
setup(
    name="Camvas",
    version="0.1",
    description="Camvas",
    options={
        'build_exe': {
            'includes': ['VirtualCam', 'loading_screen', 'status_app', 'frame_processor'],
            'include_files': [
                (onnx_file, onnx_file),
                (hand_model_path, hand_model_path),
                (icon_path, os.path.basename(icon_path)),
            ]
        }
    },
    executables=[Executable(main_file, base=None, target_name="Camvas.exe", icon=icon_path)],
)
