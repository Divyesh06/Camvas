from cx_Freeze import setup, Executable
import os
from _meta import NAME, VERSION, DESCRIPTION
import sys

sys.path.insert(0, os.path.abspath("app"))
# Specify the path to the .onnx file
onnx_file = os.path.join('ml', 'models', 'ShapeDetection.onnx')
icon_path = os.path.join('assets', 'Camvas.ico')
main_file = os.path.join('app', 'main.py')
hand_model = os.path.join('ml', 'models', 'hand_landmarker.task')
placeholder_bmp = os.path.join('assets', 'Camvas.bmp')

# Modify setup to include the .onnx file
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    options={
        'build_exe': {
            'includes': ['VirtualCam', 'loading_screen', 'status_app', 'frame_processor', 'camera_worker'],
            'excludes': ['onnxruntime.training', 'sympy', 'scipy', 'torch', 'tensorflow'],
            'include_files': [
                onnx_file,
                hand_model,
                icon_path,
                placeholder_bmp,
            ]
        }
    },
    executables=[Executable(main_file, base="gui", target_name=f"{NAME}.exe", icon=icon_path)],
)
