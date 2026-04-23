import tensorflow as tf
import tf2onnx

KERAS_PATH = "ml/models/ShapeDetection.keras"
TFLITE_PATH = "ml/models/ShapeDetection.tflite"
ONNX_PATH = "ml/models/ShapeDetection.onnx"

model = tf.keras.models.load_model(KERAS_PATH)
model.output_names = ['output']

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)
print(f"Saved TFLite model to {TFLITE_PATH}")

spec = (tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="input"),)
tf2onnx.convert.from_keras(model, input_signature=spec, output_path=ONNX_PATH)
print(f"Saved ONNX model to {ONNX_PATH}")
