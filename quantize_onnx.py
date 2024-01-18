import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Load your existing ONNX model
model_path = './models/encoder_L12.onnx'  # Replace with your model's file path
model = onnx.load(model_path)

# Specify the output path for the quantized model
quantized_model_path = './models/encoder_L12_float16.onnx'  # Replace with desired output path

# Quantize the model from float32 to float16
quantized_model = quantize_dynamic(model_path,
                                   quantized_model_path,
                                   weight_type=QuantType.QUInt8)

print(f"Quantized model saved to {quantized_model_path}")
