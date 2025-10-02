from optimum.onnxruntime import ORTModelForTokenClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer


model_id = "rubert-model"
ort_model = ORTModelForTokenClassification.from_pretrained(model_id, export=True)
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
quantizer = ORTQuantizer.from_pretrained(ort_model)


quantizer.quantize(
    quantization_config=qconfig,
    save_dir="rubert-model-int8"
)