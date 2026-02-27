from src.neural_networks import tiny_test_model
from src.bobs_calculator import compare_model
from src.utils_quantization import toggle_quantization, attach_weight_quantizers
from src.quantizer import DeadZoneLDZCompander
import torch
from fvcore.nn import FlopCountAnalysis 

QUANT_ARGS = {"fixed_bit_val": 8, "init_deadzone_logit": 3.0, "max_bits": 8, "learnable_deadzone": True, "learnable_bit": False}

example = torch.tensor([1.0])
model_pruned = tiny_test_model(remove_column=True)
model_non_pruned = tiny_test_model(remove_column=False)

models = [model_non_pruned,model_pruned]

for model in models:
    print(FlopCountAnalysis(model,example).by_module())
    toggle_quantization(model=model, enabled=True)
    attach_weight_quantizers(model=model,exclude_layers=[],quantizer=DeadZoneLDZCompander,quantizer_kwargs=QUANT_ARGS,enabled=True)
    print(" \n quantized model \n")
    result = compare_model(model)
    result.print()



