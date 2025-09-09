import torch

from helper import DummyModel
from helper import load_image, get_generation
from IPython.display import display
from copy import deepcopy
from transformers import BlipForConditionalGeneration
from transformers import BlipProcessor


## ------------------------------------------------------##
model = DummyModel()

## ------------------------------------------------------##
model

## ------------------------------------------------------##
def print_param_dtype(model):
    for name, param in model.named_parameters():
        print(f"{name} is loaded in {param.dtype}")

## ------------------------------------------------------##
print_param_dtype(model)

## ------------------------------------------------------##
model_fp16 = DummyModel().half()

## ------------------------------------------------------##
print_param_dtype(model_fp16)

## ------------------------------------------------------##
model_fp16

## ------------------------------------------------------##
dummy_input = torch.LongTensor([[1, 0], [0, 1]])

## ------------------------------------------------------##
logits_fp32 = model(dummy_input)

## ------------------------------------------------------##
logits_fp32

## ------------------------------------------------------##
try:
    logits_fp16 = model_fp16(dummy_input)

except Exception as error:
    print("\033[91m", type(error).__name__, ": ", error, "\033[0m")

## ------------------------------------------------------##
model_bf16 = deepcopy(model)

## ------------------------------------------------------##
model_bf16 = model_bf16.to(torch.bfloat16)

## ------------------------------------------------------##
print_param_dtype(model_bf16)

## ------------------------------------------------------##
logits_bf16 = model_bf16(dummy_input)

## ------------------------------------------------------##
mean_diff = torch.abs(logits_bf16 - logits_fp32).mean().item()
max_diff = torch.abs(logits_bf16 - logits_fp32).max().item()

print(f"Mean diff: {mean_diff} | Max diff: {max_diff}")

## ------------------------------------------------------##
model_name = "Salesforce/blip-image-captioning-base"

## ------------------------------------------------------##
model = BlipForConditionalGeneration.from_pretrained(model_name)

## ------------------------------------------------------##
# print_param_dtype(model)

## ------------------------------------------------------##
fp32_mem_footprint = model.get_memory_footprint()

## ------------------------------------------------------##
print("Footprint of the fp32 model in bytes: ",
      fp32_mem_footprint)
print("Footprint of the fp32 model in MBs: ",
      fp32_mem_footprint / 1e+6)

## ------------------------------------------------------##
model_bf16 = BlipForConditionalGeneration.from_pretrained(
                                                        model_name,
                                                        torch_dtype = torch.bfloat16
                                                        )

## ------------------------------------------------------##
bf16_mem_footprint = model_bf16.get_memory_footprint()

## ------------------------------------------------------##
relative_diff = bf16_mem_footprint / fp32_mem_footprint

print("Footprint of the bf16 model in MBs: ",
      bf16_mem_footprint / 1e+6)
print(f"Relative diff: {relative_diff}")

## ------------------------------------------------------##
processor = BlipProcessor.from_pretrained(model_name)

## ------------------------------------------------------##
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'

image = load_image(img_url)
display(image.resize((500, 350)))

## ------------------------------------------------------##
results_fp32 = get_generation(model,
                              processor,
                              image,
                              torch.float32)

## ------------------------------------------------------##
print("fp32 Model Results:\n", results_fp32)

## ------------------------------------------------------##
results_bf16 = get_generation(model_bf16,
                              processor,
                              image,
                              torch.bfloat16)

## ------------------------------------------------------##
print("bf16 Model Results:\n", results_bf16)

## ------------------------------------------------------##
desired_dtype = torch.bfloat16
torch.set_default_dtype(desired_dtype)

## ------------------------------------------------------##
dummy_model_bf16 = DummyModel()

## ------------------------------------------------------##
print_param_dtype(dummy_model_bf16)

## ------------------------------------------------------##
torch.set_default_dtype(torch.float32)

## ------------------------------------------------------##
print_param_dtype(dummy_model_bf16)
