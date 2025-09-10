model_name = "google/flan-t5-small"

## ------------------------------------------------------##
import sentencepiece as spm
import torch

from quanto import quantize, freeze
from transformers import T5Tokenizer, T5ForConditionalGeneration
from helper import compute_module_sizes


tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

## ------------------------------------------------------##
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

## ------------------------------------------------------##
input_text = "Hello, my name is "
input_ids = tokenizer(input_text, return_tensors = "pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))

## ------------------------------------------------------##
module_sizes = compute_module_sizes(model)
print(f"The model size is {module_sizes[''] * 1e-9} GB")

## ------------------------------------------------------##
quantize(model, weights = torch.int8, activations = None)

## ------------------------------------------------------##
print(model)

## ------------------------------------------------------##
freeze(model)

## ------------------------------------------------------##
module_sizes = compute_module_sizes(model)
print(f"The model size is {module_sizes[''] * 1e-9} GB")

## ------------------------------------------------------##
input_text = "Hello, my name is "
input_ids = tokenizer(input_text, return_tensors = "pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
