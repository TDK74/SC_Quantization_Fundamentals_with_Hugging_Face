import torch


## ------------------------------------------------------##
torch.iinfo(torch.uint8)

## ------------------------------------------------------##
torch.iinfo(torch.int8)

## ------------------------------------------------------##
torch.iinfo(torch.int64)

## ------------------------------------------------------##
torch.iinfo(torch.int32)

## ------------------------------------------------------##
torch.iinfo(torch.int16)

## ------------------------------------------------------##
value = 1 / 3

## ------------------------------------------------------##
format(value, '.60f')

## ------------------------------------------------------##
tensor_fp64 = torch.tensor(value, dtype = torch.float64)

## ------------------------------------------------------##
print(f"fp64 tensor: {format(tensor_fp64.item(), '.60f')}")

## ------------------------------------------------------##
tensor_fp32 = torch.tensor(value, dtype = torch.float32)
tensor_fp16 = torch.tensor(value, dtype = torch.float16)
tensor_bf16 = torch.tensor(value, dtype = torch.bfloat16)

## ------------------------------------------------------##
print(f"fp64 tensor: {format(tensor_fp64.item(), '.60f')}")
print(f"fp32 tensor: {format(tensor_fp32.item(), '.60f')}")
print(f"fp16 tensor: {format(tensor_fp16.item(), '.60f')}")
print(f"bf16 tensor: {format(tensor_bf16.item(), '.60f')}")

## ------------------------------------------------------##
torch.finfo(torch.bfloat16)

## ------------------------------------------------------##
torch.finfo(torch.float32)

## ------------------------------------------------------##
torch.finfo(torch.float16)

## ------------------------------------------------------##
torch.finfo(torch.float64)

## ------------------------------------------------------##
tensor_fp32 = torch.rand(1000, dtype = torch.float32)

## ------------------------------------------------------##
tensor_fp32[ : 5]

## ------------------------------------------------------##
tensor_fp32_to_bf16 = tensor_fp32.to(dtype = torch.bfloat16)

## ------------------------------------------------------##
tensor_fp32_to_bf16[:5]

## ------------------------------------------------------##
m_float32 = torch.dot(tensor_fp32, tensor_fp32)

## ------------------------------------------------------##
m_bfloat16 = torch.dot(tensor_fp32_to_bf16, tensor_fp32_to_bf16)

## ------------------------------------------------------##
m_bfloat16  # print(m_bfloat16)
