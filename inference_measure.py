import torch
from models import *

# Ensure CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

# Load the model
model = Generator()
model = model.cuda()
state_dict = torch.load('/home/op/xinhang/synandasyn/fl_wgan_mnist_topology/generator_epoch_0.pth')
model.load_state_dict(state_dict)

# Generate random input data
input_data = torch.randn(1, 128) 
input_data = input_data.cuda()

# Measure GPU memory usage before inference
memory_allocated_before = torch.cuda.memory_allocated()
memory_reserved_before = torch.cuda.memory_reserved()
print(f'GPU memory allocated before: {memory_allocated_before / 1024 ** 2} MB')
print(f'GPU memory reserved before: {memory_reserved_before / 1024 ** 2} MB')

for i in range(10000):
    # Perform inference
    with torch.no_grad():
        output = model(input_data)

# Measure GPU memory usage after inference
memory_allocated_after = torch.cuda.memory_allocated()
memory_reserved_after = torch.cuda.memory_reserved()
print(f'GPU memory allocated after: {memory_allocated_after / 1024 ** 2} MB')
print(f'GPU memory reserved after: {memory_reserved_after / 1024 ** 2} MB')
