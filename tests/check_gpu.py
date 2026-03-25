import torch
import torch_directml

# Crea il device DirectML (il sostituto di CUDA per AMD)
device = torch_directml.device()

print(f"Device rilevato: {device}")
print(f"Nome della GPU: {torch_directml.device_name(0)}")

# Prova un piccolo calcolo per vedere se crasha
x = torch.tensor([1.0, 2.0]).to(device)
print(f"Calcolo riuscito sul device: {x * 2}")