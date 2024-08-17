import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import *
from model import *



def find_max_batch_size(device='cuda', max_batch_size=256):

    batch_size = 1
    while batch_size <= max_batch_size:
        try:
            dataloader = DataLoader(dataset, batch_size=batch_size)
            for batch in dataloader:
                inputs = batch[0].to(device)
                _ = model(inputs)  # Forward pass
            batch_size *= 2  # Double batch_size for the next iteration
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"Batch size {batch_size} is too large. Using batch size {batch_size // 2}.")
                return batch_size // 2
            else:
                raise e
    return batch_size

# Example usage
if __name__ == "__main__":

    # Create a dummy dataset
    data = torch.randn(10000, 1000)
    labels = torch.randint(0, 10, (10000,))
    dataset = TensorDataset(data, labels)

    # Define the model
    model = SimpleModel(input_size=1000, hidden_size=500, output_size=10).cuda()  # Move model to GPU

    # Find the maximum batch size
    max_batch_size = find_max_batch_size(model, dataset)
    print(f"Maximum batch size that fits in GPU memory: {max_batch_size}")