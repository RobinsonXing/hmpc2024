import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from dataset import *
from model import *

def find_max_batch_size(model, dataset, device='cuda', max_batch_size=256):

    batch_size = 1
    while batch_size <= max_batch_size:
        try:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
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

# 将一个批次的数据样本整合成一个张量
def collate_fn(batch):
    d = [item['d'] for item in batch]
    t = [item['t'] for item in batch]
    input_x = [item['input_x'] for item in batch]
    input_y = [item['input_y'] for item in batch]
    time_delta = [item['time_delta'] for item in batch]
    city = [item['city'] for item in batch]
    label_x = [item['label_x'] for item in batch]
    label_y = [item['label_y'] for item in batch]
    len_tensor = torch.tensor([item['len'] for item in batch])

    # 将样本填充至相同长度，填充值均为0
    d_padded = pad_sequence(d, batch_first=True, padding_value=0)
    t_padded = pad_sequence(t, batch_first=True, padding_value=0)
    input_x_padded = pad_sequence(input_x, batch_first=True, padding_value=0)
    input_y_padded = pad_sequence(input_y, batch_first=True, padding_value=0)
    time_delta_padded = pad_sequence(time_delta, batch_first=True, padding_value=0)
    city_padded = pad_sequence(city, batch_first=True, padding_value=0)
    label_x_padded = pad_sequence(label_x, batch_first=True, padding_value=0)
    label_y_padded = pad_sequence(label_y, batch_first=True, padding_value=0)

    # 返回字典，包含填充后的张量
    return {
        'd': d_padded,
        't': t_padded,
        'input_x': input_x_padded,
        'input_y': input_y_padded,
        'time_delta': time_delta_padded,
        'city': city_padded,
        'label_x': label_x_padded,
        'label_y': label_y_padded,
        'len': len_tensor
    }

# Example usage
if __name__ == "__main__":
    
    path_arr = [
    './dataset/cityA_groundtruthdata.csv.gz',
    './dataset/cityB_challengedata.csv.gz',
    './dataset/cityC_challengedata.csv.gz',
    './dataset/cityD_challengedata.csv.gz'
]
    # Create a dummy dataset
    dataset = TrainSet(path_arr)

    # Define the model
    model = LPBERT(4 , 8, 128).cuda()  # Move model to GPU

    # Find the maximum batch size
    max_batch_size = find_max_batch_size(model, dataset)
    print(f"Maximum batch size that fits in GPU memory: {max_batch_size}")