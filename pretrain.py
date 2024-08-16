import os
import argparse
import logging
import random
import datetime
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

from dataset import *
from model import *

path_arr = [
    './dataset/cityA_groundtruthdata.csv.gz',
    './dataset/cityB_challengedata.csv.gz',
    './dataset/cityC_challengedata.csv.gz',
    './dataset/cityD_challengedata.csv.gz',
]
cities = ['A', 'B', 'C', 'D']

# 设置随机种子以确保结果的可重复性
def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 将一个批次的数据样本整合成一个张量
def collate_fn(batch):
    d = [item['d'] for item in batch]
    t = [item['t'] for item in batch]
    input_x = [item['input_x'] for item in batch]
    input_y = [item['input_y'] for item in batch]
    time_delta = [item['time_delta'] for item in batch]
    label_x = [item['label_x'] for item in batch]
    label_y = [item['label_y'] for item in batch]
    len_tensor = torch.tensor([item['len'] for item in batch])
    city = torch.tensor([item['city'] for item in batch])

    # 将样本填充至相同长度，填充值均为0
    d_padded = pad_sequence(d, batch_first=True, padding_value=0)
    t_padded = pad_sequence(t, batch_first=True, padding_value=0)
    input_x_padded = pad_sequence(input_x, batch_first=True, padding_value=0)
    input_y_padded = pad_sequence(input_y, batch_first=True, padding_value=0)
    time_delta_padded = pad_sequence(time_delta, batch_first=True, padding_value=0)
    label_x_padded = pad_sequence(label_x, batch_first=True, padding_value=0)
    label_y_padded = pad_sequence(label_y, batch_first=True, padding_value=0)
    city_padded = pad_sequence(city, batch_first=True, padding_value=0)

    # 返回字典，包含填充后的张量
    return {
        'd': d_padded,
        't': t_padded,
        'input_x': input_x_padded,
        'input_y': input_y_padded,
        'time_delta': time_delta_padded,
        'label_x': label_x_padded,
        'label_y': label_y_padded,
        'len': len_tensor
    }

# 预训练函数
def pretrain(args):

    # 设置日志文件名
    name = f'batchsize{args.batch_size}_epochs{args.epochs}_embedsize{args.embed_size}_layersnum{args.layers_num}_headsnum{args.heads_num}_cuda{args.cuda}_lr{args.lr}_seed{args.seed}'
    current_time = datetime.datetime.now()

    # 设置存储日志文件的路径
    log_path = os.path.join('log', 'pretrain', name)
    tensorboard_log_path = os.path.join('tb_log', 'pretrain', name)
    checkpoint_path = os.path.join('checkpoint', 'pretrain', name)

    # 创建路径
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(tensorboard_log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    # 设置日志记录，保存到指定的日志文件中
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(log_path, f'{current_time.strftime("%Y_%m_%d_%H_%M_%S")}.txt'),
                        filemode='w')
    #  TensorBoard日志写入器，用于记录训练过程中的标量值
    writer = SummaryWriter(tensorboard_log_path)

    # 加载预训练集
    dataset_pretrain = HuMobDatasetPreTrain(path_arr[0], cities[0])
    dataloader_pretrain = DataLoader(task1_dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)

    # 通过cuda:<device_id>指定使用的GPU
    device = torch.device(f'cuda:{args.cuda}')

    # 实例化LP-BERT模型，并加载至GPU上
    model = LPBERT(args.layers_num, args.heads_num, args.embed_size).to(device)

    # 指定Adam优化器、CosineAnnealingLR学习率调度器、交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch_id in range(args.epochs):
        for batch_id, batch in enumerate(tqdm(dataloader_pretrain)):

            # 按批次将数据加载至GPU中
            batch['d'] = batch['d'].to(device)
            batch['t'] = batch['t'].to(device)
            batch['input_x'] = batch['input_x'].to(device)
            batch['input_y'] = batch['input_y'].to(device)
            batch['time_delta'] = batch['time_delta'].to(device)
            batch['label_x'] = batch['label_x'].to(device)
            batch['label_y'] = batch['label_y'].to(device)
            batch['len'] = batch['len'].to(device)

            # 将数据输入模型中得到输出
            output = model(batch['d'], batch['t'], batch['input_x'], batch['input_y'], batch['time_delta'], batch['len'])

            # 将x和y堆叠成一个张量
            label = torch.stack((batch['label_x'], batch['label_y']), dim=-1)

            # 创建预测掩码，并将其扩展至与label相同的维度
            pred_mask = (batch['input_x'] == 201)
            pred_mask = torch.cat((pred_mask.unsqueeze(-1), pred_mask.unsqueeze(-1)), dim=-1)

            # 计算损失，反向传播计算梯度并更新模型参数，清除累积梯度
            loss = criterion(output[pred_mask], label[pred_mask])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            
            step = epoch_id * len(task1_dataloader_train) + batch_id
            writer.add_scalar('loss', loss.detach().item(), step)
        scheduler.step()

        logging.info(f'epoch: {epoch_id}, loss: {loss.detach().item()}')

    torch.save(model.state_dict(), os.path.join(checkpoint_path, f'{current_time.strftime("%Y_%m_%d_%H_%M_%S")}.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--layers_num', type=int, default=4)
    parser.add_argument('--heads_num', type=int, default=8)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_random_seed(args.seed)

    pretrain(args)