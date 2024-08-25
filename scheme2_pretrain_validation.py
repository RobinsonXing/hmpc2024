import os
import argparse
import json
import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from dataset import *
from model2 import *

path_arr = [
    './dataset/cityA_groundtruthdata.csv.gz',
    './dataset/cityB_challengedata.csv.gz',
    './dataset/cityC_challengedata.csv.gz',
    './dataset/cityD_challengedata.csv.gz'
]

def Validation(args):

    # 设置结果的存储路径
    result_path = 'validation/scheme2/pretrain'
    os.makedirs(result_path, exist_ok=True)

    # 加载验证集
    dataset_val = ValidationSet(path_arr[0])
    dataloader_val = DataLoader(dataset_val, batch_size=1, num_workers=args.num_workers)

    # 通过cuda:<device_id>指定使用的GPU
    device = torch.device(f'cuda:{args.cuda}')

    # 实例化LP-BERT模型加载至GPU上，并加载预训练模型的参数
    model = LPBERT(args.layers_num, args.heads_num, args.embed_size).to(device)
    model.load_state_dict(torch.load(args.pth_file, map_location=device))

    # 初始化
    result = dict()
    result['generated'] = []
    result['reference'] = []

    # 模型验证
    model.eval() # 评估模式
    with torch.no_grad():
        for data in tqdm(dataloader_val):

            # 将数据加载到GPU上
            data['d'] = data['d'].to(device)
            data['t'] = data['t'].to(device)
            data['input_x'] = data['input_x'].to(device)
            data['input_y'] = data['input_y'].to(device)
            data['time_delta'] = data['time_delta'].to(device)
            data['label_x'] = data['label_x'].to(device)
            data['label_y'] = data['label_y'].to(device)
            data['len'] = data['len'].to(device)

            # 获取推测，并将标签堆叠成张量
            output = model(data['d'], data['t'], data['input_x'], data['input_y'], data['time_delta'], data['len'])
            label = torch.stack((data['label_x'], data['label_y']), dim=-1)

            # 处理输出
            assert torch.all((data['input_x'] == 201) == (data['input_y'] == 201))  # 捡查x和y的掩码是否一致
            pred_mask = (data['input_x'] == 201)
            output = output[pred_mask]
            pred = []
            pre_x, pre_y = -1, -1
            for step in range(len(output)):
                if step > 0:
                    output[step][0][pre_x] *= 0.9
                    output[step][1][pre_y] *= 0.9

                pred.append(torch.argmax(output[step], dim=-1))
                pre_x, pre_y = pred[-1][0].item(), pred[-1][1].item()

            # 生成预测结果
            pred = torch.stack(pred)
            generated = torch.cat((data['d'][pred_mask].unsqueeze(-1)-1, data['t'][pred_mask].unsqueeze(-1)-1, pred+1), dim=-1).cpu().tolist()
            generated = [tuple(x) for x in generated]

            # 生成参考结果（标签）
            reference = torch.cat((data['d'][pred_mask].unsqueeze(-1)-1, data['t'][pred_mask].unsqueeze(-1)-1, label[pred_mask]+1), dim=-1).cpu().tolist()
            reference = [tuple(x) for x in reference]
            
            result['generated'].append(generated)
            result['reference'].append(reference)

    # 保存结果
    current_time = datetime.datetime.now()
    with open(os.path.join(result_path, f'{current_time.strftime("%Y_%m_%d_%H_%M_%S")}.json'), 'w') as file:
        json.dump(result, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_file', type=str, default='')     # 改为训练完成的模型的存储地址
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--layers_num', type=int, default=4)
    parser.add_argument('--heads_num', type=int, default=8)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()

    Validation(args)