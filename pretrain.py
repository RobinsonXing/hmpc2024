import torch

from dataset import HuMobDatasetPreTrain
from model import LPBERT

path_arr = [
    './dataset/cityA_groundtruthdata.csv.gz',
    './dataset/cityB_challengedata.csv.gz',
    './dataset/cityC_challengedata.csv.gz',
    './dataset/cityD_challengedata.csv.gz',
]
cities = ['A', 'B', 'C', 'D']



