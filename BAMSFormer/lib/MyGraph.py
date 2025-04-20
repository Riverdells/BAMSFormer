import numpy as np
import pandas as pd
import torch
import yaml

import os
from fastdtw import fastdtw
from tqdm import tqdm

class MyGraph:
    def __init__(self, dataset, config_file='../configs/ext_data.yaml'):
        # 加载配置文件
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        self.config = config
        self.dataset = dataset

        print("self.dataset =", self.dataset)

        # 数据路径设置
        self.data_path = f'../ext_data/{self.dataset}/'

        # 根据数据集名称获取配置
        if self.dataset in self.config:
            dataset_config = self.config[self.dataset]
        else:
            raise ValueError(f"Dataset '{self.dataset}' not found in the configuration file.")

        # 初始化配置属性
        self.rel_file = dataset_config.get('rel_file', self.dataset)
        self.geo_file = dataset_config.get('geo_file', self.dataset)
        self.num_nodes = dataset_config.get('num_nodes', 0)
        self.set_weight_link_or_dist = dataset_config.get('set_weight_link_or_dist', 'dist')
        self.init_weight_inf_or_zero = dataset_config.get('init_weight_inf_or_zero', 'inf')
        self.weight_col = dataset_config.get('weight_col', '')
        self.bidir = dataset_config.get('bidir', False)
        self.calculate_weight_adj = dataset_config.get('calculate_weight_adj', False)
        self.type_short_path = dataset_config.get('type_short_path', "hop")
        self.far_mask_delta = dataset_config.get('far_mask_delta', 5)
        self.data_col = dataset_config.get('data_col', [])
        self.data_files = dataset_config.get('data_files', [self.dataset])
        self.dtw_delta = dataset_config.get('dtw_delta', 5)

        # 初始化其他属性
        self.sd_mx = None
        self.sh_mx = None
        self.adj_mx = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.far_mask = None
        self.geo_mask = None
        self.sem_mask = None
        self.dtw_matrix = None
        self.geo_masks = []
        self.points_per_day = 24 * 3600 // 300
        self.points_per_hour = 3600 // 300

    def compute_masks(self):
        # Load geo
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {idx: index for index, idx in enumerate(self.geo_ids)}
        print(f"Loaded file {self.geo_file}.geo, num_nodes={self.num_nodes}")

        # Load rel file and build adjacency matrix
        relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')
        if self.weight_col:
            self.distance_df = relfile[~relfile[self.weight_col].isna()][['origin_id', 'destination_id', self.weight_col]]
        else:
            if len(relfile.columns) != 5:
                raise ValueError("Don't know which column to be loaded! Please set `weight_col` parameter!")
            self.weight_col = relfile.columns[-1]
            self.distance_df = relfile[~relfile[self.weight_col].isna()][['origin_id', 'destination_id', self.weight_col]]

        self.adj_mx = np.inf * np.ones((self.num_nodes, self.num_nodes),
                                       dtype=np.float32) if self.init_weight_inf_or_zero.lower() == 'inf' and self.set_weight_link_or_dist.lower() != 'link' else np.zeros(
            (self.num_nodes, self.num_nodes), dtype=np.float32)

        for row in self.distance_df.values:
            if row[0] in self.geo_to_ind and row[1] in self.geo_to_ind:
                i, j = self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]
                self.adj_mx[i, j] = row[2] if self.set_weight_link_or_dist.lower() == 'dist' else 1
                if self.bidir:
                    self.adj_mx[j, i] = row[2] if self.set_weight_link_or_dist.lower() == 'dist' else 1

        print(f"Loaded file {self.rel_file}.rel, shape={self.adj_mx.shape}")

        if self.calculate_weight_adj:
            self._calculate_adjacency_matrix()
        print(f"Max adj_mx value = {self.adj_mx.max()}")
        self.sh_mx = self.adj_mx.copy()
        if self.type_short_path == 'hop':
            self.sh_mx[self.sh_mx > 0] = 1
            self.sh_mx[self.sh_mx == 0] = np.inf
            np.fill_diagonal(self.sh_mx, 0)

            for k in range(self.num_nodes):
                self.sh_mx = np.minimum(self.sh_mx, np.add.outer(self.sh_mx[:, k], self.sh_mx[k, :]))

            self.sh_mx[self.sh_mx > 511] = 511
            np.save(f'{self.dataset}.npy', self.sh_mx)

        # Geo mask computation
        if self.type_short_path == "dist":
            distances = self.sd_mx[~np.isinf(self.sd_mx)].flatten()
            std = distances.std()
            sd_mx = np.exp(-np.square(self.sd_mx / std))
            self.far_mask = torch.zeros(self.num_nodes, self.num_nodes, device=self.device) #dist看相似度，hop看跳数
            self.far_mask[sd_mx < self.far_mask_delta] = 1
            self.far_mask = self.far_mask.bool()
        else:
            sh_mx = torch.tensor(self.sh_mx.T, device=self.device)
            self.geo_mask = torch.ge(sh_mx, self.far_mask_delta).bool()
            for i in range(6):
                hop_distance = self.far_mask_delta + i
                geo_mask = torch.ge(sh_mx, hop_distance).bool()
                self.geo_masks.append(geo_mask)
        print(sh_mx)
        print(self.adj_mx)
        return self.geo_masks,self.adj_mx
