import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, vrange
import pickle

# ! X shape: (B, T, N, C)

def split_train_val_test( x, y):
    # print org data("x.shape ,y.shape = ",x.shape ,y.shape)
    train_rate = 0.6
    eval_rate = 0.2


    test_rate = 1 - train_rate - eval_rate
    num_samples = x.shape[0]
    num_test = round(num_samples * test_rate)
    num_train = round(num_samples * train_rate)
    num_val = num_samples - num_test - num_train
    part_train_rate = 1
    x_train, y_train = x[int(num_train * (1 - part_train_rate)):num_train], y[int(num_train * (
                1 - part_train_rate)):num_train]
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]
    print(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}")
    print(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}")
    print(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}")

    # if self.rank == 0 and self.cache_dataset:
    #     ensure_dir(self.cache_file_folder)
    #     np.savez_compressed(
    #         self.cache_file_name,
    #         x_train=x_train,
    #         y_train=y_train,
    #         x_test=x_test,
    #         y_test=y_test,
    #         x_val=x_val,
    #         y_val=y_val,
    #     )
    #     self._logger.info('Saved at ' + self.cache_file_name)
    return x_train, y_train, x_val, y_val, x_test, y_test

def generate_input_data(data):
    num_samples = data.shape[0]
    x_offsets = np.sort(np.concatenate((np.arange(-12 + 1, 1, 1),)))
    y_offsets = np.sort(np.arange(1, 12 + 1, 1))

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    print("x.shape = ", x.shape)
    print("y.shape = ", y.shape)
    return x, y

def get_dataloaders_from_index_data(
    data_dir, tod=False, dow=False, dom=False,hhod=False, batch_size=64, log=None
):
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)
    print("Data shape:", data.shape)

    features = [0]
    if tod:
        features.append(1)
    if dow:
        features.append(2)
    features.append(3)
    # if hhod:
    #     features.append(3)
    # if dom:
    #     features.append(3)
    data = data[..., features]  # 装载feature  source[..., 3]
    print("Data shape:", data.shape)
    index = np.load(os.path.join(data_dir, "index.npz"))

    train_index = index["train"]  # (num_samples, 3)
    val_index = index["val"]
    test_index = index["test"]

    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    x_train = data[x_train_index]
    y_train = data[y_train_index][..., :1]  # 装载 1
    x_val = data[x_val_index]
    y_val = data[y_val_index][..., :1]
    x_test = data[x_test_index]
    y_test = data[y_test_index][..., :1]


    # x_list, y_list = [], []
    # x, y = generate_input_data(data)
    # x_list.append(x)
    # y_list.append(y)
    # x = np.concatenate(x_list)
    # y = np.concatenate(y_list)
    # x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(x, y)
    # y_train = y_train[..., :1]
    # y_val = y_val[..., :1]
    # y_test = y_test[..., :1]


    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])



    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    trainset_loader = torch.utils.data.DataLoader(    #按照 batch_size进行划分，加载数据，以便处理
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler,
