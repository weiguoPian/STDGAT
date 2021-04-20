import numpy as np
from torch.utils.data import Dataset, DataLoader

class LoaderSTDGAT(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.data = np.load(self.args.data_path_demand_node, allow_pickle=True)
        self.od_feature = np.load(self.args.od_feature, allow_pickle=True)
        print(self.od_feature.shape)

        self.seq_len = args.seq_len

        if self.mode == 'train':
            # 5,6,7,8  (2952, 121)
            self.data = self.data[:-(30+31)*24]
            self.od_feature = self.od_feature[:-(30+31)*24]
        elif self.mode == 'val':
            # 9
            self.data = self.data[-(30+31)*24:-31*24]
            self.od_feature = self.od_feature[-(30+31)*24:-31*24]
        elif self.mode == 'test':
            # 10
            self.data = self.data[-(31*24+self.seq_len):]
            self.od_feature = self.od_feature[-(31*24+self.seq_len):]
        else:
            raise Exception('mode must be \'train\', \'val\' or \'test\'')

    def __getitem__(self, index):
        sample = []

        od_feature_list = []

        for i in range(self.seq_len+1):
            sample.append(self.data[index + i])
            od_feature_list.append(self.od_feature[index + i])
        return sample, od_feature_list[:-1]

    def __len__(self):
        return (len(self.data) - self.seq_len)