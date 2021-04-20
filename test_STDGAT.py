from loader.data_loader_GAT import LoaderSTDGAT
from Glob.glob import p_parse
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from utils.lossFunction import Loss

from model.STDGAT import STDGAT, GAT

def testNetSTDGAT(args, model_path='./save/snap_STGAT/', best_snap=0):
    test_dataset = LoaderSTDGAT(args, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False, shuffle=False)
    
    snapList = os.listdir(os.path.join(model_path))
    snapList.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))

    best_model = snapList[best_snap]
    model = STDGAT(args)

    if args.cuda:
        model = model.cuda()
    
    model.load_state_dict(torch.load(os.path.join(
        model_path, best_model), map_location='cpu')['state_dict'])
    model.eval()
    loss_func = torch.nn.MSELoss()

    val_res_list = []
    val_label_list = []

    with torch.no_grad():
        for _, pack in enumerate(test_loader):
            samples = pack[0]
            od_features = pack[1]

            samples = list(map(lambda item: item.numpy(), samples))
            seqs_sample = torch.Tensor(samples[:-1])
            labels = torch.Tensor(samples[-1])

            od_features = list(map(lambda item: item.numpy(), od_features))
            od_features = torch.Tensor(od_features)

            if args.cuda:
                seqs_sample = seqs_sample.cuda()
                od_features = od_features.cuda()
                labels = labels.cuda()
            seqs_sample = torch.transpose(seqs_sample, 0, 1)
            out = model.forward(x=seqs_sample, od_features=od_features)
            for i in out.detach().cpu().numpy():
                val_res_list.append(i)
            for i in labels.detach().cpu().numpy():
                val_label_list.append(i)
        val_res_list = torch.Tensor(val_res_list)
        val_label_list = torch.Tensor(val_label_list)
        RMSE_loss = (loss_func(val_res_list, val_label_list).item())**0.5
    MAPE_loss = MAPE(y_true=val_label_list.detach().cpu().numpy(), y_pred=val_res_list.detach().cpu().numpy())
    MAE_loss = MAE(y_true=val_label_list.detach().cpu().numpy(), y_pred=val_res_list.detach().cpu().numpy())
    print('RMSE: {}   MAPE: {}   MAE: {}'.format(RMSE_loss, MAPE_loss, MAE_loss))


def MAPE(y_true, y_pred):
    idx = (y_true>20).nonzero()
    return np.mean(np.abs(y_true[idx] - y_pred[idx]) / y_true[idx])

def MAE(y_true, y_pred):
    return np.mean(abs(y_pred-y_true))

if __name__ == "__main__":
    args = p_parse()
    testNetSTDGAT(args, best_snap=180)

    