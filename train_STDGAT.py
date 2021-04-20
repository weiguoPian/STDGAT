from loader.data_loader_GAT import LoaderSTDGAT
from Glob.glob import p_parse
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from utils.lossFunction import Loss

from model.STDGAT import Attentionlayer, STDGAT, GAT

def trainNetSTDGAT(args):
    train_dataset = LoaderSTDGAT(args, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, shuffle=True)
    
    seed = 1337
    torch.manual_seed(seed)

    model = STDGAT(args)

    if args.cuda:
        torch.cuda.manual_seed(seed)
        model = model.cuda()
    
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
    train_loss_list = []

    for epoch in range(args.max_epoches):
        train_loss = 0.0
        step = 0
        for _, pack in enumerate(train_loader):
            step += 1
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

            loss_func = torch.nn.MSELoss()
            loss = loss_func(out, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= step
        # train_loss = train_loss**0.5
        train_loss_list.append(train_loss)
        print('epoch:{} train_loss:{}'.format(epoch, train_loss))
        snap_shot = {'state_dict': model.state_dict()}
        torch.save(snap_shot, './save/snap_STDGAT/snap_{}.pth.tar'.format(epoch))

        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss')
        plt.legend()
        plt.savefig('./save/img/STDGAT/train_loss.png')
        plt.close()

def valNetSTDGAT(args, model_path='./save/snap_STDGAT/'):
    val_dataset = LoaderSTDGAT(args, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False, shuffle=True)
    
    snapList = os.listdir(os.path.join(model_path))
    snapList.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))
    val_loss_list = []
    best_loss = None
    best_model = None

    for snap in snapList:
        model = STDGAT(args)

        if args.cuda:
            model = model.cuda()
        model.load_state_dict(torch.load(os.path.join(
            model_path, snap), map_location='cpu')['state_dict'])
        model.eval()
        loss_func = torch.nn.MSELoss()

        val_res_list = []
        val_label_list = []

        with torch.no_grad():
            for _, pack in enumerate(val_loader):
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
                # val_res_list.append(out.detach().cpu().numpy())
                # val_label_list.append(labels.detach().cpu().numpy())
            val_res_list = torch.Tensor(val_res_list)
            val_label_list = torch.Tensor(val_label_list)
            # loss = (loss_func(val_res_list, val_label_list).item())**0.5
            loss = (loss_func(val_res_list, val_label_list).item())
        val_loss_list.append(loss)
        print('model: {} val_loss: {}'.format(snap, loss))

        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_model = int(snap.split('.')[0].split('_')[1])
        
        plt.figure()
        plt.plot(range(len(val_loss_list)), val_loss_list, label='val_loss')
        plt.legend()
        plt.savefig('./save/img/STDGAT/val_loss.png')
        plt.close()
    
    print('best loss: {}  model: {}'.format(best_loss, best_model))


if __name__ == "__main__":
    args = p_parse()
    trainNetSTDGAT(args)
    valNetSTDGAT(args)