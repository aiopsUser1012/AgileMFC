import os
import random
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import public_function as pf
from codes.model import LinearClassifierv2
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, f1_score, recall_score
import shap
import warnings
warnings.filterwarnings('ignore')

class UnircaDataset():
    def __init__(self, dataset_path, labels_path, shuffle=False):
        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.data = []
        self.labels = []
        self.load()

        if shuffle:
            self.shuffle()

    def load(self):
        Xs = tensor(pf.load(self.dataset_path))
        ys = tensor(pf.load(self.labels_path))

        assert Xs.shape[0] == ys.shape[0]

        self.data = Xs.reshape(Xs.size(0), -1)
        self.labels = ys

    def shuffle(self):
        data_labels = [(x, y) for x, y in zip(self.data, self.labels)]
        random.shuffle(data_labels)
        self.data = [i[0] for i in data_labels]
        self.labels = [i[1] for i in data_labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx]), self.labels[idx])
    
class RawDataProcess():
    def __init__(self, config):
        self.config = config

    def process(self):
        run_table = pd.read_csv(self.config['run_table_path'], index_col=0)
        Xs = pf.load(self.config['Xs_path'])
        print(f'file: {self.config["Xs_path"]}')
        Xs = np.array(Xs)
        label_types = 'anomaly_type'
        label_dict = {label_types: None}
        anomaly_type_list = sorted(list(set(list(run_table[label_types]))))
        anomaly_type_dict = {}
        for index, value in enumerate(anomaly_type_list):
            anomaly_type_dict[value] = index
        # print(anomaly_type_dict)

        meta_labels = sorted(list(set(list(run_table[label_types]))))
        label_dict[label_types] = self.get_label(label_types, run_table, meta_labels)
        
        save_dir = self.config['save_dir']

        train_index = np.where(run_table['data_type'].values=='train')
        test_index = np.where(run_table['data_type'].values=='test')
        train_size = len(train_index[0])

        pf.save(os.path.join(save_dir, 'train_Xs.pkl'), Xs[: train_size])
        pf.save(os.path.join(save_dir, 'test_Xs.pkl'), Xs[train_size: ])

        for _, labels in label_dict.items():
            pf.save(os.path.join(save_dir, f'train_ys_{label_types}.pkl'), labels[train_index])
            pf.save(os.path.join(save_dir, f'test_ys_{label_types}.pkl'), labels[test_index])

    def get_label(self, label_type, run_table, meta_labels):
        labels_idx = {label: idx for label, idx in zip(meta_labels, range(len(meta_labels)))}
        labels = np.array(run_table[label_type].apply(lambda label_str: labels_idx[label_str]))
        return labels

class UnircaLab():
    def __init__(self, config):
        self.config = config
        instances = config['nodes'].split()
        self.ins_dict = dict(zip(instances, range(len(instances))))
        self.demos = pd.read_csv(self.config['run_table_path'], index_col=0)
    
    def collate(self, samples):
        data, labels = map(list, zip(*samples))
        batched_data = torch.stack(data)
        batched_labels = torch.tensor(labels)
        return batched_data, batched_labels
    
    def data_batch_shape_is_double(self, batch_data):
        return batch_data.shape[0] % 2 == 0

    def train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], collate_fn=self.collate)

        in_dim = dataset.data[0].shape[0]
        out_dim = self.config['N_class']
        hid_dim_1 = self.config['hidden_dim_1']
        hid_dim_2 = self.config['hidden_dim_2']

        model = LinearClassifierv2(in_dim, hid_dim_1, hid_dim_2, out_dim)
        # print(model)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)
        
        model.train()
        losses = []
        for _ in trange(self.config['epoch']):
            epoch_loss = 0
            epoch_cnt = 0
            for data, label in dataloader:
                optimizer.zero_grad()

                if not self.data_batch_shape_is_double(data):
                    continue

                output = model(data)
                
                loss = F.cross_entropy(output, label)
                
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_cnt += 1
                
            losses.append(epoch_loss / epoch_cnt)
            scheduler.step(epoch_loss / epoch_cnt)
            if len(losses) > self.config['win_size'] and \
                    abs(losses[-self.config['win_size']] - losses[-1]) < self.config['win_threshold']:
                break

        return model

    def test(self, model, dataset):
        model.eval()

        dataloader = DataLoader(dataset, batch_size=len(dataset) + 10, collate_fn=self.collate)

        for test_data, labels in dataloader:
            output = model(test_data)
            k = output.shape[-1]

            _, indices = torch.topk(output, k=k, dim=1, largest=True, sorted=True)  
            out_dir = os.path.join(self.config['save_dir'], 'preds')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            y_pred = indices.detach().numpy()
            y_true = labels.detach().numpy().reshape(-1, 1)

            pre = precision_score(y_pred[:, 0], y_true, average='weighted')
            rec = recall_score(y_pred[:, 0], y_true, average='weighted')
            f1 = f1_score(y_pred[:, 0], y_true, average='weighted')

            # print(conf_mat)
            print('Weighted precision:', pre)
            print('Weighted recall:', rec)
            print('Weighted f1-score:', f1)
        
    def run(self):
        save_dir = self.config['save_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        RawDataProcess(self.config).process()
        
        train_dataset = UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
                                                 os.path.join(save_dir, 'train_ys_anomaly_type.pkl'),
                                                 shuffle=True)
        model = self.train(train_dataset)

        test_dataset = UnircaDataset(os.path.join(save_dir, 'test_Xs.pkl'),
                                        os.path.join(save_dir, 'test_ys_anomaly_type.pkl'))
        self.test(model, test_dataset)
    
