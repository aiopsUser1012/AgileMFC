import random
import torch.nn as nn
import torch.backends
import public_function as pf
import numpy as np
import json
import os
import torch
from tqdm import tqdm
from codes.model import CustomDataset, MMoEv2
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

class ModalityRep:
    def __init__(self, config, cases):
        self.config = config
        self.cases = cases
        self.input_dim = config['vector_dim']
        self.nodes = config['nodes'].split()

        if config['dataset'] == 'gaia':
            self.anomaly_types = sorted(np.append('[normal]', cases['anomaly_type'].unique()))
        else:
            self.anomaly_types = sorted(np.append('normal', cases['anomaly_type'].unique()))

        self.anomaly_type_labels = dict(zip(self.anomaly_types, range(len(self.anomaly_types))))
        # print(self.anomaly_type_labels)
        self.node_labels = dict(zip(self.nodes, range(len(self.nodes))))
        self.zero_batch = False
        self.cnt = 0

        self.train_data_metric, self.train_data_log, self.train_data_trace = self.prepare_data()
        self.model = MMoEv2(self.config)

    def save_to_txt(self, data: dict, keys, save_path):
        fillna = False
        with open(save_path, 'w') as f:
            for case_id in keys:
                case_id = case_id if case_id in data.keys() else str(case_id)
                for node_info in data[case_id]: # node_info: (service_name, anomaly_type)
                    text = data[case_id][node_info]
                    if isinstance(text, str):
                        text = text.replace('(', '').replace(')', '')
                        if fillna and len(text) == 0:
                            text = 'None'
                        label_name = f'__label__{self.node_labels[node_info[0]]}{self.anomaly_type_labels[node_info[1]]}'
                        f.write(f'{text}\t{label_name}\n')
                    elif isinstance(text, list):
                        text = ' '.join(text)
                        if fillna and len(text) == 0:
                            text = 'None'
                        label_name = f'__label__{self.node_labels[node_info[0]]}{self.anomaly_type_labels[node_info[1]]}'
                        f.write(f'{text}\t{label_name}\n')
                    else:
                        raise Exception('type error')
        return
    
    def prepare_data(self):
        metric_text_path = self.config['text_path_metric']
        log_text_path = self.config['text_path_log']
        trace_text_path = self.config['text_path_trace']

        os.makedirs(self.config['txt_save_path'], exist_ok=True)

        temp_data_metric = pf.load(metric_text_path)
        temp_data_log = pf.load(log_text_path)
        temp_data_trace = pf.load(trace_text_path)

        train = self.cases[self.cases['data_type']=='train'].index
        test = self.cases[self.cases['data_type']=='test'].index

        self.save_to_txt(temp_data_metric, train, self.config['train_path_metric'])
        self.save_to_txt(temp_data_log, train, self.config['train_path_log'])
        self.save_to_txt(temp_data_trace, train, self.config['train_path_trace'])

        with open(self.config['train_path_metric'], 'r') as f:
            train_data_metric = f.read().splitlines()
        with open(self.config['train_path_log'], 'r') as f:
            train_data_log = f.read().splitlines()
        with open(self.config['train_path_trace'], 'r') as f:
            train_data_trace = f.read().splitlines()
        
        return train_data_metric, train_data_log, train_data_trace
    
    def trans_DA(self):
        train_data_metric = self.train_data_metric.copy()
        train_data_log = self.train_data_log.copy()
        train_data_trace = self.train_data_trace.copy()
        random.seed(0)

        word_index_metric = CustomDataset(self.config['train_path_metric']).word_to_index
        word_index_log = CustomDataset(self.config['train_path_log']).word_to_index
        word_index_trace = CustomDataset(self.config['train_path_trace']).word_to_index

        emb_dim = len(word_index_metric.keys()) + len(word_index_log.keys()) + len(word_index_trace.keys())

        for anomaly_type in tqdm(list(self.anomaly_types)):
            for node in self.nodes:
                # data aug for metric
                sample_count = len([
                    text for text in train_data_metric
                    if text.split('__label__')[-1] == f"{self.node_labels[node]}{self.anomaly_type_labels[anomaly_type]}"
                ])

                if sample_count == 0:
                    continue

                anomaly_texts = [
                    text for text in train_data_metric
                    if text.split('\t')[-1] == f'__label__{self.node_labels[node]}{self.anomaly_type_labels[anomaly_type]}'
                ]

                loop = 0
                while sample_count < self.config["sample_count"]:
                    loop += 1
                    if loop >= 10 * self.config['sample_count']:
                        break

                    chosen_text, label = anomaly_texts[random.randint(0, len(anomaly_texts) - 1)].split('\t')
                    chosen_text_splits = chosen_text.split()

                    if len(chosen_text_splits) < self.config["minCount"]:
                        continue
                    
                    words = list(word_index_metric.keys())

                    edit_event_ids = random.sample(range(len(chosen_text_splits)), self.config["edit_count"])
                    for event_id in edit_event_ids:
                        nearest_events = self.get_nearest_neighbors(chosen_text_splits[event_id], word_index_metric, self.model.embedding_layer, words, k=10)
                        nearest_event = nearest_events[0][-1]
                        chosen_text_splits[event_id] = nearest_event

                    train_data_metric.append(
                        ' '.join(chosen_text_splits) + f'\t__label__{self.node_labels[node]}{self.anomaly_type_labels[anomaly_type]}'
                    )
                    sample_count += 1

                # data aug for log
                sample_count = len([
                    text for text in train_data_log
                    if text.split('__label__')[-1] == f"{self.node_labels[node]}{self.anomaly_type_labels[anomaly_type]}"
                ])

                if sample_count == 0:
                    continue

                anomaly_texts = [
                    text for text in train_data_log
                    if text.split('\t')[-1] == f'__label__{self.node_labels[node]}{self.anomaly_type_labels[anomaly_type]}'
                ]

                loop = 0
                while sample_count < self.config["sample_count"]:
                    loop += 1
                    if loop >= 10 * self.config['sample_count']:
                        break
                    
                    chosen_text, label = anomaly_texts[random.randint(0, len(anomaly_texts) - 1)].split('\t')
                    chosen_text_splits = chosen_text.split()

                    if len(chosen_text_splits) < self.config["minCount"]:
                        continue
                    
                    words = list(word_index_log.keys())

                    edit_event_ids = random.sample(range(len(chosen_text_splits)), self.config["edit_count"])
                    for event_id in edit_event_ids:
                        nearest_events = self.get_nearest_neighbors(chosen_text_splits[event_id], word_index_log, self.model.embedding_layer, words, k=10)
                        nearest_event = nearest_events[0][-1]
                        chosen_text_splits[event_id] = nearest_event

                    train_data_log.append(
                        ' '.join(chosen_text_splits) + f'\t__label__{self.node_labels[node]}{self.anomaly_type_labels[anomaly_type]}'
                    )
                    sample_count += 1

                # data aug for trace
                sample_count = len([
                    text for text in train_data_trace
                    if text.split('__label__')[-1] == f"{self.node_labels[node]}{self.anomaly_type_labels[anomaly_type]}"
                ])

                if sample_count == 0:
                    continue

                anomaly_texts = [
                    text for text in train_data_trace
                    if text.split('\t')[-1] == f'__label__{self.node_labels[node]}{self.anomaly_type_labels[anomaly_type]}'
                ]

                loop = 0
                while sample_count < self.config["sample_count"]:
                    loop += 1
                    if loop >= 10 * self.config['sample_count']:
                        break
                    
                    chosen_text, label = anomaly_texts[random.randint(0, len(anomaly_texts) - 1)].split('\t')
                    chosen_text_splits = chosen_text.split()

                    if len(chosen_text_splits) < self.config["minCount"]:
                        continue
                    
                    words = list(word_index_trace.keys())

                    edit_event_ids = random.sample(range(len(chosen_text_splits)), self.config["edit_count"])
                    for event_id in edit_event_ids:
                        nearest_events = self.get_nearest_neighbors(chosen_text_splits[event_id], word_index_trace, self.model.embedding_layer, words, k=10)
                        nearest_event = nearest_events[0][-1]
                        chosen_text_splits[event_id] = nearest_event

                    train_data_trace.append(
                        ' '.join(chosen_text_splits) + f'\t__label__{self.node_labels[node]}{self.anomaly_type_labels[anomaly_type]}'
                    )
                    sample_count += 1

        with open(self.config['train_da_metric_path'], 'w') as f:
            for text in train_data_metric:
                f.write(text + '\n')
        with open(self.config['train_da_log_path'], 'w') as f:
            for text in train_data_log:
                f.write(text + '\n')
        with open(self.config['train_da_trace_path'], 'w') as f:
            for text in train_data_trace:
                f.write(text + '\n')

        words = list(word_index_metric.keys()) + list(word_index_log.keys()) + list(word_index_trace.keys())

        with open(self.config['word_list_path'], 'w') as f:
            json.dump(words, f)

    def custom_collate(self, batch):
        target_length = self.input_dim

        padded_data = []
        labels = []
        
        for item, label in batch:
            item = torch.tensor(item, dtype=torch.int64)

            if len(item) > target_length:
                item = item[:target_length]
            
            if len(item) < target_length:
                padding_length = target_length - len(item)
                padded_item = torch.cat([item, torch.zeros(padding_length, dtype=torch.int64)])
            else:
                padded_item = item
            
            padded_data.append(padded_item)
            labels.append(label)

        return torch.stack(padded_data), torch.tensor(labels, dtype=torch.int64)
    
    def batch_size_not_equal(self, metric, log, trace, batch_size):
        if (metric.shape[0] == log.shape[0] == trace.shape[0] == batch_size):
            return False 
        else:
            return True

    def fit(self, metric_train_dataset, log_train_dataset, trace_train_dataset, model_save_path):
        config = self.config
        # print(config)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        
        metric_train_dataloader = DataLoader(metric_train_dataset, batch_size=config['batch'], shuffle=True, collate_fn=self.custom_collate)
        log_train_dataloader = DataLoader(log_train_dataset, batch_size=config['batch'], shuffle=True, collate_fn=self.custom_collate)
        trace_train_dataloader = DataLoader(trace_train_dataset, batch_size=config['batch'], shuffle=True, collate_fn=self.custom_collate)
        
        model = MMoEv2(config).to(device)
        # print(model)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1, verbose=True)

        num_epochs = config['epoch']
        model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            cnt_loss = 0
            dataloader = zip(metric_train_dataloader, log_train_dataloader, trace_train_dataloader)
            for metric_batch, log_batch, trace_batch in tqdm(list(dataloader)):
                metric_inputs, label = metric_batch
                log_inputs, _ = log_batch
                trace_inputs, _ = trace_batch

                if self.batch_size_not_equal(metric_inputs, log_inputs, trace_inputs, config['batch']):
                    continue
                
                metric_inputs, log_inputs, trace_inputs = metric_inputs.to(device), log_inputs.to(device), trace_inputs.to(device)
                label = label.to(device)
                
                optimizer.zero_grad()
                outputs = model(metric_inputs, log_inputs, trace_inputs)

                loss = criterion(outputs, label)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                cnt_loss += 1

            average_loss = epoch_loss / cnt_loss
            lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch + 1}/{num_epochs}, lr: {lr}, Loss: {average_loss},")

            # os.makedirs(self.config['check_point_save_path'], exist_ok=True)
            # checkpoint_save_path = self.config['check_point_save_path'] + f'/checkpoint_{epoch + 1}.pth'
            # torch.save(model.state_dict(), checkpoint_save_path)
            # print(f'checkpoint_{epoch+1}.pth saved')
            
            scheduler.step(average_loss)

        # 保存模型
        if config['save_model']:
            torch.save(model.state_dict(), model_save_path)
            var: str = 'Model saved'
            print(f'{var:=^51}')

        return model
    
    def get_nearest_neighbors(self, word, word_to_index, embedding_layer, words, k):
        word_embedding = self.get_embbedding(word, word_to_index, embedding_layer)
        
        similarities = []
        for other_word in words:
            if other_word == word:
                continue
            
            other_embedding = self.get_embbedding(other_word, word_to_index, embedding_layer)

            if word_embedding.dim() == 1:
                word_embedding = word_embedding.unsqueeze(0)
            if other_embedding.dim() == 1:
                other_embedding = other_embedding.unsqueeze(0)
            
            cosine_similarity = F.cosine_similarity(word_embedding, other_embedding).item()
            similarities.append((cosine_similarity, other_word))
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        return similarities[:k]
    
    def get_embbedding(self, word, word_to_index, embedding_layer):
        if word not in word_to_index:
            raise ValueError(f"Word '{word}' not in vocabulary")
        
        word_index = word_to_index[word]
        return embedding_layer.weight[word_index]
    
    def event_emb_transformer(self, metric_data_path, log_data_path, trace_data_path, model_save_path, word_list_path):
        if self.config['save_model']:
            trained_model = self.fit(CustomDataset(metric_data_path, shuffle=True),
                                    CustomDataset(log_data_path, shuffle=True),
                                    CustomDataset(trace_data_path, shuffle=True),
                                    model_save_path)
        else:
            trained_model = self.model
            trained_model.load_state_dict(torch.load(model_save_path))
            trained_model.eval()

        with open(word_list_path, 'r') as f:
            words = {word: idx for idx, word in enumerate(json.load(f))}

        embedding_layer = trained_model.embedding_layer
        event_dict = dict()
        for word in words:
            word_embedding = self.get_embbedding(word, words, embedding_layer)
            event_dict[word] = word_embedding.cpu().detach().numpy()

        pf.save(self.config['event_save_path'], event_dict)
    
    def join_strings(self, *strings):
        return ' '.join(s for s in strings if s)

    def read_text(self, path1, path2, path3):
        text = []
        f1 = open(path1, 'r')
        f2 = open(path2, 'r')
        f3 = open(path3, 'r')
        line1 = f1.readline()
        line2 = f2.readline()
        line3 = f3.readline()

        line_text = self.join_strings(line1.split('\t')[0], line2.split('\t')[0], line3.split('\t')[0])
        text.append(line_text)
        while line1:
            line1 = f1.readline()
            line2 = f2.readline()
            line3 = f3.readline()
            line_text = self.join_strings(line1.split('\t')[0], line2.split('\t')[0], line3.split('\t')[0])
            text.append(line_text)
        f1.close()
        f2.close()
        f3.close()

        return text[:-1]

    def tfidf_word_embedding(self, weight, data_dict, texts, word_dict, service_num):
        length = len(data_dict[list(data_dict.keys())[0]])
        count = 0
        case_embedding = []
        sentence_embedding = []
        for text in texts:
            temp = np.array([0] * length, 'float32')
            if text != '':
                words = list(set(text.split(' ')))

                for word in words:
                    if word in word_dict:
                        temp = temp + weight[count][word_dict[word]] * np.array(data_dict[word])

            case_embedding.append(temp)
            if (count + 1) % service_num == 0:
                sentence_embedding.append(case_embedding)
                case_embedding = []
            count += 1
        return sentence_embedding

    def sentence_embedding(self, file_dict, train_path_metric, test_path_metric, train_path_log, test_path_log, train_path_trace, test_path_trace, save_path, service_num):
        data_dict = pf.load(file_dict)
        train_text = self.read_text(train_path_metric, train_path_log, train_path_trace)
        test_text = self.read_text(test_path_metric, test_path_log, test_path_trace)
        vectorizer = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\S\S+')
        transformer = TfidfTransformer()

        vec_train = vectorizer.fit_transform(train_text)
        tfidf_train = transformer.fit_transform(vec_train)

        vec_test = vectorizer.transform(test_text)
        tfidf_test = transformer.transform(vec_test)

        weight_train = tfidf_train.toarray()
        weight_test = tfidf_test.toarray()

        word = vectorizer.get_feature_names_out()
        word_dict = {word[i]: i for i in range(len(word))}

        train_embedding = self.tfidf_word_embedding(weight_train, data_dict, train_text, word_dict, service_num)
        test_embedding = self.tfidf_word_embedding(weight_test, data_dict, test_text, word_dict, service_num)

        train_embedding.extend(test_embedding)

        pf.save(save_path, train_embedding)

    def run(self):
        # self.trans_DA()
        self.event_emb_transformer(self.config['train_da_metric_path'],
                                          self.config['train_da_log_path'],
                                          self.config['train_da_trace_path'],
                                          self.config['model_save_path'],
                                          self.config['word_list_path'])
        self.sentence_embedding(self.config['event_save_path'],
                       self.config['train_path_metric'], self.config['test_path_metric'],
                       self.config['train_path_log'], self.config['test_path_log'],
                       self.config['train_path_trace'], self.config['test_path_trace'],
                       self.config['semb_save_path'], self.config['service_num'])

def run_trans(config, labels):
    print(f"experts: {config['num_experts']}, blocks: {config['num_layers']}, gate-weight: {config['gate_weight']}, aug-sample: {config['sample_count']}")
    lab2 = ModalityRep(config, labels)
    lab2.run()
