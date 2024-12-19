#!/usr/bin/python2.7

import torch
import numpy as np
import random
from collections import defaultdict

class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, progress_path, sample_rate, dataset,
            feature_transpose=False):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.progress_path = progress_path
        self.sample_rate = sample_rate
        self.feature_transpose = feature_transpose
        self.dataset = dataset

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_progress = []
        for vid in batch:
            if self.dataset == 'egoprocel':
                features = np.load(self.features_path + vid + '.npy')
                progress_values = np.load(self.progress_path + vid + '.npy')
                file_ptr = open(self.gt_path + vid + '.txt', 'r')
            else:

                features = np.load(self.features_path + vid.split('.')[0] + '.npy')
                
                progress_values = np.load(self.progress_path + vid.split('.')[0] + '.npy')
                file_ptr = open(self.gt_path + vid, 'r')
            if self.feature_transpose:
                features = features.T
            
            content = file_ptr.read().split('\n')[:-1]
            np.set_printoptions(threshold=np.inf)
            if len(content) < np.shape(features)[1]:
                features = features[:, :len(content)]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            
            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])
            batch_progress.append(progress_values[:, ::self.sample_rate])
        
        length_of_sequences = list(map(len, batch_target))

        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        batch_progress_tensor = torch.zeros(len(batch_input), np.shape(batch_progress[0])[0], max(length_of_sequences), dtype=torch.float)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            
            batch_progress_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_progress[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, batch_progress_tensor, mask
