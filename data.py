#This is taken from https://github.com/yuzhiyang123/FL-BNN 
#with a bit of modification
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import random
import torch
from collections import defaultdict
import json
from PIL import Image
import numpy as np


_DATASETS_MAIN_PATH ='./data/'


#os.makedirs(_DATASETS_MAIN_PATH, exist_ok=True)
_dataset_path = {
    'cifar10': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR10'),
    'cifar100': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR100'),
    'stl10': os.path.join(_DATASETS_MAIN_PATH, 'STL10'),
    'mnist': os.path.join(_DATASETS_MAIN_PATH, 'MNIST'),
    'imagenet': {
        'train': os.path.join(_DATASETS_MAIN_PATH, 'ImageNet/train'),
        'val': os.path.join(_DATASETS_MAIN_PATH, 'ImageNet/val')
    },
    'celeba': {
        'train': os.path.join(_DATASETS_MAIN_PATH, 'celeba/data/niid_train_'),
        'val': os.path.join(_DATASETS_MAIN_PATH, 'celeba/data/niid_test_')
    }
}


class MyDataset(data.Dataset):
    def __init__(self, base_dataset):
        super(MyDataset).__init__()
        self.base_dataset = base_dataset
        self.path = os.path.join(_DATASETS_MAIN_PATH, 'celeba/data/raw/img_align_celeba')
    
    def __len__(self):
        # return 500
        return len(self.base_dataset['y'])

    def __getitem__(self, i):
        img_name = self.base_dataset['x'][i]
        print('\n======\ninside get item: img_name: ',img_name)
        print('\n======\ninside get item: self.path: ',self.path,'\n===========\n')
        img = Image.open(os.path.join(self.path, img_name))
        img = img.resize((84,84)).convert('RGB')
        return (torch.Tensor(np.array(img).transpose(2, 0, 1)), self.base_dataset['y'][i])
        # return (torch.Tensor(self.base_dataset['x'][i]), self.base_dataset['y'][i])


class dataset_index(data.Dataset):
    def __init__(self, base_dataset, indexes=None):
        super(dataset_index, self).__init__() 
        self.base_dataset = base_dataset
        self.indexes = indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, i):
        index_ori = self.indexes[i]
        return (self.base_dataset[index_ori])


def distribute(dataset, distribution, numclients, numclasses):
    if distribution == 'noniid':
        nc = 3
        n = int(len(dataset) / (nc * numclients))
        distribution = []
        for _ in range(numclients):
            distribution.append([0] * numclasses)
        tmp = [i for i in range(numclasses)] * int(numclients * nc / numclasses)
        random.seed(0)
        random.shuffle(tmp)
        i = 0
        for _tmp in tmp:
            distribution[i // nc][_tmp] += n
            i += 1
        # print(distribution)
#the data is not evenly distributed among clients, either in terms of the total number of samples or the distribution of classes
    elif distribution == 'unbalanced':
        distribution = []
        for _ in range(20):
          #First 20 Clients: For the first 20 clients, it appends a list containing 120 samples for each of the 10 classes ([120]*10).
           #This means each of these clients will get 120 samples from every class.
            distribution.append([120]*10)
        for _ in range(40):
          #First 20 Clients: For the first 20 clients, it appends a list containing 120 samples for each of the 10 classes ([120]*10).
           #This means each of these clients will get 120 samples from every class.
            distribution.append([60]*10)
        for _ in range(40):
            distribution.append([30]*10)
    elif distribution is None or not (len(distribution) - 1 == numclients):
      # when either no specific distribution is provided (distribution is None) or the provided distribution does not match the number of clients (not (len(distribution) - 1 == numclients)).
      # In this case, the dataset is distributed in a balanced way across clients.
        n = int(len(dataset) / (numclasses * numclients))
        distribution = []
        for _ in range(numclients):
            distribution.append([n] * numclasses)
    #This list will be used to store the indices of the dataset that are allocated to each client.
    indexes_ = [[]]
    for _ in range(len(distribution) - 1):
        indexes_.append([])
    index = 0
    #For each data item in the dataset
    for _, y in dataset:
        for i in range(len(distribution)): # iterates over each client's distribution.
            if distribution[i][y] > 0: #checks if the current client i should still receive more samples of class y.
                distribution[i][y] -= 1 #decrements the count of remaining samples of class y for client i
                (indexes_[i]).append(index) #appends the current index from the dataset to the list corresponding to client i in indexes_. This assigns the data sample at this index to client i.
                break #The break statement ensures that once a data sample is assigned to a client, the inner loop exits, and the function does not attempt to assign the same sample to multiple clients.
        index += 1  #moving to the next data sample in the dataset
    return indexes_  #filled with indices of the dataset, divided among clients according to the specified distribution

def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True, distribution=None, numclients=4, dataset_path=None):
    train = (split == 'train')
    print('get_dataset--> trainFlag: ',train)
    if name == 'cifar10':
        print("CIFAR-10 dataset is being initialized.")
        dataset = datasets.CIFAR10(root=_dataset_path['cifar10'],
                                   train=train,
                                   transform=transform,
                                   target_transform=target_transform,
                                   download=download)
        numclasses = 10
    elif name == 'cifar100':
        dataset = datasets.CIFAR100(root=_dataset_path['cifar100'],
                                    train=train,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=download)
        numclasses = 100
    elif name == 'imagenet':
        path = _dataset_path[name][split]
        # Ensure the directory exists:
        os.makedirs(path, exist_ok=True)
        # dataset = datasets.ImageFolder(root=path,
        #                                transform=transform,
        #                                target_transform=target_transform)
        dataset = datasets.ImageNet(root=path,
                                       download=True,
                                       transform=transform,
                                       target_transform=target_transform)
        numclasses = 21841
    elif name == 'MNIST':
        dataset = datasets.MNIST(root=_dataset_path['mnist'],
                                 train=train,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]),
                                 download=download)
        numclasses = 10
    dataset_ = []
    if name == 'femnist' or name == 'celeba':
        path = _dataset_path[name][split]
        print('\n\npath for femnist:',path,'\n\n')
        # if dataset_path is not None:
        #     path = path+dataset_path
        #     print('\n\npath for femnist_2:',path,'\n\n')
        c, g, data = read_dir(path)
        print(len(c))
        if distribution == 'noniid':
            k = 0  #initializes a counter to keep track of the accumulated number of samples in the current batch of datasets
            d_ = [] #initializes an empty list to temporarily hold a batch of client datasets
            for u in c: #iterates over each client. The variable c contains the list of clients, likely obtained from reading the dataset directory.
                d = MyDataset(data[u]) #creates a dataset for the current client u. The MyDataset class is a custom dataset wrapper that handles specific preprocessing or dataset configurations.
                d_.append(d)
                k += d.__len__()
                if k > 500:
                    k = 0
                    dataset_.append(torch.utils.data.ConcatDataset(d_))
                    d_ = []
        elif distribution == 'unbalanced':
            k = 0
            d_ = []
            for u in c:
                d = MyDataset(data[u])
                d_.append(d)
                k += 1
                if k < 37 or (k % 2 == 0 and k < 107) or k % 4 ==2:
                    dataset_.append(torch.utils.data.ConcatDataset(d_))
                    d_ = []
        else:
            for u in c:
                dataset_.append(MyDataset(data[u]))
    else:
        #      #The distribute function returns a list of lists (indexes_), where each sublist contains the indices of the dataset that are allocated to each client.
        indexes_ = distribute(dataset=dataset, distribution=distribution, \
                              numclients=numclients, numclasses=numclasses)
        for indexes in indexes_:
            dataset_.append(dataset_index(dataset, indexes=indexes))
    print('Dataset loaded')
    return dataset_

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data
