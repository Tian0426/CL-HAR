'''
Data Pre-processing on SHAR dataset.

'''

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import pickle as cp
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split
import scipy.io
from data_preprocess.base_loader import base_loader

def load_domain_data(domain_idx):
    """ to load all the data from the specific domain with index domain_idx
    :param domain_idx: index of a single domain
    :return: X and y data of the entire domain
    """
    data_dir = './data/UniMiB-SHAR/'
    saved_filename = 'shar_domain_' + domain_idx + '_wd.data' # "wd": with domain label

    if os.path.isfile(data_dir + saved_filename) == True:
        data = np.load(data_dir + saved_filename, allow_pickle=True)
        X = data[0][0]
        y = data[0][1]
        d = data[0][2]
    else:
        str_folder = './data/UniMiB-SHAR/data/'
        data_all = scipy.io.loadmat(str_folder + 'acc_data.mat')
        y_id_all = scipy.io.loadmat(str_folder + 'acc_labels.mat')
        y_id_all = y_id_all['acc_labels'] # (11771, 3)

        X_all = data_all['acc_data'] # data: (11771, 453)
        y_all = y_id_all[:, 0] - 1 # to map the labels to [0, 16]
        id_all = y_id_all[:, 1]

        print('\nProcessing domain {0} files...\n'.format(domain_idx))

        target_idx = np.where(id_all == int(domain_idx))
        X = X_all[target_idx]
        y = y_all[target_idx]

        domain_idx_map = {'1':0, '2':1, '3':2, '5':3}
        domain_idx_int = domain_idx_map[domain_idx]

        d = np.full(y.shape, domain_idx_int, dtype=int)

        print('\nProcessing domain {0} files | X: {1} y: {2} d:{3} \n'.format(domain_idx, X.shape, y.shape, d.shape))
        obj = [(X, y, d)]
        f = open(os.path.join(data_dir, saved_filename), 'wb')
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()
    return X, y, d

def load_domain_data_large(domain_idx):
    """ to load all the data from the specific domain
    :param domain_idx:
    :return: X and y data of the entire domain
    """
    data_dir = './data/UniMiB-SHAR/'
    saved_filename = 'shar_domain_' + domain_idx + '_wd.data' # with domain label

    if os.path.isfile(data_dir + saved_filename) == True:
        data = np.load(data_dir + saved_filename, allow_pickle=True)
        X = data[0][0]
        y = data[0][1]
        d = data[0][2]
    else:
        str_folder = './data/UniMiB-SHAR/data/'
        data_all = scipy.io.loadmat(str_folder + 'acc_data.mat')
        y_id_all = scipy.io.loadmat(str_folder + 'acc_labels.mat')
        y_id_all = y_id_all['acc_labels'] # (11771, 3)

        X_all = data_all['acc_data'] # data: (11771, 453)
        y_all = y_id_all[:, 0] - 1 # to map the labels to [0, 16]
        id_all = y_id_all[:, 1]

        print('\nProcessing domain {0} files...\n'.format(domain_idx))

        target_idx = np.where(id_all == int(domain_idx))
        X = X_all[target_idx]
        y = y_all[target_idx]
        # note: to change domain ID
        # source_domain_list = ['1', '2', '3', '5', '6', '9',
        #                       '11', '13', '14', '15', '16', '17', '19', '20',
        #                       '21', '22', '23', '24', '25', '29']
        domain_idx_map = {'1':0, '2':1, '3':2, '5':3, '6':4, '9':5,
                          '11':6, '13':7, '14':8, '15':9, '16':10, '17':11, '19':12, '20':13,
                          '21':14, '22':15, '23':16, '24':17, '25':18, '29':19}
        domain_idx_int = domain_idx_map[domain_idx]

        d = np.full(y.shape, domain_idx_int, dtype=int)

        print('\nProcessing domain {0} files | X: {1} y: {2} d:{3} \n'.format(domain_idx, X.shape, y.shape, d.shape))

        obj = [(X, y, d)]
        f = open(os.path.join(data_dir, saved_filename), 'wb')
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()
    return X, y, d


class data_loader_shar(base_loader):
    def __init__(self, samples, labels, domains):
        super(data_loader_shar, self).__init__(samples, labels, domains)


def prep_domains_shar_subject(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):

    # info: for SHAR dataset, the following domains have incomplete classes: 4,7,8,10
    source_domain_list = ['1', '2', '3', '5']
    source_domain_list.remove(args.target_domain)

    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        print('source_domain:', source_domain)
        x, y, d = load_domain_data(source_domain)

        x = x.reshape(-1, 151, 3)
        print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    unique_y, counts_y = np.unique(y_win_all, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()

    sample_weights = get_sample_weights(y_win_all, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights), replacement=True)

    data_set = data_loader_shar(x_win_all, y_win_all, d_win_all)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    print('source_loader batch: ', len(source_loader))
    source_loaders = [source_loader]

    # target domain data prep
    print('target_domain:', args.target_domain)
    x, y, d = load_domain_data(args.target_domain)

    x = x.reshape(-1, 151, 3)

    print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

    unique_y, counts_y = np.unique(y, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)

    data_set = data_loader_shar(x, y, d)
    # shuffle is forced to be False when sampler is available
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)
    print('target_loader batch: ', len(target_loader))
    return source_loaders, None, target_loader


def prep_domains_shar_subject_large(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    # note: for SHAR dataset with total 30 domains,
    # note: for SHAR dataset, the following domains have incomplete classes: 4, 7, 8, 10, 12, 18, 26, 27, 28, 30
    source_domain_list = ['1', '2', '3', '5', '6', '9',
                          '11', '13', '14', '15', '16', '17', '19', '20',
                          '21', '22', '23', '24', '25', '29']

    source_domain_list.remove(args.target_domain)

    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        print('source_domain:', source_domain)
        # todo: index change of domain ID is different from smaller indices; can be combined to a function when time is more available
        x, y, d = load_domain_data_large(source_domain)

        x = x.reshape(-1, 151, 3)
        print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    unique_y, counts_y = np.unique(y_win_all, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()

    sample_weights = get_sample_weights(y_win_all, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    data_set = data_loader_shar(x_win_all, y_win_all, d_win_all)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    print('source_loader batch: ', len(source_loader))
    source_loaders = [source_loader]

    # target domain data prep
    print('target_domain:', args.target_domain)
    x, y, d = load_domain_data_large(args.target_domain)

    x = x.reshape(-1, 151, 3)

    print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

    data_set = data_loader_shar(x, y, d)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)
    print('target_loader batch: ', len(target_loader))
    return source_loaders, None, target_loader

def prep_domains_shar_random(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    # note: for SHAR dataset, the following domains have incomplete classes: 4, 7, 8, 10, 12, 18, 26, 27, 28, 30

    source_domain_list = ['1', '2', '3', '5', '6', '9',
                          '11', '13', '14', '15', '16', '17', '19', '20',
                          '21', '22', '23', '24', '25', '29']
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    n_train, n_test, split_ratio = [], 0, 0.0

    for source_domain in source_domain_list:
        x_win, y_win, d_win = load_domain_data_large(source_domain)

        x_win = x_win.reshape(-1, 151, 3)

        x_win_all = np.concatenate((x_win_all, x_win), axis=0) if x_win_all.size else x_win
        y_win_all = np.concatenate((y_win_all, y_win), axis=0) if y_win_all.size else y_win
        d_win_all = np.concatenate((d_win_all, d_win), axis=0) if d_win_all.size else d_win

        n_train.append(x_win.shape[0])

    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_split(x_win_all, y_win_all, d_win_all,
                                                              split_ratio=args.split_ratio)

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()
    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights), replacement=True)

    train_set_r = data_loader_shar(x_win_train, y_win_train, d_win_train)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    val_set_r = data_loader_shar(x_win_val, y_win_val, d_win_val)
    val_loader_r = DataLoader(val_set_r, batch_size=args.batch_size, shuffle=False)
    test_set_r = data_loader_shar(x_win_test, y_win_test, d_win_test)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size, shuffle=False)

    return [train_loader_r], val_loader_r, test_loader_r

def prep_shar(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    if args.cases == 'subject':
        return prep_domains_shar_subject(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'subject_large':
        return prep_domains_shar_subject_large(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'random':
        return prep_domains_shar_random(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == '':
        pass
    else:
        return 'Error!\n'

