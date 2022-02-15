'''
Data Pre-processing on HHAR dataset.

'''

import numpy as np
import pandas as pd
import sys
import os
from scipy.interpolate import interp1d
import torch
import pickle5 as cp
from torch.utils.data import Dataset, DataLoader
from data_preprocess.data_preprocess_utils import get_sample_weights, opp_sliding_window, normalize, train_test_val_split

NUM_FEATURES = 6


class data_loader_hhar(Dataset):
    def __init__(self, samples, labels, domains):
        self.samples = samples
        self.labels = labels
        self.domains = domains

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return sample, target, domain

    def __len__(self):
        return len(self.samples)

def sep_user_device_gt():
    user_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    gt_list = ['bike', 'sit', 'stand', 'walk', 'stairsup', 'stairsdown', 'null']
    watch_device = ['gear_1', 'gear_2', 'lgwatch_1', 'lgwatch_2']
    phone_device = ['nexus4_1', 'nexus4_2', 's3_1', 's3_2', 's3mini_1', 's3mini_2', 'samsungold_1', 'samsungold_2']
    dataDir = 'data/HHAR/Activity recognition exp/'
    fileInList = os.listdir(dataDir)

    saveDir = 'data/HHAR/avtivity_data_separated/'
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
        
    for fileName in fileInList:
        print(fileName)
        if not '.csv' in fileName:
            continue
        if 'Watch' in fileName:
            device_list = watch_device
        else:
            device_list = phone_device
        df = pd.read_csv(dataDir + fileName, index_col=0)
        for user in user_list:
            for device in device_list:
                for gt in gt_list:
                    sep_df = df[(df['User']==user) & (df['Device']==device) & (df['gt']==gt)]
                    if sep_df.shape[0] == 0:
                        continue
                    cur_file_name = user+'_'+device+'_'+gt+'_'+fileName
                    print(cur_file_name)
                    sep_df.to_csv(saveDir+cur_file_name)

def combine_acc_gyr():
    dataDir = 'data/HHAR/avtivity_data_separated/'
    saveDir = 'data/HHAR/avtivity_data_acc_gyr/'
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    user_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    gt_list = ['bike', 'sit', 'stand', 'walk', 'stairsup', 'stairsdown'] # exclude 'null'
    watch_device = ['gear_1', 'gear_2', 'lgwatch_1', 'lgwatch_2']
    phone_device = ['nexus4_1', 'nexus4_2', 's3_1', 's3_2', 's3mini_1', 's3mini_2'] # exclude'samsungold_1', 'samsungold_2'
    device_list = watch_device + phone_device
    timeScale = 100000000.
    for user in user_list:
        for i, device in enumerate(device_list):
            if i <= 3:
                d = 'Watch'
            else: 
                d = 'Phones'
            for gt in gt_list:
                idx1, idx2 = 0, 0
                cur_file_name = user+'_'+device+'_'+gt+'_'+d+'_acc_gyr.csv'
                print(cur_file_name)
                if os.path.isfile(saveDir+cur_file_name):
                    continue
                acc_dir = dataDir+user+'_'+device+'_'+gt+'_'+d+'_accelerometer.csv'
                gyr_dir = dataDir+user+'_'+device+'_'+gt+'_'+d+'_gyroscope.csv'
                df_acc_gyr = pd.DataFrame(columns=['Time', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z'])
                if os.path.isfile(acc_dir) and os.path.isfile(gyr_dir):
                    df_acc = pd.read_csv(acc_dir, index_col=0)
                    df_gyr = pd.read_csv(gyr_dir, index_col=0)
                else: 
                    continue
                while idx1 < df_acc.shape[0] and idx2 < df_gyr.shape[0]:
                    curTime1 = df_acc.loc[df_acc.index[idx1], 'Creation_Time']
                    curTime2 = df_gyr.loc[df_gyr.index[idx2], 'Creation_Time']
                    if abs(curTime1 - curTime2) < 0.1*timeScale:
                        df_acc_gyr = df_acc_gyr.append({'Time': (curTime1 + curTime2) * 0.5, 
                                                        'acc_x': df_acc.loc[df_acc.index[idx1], 'x'], 
                                                        'acc_y': df_acc.loc[df_acc.index[idx1], 'y'], 
                                                        'acc_z': df_acc.loc[df_acc.index[idx1], 'z'],
                                                        'gyr_x': df_gyr.loc[df_gyr.index[idx2], 'x'],
                                                        'gyr_y': df_gyr.loc[df_gyr.index[idx2], 'y'], 
                                                        'gyr_z': df_gyr.loc[df_gyr.index[idx2], 'z']}, 
                                                        ignore_index=True)
                        idx1 += 1
                        idx2 += 1
                    elif curTime1 - curTime2 >= 0.1:
                        idx2 += 1
                    else:
                        idx1 += 1
                print(df_acc.shape[0], df_gyr.shape[0], df_acc_gyr.shape[0])
                df_acc_gyr.to_csv(saveDir+cur_file_name)

def interpolate():
    dataDir = 'data/HHAR/avtivity_data_acc_gyr/'
    saveDir = 'data/HHAR/avtivity_data_acc_gyr_interp/'
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    user_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    gt_list = ['bike', 'sit', 'stand', 'walk', 'stairsup', 'stairsdown'] # exclude 'null'
    watch_device = ['gear_1', 'gear_2', 'lgwatch_1', 'lgwatch_2']
    phone_device = ['nexus4_1', 'nexus4_2', 's3_1', 's3_2', 's3mini_1', 's3mini_2'] # exclude'samsungold_1', 'samsungold_2'
    device_list = watch_device + phone_device
    for user in user_list:
        for i, device in enumerate(device_list):
            if i <= 3:
                d = 'Watch'
            else: 
                d = 'Phones'
            for gt in gt_list:
                file_name = user+'_'+device+'_'+gt+'_'+d+'_acc_gyr.csv'
                print(file_name)
                if not os.path.isfile(dataDir+file_name):
                    continue
                df = pd.read_csv(dataDir+file_name, index_col=0)
                
                curTimeList = df[['Time']].to_numpy().squeeze()
                if curTimeList.shape[0] == 0:
                    continue
                n_samples = len(curTimeList)
                print(curTimeList[0], curTimeList[-1], n_samples)
                interval = int((curTimeList[-1] - curTimeList[0]) / n_samples)
                print(interval)
                resample_ratio = 1 # TODO
                endTime = curTimeList[0] + n_samples * interval
                print(endTime)
                InterpTime = np.linspace(curTimeList[0], endTime, n_samples*resample_ratio).squeeze() # TODO n_samples*1 decides downsampling
                print(InterpTime.ndim, InterpTime.shape)
                
                curAccList = df[['acc_x', 'acc_y', 'acc_x']].to_numpy()
                curGyrList = df[['gyr_x', 'gyr_y', 'gyr_z']].to_numpy()
                print(curAccList.shape)
                accInterp = interp1d(curTimeList, curAccList, axis=0)
                accInterpVal = accInterp(InterpTime)
                gyroInterp = interp1d(curTimeList, curGyrList, axis=0)
                gyroInterpVal = gyroInterp(InterpTime)
                df_interp = pd.DataFrame(columns=['Time', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z'])
                df_interp.loc[:, 'Time'] = InterpTime
                df_interp.loc[:, ['acc_x', 'acc_y', 'acc_z']] = accInterpVal
                df_interp.loc[:, ['gyr_x', 'gyr_y', 'gyr_z']] = gyroInterpVal
                
                cur_file_name = user+'_'+device+'_'+gt+'_'+d+'_acc_gyr_interp.csv'
                print(cur_file_name)
                df_interp.to_csv(saveDir + cur_file_name)

def downsampling(data_t, data_x, data_y, freq):
    """Recordings are downsamplied to 50Hz
    
    :param data_t: numpy integer array
        time array
    :param data_x: numpy integer array
        sensor recordings
    :param data_y: numpy integer array
        labels
    :return: numpy integer array
        Downsampled input
    """
    idx = np.arange(0, data_t.shape[0], int(freq/50))

    return data_t[idx], data_x[idx], data_y[idx]

def preprocess():
    print('preprocessing the data...')
    sep_user_device_gt()
    combine_acc_gyr()
    interpolate()
    print('preprocessing done!')

def split_train_test_subject(train_user, test_user, device, SLIDING_WINDOW_LEN=100, SLIDING_WINDOW_STEP=50):
    # todo besides user domain, consider device domain
    dataDir = './data/HHAR/avtivity_data_acc_gyr_interp/'
    if os.path.isdir(dataDir) == False:
        preprocess()
    user_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    gt_list = ['bike', 'sit', 'stand', 'walk', 'stairsup', 'stairsdown'] # exclude 'null'
    watch_device = ['gear_1', 'gear_2', 'lgwatch_1', 'lgwatch_2']
    phone_device = ['nexus4_1', 'nexus4_2', 's3_1', 's3_2', 's3mini_1', 's3mini_2'] # exclude'samsungold_1', 'samsungold_2'
    watch_freq_list = [100, 100, 200, 200]
    phone_freq_list = [200, 200, 150, 150, 100, 100]

    if device =='Watch':
        devices = watch_device
        freqs = watch_freq_list
    else:
        devices = phone_device
        freqs = phone_freq_list
    x_train, x_test, y_train, y_test, d_train, d_test = None, None, None, None, None, None
    for user in user_list:
        if user != test_user and user not in test_user and user != train_user and user not in train_user:
            continue
        print(user)
        for gt in gt_list:
            for d in devices:
                file_name = user+'_'+d+'_'+gt+'_'+device+'_acc_gyr_interp.csv'
                if not os.path.isfile(dataDir+file_name):
                    continue
                data = pd.read_csv(dataDir+file_name, index_col=0).to_numpy()
                gt_label = gt_list.index(gt)
                freq = freqs[devices.index(d)]
                _, data, label = downsampling(data[:, 0], data[:, 1:], np.full((data.shape[0]), gt_label), freq)
                # print(np.any(np.isnan(data)))
                if data.shape[0] < SLIDING_WINDOW_LEN:
                    continue
                if np.any(np.isnan(data)):
                    continue
                data_sw, label_sw = opp_sliding_window(data, label, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
                d_sw = np.full((data_sw.shape[0]), user)
                if user == test_user or user in test_user:
                    if x_test is None:
                        x_test = data_sw
                        y_test = label_sw
                        d_test = d_sw
                    else:
                        x_test = np.concatenate((x_test, data_sw), axis=0)
                        y_test = np.concatenate((y_test, label_sw), axis=None)
                        d_test = np.concatenate((d_test, d_sw), axis=None)
                if user == train_user or user in train_user:
                    if x_train is None:
                        x_train = data_sw
                        y_train = label_sw
                        d_train = d_sw
                    else:
                        x_train = np.concatenate((x_train, data_sw), axis=0)
                        y_train = np.concatenate((y_train, label_sw), axis=None)
                        d_train = np.concatenate((d_train, d_sw), axis=None)
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    print(x_train.shape, d_train.shape)
    return x_train, x_test, y_train, y_test, d_train, d_test

def split_train_test(device, args, SLIDING_WINDOW_LEN=100, SLIDING_WINDOW_STEP=50):
    # device: 'Watch' or 'Phones'
    dataDir = './data/HHAR/avtivity_data_acc_gyr_interp/'
    preprocess_Dir = 'data/HHAR/hhar_processed_'+device+'.data'
    if os.path.isfile(preprocess_Dir) == True:
        print('data is preprocessed in advance! Loading...')
        data = np.load(preprocess_Dir, allow_pickle=True)
        x_train = data[0][0]
        y_train = data[0][1]
        d_train = data[0][2]
        x_val = data[1][0]
        y_val = data[1][1]
        d_val = data[1][2]
        x_test = data[2][0]
        y_test = data[2][1]
        d_test = data[2][2]
    else:
        if os.path.isdir(dataDir) == False:
            preprocess()
        user_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        gt_list = ['bike', 'sit', 'stand', 'walk', 'stairsup', 'stairsdown'] # exclude 'null'
        watch_device = ['gear_1', 'gear_2', 'lgwatch_1', 'lgwatch_2']
        phone_device = ['nexus4_1', 'nexus4_2', 's3_1', 's3_2', 's3mini_1', 's3mini_2'] # exclude'samsungold_1', 'samsungold_2'
        watch_freq_list = [100, 100, 200, 200]
        phone_freq_list = [200, 200, 150, 150, 100, 100]

        if device =='Watch':
            devices = watch_device
            freqs = watch_freq_list
        else:
            devices = phone_device
            freqs = phone_freq_list
        x_sw, y_sw, d_sw = None, None, None
        for d in devices:
            for gt in gt_list:
                for user in user_list:
                    file_name = user+'_'+d+'_'+gt+'_'+device+'_acc_gyr_interp.csv'
                    print(file_name)
                    if not os.path.isfile(dataDir+file_name):
                        continue
                    data = pd.read_csv(dataDir+file_name, index_col=0).to_numpy()

                    gt_label = gt_list.index(gt)
                    freq = freqs[devices.index(d)]
                    _, data, label = downsampling(data[:, 0], data[:, 1:], np.full((data.shape[0]), gt_label), freq)
                    # print(np.any(np.isnan(data)))
                    if np.any(np.isnan(data)):
                        continue

                    if data.shape[0] < SLIDING_WINDOW_LEN:
                        continue
                    if x_sw is None:
                        x_sw, y_sw = opp_sliding_window(data, label, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
                        d_sw = np.full((x_sw.shape[0]), user)
                    else:
                        _x, _y = opp_sliding_window(data, label, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
                        x_sw = np.concatenate((x_sw, _x), axis=0)
                        y_sw = np.concatenate((y_sw, _y), axis=None)
                        d_sw = np.concatenate((d_sw, np.full((_x.shape[0]), user)), axis=None)

        x_train, x_val, x_test, \
        y_train, y_val, y_test, \
        d_train, d_val, d_test = train_test_val_split(x_sw, y_sw, d_sw, split_ratio=args.split_ratio)

        x_train = normalize(x_train)
        x_val = normalize(x_val)
        x_test = normalize(x_test)
        obj = [(x_train, y_train, d_train),(x_val, y_val, d_val), (x_test, y_test, d_test)]
        f = open(preprocess_Dir, 'wb')
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()
    return x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test

def prep_hhar(args, SLIDING_WINDOW_LEN=100, SLIDING_WINDOW_STEP=50, device='Phones', train_user=None, test_user=None):
    # device: 'Watch' or 'Phones'
    # test_user: only applicable when split_mode == 'subject
    if args.cases == 'random':
        x_train, x_val, x_test, y_train, y_val, y_test, d_train, d_val, d_test = split_train_test(device, args, SLIDING_WINDOW_LEN=SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP=SLIDING_WINDOW_STEP)
        assert x_train.shape[0] == y_train.shape[0] == d_train.shape[0]
    elif args.cases == 'subject':
        if test_user is None or train_user is None:
            raise ValueError('Please specify train/test user')
        x_train, x_test, y_train, y_test, d_train, d_test = split_train_test_subject(train_user, test_user, device, SLIDING_WINDOW_LEN=SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP=SLIDING_WINDOW_STEP)
    elif args.cases == 'cross_device':
        x_train, _, _, y_train, _, _, d_train, _, _ = split_train_test(device, args, SLIDING_WINDOW_LEN=SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP=SLIDING_WINDOW_STEP)
        test_device = ['Phones', 'Watch']
        test_device.remove(device)
        print('source device:', device, 'target device', test_device[0])
        x_test_1, x_test_2, x_val, y_test_1, y_test_2, y_val, d_test_1, d_test_2, d_val = split_train_test(test_device[0], args, SLIDING_WINDOW_LEN=SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP=SLIDING_WINDOW_STEP)
        x_test = np.concatenate((x_test_1, x_test_2), axis=0)
        y_test = np.concatenate((y_test_1, y_test_2), axis=None)
        d_test = np.concatenate((d_test_1, d_test_2), axis=None)
    elif args.cases == 'joint_device':
        x_train_1, x_val, x_test, y_train_1, y_val, y_test, d_train_1, d_val, d_test = split_train_test(device, args, SLIDING_WINDOW_LEN=SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP=SLIDING_WINDOW_STEP)
        test_device = ['Phones', 'Watch']
        test_device.remove(device)
        print('source device:', device, 'target device', test_device[0])
        x_train_2, _, _, y_train_2, _, _, d_train_2, _, _ = split_train_test(test_device[0], args, SLIDING_WINDOW_LEN=SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP=SLIDING_WINDOW_STEP)
        x_train = np.concatenate((x_train_1, x_train_2), axis=0)
        y_train = np.concatenate((y_train_1, y_train_2), axis=None)
        d_train = np.concatenate((d_train_1, d_train_2), axis=None)

    unique_ytrain, counts_ytrain = np.unique(y_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_ytrain, counts_ytrain)))

    weights = 100.0 / torch.Tensor(counts_ytrain)
    weights = weights.double()
    print('weights of sampler: ', weights)
    sample_weights = get_sample_weights(y_train, weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_set = data_loader_hhar(x_train, y_train, d_train)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    test_set = data_loader_hhar(x_test, y_test, d_test)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    print('train_loader batch: ', len(train_loader), 'test_loader batch: ', len(test_loader))

    if args.cases in ['random', 'cross_device', 'joint_device']:
        val_set = data_loader_hhar(x_val, y_val, d_val)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        print('val_loader batch: ', len(val_loader))
        return [train_loader], val_loader, test_loader
    elif args.cases == 'subject':
        return [train_loader], None, test_loader

