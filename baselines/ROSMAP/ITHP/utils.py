import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dataset import *
from collections import Counter
import torch
import random
import pandas as pd
def drawGraph(inputs, title, y_label, legend_name, save_path):
    for item in inputs:
        plt.plot(item)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(y_label)
    plt.legend(legend_name, loc='upper left')
    plt.savefig(save_path)
    # plt.show()
    plt.clf()
    return 0


def datasetConfig(x):

    if x=="crisismmd1":
        return "../datasets/crisismmd_data/task1/"
    elif x=="crisismmd2":
        return "../datasets/upper_task2/"
    elif x=="mvsa_single":
        return "../datasets/mvsa_single_old_annotation/"
    elif x=="rosmap":
        return "../../../datasets/ROSMAP/"



def split_train_meta(im_dir, txt_dir, y_dir, split_ratio, ifWeibo=False, ifYelp=False):

    im_data = np.load(im_dir, allow_pickle=True)
    if not ifYelp:
        txt_data = np.load(txt_dir, allow_pickle=True).astype("float32")
    else:
        txt_data = np.load(txt_dir, allow_pickle=True).astype("int64")
    y_data = np.int64(np.load(y_dir, allow_pickle=True))

    x_data = np.arange(im_data.shape[0])
    txt_dim = txt_data.shape[1]

    if ifWeibo:
        train_index,val_index, y_train, y_val = train_test_split( x_data, y_data, stratify = y_data,
                                                                 test_size=0.1, random_state=233)
        train_index, meta_index, y_train, y_meta = train_test_split( train_index, y_train, stratify = y_train,
                                                                 test_size=split_ratio, random_state=233)
        #train_dataset = DatasetBERT(im_data[train_index], txt_data[train_index], y_data[train_index])
        val_dataset = DatasetBERT(im_data[val_index], txt_data[val_index], y_data[val_index])
        #meta_dataset = DatasetBERT(im_data[meta_index], txt_data[meta_index], y_data[meta_index])

        return (im_data[train_index], txt_data[train_index], y_data[train_index]),val_dataset
        #return train_dataset, meta_dataset, val_dataset
    else:
        return (im_data, txt_data, y_data)
        # train_index, meta_index, y_train, y_meta = train_test_split( x_data, y_data, stratify = y_data,
        #                                                              test_size=split_ratio, random_state=233)
        # train_dataset = DatasetBERT(im_data[train_index], txt_data[train_index], y_data[train_index])
        #
        # meta_dataset = DatasetBERT(im_data[meta_index], txt_data[meta_index], y_data[meta_index])
    
        # return train_dataset, meta_dataset


def split_for_crisismmd(im_dir, txt_dir, y_dir, events, split_ratio,
                         isBNN=True, te = 0, tr = 1):

    im_data = np.load(im_dir, allow_pickle=True)
    txt_data = np.load(txt_dir, allow_pickle=True).astype("float32")
    y_data = np.int64(np.load(y_dir, allow_pickle=True))
    events = np.load(events, allow_pickle=True)

    event_list = ['california_wildfires',
                 'hurricane_harvey',
                 'hurricane_irma',
                 'hurricane_maria',
                 'iraq_iran_earthquake',
                 'mexico_earthquake',
                 'srilanka_floods']
    train_event = event_list[tr]
    print("Training set:", train_event)
    test_event = event_list[te]

    test_index = []
    for idx, value in enumerate(events):
        if value == test_event:
                test_index.append(idx)
    te_im_data = im_data[test_index]
    te_txt_data = txt_data[test_index]
    te_data = np.arange(txt_data[test_index].shape[0])
    te_y_data = y_data[test_index]
    _, test_idx, _, test_label = train_test_split( te_data, te_y_data, stratify = te_y_data,
                                                                         test_size=0.2, random_state=233)
    test_dataset = DatasetBERT(te_im_data[test_idx], te_txt_data[test_idx], te_y_data[test_idx])
    print("@#@#@#@#@#@#@#@#@#@#@#@#@ Test set: ", test_event)
    # print("test on ", len(test_idx))

    train_index = []
    for idx, value in enumerate(events):
        # if value in train_event:
        if value == train_event:
                train_index.append(idx)

    tr_im_data = im_data[train_index]
    tr_txt_data = txt_data[train_index]
    tr_data = np.arange(txt_data[train_index].shape[0])
    tr_y_data = y_data[train_index]
    train_index, _, train_label, _ = train_test_split( tr_data, tr_y_data, stratify = tr_y_data,
                                                                         test_size=0.2, random_state=233)
    print("Currently the train set is : ", train_event)


    im_data = im_data[train_index]
    txt_data = txt_data[train_index]
    y_data = y_data[train_index]
    x_data = np.arange(txt_data.shape[0])

    if isBNN:
        train_index, val_index, y_train, y_val = train_test_split( x_data, y_data, stratify = y_data,
                                                                         test_size=0.05, random_state=233)
        # train_index, meta_index, y_train, y_meta = train_test_split( train_index, y_train, stratify = y_train,
        #                                                                  test_size=split_ratio, random_state=233)
        # train_dataset = DatasetBERT(im_data[train_index], txt_data[train_index], y_data[train_index])
        # meta_dataset = DatasetBERT(im_data[meta_index], txt_data[meta_index], y_data[meta_index])

        val_dataset = DatasetBERT(im_data[val_index], txt_data[val_index], y_data[val_index])

        train_dataset = (im_data[train_index], txt_data[train_index], y_data[train_index])
        return train_dataset, val_dataset, test_dataset
        # return train_dataset, meta_dataset, val_dataset, test_dataset
    else:
        train_index, val_index, y_train, y_val = train_test_split( x_data, y_data, stratify = y_data,
                                                                         test_size=0.05, random_state=233)
        train_dataset = DatasetBERT(im_data, txt_data, y_data)
        val_dataset = DatasetBERT(im_data[val_index], txt_data[val_index], y_data[val_index])
        return train_dataset, val_dataset, test_dataset


def split_val(im_dir, txt_dir, y_dir):

    im_data = np.load(im_dir, allow_pickle=True)
    txt_data = np.load(txt_dir, allow_pickle=True).astype("float32")
    y_data = np.int64(np.load(y_dir, allow_pickle=True))

    x_data = np.arange(im_data.shape[0])
    txt_dim = txt_data.shape[1]

    train_index, val_index, y_train, y_val = train_test_split( x_data, y_data, stratify = y_data,
                                                                 test_size=0.1, random_state=233)
    train_dataset = DatasetBERT(im_data[train_index], txt_data[train_index], y_data[train_index])

    val_dataset = DatasetBERT(im_data[val_index], txt_data[val_index], y_data[val_index])

    return train_dataset, val_dataset

def split_for_yelp(im_dir, txt_dir, y_dir, events,
                         isBNN=True, id = 0):

    im_data = np.load(im_dir, allow_pickle=True)
    txt_data = np.load(txt_dir, allow_pickle=True).astype("int64")
    y_data = np.int64(np.load(y_dir, allow_pickle=True))
    events = np.load(events, allow_pickle=True)

    event_list = ['Boston', 'Chicago', 'Los Angeles', 'New York', 'San Francisco']
    this_event = event_list[id]
    print("################$  This event is ",this_event,"  $################")

    index = []
    for idx, value in enumerate(events):
        if value == this_event:
                index.append(idx)
    this_im_data = im_data[index]
    this_txt_data = txt_data[index]
    this_y_data = y_data[index]


    if isBNN:
        return (this_im_data, this_txt_data, this_y_data)
    else:
        dataset = DatasetBERT(this_im_data, this_txt_data, this_y_data)
        return dataset


def underSampling(y):
    SEED = 999
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print("Classes distribution of this dataset : ", Counter(y))
    num_of_instances = min(Counter(y).values())
    classes = Counter(y).keys()
    idx = np.arange(len(y))
    idx_by_classes = {}

    for cls in classes:
        for i in idx:
            if y[i] == cls:
                if cls in idx_by_classes.keys():
                    idx_by_classes[cls].append(i)
                else:
                    idx_by_classes[cls] = [i]

    new_idx = np.array([], dtype=int)
    for key, value in idx_by_classes.items():
        this_idx = np.random.choice(value, num_of_instances, replace = False)
        new_idx = np.append(new_idx, this_idx)
    print("Balanced dataset : ", Counter(np.array(y)[new_idx]))
    return new_idx.tolist()

