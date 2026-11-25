import copy
from random import randrange
import torch as torch
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from dataset import *
import random
from multi_stage_3 import *
import os
import argparse
from utils import *
from optimizer import MetaSGD
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings

Min_loss = 999
Num_loss = 0
scheduler = 0
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--lr', type=float)
parser.add_argument('--blr', type=float)
parser.add_argument('--nepochs', type=int)
parser.add_argument('--nForwardPasses', type=int)
parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--train_event', type=int)
parser.add_argument('--test_event', type=int)
parser.add_argument('--dim', type=int)
parser.add_argument('--length', type=int)
parser.add_argument('--ratio', type=float)
parser.add_argument('--functionalp', type=float)
parser.add_argument('--bnnp', type=float)
parser.add_argument('--isMean', default=False, action='store_true')
parser.add_argument("--textLabel", default=False, action='store_true')
parser.add_argument('--fold', type=int)
parser.add_argument('--earlyStop', type=int)
parser.add_argument('--scheduler', type=int)
parser.add_argument('--numFirst', type=int)
parser.add_argument('--numSec', type=int)
args = parser.parse_args()

def l2_reg(model):
    l2_lambda = 1e-3
    l2_reg = torch.tensor(0.).to(device)
    for name, param in model.named_parameters():
        if name.split(".")[0] != "early":
            l2_reg += torch.norm(param, 2).to(device)
    return l2_lambda * l2_reg

def train(learning_rate, learning_rate_b):

    torch.autograd.set_detect_anomaly(True)
    bnn_loss_list_train = []
    functional_loss_list_train = []
    bnn_loss_list_val = []
    functional_loss_list_val = []
    test_acc_list = []
    for epoch in range(n_epochs):
        model.train()
        meta_model_0.eval()
        #meta_model_1.eval()
        #meta_model_2.eval()
        functional_loss = 0.0
        bnn_loss = 0.0

        modality1_dir, modality2_dir, modality3_dir, y_dir = train_dataset_data

        modality1_data = pd.read_csv(modality1_dir, header=None).astype(np.float32).to_numpy()
        modality2_data = pd.read_csv(modality2_dir, header=None).astype(np.float32).to_numpy()
        modality3_data = pd.read_csv(modality3_dir, header=None).astype(np.float32).to_numpy()
        y_data = pd.read_csv(y_dir, header=None).astype(int).to_numpy().ravel()

        x_data = np.arange(modality1_data.shape[0])
        train_index, meta_index, y_train, y_meta = train_test_split( x_data, y_data, stratify = y_data,test_size=split_ratio)
        train_dataset = DatasetBERT(modality1_data[train_index], modality2_data[train_index], modality3_data[train_index], y_data[train_index])
        train_meta_dataset = DatasetBERT(modality1_data[meta_index], modality2_data[meta_index], modality3_data[meta_index], y_data[meta_index])
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        train_meta_loader = DataLoader(dataset=train_meta_dataset, batch_size=math.floor(batch_size*split_ratio), drop_last=True, shuffle=True)
        meta_dataloader_iter = iter(train_meta_loader)
        meta_samples = 0

        for modality1, modality2, modality3, labels in train_loader:

            meta_modality1, meta_modality2, meta_modality3, meta_labels = next(meta_dataloader_iter)

            meta_samples += meta_labels.size(0)
            
            modality1 = modality1.to(device)
            modality2 = modality2.to(device)
            modality3 = modality3.to(device)

            meta_modality1 = meta_modality1.to(device)
            meta_modality2 = meta_modality2.to(device)
            meta_modality3 = meta_modality3.to(device)

            model_hat = Early_fusion_new_0(length=length, functional_p=functional_p, isBnn=[True, False, False], num_f=num_first, num_sec = num_sec).to(device)
            model_hat.load_state_dict(model.state_dict())
            model_hat.train()
            # enable_dropout(meta_model_txt)
            # enable_dropout(meta_model_im)
            loss_hat = 0
            for k in range(num_forward_pass):
                #output_hat = model_hat(modality1, modality2, modality3, [meta_model_0], [meta_model_cat,meta_model_cat2,meta_model_cat3])
                output_hat = model_hat(modality1, modality2, modality3, [meta_model_0], [meta_model_cat])

                loss_hat += cross_entropy(output_hat.cpu(), labels.long())
            #functional_loss += loss_hat
            loss_hat = loss_hat/num_forward_pass
            grads_hat = torch.autograd.grad(loss_hat, model_hat.parameters(),
                                            create_graph=True)

            optimizer_hat = MetaSGD(model_hat, model_hat.parameters(), lr=learning_rate)
            optimizer_hat.load_state_dict(optimizer.state_dict())
            optimizer_hat.meta_step(grads_hat)
            del grads_hat

     #         update functional model with training data and new weight from BNN
            model.eval()
            # meta_model_txt.train()
            meta_model_0.train()
            #meta_model_1.train()
            #meta_model_2.train()
            meta_model_cat.train()
            #meta_model_cat2.train()
            #meta_model_cat3.train()
            loss = 0
            for k in range(num_forward_pass):
                # uncertainty_im_meta = meta_model_im(meta_images)
                # uncertainty_txt_meta = meta_model_txt(meta_text)

                #output = model_hat(meta_modality1, meta_modality2, meta_modality3, [meta_model_0], [meta_model_cat,meta_model_cat2,meta_model_cat3])
                output = model_hat(meta_modality1, meta_modality2, meta_modality3, [meta_model_0], [meta_model_cat])
                # print(output_hat)
                loss += cross_entropy(output.cpu(), meta_labels.long())
            loss = loss / num_forward_pass
            bnn_loss += loss*meta_labels.size()[0]
            meta_loss = loss

            meta_optimizer_0.zero_grad()
            #meta_optimizer_1.zero_grad()
            #meta_optimizer_2.zero_grad()
            meta_loss.backward( retain_graph = True,inputs = list(meta_model_0.parameters()))
            #meta_loss.backward( retain_graph = True,inputs = list(meta_model_1.parameters()))
            #meta_loss.backward( retain_graph = True,inputs = list(meta_model_2.parameters()))

            meta_loss_cat = loss
            #bnn_loss += meta_loss_cat
            meta_optimizer_cat.zero_grad()
            #meta_optimizer_cat2.zero_grad()
            #meta_optimizer_cat3.zero_grad()
            #meta_loss_cat.backward(retain_graph = True, inputs = list(meta_model_cat3.parameters()))
            #meta_loss_cat.backward(retain_graph = True, inputs = list(meta_model_cat2.parameters()))
            meta_loss_cat.backward(retain_graph = True, inputs = list(meta_model_cat.parameters()))

            meta_optimizer_0.step()
            #meta_optimizer_1.step()
            #meta_optimizer_2.step()
            meta_optimizer_cat.step()
            #meta_optimizer_cat2.step()
            #meta_optimizer_cat3.step()

            model.train()
            meta_model_0.eval()
            #meta_model_1.eval()
            #meta_model_2.eval()
            meta_model_cat.eval()
            #meta_model_cat2.eval()
            #meta_model_cat3.eval()
            loss = 0
            # uncertainty_im = meta_model_im(images)
            # uncertainty_txt = meta_model_txt(text)
            for i in range(num_forward_pass):
                #output = model(modality1, modality2, modality3, [meta_model_0], [meta_model_cat,meta_model_cat2,meta_model_cat3])
                output = model(modality1, modality2, modality3, [meta_model_0], [meta_model_cat])
                loss += cross_entropy(output.cpu(), labels.long())
            loss = loss/num_forward_pass
            functional_loss += loss*labels.size()[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        print("#########################################################################################")
        print(f'epoch {epoch}: functional loss: {functional_loss/len(train_dataset):.3f}')
        functional_loss_list_train.append(functional_loss.cpu().detach().numpy()/len(train_dataset))
        bnn_loss_list_train.append(bnn_loss.cpu().detach().numpy()/meta_samples)

        (acc, f1_weighted, f1_macro), test_loss = test(test_loader, 500, 99999, isVal=False)
        functional_loss_list_val.append(test_loss.cpu().detach().numpy())
        test_acc_list.append(acc)

    return (acc, f1_weighted, f1_macro)

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def test(test_loader, test_set_length, epoch, isVal=True):
    n_correct = 0
    n_samples = 0
    predict_list = torch.zeros(0)
    real_list = torch.zeros(0)
    predict_score = []
    model.eval()
    meta_model_0.eval()
    #meta_model_1.eval()
    #meta_model_2.eval()
    meta_model_cat.eval()
    #meta_model_cat2.eval()
    #meta_model_cat3.eval()
    # enable_dropout(meta_model_txt)
    # enable_dropout(meta_model_im)
    bnn_loss = 0
    loss = 0
    for m1, m2, m3, labels in test_loader:

        m1 = m1.to(device)
        m2 = m2.to(device)
        m3 = m3.to(device)

        with torch.no_grad():
            # uncertainty_im = meta_model_im(images)
            # uncertainty_txt = meta_model_txt(text)

            #output = model(m1, m2, m3, [meta_model_0], [meta_model_cat,meta_model_cat2,meta_model_cat3])
            output = model(m1, m2, m3, [meta_model_0], [meta_model_cat])
            #predict_score = predict_score + torch.nn.functional.softmax(output).numpy().tolist()
            _, prediction = torch.max(output, 1)
            n_samples += labels.size(0)
            n_correct += (prediction.cpu() == labels).sum().item()
            predict_list = torch.cat([predict_list, prediction.view(-1).cpu()])
            real_list = torch.cat([real_list, labels.view(-1).cpu()])
            loss += cross_entropy(output.cpu(), labels.long())*labels.size()[0]
            # meta_loss = criterion(output, labels)
            # print(output)
            # bnn_loss += meta_loss

    if not isVal:
        f1_weighted = f1_score(real_list.numpy(),predict_list.numpy(),average="weighted")
        precision = precision_score(real_list.numpy(),predict_list.numpy(),average="weighted")
        recall = recall_score(real_list.numpy(),predict_list.numpy(),average="weighted")

        f1_macro = f1_score(real_list.numpy(),predict_list.numpy(),average="macro")
        precision_macro = precision_score(real_list.numpy(),predict_list.numpy(),average="macro")
        recall_macro = recall_score(real_list.numpy(),predict_list.numpy(),average="macro")

        acc = 100.0 * n_correct / n_samples

        print("---------------------Test set------------------------")
        print(f'functional loss: {loss/test_set_length:.3f}, Acc: {acc:.4f} %,')
        print(f'Weighted results: f1 : {f1_weighted:.4f}, precision : {precision:.4f}, recall : {recall:.4f}')
        print(f'Macro results: f1 : {f1_macro:.4f}, precision : {precision_macro:.4f}, recall : {recall_macro:.4f}')

        #auc_macro =  roc_auc_score(real_list.numpy(), np.array(predict_score), multi_class = 'ovr', average="macro")
        #auc_weighted =  roc_auc_score(real_list.numpy(), np.array(predict_score), multi_class = 'ovr', average="weighted")

        #print(f"average ROC AUC OvR: macro : {auc_macro:.4f}, weighted : {auc_weighted:.4f}")

    return (acc, f1_weighted, f1_macro),loss/test_set_length



if __name__ == '__main__':

    print("sigmoid addition")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if device != torch.device("cpu"):
        print(torch.cuda.get_device_name(0))
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    learning_rate = args.lr if not args.lr == None else 1e-3
    learning_rate_b = args.blr * learning_rate if not args.blr == None else learning_rate
    n_epochs = args.nepochs if not args.nepochs == None else 100
    num_forward_pass = args.nForwardPasses if not args.nForwardPasses == None else 10
    this_model = args.model if not args.model == None else "early"
    dim = args.dim if not args.dim == None else 1
    length = args.length if not args.length == None else 100
    split_ratio = args.ratio if not args.ratio == None else 0.2
    functional_p = args.functionalp if not args.functionalp == None else 0.5
    bnn_p = args.bnnp if not args.bnnp == None else 0.2
    train_event = args.train_event if not args.train_event == None else 4
    test_event = args.test_event if not args.test_event == None else train_event
    isMean = args.isMean
    fold = args.fold if not args.fold == None else 0
    textLabel = args.textLabel
    num_scheduler = args.scheduler if not args.scheduler == None else 10
    num_earlyStop = args.earlyStop if not args.earlyStop == None else 30
    num_first = args.numFirst if not args.numFirst == None else 6
    num_sec = args.numSec if not args.numSec == None else 3
    dataset_name = "brca"
    dataset = datasetConfig(dataset_name)
 
    test_acc = []
    batch_size = 64
    print(f"running the script with learning rate = {learning_rate}, bnn lr = {learning_rate_b}, #epcohs = {n_epochs},"
          f" #forward passes: {num_forward_pass}, current model {this_model}, dim {dim},"
          f" length {length}, split ratio {split_ratio}, functional dropout rate, {functional_p},"
          f" bnn dropout rate {bnn_p},"
          f"num_first {num_first} num_sec {num_sec}")

    acc_list = []
    wf1_list = []
    mf1_list = []

    best_epoch = []
    best_accs = []

    SEED = randrange(999)
    print("RANDOM SEED IS ############# ", SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    for i in np.arange(20):

        #if i!= fold:
         #   continue

        #SEED = 999
        #random.seed(SEED)
        #np.random.seed(SEED)
        #torch.manual_seed(SEED)

        modality1_dir = dataset + r"1_tr.csv"
        modality2_dir = dataset + r"2_tr.csv"
        modality3_dir = dataset + r"3_tr.csv"
        y_dir = dataset + r"labels_tr.csv"

        train_dataset_data = (modality1_dir, modality2_dir, modality3_dir, y_dir)


        train_dataset_data = (modality1_dir, modality2_dir, modality3_dir, y_dir)

        model = Early_fusion_new_0(length=length, functional_p=functional_p , isBnn=[True, False, False], num_f=num_first, num_sec = num_sec).to(device)
        meta_model_0 = Meta_net(bnn_p=bnn_p,dims = [1000, 1000, 503], length = num_first*length).to(device)
        #meta_model_1 = Meta_net_static(bnn_p=bnn_p,dims = [num_first*length,num_first*length,num_first*length], length = num_sec*length).to(device)
        #meta_model_2 = Meta_net_static(bnn_p=bnn_p,dims = [num_sec*length,num_sec*length,num_sec*length], length = length).to(device)
        # meta_model_txt = Meta_net(input_length=96, length=length, bnn_p=bnn_p).to(device)
        meta_model_cat = Meta_net_cat(length=num_first*length, bnn_p=bnn_p).to(device)
        #meta_model_cat2 = Meta_net_cat(length=num_sec*length, bnn_p=bnn_p).to(device)
        #meta_model_cat3 = Meta_net_cat(length=length, bnn_p=bnn_p).to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate, momentum=0.9, weight_decay=1e-3)
        meta_optimizer_0 = torch.optim.SGD(meta_model_0.parameters(), lr= learning_rate_b, momentum=0.9, weight_decay=1e-3)
        #meta_optimizer_1 = torch.optim.SGD(meta_model_1.parameters(), lr= learning_rate_b, momentum=0.9, weight_decay=1e-3)
        #meta_optimizer_2 = torch.optim.SGD(meta_model_2.parameters(), lr= learning_rate_b, momentum=0.9, weight_decay=1e-3)
        # meta_optimizer_txt = torch.optim.SGD(meta_model_txt.parameters(), lr= learning_rate_b, momentum=0.9, weight_decay=1e-3)
        meta_optimizer_cat = torch.optim.SGD(meta_model_cat.parameters(), lr= learning_rate_b, momentum=0.9, weight_decay=1e-3)
        #meta_optimizer_cat2 = torch.optim.SGD(meta_model_cat2.parameters(), lr= learning_rate_b, momentum=0.9, weight_decay=1e-3)
        #meta_optimizer_cat3 = torch.optim.SGD(meta_model_cat3.parameters(), lr= learning_rate_b, momentum=0.9, weight_decay=1e-3)
        
        criterion_1 = nn.CrossEntropyLoss()
        criterion_2 = nn.CrossEntropyLoss()
        criterion_3 = nn.CrossEntropyLoss()
        criterion_cat = nn.CrossEntropyLoss()

        modality1_dir = dataset + r"1_te.csv"
        modality2_dir = dataset + r"2_te.csv"
        modality3_dir = dataset + r"3_te.csv"
        y_dir = dataset + r"labels_te.csv"

        modality1_data = pd.read_csv(modality1_dir, header=None).astype(np.float32).to_numpy()
        modality2_data = pd.read_csv(modality2_dir, header=None).astype(np.float32).to_numpy()
        modality3_data = pd.read_csv(modality3_dir, header=None).astype(np.float32).to_numpy()
        y_data = pd.read_csv(y_dir, header=None).astype(int).to_numpy().ravel()
        test_dataset= DatasetBERT(modality1_data, modality2_data, modality3_data, y_data)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle= True)

        acc, wf1, mf1 = train(learning_rate,learning_rate_b)

        acc_list.append(acc)
        wf1_list.append(wf1)
        mf1_list.append(mf1)

    print("acc mean: ", round(np.mean(acc_list),4), "  std: ", round( np.std(acc_list),4))
    print("wf1 mean: ", round(np.mean(wf1_list),4), "  std: ", round( np.std(wf1_list),4))
    print("mf1 mean: ", round(np.mean(mf1_list),4), "  std: ", round( np.std(mf1_list),4))

