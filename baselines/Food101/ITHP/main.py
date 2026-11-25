from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from dataset_gp import *
import random
from models import *
import os
import argparse
from utils import *
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
import pickle
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
parser.add_argument('--textLabel', default=False, action='store_true')
parser.add_argument('--earlyStop', type=int)
parser.add_argument('--scheduler', type=int)
parser.add_argument('--p_beta', type=float)
parser.add_argument('--p_gamma', type=float)
parser.add_argument('--p_lambda', type=float)
parser.add_argument('--gate', type=str)

args = parser.parse_args()

def l2_reg(model):
    l2_lambda = 1e-3
    l2_reg = torch.tensor(0.)
    for name, param in model.named_parameters():
        if name.split(".")[0] != "early":
            l2_reg += torch.norm(param, 2).to(device)
    return l2_lambda * l2_reg

def train(learning_rate):
    bnn_loss_list_train = []
    functional_loss_list_train = []
    bnn_loss_list_val = []
    functional_loss_list_val = []
    for epoch in range(n_epochs):
        model.train()

        functional_loss = 0.0

        for images, text, labels in train_loader:

            images = images.to(device)
            text = text.to(device)
            labels = labels.to(device)
            
            output, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1  = model(images, text)
            loss = cross_entropy(output, labels)
            loss += 2 / (p_beta + p_gamma) * IB_loss
            functional_loss += loss
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for name, params in model.named_parameters():
            #         print(name, " - ", params.grad)

            optimizer.zero_grad()



        print("#########################################################################################")
        print(f'epoch {epoch}: functional loss: {functional_loss*(batch_size/32)/len(train_dataset):.3f}')
        functional_loss_list_train.append(functional_loss.cpu().detach().numpy()/len(train_dataset))
        # bnn_loss_list_train.append(bnn_loss.detach().numpy()/meta_samples)
        # val_bnn_loss, val_functional_loss= test(val_loader, len(val_dataset))

        test(val_loader, len(val_dataset), 99999, isVal=True)
        acc,wf1,mf1 = test(test_loader, len(test_dataset), 99999, isVal=False)

    return acc,wf1,mf1

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
    loss = 0
    for images, text, labels in test_loader:

        images = images.to(device)
        text = text.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1  = model(images, text)
            predict_score = predict_score + torch.nn.functional.softmax(output).numpy().tolist()
            _, prediction = torch.max(output, 1)
            n_samples += labels.size(0)
            n_correct += (prediction == labels).sum().item()
            predict_list = torch.cat([predict_list, prediction.view(-1).cpu()])
            real_list = torch.cat([real_list, labels.view(-1).cpu()])
            loss += cross_entropy(output, labels)

    if not isVal:
        f1_weighted = f1_score(real_list.numpy(),predict_list.numpy(),average="weighted")
        precision = precision_score(real_list.numpy(),predict_list.numpy(),average="weighted")
        recall = recall_score(real_list.numpy(),predict_list.numpy(),average="weighted")

        f1_macro = f1_score(real_list.numpy(),predict_list.numpy(),average="macro")
        precision_macro = precision_score(real_list.numpy(),predict_list.numpy(),average="macro")
        recall_macro = recall_score(real_list.numpy(),predict_list.numpy(),average="macro")

        acc = 100.0 * n_correct / n_samples

        print("---------------------Test set------------------------")
        print(f'functional loss: {loss*(batch_size/32)/test_set_length:.3f}, Acc: {acc:.4f} %,')
        print(f'Weighted results: f1 : {f1_weighted:.4f}, precision : {precision:.4f}, recall : {recall:.4f}')
        print(f'Macro results: f1 : {f1_macro:.4f}, precision : {precision_macro:.4f}, recall : {recall_macro:.4f}')

        return acc, f1_weighted, f1_macro
if __name__ == '__main__':

    print("sigmoid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(torch.cuda.get_device_name(0))
    learning_rate = args.lr if not args.lr == None else 1e-5
    learning_rate_b = args.blr * learning_rate if not args.blr == None else learning_rate
    n_epochs = args.nepochs if not args.nepochs == None else 100
    num_forward_pass = args.nForwardPasses if not args.nForwardPasses == None else 1
    this_model = args.model if not args.model == None else "early"
    gate = args.gate if not args.gate == None else "im"
    dim = args.dim if not args.dim == None else 1
    length = args.length if not args.length == None else 200
    split_ratio = args.ratio if not args.ratio == None else 0.1
    functional_p = args.functionalp if not args.functionalp == None else 0.2
    bnn_p = args.bnnp if not args.bnnp == None else 0.1
    train_event = args.train_event if not args.train_event == None else 4
    test_event = args.test_event if not args.test_event == None else train_event
    isMean = args.isMean
    textLabel = args.textLabel
    num_scheduler = args.scheduler if not args.scheduler == None else 10
    num_earlyStop = args.earlyStop if not args.earlyStop == None else 30
    p_gamma = args.p_gamma if not args.p_gamma == None else 8
    p_beta = args.p_beta if not args.p_beta == None else 32
    p_lambda = args.p_lambda if not args.p_lambda == None else 0.3

    if textLabel:
        label_label = "text"
    else:
        label_label = "image"


    test_acc = []
    batch_size = 128
    print(f"running the script with learning rate = {learning_rate}, bnn lr = {learning_rate_b}, #epcohs = {n_epochs},"
          f" #forward passes: {num_forward_pass}, current model {this_model}, dim {dim},"
          f" length {length}, split ratio {split_ratio}, functional dropout rate, {functional_p},"
          f" bnn dropout rate {bnn_p}")

    best_epoch = []
    best_accs = []
    display_results = []

    acc_list = []
    wf1_list = []
    mf1_list = []

        
    for i in range(1):

        with open((str(__import__("pathlib").Path(__file__).resolve().parents[3] / "datasets" / "Food101") + "/" + 'text_train.pkl'), 'rb') as f:
             m2 = pickle.load(f)
        with open((str(__import__("pathlib").Path(__file__).resolve().parents[3] / "datasets" / "Food101") + "/" + 'image_train.pkl'), 'rb') as f:
             m1 = pickle.load(f)
        with open((str(__import__("pathlib").Path(__file__).resolve().parents[3] / "datasets" / "Food101") + "/" + 'label_train.pkl'), 'rb') as f:
             y = pickle.load(f)

        X = np.arange(len(m1))
        print("X ",X)
        X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=99
,stratify=y,test_size=0.1)

        m1_train = np.array(m1)[X_train]
        m2_train = np.array(m2)[X_train]
        train_y = np.array(y)[X_train]

        m1_test = np.array(m1)[X_test]
        m2_test = np.array(m2)[X_test]
        test_y = np.array(y)[X_test]

        train_dataset = DatasetBERT(m1_train,m2_train,train_y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

        val_dataset = DatasetBERT(m1_test,m2_test,test_y)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

        with open((str(__import__("pathlib").Path(__file__).resolve().parents[3] / "datasets" / "Food101") + "/" + 'text_test.pkl'), 'rb') as f:
             text_test = pickle.load(f)
        with open((str(__import__("pathlib").Path(__file__).resolve().parents[3] / "datasets" / "Food101") + "/" + 'image_test.pkl'), 'rb') as f:
             image_test = pickle.load(f)
        with open((str(__import__("pathlib").Path(__file__).resolve().parents[3] / "datasets" / "Food101") + "/" + 'label_test.pkl'), 'rb') as f:
             label_test = pickle.load(f)

        test_dataset = DatasetBERT(np.array(image_test),np.array(text_test),np.array(label_test))
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

        #if i!=0:
        #    break
        SEED = 999
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        txt_dim = 40
        im_dim = 512

        model = IB(length= length, functional_p=functional_p, p_gamma = p_gamma, p_beta = p_beta, p_lambda = p_lambda).to(device)

        optimizer = torch.optim.SGD(model.parameters(),momentum=0.9,weight_decay=1e-3, lr= learning_rate)
        acc, mf1,wf1 = train(learning_rate=learning_rate)
        acc_list.append(acc)
        wf1_list.append(wf1)
        mf1_list.append(mf1)
    
    print("acc mean: ", round(np.mean(acc_list),4), "  std: ", round( np.std(acc_list),4))
    print("wf1 mean: ", round(np.mean(wf1_list),4), "  std: ", round( np.std(wf1_list),4))
    print("mf1 mean: ", round(np.mean(mf1_list),4), "  std: ", round( np.std(mf1_list),4))
