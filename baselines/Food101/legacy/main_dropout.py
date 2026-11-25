import pickle
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from dataset_gp import *
import random
from models import *
from models_baselines2 import *
import os
import argparse
from utils import *
from optimizer import MetaSGD
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import pickle

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

args = parser.parse_args()

Min_loss = 999
Num_loss = 0
scheduler = 0
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def l2_reg(model):
    l2_lambda = 1e-3
    l2_reg = torch.tensor(0.)
    for name, param in model.named_parameters():
        if name.split(".")[0] != "early":
            l2_reg += torch.norm(param, 2).to(device)
    return l2_lambda * l2_reg

def train(isText = True, learning_rate=1e-3):
    bnn_loss_list_train = []
    functional_loss_list_train = []
    bnn_loss_list_val = []
    functional_loss_list_val = []
    for epoch in range(n_epochs):
        model.train()

        functional_loss = 0.0

        if isText:
            for _, text, labels in train_loader:

                output = model(text)
                loss = cross_entropy(output, labels)
                functional_loss += loss
                # optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        else:
            for images, _, labels in train_loader:

                output = model(images)
                loss = cross_entropy(output, labels)
                functional_loss += loss
                # optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()



        print("#########################################################################################")
        print(f'epoch {epoch}: functional loss: {functional_loss/len(train_dataset):.3f}')
        functional_loss_list_train.append(functional_loss.cpu().detach().numpy()/len(train_dataset))

        test_acc = []

        test_loss = test(test_loader, len(test_dataset), 99999, isVal=False, isText=isText)
       
    return 0

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def test(test_loader, test_set_length, epoch, isVal=True, isText = True):
    n_correct = 0
    n_samples = 0
    predict_list = torch.zeros(0)
    real_list = torch.zeros(0)
    model.eval()

    loss = 0
    if isText:
        for _, text, labels in test_loader:

            with torch.no_grad():
                output = model(text)
                _, prediction = torch.max(output, 1)
                n_samples += labels.size(0)
                n_correct += (prediction == labels).sum().item()
                predict_list = torch.cat([predict_list, prediction.view(-1).cpu()])
                real_list = torch.cat([real_list, labels.view(-1).cpu()])
                loss += cross_entropy(output, labels)
    else:
        for images, _, labels in test_loader:

            with torch.no_grad():
                output = model(images)
                _, prediction = torch.max(output, 1)
                n_samples += labels.size(0)
                n_correct += (prediction == labels).sum().item()
                predict_list = torch.cat([predict_list, prediction.view(-1).cpu()])
                real_list = torch.cat([real_list, labels.view(-1).cpu()])
                loss += cross_entropy(output, labels)

    f1_weighted = f1_score(real_list.numpy(),predict_list.numpy(),average="weighted")
    acc = 100.0 * n_correct / n_samples

    #print(f'functional loss: {loss/test_set_length:.3f} Acc: {acc:.3f} %, f1 weighted : {f1_weighted:.4f}')

    return loss/test_set_length


def test_ensemble(test_loader, test_set_length, model_list, isVal=True):
    n_correct = 0
    n_samples = 0
    predict_list = torch.zeros(0)
    real_list = torch.zeros(0)
    predict_score = []
    loss = 0
    for images, text, labels in test_loader:
        output_im = []
        output_txt = []
        # print("This is the model list ",model_list)
        with torch.no_grad():
            for idx in range(10):
                model = model_list[1]
                model.eval()
                enable_dropout(model)
                this_output = model(images)
                output_im.append( this_output)
            for idx in range(10):
                model = model_list[0]
                model.eval()
                enable_dropout(model)
                this_output = model(text)
                output_txt.append( this_output)

        output_im = torch.stack(output_im)
        output_txt = torch.stack(output_txt)

        mean_im = torch.mean(output_im, dim=0)
        mean_txt = torch.mean(output_txt, dim=0)

        output_ensemble = mean_im + mean_txt
        predict_score = predict_score + torch.nn.functional.softmax(output_ensemble).numpy().tolist()
        _, prediction = torch.max(output_ensemble, 1)
        n_samples += labels.size(0)
        n_correct += (prediction == labels).sum().item()
        predict_list = torch.cat([predict_list, prediction.view(-1).cpu()])
        real_list = torch.cat([real_list, labels.view(-1).cpu()])
        loss += cross_entropy(output_ensemble, labels)

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

        #auc_macro =  roc_auc_score(real_list.numpy(), np.array(predict_score), multi_class = 'ovr', average="macro")
        #auc_weighted =  roc_auc_score(real_list.numpy(), np.array(predict_score), multi_class = 'ovr', average="weighted")
      
        #print(f"average ROC AUC OvR: macro : {auc_macro:.4f}, weighted : {auc_weighted:.4f}")

    return acc, f1_weighted,f1_macro

if __name__ == '__main__':

    print("sigmoid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(torch.cuda.get_device_name(0))
    learning_rate = args.lr if not args.lr == None else 1e-3
    learning_rate_b = args.blr * learning_rate if not args.blr == None else learning_rate
    n_epochs = args.nepochs if not args.nepochs == None else 1
    num_forward_pass = args.nForwardPasses if not args.nForwardPasses == None else 1
    this_model = args.model if not args.model == None else "early"
    dim = args.dim if not args.dim == None else 1
    length = args.length if not args.length == None else 100
    split_ratio = args.ratio if not args.ratio == None else 0.1
    functional_p = args.functionalp if not args.functionalp == None else 0.2
    bnn_p = args.bnnp if not args.bnnp == None else 0.1
    train_event = args.train_event if not args.train_event == None else 4
    test_event = args.test_event if not args.test_event == None else train_event
    isMean = args.isMean
    textLabel = args.textLabel
    num_earlyStop = args.earlyStop if not args.earlyStop == None else 30
    num_scheduler = args.scheduler if not args.scheduler == None else 10
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

    acc_list = []
    wf1_list = []
    mf1_list = []


    for i in range(1):

        with open('./data/text_train.pkl', 'rb') as f:
             m2 = pickle.load(f)
        with open('./data/image_train.pkl', 'rb') as f:
             m1 = pickle.load(f)
        with open('./data/label_train.pkl', 'rb') as f:
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

        with open('./data/text_test.pkl', 'rb') as f:
             text_test = pickle.load(f)
        with open('./data/image_test.pkl', 'rb') as f:
             image_test = pickle.load(f)
        with open('./data/label_test.pkl', 'rb') as f:
             label_test = pickle.load(f)

        test_dataset = DatasetBERT(np.array(image_test),np.array(text_test),np.array(label_test))
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

        model_list = []
        models = [True, False]
        for isTextModel in models:
            print("@##$%#$^$%&^%^*%*^&($TRAINING MODEL ")

            if isTextModel:
                model = Text( length=length, functional_p=functional_p).to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)
                train(isText=True, learning_rate = learning_rate)
            else:
                model = Image( length=length, functional_p=functional_p).to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)
                train(isText=False, learning_rate = learning_rate)
            
            Min_loss = 999
            Num_loss = 0
            scheduler = 0
            model_list.append(model)

        print("&&&&&&&&&&&&&&READY TO TEST THE ENSEMBLE MODEL for fold ", i," &&&&&&&&&&&&&&&&&")


        acc,wf1,mf1 = test_ensemble(test_loader, len(test_dataset), model_list=model_list, isVal=False)
        acc_list.append(acc)
        wf1_list.append(wf1)
        mf1_list.append(mf1)

    print("acc mean: ", round(np.mean(acc_list),4), "  std: ", round( np.std(acc_list),4))
    print("wf1 mean: ", round(np.mean(wf1_list),4), "  std: ", round( np.std(wf1_list),4))
    print("mf1 mean: ", round(np.mean(mf1_list),4), "  std: ", round( np.std(mf1_list),4))
