from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from dataset import *
import random
from models import *
from models_baselines import *
import os
import argparse
from utils import *
from optimizer import MetaSGD
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
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
parser.add_argument('--isText', default=False, action='store_true')
parser.add_argument('--textLabel', default=False, action='store_true')
parser.add_argument('--earlyStop', type=int)
parser.add_argument('--scheduler', type=int)

args = parser.parse_args()

Min_loss = 999
Num_loss = 0
scheduler = 0
def l2_reg(model):
    l2_lambda = 1e-3
    l2_reg = torch.tensor(0.)
    for name, param in model.named_parameters():
        if name.split(".")[0] != "early":
            l2_reg += torch.norm(param, 2).to(device)
    return l2_lambda * l2_reg

def train():

    functional_loss_list_train = []
    bcl = torch.nn.BCELoss()
    for epoch in range(n_epochs):
        model.train()

        functional_loss = 0.0

        for m1, m2, m3, labels in train_loader:
              
            if modal == 0:
               output = model(m1)
            elif modal == 1:
               output = model(m2)
            elif modal == 2:
               output = model(m3)
            loss = cross_entropy(output, labels)
            functional_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("#########################################################################################")
        print(f'epoch {epoch}: functional loss: {functional_loss/len(train_dataset):.3f}')
        functional_loss_list_train.append(functional_loss.cpu().detach().numpy()/len(train_dataset))

        modality1_dir = dataset + r"1_te.csv"
        modality2_dir = dataset + r"2_te.csv"
        modality3_dir = dataset + r"3_te.csv"
        y_dir = dataset + r"labels_te.csv"

        modality1_data = pd.read_csv(modality1_dir, header=None).astype(np.float32).to_numpy()
        modality2_data = pd.read_csv(modality2_dir, header=None).astype(np.float32).to_numpy()
        modality3_data = pd.read_csv(modality3_dir, header=None).astype(np.float32).to_numpy()
        y_data = pd.read_csv(y_dir, header=None).astype(int).to_numpy().ravel()
        test_dataset = DatasetBERT(modality1_data, modality2_data, modality3_data, y_data)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle= True)
        acc, wf1, mf1 = test(test_loader, len(test_dataset), 99999, isVal=False, isText=isText)
    return acc, wf1, mf1

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
    predict_score = []
    loss = 0
    bcl = torch.nn.BCELoss()
    for m1, m2, m3, labels in test_loader:
         with torch.no_grad():   
            if modal == 0:
               output = model(m1)
            elif modal == 1:
               output = model(m2)
            elif modal == 2:
               output = model(m3)
            _, prediction = torch.max(output, 1)
            n_samples += labels.size(0)
            n_correct += (prediction == labels).sum().item()
            predict_list = torch.cat([predict_list, prediction.view(-1).cpu()])
            real_list = torch.cat([real_list, labels.view(-1).cpu()])
            loss = cross_entropy(output, labels)
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

    return acc, f1_weighted, f1_macro

if __name__ == '__main__':

    print("sigmoid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(torch.cuda.get_device_name(0))
    learning_rate = args.lr if not args.lr == None else 1e-2
    learning_rate_b = args.blr * learning_rate if not args.blr == None else learning_rate
    n_epochs = args.nepochs if not args.nepochs == None else 2
    num_forward_pass = args.nForwardPasses if not args.nForwardPasses == None else 1
    this_model = args.model if not args.model == None else "early"
    dim = args.dim if not args.dim == None else 1
    length = args.length if not args.length == None else 100
    split_ratio = args.ratio if not args.ratio == None else 0.1
    functional_p = args.functionalp if not args.functionalp == None else 0.5
    bnn_p = args.bnnp if not args.bnnp == None else 0.1
    train_event = args.train_event if not args.train_event == None else 4
    test_event = args.test_event if not args.test_event == None else train_event
    isText = args.isText
    num_earlyStop = args.earlyStop if not args.earlyStop == None else 30
    num_scheduler = args.scheduler if not args.scheduler == None else 10
    

    dataset_name = "rosmap"
    dataset = datasetConfig(dataset_name)
    test_acc = []
    batch_size = 16
    print(f"running the script with learning rate = {learning_rate}, bnn lr = {learning_rate_b}, #epcohs = {n_epochs},"
          f" #forward passes: {num_forward_pass}, current model {this_model}, dim {dim},"
          f" length {length}, split ratio {split_ratio}, functional dropout rate, {functional_p},"
          f" bnn dropout rate {bnn_p}")

    model_name = [0, 1, 2]    

    for modal in model_name:
        acc_list = []
        wf1_list = []
        mf1_list = [] 

        for i in np.arange(20):

            modality1_dir = dataset + r"1_tr.csv"
            modality2_dir = dataset + r"2_tr.csv"
            modality3_dir = dataset + r"3_tr.csv"
            y_dir = dataset + r"labels_tr.csv"

            modality1_data = pd.read_csv(modality1_dir, header=None).astype(np.float32).to_numpy()
            modality2_data = pd.read_csv(modality2_dir, header=None).astype(np.float32).to_numpy()
            modality3_data = pd.read_csv(modality3_dir, header=None).astype(np.float32).to_numpy()
            y_data = pd.read_csv(y_dir, header=None).astype(int).to_numpy().ravel()
            train_dataset = DatasetBERT(modality1_data, modality2_data, modality3_data, y_data)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
            if modal == 2:
                   modal_length = 200
            else:
                   modal_length = 200
            model = Unimodal(input_length=modal_length, length=length, functional_p=functional_p).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)
            acc, wf1, mf1 = train()
            acc_list.append(acc)
            wf1_list.append(wf1)
            mf1_list.append(mf1)
        print("REPORTING THE RESULT FOR MODAL ", str(modal)," ###########")
        print("acc mean: ", round(np.mean(acc_list),4), "  std: ", round( np.std(acc_list),4))
        print("wf1 mean: ", round(np.mean(wf1_list),4), "  std: ", round( np.std(wf1_list),4))
        print("mf1 mean: ", round(np.mean(mf1_list),4), "  std: ", round( np.std(mf1_list),4))

