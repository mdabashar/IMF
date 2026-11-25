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
parser.add_argument('--gate', type=str)
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

            output = model(images, text)
            loss = cross_entropy(output, labels)
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

        #txt_dir_val = dataset + 'dev_text'+".npy"
        #im_dir_val = dataset + 'dev_img'+".npy"
        #y_dir_val = dataset + 'dev_labels' + ".npy"

        #im_data_val = np.load(im_dir_val, allow_pickle=True)
        #txt_data_val = np.load(txt_dir_val, allow_pickle=True)
        #y_data_val = np.int64(np.load(y_dir_val, allow_pickle=True))
        
        txt_dir_te = dataset + 'test_text_fold'+"_"+str(fold)+".npy"
        im_dir_te = dataset + 'test_img_fold'+"_"+str(fold)+".npy"
        y_dir_te = dataset + 'test_labels_fold'+"_"+str(fold)+ ".npy"

        im_data_te = np.load(im_dir_te, allow_pickle=True)
        txt_data_te = np.load(txt_dir_te, allow_pickle=True)
        y_data_te = np.int64(np.load(y_dir_te, allow_pickle=True))
        x_data = np.arange(txt_data_te.shape[0])
        
        dev_index, test_index, dev_la, test_la = train_test_split( x_data, y_data_te, stratify = y_data_te, random_state=100,test_size=0.5)
         
        val_dataset = DatasetBERT(im_data_te[dev_index], txt_data_te[dev_index], y_data_te[dev_index])
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle= True)
        val_loss= test(val_loader, len(val_dataset), 99999, isVal=True)

        print(f"######## validation loss is {val_loss*(batch_size/32):.4f}")

        test_dataset = DatasetBERT(im_data_te[test_index], txt_data_te[test_index], y_data_te[test_index])
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle= True)
        test_loss = test(test_loader, len(test_dataset), 99999, isVal=False)

        global Min_loss
        global Num_loss
       

        if val_loss < Min_loss:
           Min_loss = val_loss
           Num_loss = 0
           scheduler = 0
        else:
          Num_loss += 1
          scheduler += 1

        if Num_loss > num_earlyStop:
           print("Have not improved for ",num_earlyStop," epochs, stop training")
           break
        else:
           print("Have not improved from the min loss for ", Num_loss)

        if scheduler == num_scheduler:
            learning_rate = learning_rate*0.1

            for g in optimizer.param_groups:
                g['lr'] = learning_rate
        
            scheduler = 0

            print("have not improved for ",num_scheduler ," epochs, reduce learning rate to ", learning_rate)

    return 0

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
            output = model(images, text)
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

        auc_macro =  roc_auc_score(real_list.numpy(), np.array(predict_score), multi_class = 'ovr', average="macro")
        auc_weighted =  roc_auc_score(real_list.numpy(), np.array(predict_score), multi_class = 'ovr', average="weighted")
        print(f"average ROC AUC OvR: macro : {auc_macro:.4f}, weighted : {auc_weighted:.4f}")

    return loss/test_set_length
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

    if textLabel:
        label_label = "text"
    else:
        label_label = "image"


    dataset_name = "crisismmd3"
    dataset = datasetConfig(dataset_name)

    test_acc = []
    batch_size = 128
    print(f"running the script with learning rate = {learning_rate}, bnn lr = {learning_rate_b}, #epcohs = {n_epochs},"
          f" #forward passes: {num_forward_pass}, current model {this_model}, dim {dim},"
          f" length {length}, split ratio {split_ratio}, functional dropout rate, {functional_p},"
          f" bnn dropout rate {bnn_p}")

   
    best_epoch = []
    best_accs = []
    display_results = []

    for fold in np.arange(5):

        if fold!=4:
            continue
        SEED = 999
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        txt_dim = 40
        im_dim = 512

        txt_dir = dataset + 'train_text_fold_'+str(fold)+".npy"
        im_dir = dataset + 'train_img_fold'+"_"+str(fold) + ".npy"
        y_dir = dataset + 'train_labels_fold'+"_"+str(fold)+ ".npy"

        im_data = np.load(im_dir, allow_pickle=True)
        txt_data = np.load(txt_dir, allow_pickle=True)
        y_data = np.int64(np.load(y_dir, allow_pickle=True))

        train_dataset = DatasetBERT(im_data, txt_data, y_data)
        train_loader = DataLoader(dataset= train_dataset, batch_size = batch_size, drop_last=True, shuffle=True)

        if this_model=="early":
            print("TESTING ON MODEL EARLY FUSION")
            model = Early_fusion(txt_dim=txt_dim, dim = dim, length=length, functional_p=functional_p).to(device)
        elif this_model=="late":
            print("TESTING ON MODEL LATE FUSION")
            model = Late_fusion(txt_dim=txt_dim, dim = dim, length=length, functional_p=functional_p).to(device)
        elif this_model == "cross":
            print("TESTING ON MODEL CROSS_ATTENTION")
            model = Cross(txt_dim=txt_dim, dim = dim, length=length, functional_p=functional_p).to(device)
        elif this_model == "gmu":
            print("TESTING ON MODEL GMU")
            model = GMU(txt_dim=txt_dim, dim = dim, length=length, functional_p=functional_p).to(device)
        elif this_model == "tensor":
            print("TESTING ON MODEL TENSOR")
            model = Tensor(txt_dim=txt_dim, dim = dim, length=length, functional_p=functional_p).to(device)
        elif this_model == "tensorgate":
            print("TESTING ON MODEL TENSOR GATE")
            model = TensorGate(txt_dim=txt_dim, dim = dim, length=length, functional_p=functional_p, gate = gate).to(device)


        print("TXT DIM IS ",txt_dim)
        print("######################BEGIN TRAINING ON FOLD ", fold)
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr= learning_rate)
       
        best_results = train(learning_rate=learning_rate)
        Min_loss = 999
        Num_loss = 0
        scheduler = 0 
