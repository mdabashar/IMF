from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from dataset import *
import random
from models_gm import *
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
parser.add_argument('--alpha', type=float)
args = parser.parse_args()

def l2_reg(model):
    l2_lambda = 1e-3
    l2_reg = torch.tensor(0.)
    for name, param in model.named_parameters():
        if name.split(".")[0] != "early":
            l2_reg += torch.norm(param, 2).to(device)
    return l2_lambda * l2_reg

def train(learning_rate=1e-3):
    bnn_loss_list_train = []
    functional_loss_list_train = []
    bnn_loss_list_val = []
    functional_loss_list_val = []
    bcl = torch.nn.BCELoss()
    for epoch in range(n_epochs):
        model.train()

        functional_loss = 0.0

        for modality1, modality2, modality3, labels in train_loader:

            output_multi, o1, o2, o3 = model(modality1, modality2, modality3)
            loss = bcl(output_multi.float(), torch.unsqueeze(labels, 1).float()) 
            functional_loss += loss
            
            optimizer.zero_grad()
            loss.backward()

            score_1 = []
            score_2 = []
            score_3 = []
            for i in range(torch.Tensor(o1).size(0)):
                if labels[i]:
                    score_1.append(torch.Tensor(o1)[i])
                    score_2.append(torch.Tensor(o2)[i])
                    score_3.append(torch.Tensor(o3)[i])
                else:
                    score_1.append(1-torch.Tensor(o1)[i])
                    score_2.append(1-torch.Tensor(o2)[i])
                    score_3.append(1-torch.Tensor(o3)[i])
            score_1 = sum(score_1)
            score_2 = sum(score_2)
            score_3 = sum(score_3)


            ratio_1 = 0.5*(score_1 / score_2 + score_1 /score_3)
            ratio_2 = 0.5*(score_2 / score_1 + score_2 /score_3)
            ratio_3 = 0.5*(score_3 / score_1 + score_3 /score_2)
            print("ratio_1",ratio_1)
            print("ratio_2",ratio_2)
            print("ratio_3",ratio_3)
            """
            Below is the Eq.(10) in our CVPR paper:
                    1 - tanh(alpha * rho_t_u), if rho_t_u > 1
            k_t_u =
                    1,                         else
            coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
            """
            tanh = nn.Tanh()
            relu = nn.ReLU(inplace=True)
            if ratio_1 > 1:
                coeff_1 = 1 - tanh(alpha * relu(ratio_1))
            else:
                coeff_1 = torch.tensor(1)

            if ratio_2 > 1:
                coeff_2 = 1 - tanh(alpha * relu(ratio_2))
            else:
                coeff_2 = torch.tensor(1)

            if ratio_3 > 1:
                coeff_3 = 1 - tanh(alpha * relu(ratio_3))
            else:
                coeff_3 = torch.tensor(1)

            if 0 <= epoch <= n_epochs: # bug fixed
                for name, parms in model.named_parameters():
                    layer = str(name).split('.')[0]
                    if parms.grad == None:
                        continue
                    if '1' in layer:
                        # if args.modulation == 'OGM_GE':  # bug fixed
                        parms.grad = parms.grad * coeff_1
                        # elif args.modulation == 'OGM':
                        #     parms.grad *= coeff_a
                        # print("Yes text ",coeff_a)

                    if '2' in layer:
                        # if args.modulation == 'OGM_GE':  # bug fixed
                        parms.grad = parms.grad * coeff_2 

                    if '3' in layer:
                        # if args.modulation == 'OGM_GE':  # bug fixed
                        parms.grad = parms.grad * coeff_3 

            #print("coeff_v ",coeff_v, "coeff_a",coeff_a)
            optimizer.step()

            functional_loss += loss

        print("#########################################################################################")
        print(f'epoch {epoch}: functional loss: {functional_loss/len(train_dataset):.4f}')
        functional_loss_list_train.append(functional_loss.cpu().detach().numpy()/len(train_dataset))
        acc, wf1, mf1 = test(test_loader, len(test_dataset), 99999, isVal=False)
 
    return (acc, wf1, mf1)

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
    bcl = torch.nn.BCELoss()
    for m1, m2, m3, labels in test_loader:

        with torch.no_grad():
            output = model(m1, m2, m3)
            prediction = torch.round(output[0])
            predict_score = predict_score + output[0].numpy().tolist()
            #_, prediction = torch.max(output, 1)
            n_samples += labels.size(0)
            n_correct += (prediction == labels.unsqueeze(1)).sum().item()
            predict_list = torch.cat([predict_list, prediction.view(-1).cpu()])
            real_list = torch.cat([real_list, labels.view(-1).cpu()])
            loss += bcl(output[0].float(), labels.unsqueeze(1).float())

    if not isVal:
        f1_weighted = f1_score(real_list.numpy(),predict_list.numpy(),average="weighted")
        precision = precision_score(real_list.numpy(),predict_list.numpy(),average="weighted")
        recall = recall_score(real_list.numpy(),predict_list.numpy(),average="weighted")

        f1_macro = f1_score(real_list.numpy(),predict_list.numpy(),average="macro")
        precision_macro = precision_score(real_list.numpy(),predict_list.numpy(),average="macro")
        recall_macro = recall_score(real_list.numpy(),predict_list.numpy(),average="macro")

        acc = 100.0 * n_correct / n_samples

        print("---------------------Test set------------------------")
        print(f'functional loss: {loss:.3f}, Acc: {acc:.4f} %,')
        print(f'Weighted results: f1 : {f1_weighted:.4f}, precision : {precision:.4f}, recall : {recall:.4f}')
        print(f'Macro results: f1 : {f1_macro:.4f}, precision : {precision_macro:.4f}, recall : {recall_macro:.4f}')

        auc_macro =  roc_auc_score(real_list.numpy(), np.array(predict_score), average="macro")
        auc_weighted =  roc_auc_score(real_list.numpy(), np.array(predict_score), average="weighted")
      
        print(f"average ROC AUC OvR: macro : {auc_macro:.4f}, weighted : {auc_weighted:.4f}")

    return (acc, f1_weighted, f1_macro)
if __name__ == '__main__':

    print("sigmoid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(torch.cuda.get_device_name(0))
    learning_rate = args.lr if not args.lr == None else 1e-5
    learning_rate_b = args.blr * learning_rate if not args.blr == None else learning_rate
    n_epochs = args.nepochs if not args.nepochs == None else 100
    num_forward_pass = args.nForwardPasses if not args.nForwardPasses == None else 1
    this_model = args.model if not args.model == None else "early"
    gate = args.gate if not args.gate == None else "m1"
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
    alpha = args.alpha if not args.alpha == None else 0.01
    if textLabel:
        label_label = "text"
    else:
        label_label = "image"

    test_acc = []
    batch_size = 16
    print(f"running the script with learning rate = {learning_rate}, bnn lr = {learning_rate_b}, #epcohs = {n_epochs},"
          f" #forward passes: {num_forward_pass}, current model {this_model}, dim {dim},"
          f" length {length}, split ratio {split_ratio}, functional dropout rate, {functional_p},"
          f" bnn dropout rate {bnn_p}")


    acc_list = []
    wf1_list = []
    mf1_list = []
    for i in np.arange(5):

        SEED = 999
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        train_dataset_data, test_dataset = getDataloder(fold = i)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle= True)

        modality1_data, modality2_data, modality3_data, y_data = train_dataset_data
        train_dataset = DatasetBERT(modality1_data, modality2_data, modality3_data, y_data)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=True, shuffle=True) 
        model = EarlyFusionModel(length = length, functional_p = functional_p)
        optimizer = torch.optim.SGD(model.parameters(),momentum=0.9, lr= learning_rate)
        acc, wf1, mf1 = train(learning_rate=learning_rate)
        acc_list.append(acc)
        wf1_list.append(wf1)
        mf1_list.append(mf1)
    
    print("acc mean: ", round(np.mean(acc_list),4), "  std: ", round( np.std(acc_list),4))
    print("wf1 mean: ", round(np.mean(wf1_list),4), "  std: ", round( np.std(wf1_list),4))
    print("mf1 mean: ", round(np.mean(mf1_list),4), "  std: ", round( np.std(mf1_list),4))
