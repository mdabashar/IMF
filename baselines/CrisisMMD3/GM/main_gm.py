from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from datasets_local import *
import random
from model_gm import *
import os
import argparse
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
from tools import *
from sklearn.model_selection import train_test_split
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
parser.add_argument('--alpha', type=int)
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
    criterion = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        model.train()

        functional_loss = 0.0

        for images, text, labels in train_loader:

            text = text.to(device)
            labels = labels.to(device)

            output_multi, output_im, output_txt = model(images, text)
            loss = criterion(output_multi, labels)
            optimizer.zero_grad()
            loss.backward()

            #output_im = (torch.mm(output_im, torch.transpose(model.im_dim_1.weight, 0, 1)))
            #output_txt = (torch.mm(output_txt, torch.transpose(model.text_dim_1.weight, 0, 1)))

            score_v = sum([output_im[i][labels[i]] for i in range(output_im.size(0))])
            score_a = sum([output_txt[i][labels[i]] for i in range(output_txt.size(0))])
            #score_v = sum([torch.Tensor(output_im)[i][labels[i]] for i in range(torch.Tensor(output_im).size(0))])
            #score_a = sum([torch.Tensor(output_txt)[i][labels[i]] for i in range(torch.Tensor(output_txt).size(0))])

            ratio_v = score_v / score_a
            ratio_a = 1 / ratio_v

            """
            Below is the Eq.(10) in our CVPR paper:
                    1 - tanh(alpha * rho_t_u), if rho_t_u > 1
            k_t_u =
                    1,                         else
            coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
            """
            tanh = nn.Tanh()
            relu = nn.ReLU(inplace=True)
            if ratio_v > 1:
                coeff_v = 1 - tanh(alpha * relu(ratio_v))
                coeff_a = torch.tensor(1)
            else:
                coeff_a = 1 - tanh(alpha * relu(ratio_a))
                coeff_v = torch.tensor(1)

            if 0 <= epoch <= n_epochs: # bug fixed
                for name, parms in model.named_parameters():
                    layer = str(name).split('.')[0]
                    if parms.grad == None:
                        continue
                    if 'text' in layer:
                        # if args.modulation == 'OGM_GE':  # bug fixed
                        parms.grad = parms.grad * coeff_a + \
                                         torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        # elif args.modulation == 'OGM':
                        #     parms.grad *= coeff_a
                        # print("Yes text ",coeff_a)

                    if 'im' in layer:
                        # if args.modulation == 'OGM_GE':  # bug fixed
                        parms.grad = parms.grad * coeff_v + \
                                         torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

            #print("coeff_v ",coeff_v, "coeff_a",coeff_a)
            optimizer.step()

            functional_loss += loss*labels.size()[0]

        print("#########################################################################################")
        print(f'epoch {epoch}: functional loss: {functional_loss/len(train_dataset):.3f}')
        functional_loss_list_train.append(functional_loss.cpu().detach().numpy()/len(train_dataset))
        val_loss= test(val_loader, len(val_dataset), 99999, isVal=True)

        print(f"######## validation loss is {val_loss/len(val_dataset):.4f}")

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

        text = text.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output,_,_ = model(images, text)
            predict_score = predict_score + torch.nn.functional.softmax(output).cpu().numpy().tolist()
            _, prediction = torch.max(output, 1)
            n_samples += labels.size(0)
            n_correct += (prediction == labels).sum().item()
            predict_list = torch.cat([predict_list, prediction.view(-1).cpu()])
            real_list = torch.cat([real_list, labels.view(-1).cpu()])
            loss += cross_entropy(output, labels)*labels.size()[0]

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
        try:
            auc_macro =  roc_auc_score(real_list.numpy(), np.array(predict_score), multi_class = 'ovr', average="macro")
        except:
            auc_macro = 0
        try:
            auc_weighted =  roc_auc_score(real_list.numpy(), np.array(predict_score), multi_class = 'ovr', average="weighted")
        except:
            auc_weighted = 0
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
    num_earlyStop = args.earlyStop if not args.earlyStop == None else 3000
    alpha = args.alpha if not args.alpha == None else 0.1
    if textLabel:
        label_label = "text"
    else:
        label_label = "image"


    dataset_name = "crisismmd"
    dataset = datasetConfig(dataset_name)

    test_acc = []
    batch_size = 32
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

        #train_dataset = DatasetBERT(im_data, txt_data, y_data)
        train_dataset = DatasetBERT(im_data, txt_data, y_data)
        train_loader = DataLoader(dataset= train_dataset, batch_size = batch_size, drop_last=True, shuffle=True)


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

        test_dataset = DatasetBERT(im_data_te[test_index], txt_data_te[test_index], y_data_te[test_index])
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle= True)
        print("y_data.shape ",len(y_data))
        dataset_len = {"train": len(y_data),'val': len(val_dataset)}
        print(dataset_len)

        
        model = EarlyFusionModel(length = length, out_dim=3, functional_p = functional_p).to(device)

        print("TXT DIM IS ",txt_dim)
        print("######################BEGIN TRAINING ON FOLD ", fold)
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr= learning_rate)
       
        best_results = train(learning_rate=learning_rate)
        Min_loss = 999
        Num_loss = 0
        scheduler = 0 
