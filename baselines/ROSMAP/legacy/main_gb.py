import torch
from torch import nn
import copy
from torch.utils.data import DataLoader, Subset

criterion = nn.CrossEntropyLoss()
delta = False
from torch.utils.data import DataLoader
from dataset import *
import random
from gb_models import *
import os
import argparse
from utils import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

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
args = parser.parse_args()

criterion = nn.CrossEntropyLoss()
delta = False


def getloss(model, head, data, monum, batch_size, modality= 1):
    """Get loss for model given classification head.
    Args:
        model (nn.Module): Module to evaluate
        head (nn.Module): Classification head.
        data (torch.utils.data.Dataloader): Dataloader to evaluate on.
        monum (int): Unimodal model index.
        batch_size (int): (unused) Batch Size
    Returns:
        float: Average loss per sample.
    """
    losses = 0.0
    total = 0
    with torch.no_grad():
        for m1, m2, m3, labels in data:
            total += m1.size()[0]
            if modality ==0:
               input = m1
            elif modality ==1:
               input = m2
            else:
               input = m3
            out = model(input)
            out = head(out)
            loss = criterion(out, labels)
            losses += loss*m1.size()[0]
    return losses/total


def train_unimodal(model, head, optim, trains, valids, monum, epoch, batch_size, modality):
    """Train unimodal gradient blending module.
    Args:
        model (nn.Module): Unimodal encoder
        head (nn.Module): Classifier instance
        optim (torch.optim.Optimizer): Optimizer instance
        trains (torch.utils.data.DataLoader):  Training Dataloader Instance
        valids (torch.utils.data.DataLoader):  Validation DataLoader Instance
        monum (int): Modality index
        epoch (int): Number of epochs to train on
        batch_size (int): Batch size of data loaders
    Returns:
        float: Metric
    """
    ltN = getloss(model, head, trains, monum, batch_size, modality)
    lvN = getloss(model, head, valids, monum, batch_size, modality)
    for i in range(epoch):
        totalloss = 0.0
        total = 0
        for m1, m2, m3, labels in trains:
            total += m1.size()[0]
            if modality ==0:
               input = m1
            elif modality ==1:
               input = m2
            else:
               input = m3
            optim.zero_grad()
            out = model(input)
            out = head(out)
            loss = criterion(out,labels)
            totalloss += loss * m1.size()[0]
            loss.backward()
            optim.step()
        #print("Epoch "+str(i)+" loss: "+str(totalloss / total))

    ltNn = getloss(model, head, trains, monum, batch_size, modality)
    lvNn = getloss(model, head, valids, monum, batch_size, modality)
    #print("Final train loss: "+str(ltNn)+" valid loss: "+str(lvNn))
    oNn = lvNn-ltNn
    oN = lvN-ltN
    if delta:
        oi = oNn-oN
        g = lvNn-lvN
    else:
        oi = oNn
        if oi < 0:
            oi = 0.0001
        g = lvNn
    #print("raw: "+str(g/(oi*oi)))
    return abs(g/(oi*oi))


def multimodalcondense(models, fuse, train_x):
    """Compute fusion encoded output.
    Args:
        models (List): List of nn.Modules for each encoder
        fuse (nn.Module): Fusion instance
        train_x (List): List of Input Tensors
    Returns:
        torch.Tensor: Fused output
    """
    outs = multimodalcompute(models, train_x)
    return fuse(outs)


def multimodalcompute(models, train_x):
    """Compute encoded representation for each modality in train_x using encoders in models.
    Args:
        models (list): List of encoder instances
        train_x (List): List of Input Tensors
    Returns:
        List: List of encoded tensors
    """
    outs = []
    for i in range(len(models)):
        outs.append(models[i](train_x[i]))
    return outs


def getmloss(models, head, fuse, data, batch_size):
    """Calculate multimodal loss.
    Args:
        models (list): List of encoder models
        head (nn.Module): Classifier module
        fuse (nn.Module): Fusion module
        data (torch.utils.data.Dataloader): Data loader to calculate loss on.
        batch_size (int): Batch size of dataloader
    Returns:
        float: Average loss
    """
    losses = 0.0
    total = 0
    with torch.no_grad():
        for m1, m2, m3, labels in data:
            total += m1.size()[0]
            out = head(multimodalcondense(models, fuse, (m1, m2, m3)))
            loss = criterion(out, labels)
            losses += loss*m1.size()[0]
    return losses/float(total)


def train_multimodal(models, head, fuse, optim, trains, valids, epoch, batch_size):
    """Train multimodal gradient-blending model.
    Args:
        models (list): List of nn.modules for the encoders
        head (nn.Module): Classifier, post fusion layer
        fuse (nn.Module): Fusion module
        optim (torch.optim.Optimizer): Optimizer instance.
        trains (torch.utils.data.Dataloader): Training data dataloader
        valids (torch.utils.data.Dataloader): Validation data dataloader
        epoch (int): Number of epochs to train on
        batch_size (int): Batch size
    Returns:
        float: metric
    """
    ltN = getmloss(models, head, fuse, trains, batch_size)
    lvN = getmloss(models, head, fuse, valids, batch_size)
    for i in range(epoch):
        totalloss = 0.0
        total = 0
        for m1, m2, m3, labels in trains:
            total += m1.size()[0]
            optim.zero_grad()
            out = head(multimodalcondense(models, fuse, (m1, m2, m3)))
            loss = criterion(out, labels)
            totalloss += loss*m1.size()[0]
            loss.backward()
            optim.step()
        #print("Epoch "+str(i)+" loss: "+str(totalloss/total))
    ltNn = getmloss(models, head, fuse, trains, batch_size)
    lvNn = getmloss(models, head, fuse, valids, batch_size)
    #print("Final train loss: "+str(ltNn)+" valid loss: "+str(lvNn))
    oNn = lvNn-ltNn
    oN = lvN-ltN
    if delta:
        oi = oNn-oN
        g = lvNn-lvN
    else:
        oi = oNn
        if oi < 0:
            oi = 0.0001
        g = lvNn
    #print("raw: "+str(g/(oi*oi)))
    return abs(g/(oi*oi))


def gb_estimate(unimodal_models, multimodal_classification_head, fuse, unimodal_classification_heads, train_dataloader, gb_epoch,
                batch_size, v_dataloader, lr, weight_decay=0.0, optimtype=torch.optim.SGD):
    """Compute estimate of gradient-blending score.
    Args:
        unimodal_models (list): List of encoder modules
        multimodal_classification_head (nn.Module): Classifier given fusion instance
        fuse (nn.Module): Fusion module
        unimodal_classification_heads (list): List of unimodal classifiers
        train_dataloader (torch.utils.data.Dataloader): Training data loader
        gb_epoch (int): Number of epochs for gradient-blending
        batch_size (int): Batch size
        v_dataloader (torch.utils.data.Dataloader): Validation dataloader
        lr (float): Learning Rate
        weight_decay (float, optional): Weight decay parameter. Defaults to 0.0.
        optimtype (torch.optim.Optimizer, optional): Optimizer instance. Defaults to torch.optim.SGD.
    Returns:
        float: Normalized weights between unimodal and multimodal models
    """
    weights = []
    for i in range(len(unimodal_models)):
        #print("At gb_estimate unimodal "+str(i))
        model = copy.deepcopy(unimodal_models[i]).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        head = copy.deepcopy(unimodal_classification_heads[i]).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        optim = optimtype(list(model.parameters()) +
                          list(head.parameters()), lr=lr, weight_decay=weight_decay)
        w = train_unimodal(model, head, optim, train_dataloader,
                           v_dataloader, i, gb_epoch, batch_size, modality=i)
        weights.append(w)
    #print("At gb_estimate multimodal ")
    allcopies = [copy.deepcopy(x).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) for x in unimodal_models]
    mmcopy = copy.deepcopy(multimodal_classification_head).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    fusecopy = copy.deepcopy(fuse).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    params = []
    for model in allcopies:
        params.extend(list(model.parameters()))
    params.extend(list(mmcopy.parameters()))
    if fusecopy.parameters() is not None:
        params.extend(list(fusecopy.parameters()))
    optim = optimtype(params, lr=lr, weight_decay=weight_decay)
    weights.append(train_multimodal(allcopies, mmcopy, fusecopy,
                                    optim, train_dataloader, v_dataloader, gb_epoch, batch_size))
    z = sum(weights)
    return [(w/z).item() for w in weights]


softmax = nn.Softmax()


class completeModule(nn.Module):
    """Implements and combines sub-modules into a full classifier."""
    def __init__(self, encoders, fuse, head):
        """Instantiate completeModule instance.
        Args:
            encoders (list): List of nn.Module encoders
            fuse (nn.Module): Fusion module
            head (nn.Module): Classifier module
        """
        super(completeModule, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fuse
        self.head = head

    def forward(self, x):
        """Apply classifier to output.
        Args:
            x (list[torch.Tensor]): List of input tensors
        Returns:
            torch.Tensor: Classifier output
        """
        outs = multimodalcondense(self.encoders, self.fuse, x)
        return self.head(outs)


def train(unimodal_models,  multimodal_classification_head,
          unimodal_classification_heads, fuse, train_dataloader,
          num_epoch, lr, gb_epoch=20, v_rate=0.08, weight_decay=0.0, optimtype=torch.optim.SGD,
          finetune_epoch=25, classification=True, AUPRC=False, savedir='best.pt', track_complexity=True):
    """Train model using gradient_blending.
    Args:
        unimodal_models (list): List of modules, unimodal encoders for each input modality in the order of the modality input data.
        multimodal_classification_head (nn.Module): Classification head that takes in fused output of unimodal models of all modalities
        unimodal_classification_heads (list[nn.Module]): List of classification heads that each takes in output of one unimodal model (must be in the same modality order as unimodal_models)
        fuse (nn.Module): Fusion module that takes in a list of outputs from unimodal_models and generate a fused representation
        train_dataloader (torch.utils.data.DataLoader): Training data loader
        valid_dataloader (torch.utils.data.DataLoader): Validation data loader
        num_epoch (int): Number of epochs to train this model on.
        lr (float): Learning rate.
        gb_epoch (int, optional): Number of epochs between re-evaluation of weights of gradient blend. Defaults to 20.
        v_rate (float, optional): Portion of training set used as validation for gradient blend weight estimation. Defaults to 0.08.
        weight_decay (float, optional): Weight decay of optimizer. Defaults to 0.0.
        optimtype (torch.optim.Optimizer, optional):  Type of optimizer to use. Defaults to torch.optim.SGD.
        finetune_epoch (int, optional): Number of epochs to finetune the classification head. Defaults to 25.
        classification (bool, optional): Whether the task is a classification task. Defaults to True.
        AUPRC (bool, optional): Whether to compute auprc score or not. Defaults to False.
        savedir (str, optional): The name of the saved file for the model with current best validation performance. Defaults to 'best.pt'.
        track_complexity (bool, optional): Whether to track complexity or not. Defaults to True.
    """
    def _trainprocess():
        nonlocal train_dataloader
        params = []
        for model in unimodal_models:
            params.extend(model.parameters())
        for model in unimodal_classification_heads:
            params.extend(model.parameters())
        params.extend(multimodal_classification_head.parameters())
        params.extend(fuse.parameters())
        optim = optimtype(params, lr=lr, weight_decay=weight_decay)
        train_datas = train_dataloader.dataset
        splitloc = int(len(train_datas)*v_rate)
        inds = list(range(len(train_datas)))
        train_inds = inds[splitloc:]
        v_inds = inds[0:splitloc]
        train_data = Subset(train_datas, train_inds)
        v_data = Subset(train_datas, v_inds)
        train_dataloader = DataLoader(
            train_data, shuffle=True, batch_size=train_dataloader.batch_size)
        tv_dataloader = DataLoader(
            v_data, shuffle=False, batch_size=train_dataloader.batch_size)
        finetunehead = copy.deepcopy(multimodal_classification_head).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        fusehead = copy.deepcopy(fuse).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        params = list(finetunehead.parameters())
        if fuse.parameters() is not None:
            params.extend(list(fuse.parameters()))
        optimi = optimtype(params, lr=lr, weight_decay=weight_decay)
        bestvalloss = 10000.0
        #print("gb_epoch")
        for i in range(num_epoch):
            # """
            weights = gb_estimate(unimodal_models,  multimodal_classification_head, fuse,
                                  unimodal_classification_heads, train_dataloader, gb_epoch, train_dataloader.batch_size, tv_dataloader, lr, weight_decay, optimtype)
            # """
            # weights=(1.0,1.0,1.0)
            #print("epoch "+str(i*gb_epoch)+" weights: "+str(weights))
            for jj in range(gb_epoch):
                totalloss = 0.0
                for m1, m2, m3, labels in train_dataloader:
                    optim.zero_grad()
                    outs = multimodalcompute(unimodal_models, (m1, m2, m3))
                    fuse.train()
                    multimodal_classification_head.train()
                    catout = fuse(outs)
                    blendloss = criterion(multimodal_classification_head(
                        catout), labels)*weights[-1]
                    for ii in range(len(unimodal_models)):
                        loss = criterion(unimodal_classification_heads[ii](
                            outs[ii]), labels)
                        blendloss += loss * weights[ii]
                    totalloss += blendloss*m1.size()[0]
                    blendloss.backward()
                    optim.step()
                #print("epoch "+str(jj+i*gb_epoch)+" blend train loss: " +
                    #str(totalloss/len(train_data)))
            # finetunes classification head:
            # input -> output of each modality;
            # output -> labels
            #finetunetrains = []
            #with torch.no_grad():
            #    for m1, m2, m3,labels in train_dataloader:
            #        outs = multimodalcompute(unimodal_models, (m1, m2, m3))
            #        for iii in range(len(labels)):
            #            aa = [x[iii].cpu() for x in outs]
            #            aa.append(labels[iii].cpu())
            #            finetunetrains.append(aa)

            #print("Length of ftt_dataloader: "+str(len(finetunetrains)))
            #ftt_dataloader = DataLoader(
            #    finetunetrains, shuffle=True, batch_size=train_dataloader.batch_size)
            #for jj in range(finetune_epoch):
            #    totalloss = 0.0
            #    for m1, m2, m3, labels in ftt_dataloader:
            #        optimi.zero_grad()
            #        finetunehead.train()
            #        fusehead.train()
            #        blendloss = criterion(finetunehead(
            #            fusehead((m1, m2, m3))), labels)
            #        totalloss += blendloss * m1.size()[0]
            #        blendloss.backward()
            #        optimi.step()
                #print("finetune train loss: "+str(totalloss/len(train_data)))

            print("&&&&&&&Preparing to test on TESTING dataset$$$$$$$$")
            with torch.no_grad():
                totalloss = 0.0
                total = 0
                corrects = 0
                predict_list = torch.zeros(0)
                real_list = torch.zeros(0)
                for m1, m2, m3, labels in test_loader:
                    outs = multimodalcompute(unimodal_models, (m1, m2, m3))
                    fusehead.eval()
                    catout = fusehead(outs)
                    finetunehead.eval()
                    predicts = multimodal_classification_head(catout)
                    blendloss = criterion(predicts, labels)
                    totalloss += blendloss*m1.size()[0]
                    predictlist = predicts.tolist()
                    for ii in range(m1.size()[0]):
                        total += 1
                        if predictlist[ii].index(max(predictlist[ii])) == labels[ii]:
                                corrects += 1

                    _, prediction = torch.max(predicts, 1)
                    predict_list = torch.cat([predict_list, prediction.view(-1).cpu()])
                    real_list = torch.cat([real_list, labels.view(-1).cpu()])
                #print("PREDICTION LIST ",predict_list)
                #print("REAL LIST ", real_list)
                valoss = totalloss/total

                f1_weighted = f1_score(real_list.numpy(),predict_list.numpy(),average="weighted")
                precision = precision_score(real_list.numpy(),predict_list.numpy(),average="weighted")
                recall = recall_score(real_list.numpy(),predict_list.numpy(),average="weighted")

                f1_macro = f1_score(real_list.numpy(),predict_list.numpy(),average="macro")
                precision_macro = precision_score(real_list.numpy(),predict_list.numpy(),average="macro")
                recall_macro = recall_score(real_list.numpy(),predict_list.numpy(),average="macro")

                acc = float(corrects)/total
                print("---------------------Test set------------------------")
                print("epoch "+str((i+1)*gb_epoch-1)+" test loss: "+str(totalloss/total) +
                      ((" acc: "+str(float(corrects)/total)) if classification else ''))
                print(f'Weighted results: f1 : {f1_weighted:.4f}, precision : {precision:.4f}, recall : {recall:.4f}')
                print(f'Macro results: f1 : {f1_macro:.4f}, precision : {precision_macro:.4f}, recall : {recall_macro:.4f}')

        return (acc, f1_weighted, f1_macro)
    acc, f1_weighted, f1_macro = _trainprocess()
    return (acc, f1_weighted, f1_macro)

def single_test(model, test_dataloader, auprc=False, classification=True):
    """Run single test with model and test data loader.
    Args:
        model (nn.Module): Model to evaluate.
        test_dataloader (torch.utils.data.DataLoader): Test data loader
        auprc (bool, optional): Whether to return AUPRC scores or not. Defaults to False.
        classification (bool, optional): Whether to return classification accuracy or not. Defaults to True.
    Returns:
        dict: Dictionary of (metric, value) pairs
    """
    with torch.no_grad():
        totalloss = 0.0
        total = 0
        corrects = 0
        auprclist = []
        for j in test_dataloader:
            valid_x = [x.float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) for x in j[:-1]]
            valid_y = j[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            predicts = model(valid_x)
            blendloss = criterion(predicts, valid_y.squeeze())
            totalloss += blendloss*len(j[0])
            predictlist = predicts.tolist()
            for ii in range(len(j[0])):
                total += 1
                if predictlist[ii].index(max(predictlist[ii])) == valid_y[ii]:
                        corrects += 1
        print("test loss: "+str(totalloss/total) +
              ((" acc: "+str(float(corrects)/total)) if classification else ''))

        print('#######Accuracy :', float(corrects)/total)


if __name__ == '__main__':

    print("sigmoid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(torch.cuda.get_device_name(0))
    learning_rate = args.lr if not args.lr == None else 1e-5
    learning_rate_b = args.blr * learning_rate if not args.blr == None else learning_rate
    n_epochs = args.nepochs if not args.nepochs == None else 100
    num_forward_pass = args.nForwardPasses if not args.nForwardPasses == None else 1
    this_model = args.model if not args.model == None else "early"
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


    dataset_name = "rosmap"
    dataset = datasetConfig(dataset_name)
    test_acc = []
    batch_size = 64
    print(f"running the script with learning rate = {learning_rate}, bnn lr = {learning_rate_b}, #epcohs = {n_epochs},"
          f" #forward passes: {num_forward_pass}, current model {this_model}, dim {dim},"
          f" length {length}, split ratio {split_ratio}, functional dropout rate, {functional_p},"
          f" bnn dropout rate {bnn_p}")

    acc_list = []
    wf1_list = []
    mf1_list = []

    for fold in np.arange(20):
        
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
 
        encoders = [Unimodal(input_length=200, length=length, functional_p=functional_p), Unimodal(input_length=200, length=length, functional_p=functional_p), Unimodal(input_length=200, length=length, functional_p=functional_p)]
        mult_head = MLP(3*length, 2)
        uni_head = [MLP(length, 2), MLP(length, 2), MLP(length, 2)]
        fusion = Concat()

        acc, wf1, mf1 = train(encoders, mult_head, uni_head, fusion, train_loader, n_epochs,
      gb_epoch=5, optimtype=torch.optim.SGD, lr=learning_rate)
        acc_list.append(acc)
        wf1_list.append(wf1)
        mf1_list.append(mf1)

    print("acc mean: ", round(np.mean(acc_list),4), "  std: ", round( np.std(acc_list),4))
    print("wf1 mean: ", round(np.mean(wf1_list),4), "  std: ", round( np.std(wf1_list),4))
    print("mf1 mean: ", round(np.mean(mf1_list),4), "  std: ", round( np.std(mf1_list),4))

