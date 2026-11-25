import copy

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
#from torch.nn.functional import cross_entropy
from datasets_local import *
import random
from models import *
import os
import argparse
from tools import *
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
from optimizer import MetaSGD
from transformers import AutoImageProcessor, ViTModel
from datasets import load_dataset, Dataset,Image

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

def getImage(im, image_processor, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    with torch.no_grad():
        dataset = Dataset.from_dict({"image": ['.'+m for m in im]}).cast_column("image", Image())
        image = [image.convert('RGB') for image in dataset["image"]]
        #image = dataset["image"]
        inputs = image_processor(image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        features = torch.mean(outputs.last_hidden_state,dim=1).detach().clone()
    return features

def train(learning_rate, learning_rate_b):

    torch.autograd.set_detect_anomaly(True)
    bnn_loss_list_train = []
    functional_loss_list_train = []
    bnn_loss_list_val = []
    functional_loss_list_val = []

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model_vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_vit = model_vit.to(device)
    bcl = torch.nn.BCELoss()
    for epoch in range(n_epochs):
        model.train()
        meta_model_0.eval()
        #meta_model_1.eval()
        #meta_model_2.eval()
        functional_loss = 0.0
        bnn_loss = 0.0

        im_data, txt_data, y_data = train_dataset_data
        x_data = np.arange(txt_data.shape[0])
        train_index, meta_index, y_train, y_meta = train_test_split( x_data, y_data, stratify = y_data,
                                                    test_size=split_ratio)
        train_dataset = DatasetBERT(im_data[train_index], txt_data[train_index], y_data[train_index])
        train_meta_dataset = DatasetBERT(im_data[meta_index], txt_data[meta_index], y_data[meta_index])
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        train_meta_loader = DataLoader(dataset=train_meta_dataset, batch_size=math.floor(batch_size*split_ratio), drop_last=True, shuffle=True)
        meta_dataloader_iter = iter(train_meta_loader)
        meta_samples = 0

        for images, text, labels in train_loader:

            text = text.to(device)
            labels = labels.to(device)
            # copy weight from functional model

            meta_images, meta_text, meta_labels = next(meta_dataloader_iter)

            images = getImage(images, image_processor, model_vit).clone().detach() 
            meta_images = getImage(meta_images, image_processor, model_vit).clone().detach()

            meta_text = meta_text.to(device)
            meta_labels = meta_labels.to(device)

            meta_samples += meta_labels.size(0)

            model_hat = Early_fusion(length=length, functional_p=functional_p, isBnn=[True, True, False], num_f=num_first, num_sec = num_sec).to(device)
            model_hat.load_state_dict(model.state_dict())
            model_hat.train()
            # enable_dropout(meta_model_txt)
            # enable_dropout(meta_model_im)
            loss_hat = 0
            for k in range(num_forward_pass):
                output_hat = model_hat(images, text, [meta_model_0], [meta_model_cat,meta_model_cat2,meta_model_cat3])
                #loss_hat += cross_entropy(output_hat, labels)
                loss_hat += bcl(output_hat.float(), torch.unsqueeze(labels,1).float())
            #functional_loss += loss_hat
            loss_hat = loss_hat/num_forward_pass
            grads_hat = torch.autograd.grad(loss_hat, model_hat.parameters(),
                                         allow_unused=True)
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
            meta_model_cat2.train()
            meta_model_cat3.train()
            bcl_loss = torch.nn.BCELoss()
            loss = 0
            for k in range(num_forward_pass):
                # uncertainty_im_meta = meta_model_im(meta_images)
                # uncertainty_txt_meta = meta_model_txt(meta_text)

                output = model_hat(meta_images, meta_text, [meta_model_0], [meta_model_cat,meta_model_cat2,meta_model_cat3])
                # print(output_hat)
                #loss += cross_entropy(output, meta_labels)
                loss += bcl_loss(output.float(), torch.unsqueeze(meta_labels,1).float())
            loss = loss / num_forward_pass

            meta_loss = loss

            meta_optimizer_0.zero_grad()
            #meta_optimizer_1.zero_grad()
            #meta_optimizer_2.zero_grad()
            meta_loss.backward( retain_graph = True,inputs = list(meta_model_0.parameters()))
            #meta_loss.backward( retain_graph = True,inputs = list(meta_model_1.parameters()))
            #meta_loss.backward( retain_graph = True,inputs = list(meta_model_2.parameters()))

            meta_loss_cat = loss
            bnn_loss += meta_loss_cat
            meta_optimizer_cat.zero_grad()
            meta_loss_cat.backward(retain_graph = True, inputs = list(meta_model_cat.parameters()))
            meta_loss_cat.backward(retain_graph = True, inputs = list(meta_model_cat2.parameters()))
            meta_loss_cat.backward(retain_graph = True, inputs = list(meta_model_cat3.parameters()))
            meta_optimizer_0.step()
            #meta_optimizer_1.step()
            #meta_optimizer_2.step()
            meta_optimizer_cat.step()
            meta_optimizer_cat2.step()
            meta_optimizer_cat3.step()

            model.train()
            meta_model_0.eval()
            #meta_model_1.eval()
            #meta_model_2.eval()
            meta_model_cat.eval()
            meta_model_cat2.eval()
            meta_model_cat3.eval()
            loss = 0
            # uncertainty_im = meta_model_im(images)
            # uncertainty_txt = meta_model_txt(text)
            for i in range(num_forward_pass):
                output = model(images, text, [meta_model_0], [meta_model_cat,meta_model_cat2,meta_model_cat3])
                #loss += cross_entropy(output, labels)
                loss += bcl(output.float(), torch.unsqueeze(labels,1).float())
            loss = loss/num_forward_pass
            functional_loss += loss*labels.size()[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        print("#########################################################################################")
        print(f'epoch {epoch}: functional loss: {functional_loss/len(train_dataset):.3f}')
        functional_loss_list_train.append(functional_loss.cpu().detach().numpy()/len(train_dataset))
        bnn_loss_list_train.append(bnn_loss.cpu().detach().numpy()/meta_samples)

        val_loss= test(val_loader, 500, 99999, isVal=True)

        print(f"######## validation loss is {val_loss:.4f}")
        test_loss = test(test_loader, 500, 99999, isVal=False)

        global Min_loss
        global Num_loss


        #if val_loss < Min_loss:
        #   Min_loss = val_loss
        #   Num_loss = 0
        #   scheduler = 0
        #else:
        #  Num_loss += 1
        #  scheduler += 1

        if Num_loss > num_earlyStop:
           print("Have not improved for",num_earlyStop," epochs, stop training")
           break
        else:
           print("Have not improved from the min loss for ", Num_loss)

        #if scheduler == num_scheduler:
        if n_epochs%(epoch+1) == 8:
            learning_rate_b = learning_rate_b*0.003
            learning_rate = learning_rate*0.003

            for g in optimizer.param_groups:
                g['lr'] = learning_rate
            learning_rate_d = learning_rate_b
            for g in meta_optimizer_0.param_groups:
                g['lr'] = learning_rate_d
            #for g in meta_optimizer_1.param_groups:
            #    g['lr'] = learning_rate_d
            #for g in meta_optimizer_2.param_groups:
            #    g['lr'] = learning_rate_d
            #for g in meta_optimizer_txt.param_groups:
            #     g['lr'] = learning_rate_b/2.6
            learning_rate_c = learning_rate_b
            for g in meta_optimizer_cat.param_groups:
                g['lr'] = learning_rate_c
            for g in meta_optimizer_cat2.param_groups:
                g['lr'] = learning_rate_c
            for g in meta_optimizer_cat3.param_groups:
                g['lr'] = learning_rate_c
            scheduler = 0

            print("have not improved for",num_scheduler ," epochs, reduce learning rate to ", learning_rate," for functional & ", learning_rate_b," for bnns")

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
    meta_model_0.eval()
    #meta_model_1.eval()
    #meta_model_2.eval()
    meta_model_cat.eval()
    # enable_dropout(meta_model_txt)
    # enable_dropout(meta_model_im)
    bnn_loss = 0
    bcl = torch.nn.BCELoss()
    loss = 0
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model_vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_vit = model_vit.to(device)
    for images, text, labels in test_loader:

        text = text.to(device)
        labels = labels.to(device)
        images = getImage(images, image_processor, model_vit)
        with torch.no_grad():
            # uncertainty_im = meta_model_im(images)
            # uncertainty_txt = meta_model_txt(text)

            output = model(images, text, [meta_model_0], [meta_model_cat,meta_model_cat2,meta_model_cat3])
            #predict_score = predict_score + torch.nn.functional.softmax(output).numpy().tolist()
            #_, prediction = torch.max(output, 1)
            prediction = torch.round(output.cpu())
            n_samples += labels.size(0)
            #n_correct += (prediction == labels).sum().item()
            n_correct += (prediction == labels.cpu().unsqueeze(1)).sum().item()
            predict_list = torch.cat([predict_list, prediction.view(-1).cpu()])
            real_list = torch.cat([real_list, labels.view(-1).cpu()])
            #loss += cross_entropy(output, labels)*labels.size()[0]
            loss += bcl(output.float(), torch.unsqueeze(labels,1).float())*labels.size()[0]
            #meta_loss = criterion_im(output, labels)
            # print(output)
            #bnn_loss += meta_loss

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

    return loss/test_set_length


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
    if textLabel:
        label_label = "text"
    else:
        label_label = "image"

    dataset_name = "crisismmd1"
    dataset = datasetConfig(dataset_name)
    dataset2 = "../datasets/crisismmd_no_contradict/Roberta_last4/"
    test_acc = []
    batch_size = 6
    print(f"running the script with learning rate = {learning_rate}, bnn lr = {learning_rate_b}, #epcohs = {n_epochs},"
          f" #forward passes: {num_forward_pass}, current model {this_model}, dim {dim},"
          f" length {length}, split ratio {split_ratio}, functional dropout rate, {functional_p},"
          f" bnn dropout rate {bnn_p}")

    results_list_acc = []
    results_list_f1 = []

    best_epoch = []
    best_accs = []
    fold=4
    for i in np.arange(5):

        if i!= 0:
            continue

        SEED = 999
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        txt_dim = 40
        im_dim = 768

        print("FOLD ",fold)

        txt_dir = dataset + "train_text_embed.npy"
        im_dir = dataset + "train_img_embed.npy"
        y_dir = dataset + "train_labels_embed.npy"

        im_data = np.load(im_dir, allow_pickle=True)
        txt_data = np.load(txt_dir, allow_pickle=True)
        y_data = np.int64(np.load(y_dir, allow_pickle=True))

        train_dataset = DatasetBERT(im_data, txt_data, y_data)
        train_dataset_data = (im_data, txt_data, y_data)

        model = Early_fusion(length=length, functional_p=functional_p, isBnn=[True, True, False], num_f=num_first, num_sec = num_sec).to(device)
        meta_model_0 = Meta_net(bnn_p=bnn_p,dims = [768,768], length = num_first*length).to(device)
        #meta_model_1 = Meta_net_static(bnn_p=bnn_p,dims = [4*length,4*length], length = 2*length).to(device)
        #meta_model_2 = Meta_net_static(bnn_p=bnn_p,dims = [2*length,2*length], length = length).to(device)
        # meta_model_txt = Meta_net(input_length=96, length=length, bnn_p=bnn_p).to(device)
        meta_model_cat = Meta_net_cat(length=num_first*length, bnn_p=bnn_p).to(device)
        meta_model_cat2 = Meta_net_cat(length=num_sec*length, bnn_p=bnn_p).to(device)
        meta_model_cat3 = Meta_net_cat(length=length, bnn_p=bnn_p).to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate, momentum=0.9, weight_decay=1e-3)
        meta_optimizer_0 = torch.optim.SGD(meta_model_0.parameters(), lr= learning_rate_b, momentum=0.9, weight_decay=1e-3)
        #meta_optimizer_1 = torch.optim.SGD(meta_model_1.parameters(), lr= learning_rate_b/2.6, momentum=0.9, weight_decay=1e-3)
        #meta_optimizer_2 = torch.optim.SGD(meta_model_2.parameters(), lr= learning_rate_b/2.6, momentum=0.9, weight_decay=1e-3)
        # meta_optimizer_txt = torch.optim.SGD(meta_model_txt.parameters(), lr= learning_rate_b, momentum=0.9, weight_decay=1e-3)
        meta_optimizer_cat = torch.optim.SGD(meta_model_cat.parameters(), lr= learning_rate_b, momentum=0.9, weight_decay=1e-3)
        meta_optimizer_cat2 = torch.optim.SGD(meta_model_cat2.parameters(), lr= learning_rate_b, momentum=0.9, weight_decay=1e-3)
        meta_optimizer_cat3 = torch.optim.SGD(meta_model_cat3.parameters(), lr= learning_rate_b, momentum=0.9, weight_decay=1e-3)

        criterion_im = nn.BCELoss(reduction='sum')
        criterion_txt = nn.BCELoss(reduction='sum')
        criterion_cat = nn.BCELoss(reduction='sum')

        dev_txt_dir = dataset+'dev_text_embed.npy'
        dev_y_dir = dataset+'dev_labels.npy'
        dev_im_dir = dataset+'dev_img.npy'


        im_data = np.load(dev_im_dir, allow_pickle=True)
        txt_data = np.load(dev_txt_dir, allow_pickle=True)
        y_data = np.int64(np.load(dev_y_dir, allow_pickle=True))

        val_dataset = DatasetBERT(im_data, txt_data, y_data)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle= True)

        txt_dir_te = dataset + "test_text_embed.npy"
        im_dir_te = dataset + "test_img.npy"
        y_dir_te = dataset + "test_labels.npy"

        im_data = np.load(im_dir_te, allow_pickle=True)
        txt_data = np.load(txt_dir_te, allow_pickle=True)
        y_data = np.int64(np.load(y_dir_te, allow_pickle=True))

        test_dataset = DatasetBERT(im_data, txt_data, y_data)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle= True)

        train(learning_rate,learning_rate_b)

