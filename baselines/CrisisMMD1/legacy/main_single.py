import copy

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from datasets_local import *
import random
from single import *
import os
import argparse
from tools import *
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
from optimizer import MetaSGD
from transformers import AutoImageProcessor, ViTModel
from datasets import load_dataset, Dataset,Image
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
    im_data, txt_data, y_data = train_dataset_data
    train_dataset = DatasetBERT(im_data, txt_data, y_data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        functional_loss = 0.0
        bnn_loss = 0.0

        for images, text, labels in train_loader:

            text = text.to(device)
            labels = labels.to(device)
            # copy weight from functional model


            images = getImage(images, image_processor, model_vit).clone().detach() 

            model.train()
            loss = 0
            output = model(images, text)
            loss = cross_entropy(output, labels)
            functional_loss += loss*labels.size()[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("#########################################################################################")
        print(f'epoch {epoch}: functional loss: {functional_loss/len(train_dataset):.3f}')
        functional_loss_list_train.append(functional_loss.cpu().detach().numpy()/len(train_dataset))

        val_loss= test(val_loader, 500, 99999, isVal=True)

        print(f"######## validation loss is {val_loss:.4f}")
        test_loss = test(test_loader, 500, 99999, isVal=False)

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
           print("Have not improved for",num_earlyStop," epochs, stop training")
           break
        else:
           print("Have not improved from the min loss for ", Num_loss)

        if scheduler == num_scheduler:
        #print("epoch + 1 % n_epochs ", (epoch+1)%n_epochs)
        #if (epoch+1)%n_epochs == 0:
            learning_rate_b = learning_rate_b*0.1
            learning_rate = learning_rate*0.1

            for g in optimizer.param_groups:
                g['lr'] = learning_rate
            learning_rate_d = learning_rate_b
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
    bnn_loss = 0
    loss = 0
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model_vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_vit = model_vit.to(device)
    im_list = []
    pred_list = []
    ground_list = []
    for images, text, labels in test_loader:

        text = text.to(device)
        labels = labels.to(device)
        im_list.append(images)
        images = getImage(images, image_processor, model_vit)
        with torch.no_grad():

            output = model(images, text)
            #predict_score = predict_score + torch.nn.functional.softmax(output).numpy().tolist()
            _, prediction = torch.max(output, 1)
            pred_list.append(prediction)
            ground_list.append(labels)

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
    batch_size = 16
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

        if i!= fold:
            continue

        SEED = 2988
        #SEED = np.random.randint(1, 10000 + 1)
        print("SEED IS @@@@@@@@@ ",SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        txt_dim = 40
        im_dim = 768

        print("##################################FOLD ",fold)

        print(dataset)

        txt_dir = dataset + "train_text_embed.npy"
        im_dir = dataset + "train_img_embed.npy"
        y_dir = dataset + "train_labels_embed.npy"

        im_data = np.load(im_dir, allow_pickle=True)
        txt_data = np.load(txt_dir, allow_pickle=True)
        y_data = np.int64(np.load(y_dir, allow_pickle=True))

        train_dataset_data = (im_data, txt_data, y_data)

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
        model = Early_fusion()
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate, momentum=0.9, weight_decay=1e-3) 
        train(learning_rate,learning_rate_b)

