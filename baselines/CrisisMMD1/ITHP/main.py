from collections import Counter
from torch.utils.data import DataLoader
from datasets_local import *
from models import *
import random
import os
import argparse
from tools import *
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def l2_reg(model):
    l2_lambda = 1e-3
    l2_reg = torch.tensor(0.)
    for param in model.parameters():
        l2_reg += torch.norm(param, 2)
    return l2_lambda * l2_reg

def train(fold, model, criterion, optimizer, n_epochs, dataset_len, scheduler):
    train_loss = []
    val_loss_hold = []
    train_acc = []
    val_acc = []
    train_ole_loss = []
    min_loss = 0.0
    patience = 200
    not_improving_epochs = 0
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        running_ole_loss = 0.0
        if epoch== n_epochs-1:
            running_outputs = []
            running_outputs_labels = []
        running_corrects = 0

        predict_list = torch.zeros(0)
        real_list = torch.zeros(0)

        for images, text,labels in train_loader:
            images = images
            text = text.to(device)
            outputs, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1  = model(images, text)
            #_, preds = torch.max(outputs[0].cpu(), 1)
            preds = torch.round(torch.sigmoid(outputs.cpu()))

            running_corrects += (preds == labels.unsqueeze(1)).sum()
            #running_corrects += (preds == labels).sum()

            loss = 0
            this_ole_loss = 0.0
            loss += criteria(outputs.cpu().float(), labels.unsqueeze(1).float())
            loss += 2 / (p_beta + p_gamma) * IB_loss.cpu()
            #loss += criteria(outputs[idx].cpu().float(), labels)

            # loss+= torch.sqrt(loss)
            #loss += l2_reg(model)

            #l2_lambda = 1e-5
            #l2_reg = torch.tensor(0.)
            #for name, param in model.named_parameters():
            #    if name in ['im_inter_agg2.weight', 'txt_inter_agg2.weight']:
            #       l2_reg += torch.norm(param, 2)
            #loss += l2_lambda * l2_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * text.size(0)

            if epoch== n_epochs-1:
                running_outputs.extend(outputs.cpu().detach().numpy().tolist())
                running_outputs_labels.extend(labels.detach().numpy().tolist())

            predict_list = torch.cat([predict_list, preds.view(-1)])
            real_list = torch.cat([real_list, labels.view(-1)])

        print(optimizer.state_dict()['param_groups'][0]['lr'])

        running_loss = running_loss /dataset_len["train"]

        acc = running_corrects*1.0/dataset_len['train']
        f1_macro = f1_score(real_list.detach().numpy(),predict_list.detach().numpy(),average="macro")
        f1_micro = f1_score(real_list.detach().numpy(),predict_list.detach().numpy(),average="micro")

        print(f'epoch {epoch+1} / {n_epochs}, loss:{running_loss:.3f}, acc: {acc:.3f}, f1 macro: {f1_macro:.3f}, f1 micro: {f1_micro:.3f},ole loss: {running_ole_loss:.3f}')

        train_loss.append(running_loss)
        train_acc.append(acc)
        train_ole_loss.append(running_ole_loss)
        # writer_train.add_scalar("loss", running_loss, epoch+1)
        # writer_train.add_scalar("accuracy", acc, epoch+1)
        #
        # writer_train.add_scalar("trainingLoss", running_loss, epoch+1)
        # writer_loss_ole.add_scalar("trainingLoss", running_ole_loss, epoch+1)

        # torch.save({
        #     'epoch':epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict()
        # }, os.path.join(checkpoint_path, str(epoch)+".pth"))

        model.eval()
        with torch.no_grad():
            val_corrects = 0
            val_loss = 0.0
            predict_list = torch.zeros(0)
            real_list = torch.zeros(0)
            for  images,text,labels in val_loader:

                images = images
                text = text.to(device)
                outputs, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1  = model(images, text)
                #_, preds = torch.max(outputs[0].cpu(), 1)
                preds = torch.round(torch.sigmoid(outputs.cpu()))

                loss = 0
                loss += criteria(outputs.cpu().float(), labels.unsqueeze(1).float())
                loss += 2 / (p_beta + p_gamma) * IB_loss.cpu()
                #loss += criteria(outputs[idx].cpu(), labels)
                
                #l2_reg = torch.tensor(0.)
                #for name, param in model.named_parameters():
                #    if name in ['im_inter_agg2.weight', 'txt_inter_agg2.weight']:
                #       l2_reg += torch.norm(param, 2)
                #loss += l2_lambda * l2_reg
                       
                val_loss += loss.item() * text.size(0)
                val_corrects += (preds == labels.unsqueeze(1)).sum()
                predict_list = torch.cat([predict_list, preds.view(-1)])
                real_list = torch.cat([real_list, labels.view(-1)])

            val_loss = val_loss / dataset_len['val']
            acc =  val_corrects*1.0 / dataset_len['val']
            f1_macro = f1_score(real_list.detach().numpy(),predict_list.detach().numpy(),average="macro")
            f1_micro = f1_score(real_list.detach().numpy(),predict_list.detach().numpy(),average="micro")
            print(f'val for {epoch+1} / {n_epochs}, loss:{val_loss:.3f}, acc: {acc:.3f}, f1 macro: {f1_macro:.3f}, f1 micro: {f1_micro:.3f}')
            print(confusion_matrix(predict_list.detach().numpy(), real_list.detach().numpy()))
            val_loss_hold.append(val_loss)
            val_acc.append(acc)
            # writer_test.add_scalar("loss", val_loss, epoch+1)
            # writer_test.add_scalar("accuracy", acc, epoch+1)
        scheduler.step(val_loss)
        #scheduler.step()
        if epoch == 0 or val_loss< min_loss:
            min_loss = val_loss
            not_improving_epochs = 0
        else:
            not_improving_epochs += 1

        if not_improving_epochs > patience:
            print("loss not improving for ", patience, "epochs, early stop applied!")
            break

        model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            predict_list = torch.zeros(0)
            real_list = torch.zeros(0)

            for images,text,labels in test_loader:
                images = images
                text = text.to(device)
                outputs, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1  = model(images, text)

                # checkpoint = torch.load("./ole/mvsa_single_0.0001_concat/checkpoints/89.pth")
                # model.load_state_dict(checkpoint['model_state_dict'])
                # max returns (value ,index)
                #_, predicted = torch.max(outputs[0].cpu(), 1)
                predicted = torch.round(torch.sigmoid(outputs.cpu()))
                n_samples += labels.size(0)
                n_correct += (predicted == labels.unsqueeze(1)).sum().item()
                #n_correct += (predicted == labels).sum().item()
                predict_list = torch.cat([predict_list, predicted.view(-1)])
                real_list = torch.cat([real_list, labels.view(-1)])
            print("predict_list.numpy ",predict_list.numpy())
            print("real_list.numpy ",real_list.numpy())
            print(confusion_matrix(predict_list.numpy(), real_list.numpy()))
            acc = 100.0 * n_correct / n_samples
            f1_macro = f1_score(real_list.detach().numpy(),predict_list.detach().numpy(),average="macro")
            f1_micro = f1_score(real_list.detach().numpy(),predict_list.detach().numpy(),average="micro")
            print(f'Accuracy of the network on test images: {acc:.3f} %')
            print("f1_macro ", f1_macro, " f1_micro ",f1_micro)

            #scheduler += 1
            #if scheduler == num_scheduler:
            #   learning_rate = learning_rate*0.1
            #   for g in optimizer.param_groups:
            #       g['lr'] = learning_rate
            #   print("Reduce learning rate to ",learning_rate)
            #   scheduler = 0
    #torch.save({
    #            'epoch':epoch,
    #            'model_state_dict': model.state_dict(),
    #            'optimizer_state_dict': optimizer.state_dict()
    #        }, os.path.join(checkpoint_path, str(fold) +"_"+str(epoch)+".pth"))

    #np.save(file_path+"/train_loss_"+str(fold), np.array(train_loss))
    #np.save(file_path+"/val_loss_hold_"+str(fold), np.array(val_loss_hold))
    #np.save(file_path+"/train_acc_"+str(fold), np.array(train_acc))
    #np.save(file_path+"/val_acc_"+str(fold), np.array(val_acc))

    #drawGraph([np.array(train_loss), np.array(val_loss_hold)],
    #          title = this_model, y_label= "loss", legend_name = ["train_all", "test_all"], save_path=file_path+"model_loss_"+str(fold))

    #drawGraph([np.array(train_acc), np.array(val_acc)],
    #          title = this_model, y_label= "accuracy", legend_name = ["train", "test"], save_path=file_path+"model_acc_"+str(fold))
    return model

def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        predict_list = torch.zeros(0)
        real_list = torch.zeros(0)
        
        logits = []

        for images,text,labels in test_loader:
            images = images
            text = text.to(device)
            output, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1  = model(images, text)
            # checkpoint = torch.load("./ole/mvsa_single_0.0001_concat/checkpoints/89.pth")
            # model.load_state_dict(checkpoint['model_state_dict'])
            # max returns (value ,index)
            #_, predicted = torch.max(outputs[0].cpu(), 1)
            predicted = torch.round(torch.sigmoid(outputs[0].cpu()))
            n_samples += labels.size(0)
            n_correct += (predicted == labels.unsqueeze(1)).sum().item()
            #n_correct += (predicted == labels).sum().item()
            predict_list = torch.cat([predict_list, predicted.view(-1)])
            real_list = torch.cat([real_list, labels.view(-1)])
            logits+=torch.softmax(outputs.cpu(), dim=1).tolist()
    macro_auc = roc_auc_score(real_list, logits, multi_class='ovr')
    print(confusion_matrix(predict_list.detach().numpy(), real_list.detach().numpy()))
    f1_macro = f1_score(real_list.detach().numpy(),predict_list.detach().numpy(),average="macro")
    f1_micro = f1_score(real_list.detach().numpy(),predict_list.detach().numpy(),average="micro")
    f1_weighted = f1_score(real_list.detach().numpy(),predict_list.detach().numpy(),average="weighted")
    print(f"f1 macro : {f1_macro:.4f}")
    print(f"f1 micro : {f1_micro:.4f}")
    print(f"f1 weighted : {f1_weighted:.4f}")
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on test images: {acc:.3f} %')
    return {"acc":acc, "f1": f1_weighted, "mf1": f1_macro, "mauc":macro_auc}

scheduler = 0

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--nepochs', type=int)
    # parser.add_argument('--lambda_', type=float)
    parser.add_argument('--model', type=str)
    parser.add_argument('--which_gate', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--gate', default=False, action='store_true')
    parser.add_argument('--aoa', default=False, action='store_true')
    parser.add_argument('--pregate', default=False, action='store_true')
    parser.add_argument('--functionalp', type=float)
    parser.add_argument('--p2', type=float)
    parser.add_argument('--length', type=int)
    parser.add_argument('--scheduler', type=int)
    parser.add_argument('--p_beta', type=float)
    parser.add_argument('--p_gamma', type=float)
    parser.add_argument('--p_lambda', type=float)
    args = parser.parse_args()
    p_gamma = args.p_gamma if not args.p_gamma == None else 8
    p_beta = args.p_beta if not args.p_beta == None else 32
    p_lambda = args.p_lambda if not args.p_lambda == None else 0.3

    learning_rate = args.lr if not args.lr == None else 1e-3
    n_epochs = args.nepochs if not args.nepochs == None else 75
    num_scheduler = args.scheduler if not args.scheduler == None else 10
    # this_lambda = args.lambda_ if not args.lambda_ == None else 0
    this_model = args.model if not args.model == None else "concat"
    which_gate = args.which_gate if not args.which_gate == None else "im"
    is_gate = args.gate
    is_aoa = args.aoa
    is_pregate = args.pregate
    dataset = datasetConfig("crisismmd1")
    functionalp = args.functionalp if not args.functionalp == None else 0.2
    p2 = args.p2 if not args.p2 == None else 0.2
    length = args.length if not args.length == None else 30
    
    test_acc = []
    batch_size = 128
    print(f"running the script with learning rate = {learning_rate}, #epcohs = {n_epochs}, model = {this_model}")

    #file_path = "./opl/lr_"+ str(learning_rate) + "_nepochs_" + str(n_epochs) + "_model_" + this_model + "/"
    #if not os.path.exists(file_path):
    #  os.makedirs(file_path)

    #checkpoint_path = "checkpoints/"
    #checkpoint_path = file_path + checkpoint_path
    #if not os.path.exists(checkpoint_path):
    #  os.makedirs(checkpoint_path)

    num_of_fold =1

    results_list_acc = []
    results_list_f1 = []
    results_list_mf1 =[]
    results_list_mauc =[]

    SEED = 999
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    fold = 4
    for i in range(num_of_fold):

        #if i!=0:
        #   continue

        print("#####################################this is the ",fold,"'s fold!##############################################")

        txt_dir = dataset + "train_text_embed.npy"
        im_dir = dataset + "train_img_embed.npy"
        y_dir = dataset + "train_labels_embed.npy"

        im_data = np.load(im_dir, allow_pickle=True)
        txt_data = np.load(txt_dir, allow_pickle=True)
        y_data = np.int64(np.load(y_dir, allow_pickle=True))

        train_dataset = DatasetBERT(im_data, txt_data, y_data)
        #train_dataset_data = (im_data, txt_data, y_data)
        train_loader = DataLoader(dataset= train_dataset, batch_size = batch_size, drop_last=True, shuffle=True)


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

        dataset_len = {"train": len(train_dataset),'val': len(val_dataset)}
        print(dataset_len)
         
        txt_dim=40
        model = IB(length= length, functional_p=functionalp, p_gamma = p_gamma, p_beta = p_beta, p_lambda = p_lambda).to(device)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        train_labels = [label for _,_,label in train_dataset]
        train_dist = dict(Counter(train_labels))
        print(train_dist)
        class_weights = torch.FloatTensor(weight_caculator(train_dist, "isns"))
        print(class_weights)

        criteria = nn.BCEWithLogitsLoss()
        #criterion = [nn.CrossEntropyLoss()]
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        #optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=num_scheduler ,mode='min',factor=0.1)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer,num_scheduler)
        model = train(fold, model, criteria, optimizer, n_epochs, dataset_len,
          scheduler)
        #torch.cuda.empty_cache()

        print("Ready to test the model...............")

        results = test(model, test_loader)
        results_list_acc.append(results['acc'])
        results_list_f1.append(results['f1'])
        results_list_mf1.append(results['mf1'])
        results_list_mauc.append(results['mauc'])
        torch.cuda.empty_cache()
    print("###################################################")
    print(results_list_acc)
    print("acc avg: ", np.mean(results_list_acc))
    print("acc std: ", np.std(results_list_acc))
    print(results_list_f1)
    print("wf1 avg: ", np.mean(results_list_f1))
    print("wf1 std: ", np.std(results_list_f1))
    print(results_list_mf1)
    print("mf1 avg: ", np.mean(results_list_mf1))
    print("mf1 std: ", np.std(results_list_mf1))
    print(results_list_mauc)
    print("mauc avg: ", np.mean(results_list_mauc))
    print("mauc std: ", np.std(results_list_mauc))
    # for key,value in results.items():
    #      print(key," :")
    #      avg_results = []
    #      for jj in results_list:
    #          avg_results.append(jj[key])
    #      print("avg is ", sum(avg_results)/len(avg_results))
    #      print("std is ", np.std(avg_results))
    # xs= np.arange(0, len(test_acc), 1 )
    # plt.plot(np.array(test_acc))
    # plt.title("Influence of the weight metirc "+this_model)
    # plt.xlabel("lambda")
    # plt.ylabel("test accuracy")
    # plt.legend(["test_acc"], loc='upper left')
    # for x,y in zip(xs,test_acc):
    #     label = "{:.2f}".format(y)
    #     plt.annotate(label, # this is the text
    #                  (x,y), # this is the point to label
    #                  textcoords="offset points", # how to position the text
    #                  xytext=(0,10), # distance from text to points (x,y)
    #                  ha='center') # horizontal alignment can be left, right or center
    # plt.savefig(this_model + "_lambda")
    # plt.show()
    # plt.clf()
