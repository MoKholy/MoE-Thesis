import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scripts.visualization import plot_all_losses, plot_train_validation_loss, plot_classification_report, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from tensorboardX import SummaryWriter  
from scripts.models import MixtureOfExperts
from scripts.losses import WeightedCrossEntropyLoss, FocalLoss, MSEGatingLoss
from scripts.dataset import GatingDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import numpy as np
import os 

# define function set seed for random, torch and numpy for reproduction
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# define command line parser function
def argparser():
    parser = argparse.ArgumentParser(description='Mixture of Experts Training')
    # add general arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    # add dataset arguments
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of dataset to use')
    parser.add_argument('--dataset_path', type=str, default='data/processed/', help='Path to dataset')
    parser.add_argument('--standardize', type=bool, default=True, help='Standardize dataset')
    parser.add_argument('--train_size', type=float, default=0.8, help='Train size')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation size')
    parser.add_argument('--test_size', type=float, default=0.1, help='Test size')
    # add dataloader arguments
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    # add model arguments
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension for experts')
    parser.add_argument('--num_experts', type=int, default=-1, help='Number of experts')
    parser.add_argument('--num_classes', type=int, help='Number of classes')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--activation_fn', type=str, default='relu', help='Activation function for experts')
    ## add training arguments     
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization)')
    
    # add loss function arguments
    parser.add_argument('--expert_loss_fn', choices=['ce', 'focal_loss', 'wce'], default='ce', help='Expert loss function')
    # add optimizer arguments
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam', help='Optimizer choice')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--nesterov', type=bool, default=True, help='Nesterov for SGD optimizer')
    parser.add_argument('--betas', nargs='+', type=float, default=[0.9, 0.999], help='Betas for Adam optimizer')
    # add save arguments 
    parser.add_argument('--save_dir', type=str, default='', help='Directory to save everything')
    parser.add_argument('--model_name', type=str, default='', help='Name of model')
    
    # add scheduler arguments
    parser.add_argument('--scheduler', choices=["none", 'lr_step', 'plateau'], default="none", help='Learning rate scheduler')
    parser.add_argument('--scheduler_step_size', type=int, default=20, help='Step size for lr_step scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for lr_step scheduler')
    # parser.add_argument('--patience', type=int, default=5, help='Patience for plateau scheduler')
    # parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for plateau scheduler')
    
    # eval mode arguments
    parser.add_argument('--eval_mode', type=bool, default=False, help='Evaluate model from path')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model to evaluate')
    return parser.parse_args()

# evaluate function #TODO update this function
def evaluate(model, expert_loss_fn, gating_loss_fn, dataloader, device, labels_list, model_name, save_dir):
    # set model to eval
    model.eval()
    total_gating_loss = 0.0
    total_expert_loss = 0.0
    all_preds = []
    all_labels = []
    all_pred_probs = []
    # move model to device
    model.to(device)
    
    with torch.no_grad():
        for input, true_gating_labels, labels in tqdm(dataloader, desc="Test", leave=False):
            
            # move to device 
            input, true_gating_labels, labels = input.to(device), true_gating_labels.to(device), labels.to(device)
            
            mixture_out, gating_out, expert_out = model(input)
            
            if labels.dim() == 2:
                labels = torch.squeeze(labels, dim=1)
        
            # get gating loss
            gate_loss = gating_loss_fn(gating_out, true_gating_labels.float())
            total_gating_loss += gate_loss.item()
            # get expert loss
            exprt_loss = expert_loss_fn(mixture_out, labels)
            total_expert_loss += exprt_loss.item()
            
            # calculate predictions for accuracy and F1 score
            _, preds = torch.max(mixture_out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_pred_probs.extend(nn.functional.softmax(mixture_out, dim=0).cpu().tolist())
            
    # set plot save dir
    plot_save_dir = os.path.join(save_dir, "plots", "test")
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)
    # get y_score for roc curve and precision recall curve
    y_score = np.array(all_pred_probs)
    
    avg_modes = ["micro", "macro", "weighted"]
    # Generate confusion matrix
    plot_confusion_matrix(y_true = all_preds, 
                          y_pred = all_labels, 
                          labels=labels_list, 
                          save=True, 
                          save_dir = plot_save_dir,
                          filename=f"{model_name}_confusion_matrix.png")
    
    # Generate ROC curve
    for avg_mode in avg_modes:
        plot_roc_curve(y_test = all_labels,
                    y_score = y_score,
                    labels=labels_list,
                    save=True,
                    avg_mode = avg_mode,
                    save_dir = plot_save_dir,
                    filename=f"{model_name}_{avg_mode}_roc_curve.png")

    # Generate precision recall curve
    plot_precision_recall_curve(y_test = all_labels,
                                y_score = y_score,
                                labels=labels_list,
                                save=True,
                                save_dir = plot_save_dir,
                                filename=f"{model_name}_precision_recall_curve.png")
    
    # get classification report
    plot_classification_report(y_test = all_labels,
                               y_pred = all_preds,
                               labels = labels_list,
                               save=True, 
                               save_dir = plot_save_dir,
                               filename=f"{model_name}_classification_report.png")
                               

# Training loop function
def train(args, model, expert_loss_fn, gating_loss_fn, optimizer, scheduler, dataloader, val_dataloader, device, log_dir, save_dir):
    
    best_f1_score = 0.0
    best_model = None
    writer = SummaryWriter(log_dir=log_dir)
    model.to(device)
    
    # collect gating, expert and total loss for each epoch for train and validation
    train_gating_loss = []
    train_expert_loss = []
    train_total_loss = []
    val_gating_loss = []
    val_expert_loss = []
    val_total_loss = []
    for epoch in tqdm(range(args.num_epochs), desc="Epochs"):
        
        ########  training  ########
        
        # set model to train
        model.train()
        # track losses, predictions and labels
        total_expert_loss = 0.0
        total_gating_loss = 0.0
        all_preds = []
        all_labels = []
        # loop over data from dataloader
        for input, true_gating_labels, labels in tqdm(dataloader, desc="Batches", leave=False):
            # get data to device
            input = input.to(device)
            true_gating_labels = true_gating_labels.to(device)
            labels = labels.to(device)

            # check if labels are 2D
            if labels.dim() == 2:
                labels = torch.squeeze(labels, dim=1)
            
            # get output from model
            mixture_out, gating_out, expert_out = model(input)

            # get losses
            gate_loss = gating_loss_fn(gating_out, true_gating_labels.float())
            total_gating_loss += gate_loss.item()
            exprt_loss = expert_loss_fn(mixture_out, labels)
            total_expert_loss += exprt_loss.item()
            total_loss = gate_loss + exprt_loss
            # zero gradients
            optimizer.zero_grad()
            # backpropagate
            total_loss.backward()
            # update weights
            optimizer.step()
            # update scheduler
            if scheduler is not None:
                scheduler.step()

            # calculate predictions for accuracy and F1 score
            _, preds = torch.max(mixture_out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # calculate average losses for epoch
        avg_expert_loss = total_expert_loss / len(dataloader)
        avg_gating_loss = total_gating_loss / len(dataloader)

        # append losses to list
        train_gating_loss.append(avg_gating_loss)
        train_expert_loss.append(avg_expert_loss)
        train_total_loss.append(avg_gating_loss + avg_expert_loss)
        # calculate accuracy and F1 score
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        # write to tensorboard
        writer.add_scalar("Avg_Train_Expert_Loss", avg_expert_loss, epoch)
        writer.add_scalar("Avg_Train_Gating_Loss", avg_gating_loss, epoch)
        writer.add_scalar("Train_Accuracy", accuracy, epoch)
        writer.add_scalar("Train_F1_Score", f1, epoch)

        ########  validation  ########

        # set model to eval
        model.eval()

        # track losses, predictions and labels for validation
        total_expert_loss_val = 0.0
        total_gating_loss_val = 0.0
        all_preds_val = []
        all_labels_val = []

        # loop over data from dataloader
        for input, true_gating_labels, labels in tqdm(val_dataloader, desc="Validation", leave=False):
            # get data to device
            input = input.to(device)
            true_gating_labels = true_gating_labels.to(device)
            labels = labels.to(device)

            if labels.dim() == 2:
                labels = torch.squeeze(labels, dim=1)
            # get output from model
            mixture_out, gating_out, expert_out = model(input)
            # get gating loss
            gate_loss = gating_loss_fn(gating_out, true_gating_labels.float())
            total_gating_loss_val += gate_loss.item()
            # get expert loss
            exprt_loss = expert_loss_fn(mixture_out, labels)
            total_expert_loss_val += exprt_loss.item()

            # calculate predictions for accuracy and F1 score
            _, preds = torch.max(mixture_out, dim=1)
            all_preds_val.extend(preds.cpu().numpy())
            all_labels_val.extend(labels.cpu().numpy())

        # calculate average losses for epoch
        avg_expert_loss_val = total_expert_loss_val / len(val_dataloader)
        avg_gating_loss_val = total_gating_loss_val / len(val_dataloader)

        # append losses to list
        val_gating_loss.append(avg_gating_loss_val)
        val_expert_loss.append(avg_expert_loss_val)
        val_total_loss.append(avg_gating_loss_val + avg_expert_loss_val)
        
        # calculate accuracy and F1 score
        accuracy_val = accuracy_score(all_labels_val, all_preds_val)
        f1_val = f1_score(all_labels_val, all_preds_val, average="macro")

        # write to tensorboard
        writer.add_scalar("Avg_Val_Expert_Loss", avg_expert_loss_val, epoch)
        writer.add_scalar("Avg_Val_Gating_Loss", avg_gating_loss_val, epoch)
        writer.add_scalar("Val_Accuracy", accuracy_val, epoch)
        writer.add_scalar("Val_F1_Score", f1_val, epoch)

        # check if F1 score is best
        if f1_val > best_f1_score:
            # save model
            # torch.save(model.state_dict(), "best_model.pt")
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            best_model = model
            # update best F1 score
            best_f1_score = f1
        
    writer.close()
    
    # train plots save loc
    plot_save_dir = os.path.join(save_dir, "plots", "train")
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)
    
    
    # plot train and validation losses
    plot_train_validation_loss(train_loss=train_total_loss,
                               validation_loss=val_total_loss,
                               save=True,
                               save_dir=plot_save_dir,
                               filename="train_validation_total_loss.png"
                               )
    plot_train_validation_loss(train_loss=train_expert_loss,
                               validation_loss=val_expert_loss,
                               save=True,
                               save_dir=plot_save_dir,
                               filename="train_validation_expert_loss.png"
                                 )
    plot_train_validation_loss(train_loss=train_gating_loss,
                               validation_loss=val_gating_loss,
                               save=True,
                               save_dir=plot_save_dir,
                               filename="train_validation_gating_loss.png"
                               )
    
    plot_all_losses(train_total_loss=train_total_loss,
                    val_total_loss=val_total_loss,
                    train_gating_loss=train_gating_loss,
                    val_gating_loss=val_gating_loss,
                    train_expert_loss=train_expert_loss,
                    val_expert_loss=val_expert_loss,
                    save=True,
                    save_dir=plot_save_dir,
                    filename="all_losses.png"
                    )
                    
    # return best model
    return best_model

# main function

if __name__ == "__main__":

    #parse arguments from command line
    args = argparser()
    
    # print arguments
    print(f"args : {args}")
     
    # set seed for reproducibility
    set_seed(args.seed)
    
    print(f"set seed successfully to {args.seed}")
    # set log directory 
    log_dir = os.path.join(args.save_dir, args.dataset_name, args.model_name, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # print log directory
    print(f"log_dir : {log_dir}")
    
    # set save directory
    save_dir = os.path.join(args.save_dir, args.dataset_name, args.model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"save_dir : {save_dir}")
    
    # load datasets and dataloaders
    dataset_path = args.dataset_path
    
    if not args.eval_mode:
        train_dataset = GatingDataset(dataset_name=args.dataset_name, dataset_path=dataset_path, seed=args.seed, split="train", transform=args.standardize)
        val_dataset = GatingDataset(dataset_name=args.dataset_name, dataset_path=dataset_path, seed=args.seed, split="val", transform=args.standardize)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"train dataset size : {len(train_dataset)}")
        print(f"val dataset size : {len(val_dataset)}")
        
    test_dataset = GatingDataset(dataset_name=args.dataset_name, dataset_path=dataset_path, seed=args.seed, split="test", transform=args.standardize)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"test dataset size : {len(test_dataset)}")
    # get some info from dataset
    
    if args.eval_mode:
        # get input dims from test dataset
        input_dim = test_dataset.get_input_dim()
        num_classes = test_dataset.get_num_classes()
        mapping = test_dataset.get_mapping()
        inv_mapping = test_dataset.get_inv_mapping()
    else:
        # get input dims from train dataset
        input_dim = train_dataset.get_input_dim()
        num_classes = train_dataset.get_num_classes()
        mapping = train_dataset.get_mapping()
        inv_mapping = train_dataset.get_inv_mapping()
        
    # set device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    
    if args.eval_mode: #TODO add evaluation part for model
        
        # # load args from save_dir
        # with open(os.path.join(save_dir, "train_args.txt"), "r") as f:
        #     train_args = f.read()
            
        # # set arguments for model from train_args loaded from file
        # model_args = {
        #     "input_dim" : input_dim,
            
        # }
        # # load state dictionary from model path
        # model.load_state_dict(torch.load(args.model_path))
        # # evaluate model
        pass
    
    else: #TODO add weights for focal loss and weighted cross entropy loss
        
        # set arguments for model
        model_args = {
            "input_dim" : input_dim,
            "hidden_dim" : args.hidden_dim,
            "num_experts" : num_classes if args.num_experts == -1 else args.num_experts,
            "num_classes" : num_classes,
            "activation" : args.activation_fn,
            "dropout" : args.dropout
        }
        
        # print model args
        print(f"model_args : {model_args}")
        
        model = MixtureOfExperts(**model_args)
        
        # save args in save_dir
        with open(os.path.join(save_dir, "train_args.txt"), "w") as f:
            f.write(str(args))
        
        # set loss functions
        if args.expert_loss_fn == "ce":
            expert_loss_fn = nn.CrossEntropyLoss()
        elif args.expert_loss_fn == "focal_loss":
            expert_loss_fn = FocalLoss()
        elif args.expert_loss_fn == "wce":
            expert_loss_fn = WeightedCrossEntropyLoss()
        else:
            raise ValueError("Invalid expert loss function")
        
        gating_loss_fn = MSEGatingLoss()
        
        # set optimizer
        if args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=args.betas)
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=args.nesterov)
        else:
            raise ValueError("Invalid optimizer")
        
        # set scheduler
        # if args.scheduler == "plateau":
        #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, min_lr=args.min_lr)
        
        if args.scheduler == "lr_step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.gamma)
        elif args.scheduler == "none":
            scheduler = None
        else:
            raise ValueError("Invalid scheduler")
        
        # train model
        best_model = train(args=args, model=model, expert_loss_fn=expert_loss_fn, gating_loss_fn=gating_loss_fn, optimizer=optimizer, scheduler=scheduler, dataloader=train_dataloader, val_dataloader=val_dataloader, device=device, log_dir=log_dir, save_dir=save_dir)
        
        # evaluate model
        evaluate(model=best_model, expert_loss_fn=expert_loss_fn, gating_loss_fn=gating_loss_fn, dataloader=test_dataloader, device=device, labels_list=inv_mapping, model_name=args.model_name, save_dir=save_dir)
        
        

    


