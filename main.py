import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tensorboardX import SummaryWriter  
from model import MixtureOfExperts
from losses import WeightedCrossEntropyLoss, FocalLoss, MSEGatingLoss
import matplitlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# define command line parser
def argparser():
    parser = argparse.ArgumentParser(description='Mixture of Experts Training')
    parser.add_argument('--input_dim', type=int, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension for experts')
    parser.add_argument('--num_experts', type=int, help='Number of experts')
    parser.add_argument('--num_classes', type=int, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization)')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use Focal Loss for expert loss')
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam', help='Optimizer choice')
    parser.add_argument('--scheduler', choices=[None, 'lr_step', 'plateau'], default=None, help='Learning rate scheduler')
    parser.add_argument('--scheduler_step_size', type=int, default=20, help='Step size for lr_step scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for lr_step scheduler')
    return parser.parse_args()

# evaluate function
def evaluate(model, expert_loss_fn, gating_loss_fn, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data, batch_labels in tqdm(dataloader, desc="Validation", leave=False):
            batch_data = batch_data.to(device)  # Move data to the specified device
            batch_labels = batch_labels.to(device)

            expert_outputs, _ = model(batch_data)  # We don't need gating weights for evaluation

            # Calculate expert loss (optional)
            expert_loss = expert_loss_fn(expert_outputs, batch_labels)
            total_loss += expert_loss.item()

            # Calculate predictions for accuracy and F1 score
            _, preds = torch.max(expert_outputs, 1)
            all_preds.extend(preds.cpu().tolist())  # Move predictions back to CPU
            all_labels.extend(batch_labels.cpu().tolist())

    # Calculate average loss (optional)
    avg_loss = total_loss / len(dataloader)

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Generate classification report
    classification_report_str = classification_report(all_labels, all_preds)

    # Create a confusion matrix heatmap
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # Save the heatmap as an image

    return avg_loss, accuracy, f1, classification_report_str

# Training loop function
def train(args, model, expert_loss_fn, gating_loss_fn, optimizer, scheduler, dataloader, val_dataloader, device):
    best_f1_score = 0.0
    writer = SummaryWriter()
    model.to(device)
    for epoch in tqdm(range(args.num_epochs), desc="Epochs"):

        
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

            # get output from model
            mixture_out, gating_out, expert_out = model(input)
            
            # expert out for debugging

            # get gating loss
            gate_loss = gating_loss_fn(gating_out, true_gating_labels.float())
            total_gating_loss += gate_loss.item()
            # get expert loss
            exprt_loss = expert_loss_fn(mixture_out, labels)
            total_expert_loss += exprt_loss.item()
            # get total loss
            total_loss = gate_loss + exprt_loss
            # zero gradients
            optimizer.zero_grad()
            # backpropagate
            total_loss.backward()
            # update weights
            optimizer.step()
            # update scheduler
            scheduler.step()

            # calculate predictions for accuracy and F1 score
            _, preds = torch.max(mixture_out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # calculate average losses for epoch
        avg_expert_loss = total_expert_loss / len(dataloader)
        avg_gating_loss = total_gating_loss / len(dataloader)

        # calculate accuracy and F1 score
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        # write to tensorboard
        writer.add_scalar("Train/Avg_Train_Expert_Loss", avg_expert_loss, epoch)
        writer.add_scalar("Train/Avg_Train_Gating_Loss", avg_gating_loss, epoch)
        writer.add_scalar("Train/Train_Accuracy", accuracy, epoch)
        writer.add_scalar("Train/Train_F1_Score", f1, epoch)

        # perform validation

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

        # calculate accuracy and F1 score
        accuracy_val = accuracy_score(all_labels_val, all_preds_val)
        f1_val = f1_score(all_labels_val, all_preds_val, average="macro")

        # write to tensorboard
        writer.add_scalar("Train/Avg_Val_Expert_Loss", avg_expert_loss_val, epoch)
        writer.add_scalar("Train/Avg_Val_Gating_Loss", avg_gating_loss_val, epoch)
        writer.add_scalar("Train/Val_Accuracy", accuracy_val, epoch)
        writer.add_scalar("Train/Val_F1_Score", f1_val, epoch)

        # check if F1 score is best
        if f1_val > best_f1_score:
            # save model
            torch.save(model.state_dict(), "best_model.pt")
            # update best F1 score
            best_f1_score = f1
        
    writer.close()