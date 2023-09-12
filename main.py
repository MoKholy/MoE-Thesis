import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tensorboardX import SummaryWriter  
from model import MixtureOfExperts
from losses import WeightedCrossEntropyLoss, FocalLoss, MSEGatingLoss

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
def train(args, model, expert_loss_fn, gating_loss_fn, optimizer, dataloader, val_dataloader, device):
    best_val_loss = float('inf')
    best_model_state_dict = None

    model.to(device)  # Move the model to the specified device

    # Create a SummaryWriter for TensorBoard
    writer = SummaryWriter()

    for epoch in tqdm(range(args.num_epochs), desc="Epochs"):
        model.train()
        total_expert_loss = 0.0
        total_gating_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_data, batch_labels in tqdm(dataloader, desc="Batches", leave=False):
            batch_data = batch_data.to(device)  # Move data to the specified device
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            expert_outputs, gating_weights = model(batch_data)

            # Calculate expert loss
            expert_loss = expert_loss_fn(expert_outputs, batch_labels)
            total_expert_loss += expert_loss.item()

            # Calculate gating loss
            gating_loss = gating_loss_fn(gating_weights, true_gating_labels.to(device))  # Move gating labels
            total_gating_loss += gating_loss.item()

            # Total loss for backpropagation
            total_loss = expert_loss + gating_loss
            total_loss.backward()
            optimizer.step()

            # Calculate predictions for accuracy and F1 score
            _, preds = torch.max(expert_outputs, 1)
            all_preds.extend(preds.cpu().tolist())  # Move predictions back to CPU
            all_labels.extend(batch_labels.cpu().tolist())

        # Calculate average losses for the epoch
        avg_expert_loss = total_expert_loss / len(dataloader)
        avg_gating_loss = total_gating_loss / len(dataloader)

        # Calculate accuracy and F1 score for the epoch
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        # Log metrics to TensorBoard
        writer.add_scalar('Train/Avg_Expert_Loss', avg_expert_loss, epoch)
        writer.add_scalar('Train/Avg_Gating_Loss', avg_gating_loss, epoch)
        writer.add_scalar('Train/Accuracy', accuracy, epoch)
        writer.add_scalar('Train/F1_Score', f1, epoch)

    # Save the best model checkpoint
    if best_model_state_dict is not None:
        torch.save(best_model_state_dict, 'best_model.pth')

    # Close the TensorBoard writer
    writer.close()