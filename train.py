import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from models.vgg6 import vgg6
from utils.dataset import get_cifar10_loaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    avg_loss = running_loss / total
    return avg_loss, acc


def get_optimizer(optimizer_name, model_params, lr, weight_decay, momentum):
    name = optimizer_name.lower()
    if name == 'sgd':
        return optim.SGD(model_params, lr=lr, weight_decay=weight_decay)
    elif name == 'nesterov':
        return optim.SGD(model_params, lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    elif name == 'adam':
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return optim.Adagrad(model_params, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return optim.RMSprop(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'nadam':
        return optim.NAdam(model_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def main():
    parser = argparse.ArgumentParser(description="VGG6 CIFAR-10 Training")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--batch_norm", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", type=str, default="nadam")
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", type=lambda x: (str(x).lower() == 'true'), default=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_name = f"act_{args.activation}_bs_{args.batch_size}_opt_{args.optimizer}_lr_{args.lr}_bn_{args.batch_norm}"
    if args.use_wandb:
        wandb.init(project="vgg6-cifar10", name=run_name, config=vars(args))

    train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size)
    model = vgg6(activation=args.activation, batch_norm=args.batch_norm, num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay, args.momentum)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        val_loss, val_acc = evaluate(model, test_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc
            })

    print(f"Training Complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()