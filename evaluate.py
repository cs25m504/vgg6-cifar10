import argparse
import torch
from models.vgg6 import vgg6
from utils.dataset import get_cifar10_loaders
from train import evaluate

def main():
    parser = argparse.ArgumentParser(description="Evaluate VGG6 Checkpoint")
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--batch_norm", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader = get_cifar10_loaders(batch_size=args.batch_size)

    model = vgg6(activation=args.activation, batch_norm=args.batch_norm, num_classes=10).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    val_loss, val_acc = evaluate(model, test_loader, device)
    print(f"Evaluation Results -> Test Loss: {val_loss:.4f}, Test Top-1 Accuracy: {val_acc:.2f}%")

if __name__ == "__main__":
    main()