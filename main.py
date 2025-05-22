import torch
from utils import prep
import argparse
from models.cnn import get_pretrained_model
from models.train import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement d'un modèle CNN")
    parser.add_argument('--epochs', type=int, default=10, help="Nombre d'époques d'entraînement")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.0001, help="weight decay")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help="Mode of operation: 'train' or 'eval' (default: train)")
    parser.add_argument('--cuda', action='store_true', help="Utiliser le GPU si disponible")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    train_dataloader, test_dataloader = prep.get_data()
    model = get_pretrained_model().to(device)
    if args.mode == 'eval':
        model.load_state_dict(torch.load("model.pth"))
    trainer = Trainer(model, train_dataloader, test_dataloader, args.lr, args.wd, args.epochs, device)
    if args.mode == 'train':
        trainer.train(True, True)
    trainer.evaluate()
if __name__ == '__main__':
    main()