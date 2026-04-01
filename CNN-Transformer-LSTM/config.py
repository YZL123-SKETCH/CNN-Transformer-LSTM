import argparse
import torch
import os

def get_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--filename', default="", type=str)

    # Model Parameters
    parser.add_argument('--input_size', default=1, type=int, help='Input feature dimension')
    parser.add_argument('--output_size', default=1, type=int, help='Output prediction dimension')
    parser.add_argument('--hidden_size', default=128, type=int, help='Hidden dimension size')
    parser.add_argument('--num_layers', default=2, type=int, help='Number of LSTM layers')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')

    # Training Settings
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--sequence_length', default=15, type=int, help='Input sequence length')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--epochs', default=100, type=int, help='Training epochs')

    # Device
    parser.add_argument('--useGPU', default=True, type=bool, help='Use GPU if available')
    parser.add_argument('--gpu', default="4,5,6", type=int, help='GPU device ID')
    parser.add_argument('--batch_first', default=True, type=bool, help='Batch first dimension')

    # Save Path
    parser.add_argument('--save_file', default='', type=str, help='Model save directory')

    args = parser.parse_args()

    # Device configuration
    args.device = torch.device(f"cuda:{args.gpu}")

    # Create save folder
    if not os.path.exists(args.save_file):
        os.makedirs(args.save_file)

    return args