import argparse

from dnn_framework import Network, FullyConnectedLayer, BatchNormalization, ReLU
from mnist import MnistTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=False, default=0.0065)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=False, default=128)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=False, default=50)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=False, default=".")

    parser.add_argument('--checkpoint_path', type=str, help='Choose the output path', default=None)

    args = parser.parse_args()

    network = create_network(args.checkpoint_path)
    trainer = MnistTrainer(network, args.learning_rate, args.epoch_count, args.batch_size, args.output_path)
    trainer.train()


def create_network(checkpoint_path):
    layers = [FullyConnectedLayer(784,128), BatchNormalization(128), ReLU(), FullyConnectedLayer(128,32), BatchNormalization(32), ReLU(), FullyConnectedLayer(32,10)]
    network = Network(layers)
    if checkpoint_path is not None:
        network.load(checkpoint_path)

    return network


if __name__ == '__main__':
    main()
