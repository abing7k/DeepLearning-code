import torch
from torch import nn
from d2l import torch as d2l
import time

def main():
    # Define the AlexNet model architecture
    net = nn.Sequential(
        # First convolutional layer with 96 kernels, 11x11 size, stride 4, padding 0
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # Second convolutional layer with 256 kernels, 5x5 size, padding 2
        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # Third convolutional layer with 384 kernels, 3x3 size, padding 1
        nn.Conv2d(256, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        # Fourth convolutional layer with 384 kernels, 3x3 size, padding 1
        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        # Fifth convolutional layer with 256 kernels, 3x3 size, padding 1
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # Flatten the output for the fully connected layers
        nn.Flatten(),
        # First fully connected layer with 4096 units
        nn.Linear(256 * 5 * 5, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        # Second fully connected layer with 4096 units
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        # Output layer with 10 units (for 10 classes)
        nn.Linear(4096, 10)
    )

    # Load Fashion-MNIST dataset and resize images to 224x224 as required by AlexNet
    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    # Device selection logic:
    # If running on macOS and MPS (Metal Performance Shaders) is available, use MPS.
    # Else if CUDA is available, use CUDA.
    # Otherwise, fall back to CPU.
    import sys
    if sys.platform == "darwin" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    net.to(device)


    # Train the model for 5 epochs
    num_epochs = 5
    lr = 0.01
    print("Starting training...")
    start_time = time.time()
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")
    d2l.plt.show()

if __name__ == "__main__":
    main()
