import torch
import torch.nn.functional as F
from torch import nn
from safetensors.torch import load_model
# get versions like so: python -c "import torch; print(torch.__version__)"
# python -c "import safetensors; print(safetensors.__version__)"

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)  # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)  # Apply max pooling
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)  # Apply fully connected layer
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = 10  # digits 0-9
batch_size = 64

model = CNN(in_channels=1, num_classes=num_classes).to(device)
[missing, unexpected] = load_model(model, "benchmarks/mnist/model-mnist.safetensors")
model.eval()  # Set the model to evaluation mode

sample_input = torch.rand(batch_size,1,28, 28)

traced_script_module = torch.jit.trace(model, sample_input)
traced_script_module.save("mnist_fugaku.pt") # currently manual: rename for different machine
