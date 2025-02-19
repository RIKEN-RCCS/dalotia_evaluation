# %%
# https://www.craylabs.org/docs/tutorials/ml_inference/Inference-in-SmartSim.html

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# %%
import safetensors.torch as st
import argparse


# %%
# https://doi.org/10.1016/j.compfluid.2019.104319
class SubgridLESNet(nn.Module):
    def __init__(self):
        super().__init__()
        # In this paper, the nine components of the velocity gradient tensor and the filter width  
        # are used as input features (nI = 10) in the ANN,
        #The output labels of the ANN are the six components of the SGS stress tensor
        self.fc1 = nn.Linear(10, 6)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return x


# %%
# Figure 4: Network architecture of the CNN-based policy network for N = 5. 
# The inputs of the network are either the momentum field (  ρ ̃v1, ρ ̃v2, ρ ̃v3)T 
# or the five invariants of the velocity gradient tensor λi  ∇v in a single DG element 
# with given N , for which the distribution of interpolation points is shown exemplarily. 
# The network comprises several three-dimensional convolutional layers (Conv3D) 
# with the corresponding kernel sizes k and the number of filters per layer nf . 
# The output sizes in the dimensions of convolution are given below each layer. 
# The first layer retains the input dimension by means of zero-padding, 
# while for the other layers no padding is employed in order to retain a scalar output Cs per element. 
# The scaling layer applies the sigmoid activation function σs(x) in order to scale the network output to Cs ∈ [0, 0.5] 
# and can be interpreted as the activation function of the last layer.
# https://www.sciencedirect.com/science/article/pii/S0142727X2200162X
# here: use 3-dim local momentum field vector
class DeepRLEddyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # input_size = (N, C_in, D, H, W) = (batch_size, 3, 6, 6, 6)
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        # torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
        # dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, padding=0)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, padding=0)
        self.conv4 = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = torch.flatten(x)
        output = 0.5 * F.sigmoid(x)
        return output


# %%
def load_torch_model_file(torch_module):
    loaded_net = torch.jit.load("traced_" + torch_module.__class__.__name__ + ".pt")
    loaded_inputs = st.load_file("input_" + torch_module.__class__.__name__ + ".safetensors")["random_input"]
    loaded_weights = st.load_model(torch_module, "weights_" + torch_module.__class__.__name__ + ".safetensors")
    loaded_output = st.load_file("output_" + torch_module.__class__.__name__ + ".safetensors")["output"]
    return loaded_net, loaded_inputs, loaded_weights, loaded_output

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='either SubgridLESNet or DeepRLEddyNet')
    parser.add_argument('input_size', type=int, help='number of input features')
    args = parser.parse_args()
    if args.model == "SubgridLESNet":
        n = SubgridLESNet()
    elif args.model == "DeepRLEddyNet":
        n = DeepRLEddyNet()
    else:
        raise ValueError("Model not found")

    loaded_net, example_forward_input, _, example_output = load_torch_model_file(n)

    # resize the input to args.input_size
    if len(example_forward_input) > args.input_size:
        example_forward_input = example_forward_input[:args.input_size]
        example_output = example_output[:args.input_size]
    else:
        example_forward_input = torch.cat((example_forward_input, torch.zeros(args.input_size - len(example_forward_input))))
        example_output = torch.cat((example_output, torch.zeros(args.input_size - len(example_output))))

    
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(loaded_net))

    num_iterations = 1000
    start = time.perf_counter()
    for i in range(num_iterations):
        result = frozen_mod(example_forward_input)
    end = time.perf_counter()
    print(f"Time taken for {num_iterations} iterations: {end - start} seconds")
    print(f"On average: {(end - start) / num_iterations}s")
    
    assert torch.allclose(result, example_output, atol=1e-6)

