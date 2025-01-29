# %%
# https://www.craylabs.org/docs/tutorials/ml_inference/Inference-in-SmartSim.html

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
from math import prod

# %%
import safetensors.torch as st
# %%
from typing import Tuple

# %%
def pytorch_layer_init_arange(layer, basis):
    # unfortunately we mostly don't have the true trained weights available, 
    # use inverse aranges to validate instead
    assert basis > 0
    layer.weight = torch.nn.Parameter(1. / torch.arange(start=basis, 
                                                        end=prod(layer.weight.shape) + basis, 
                                                        dtype=layer.weight.dtype).reshape(shape=layer.weight.shape))

# %%
# https://www.craylabs.org/docs/tutorials/ml_inference/Inference-in-SmartSim.html
class DefaultNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# %%


# %%
# https://doi.org/10.1016/j.compfluid.2019.104319
class SubgridLESNet(nn.Module):
    def __init__(self):
        super().__init__()
        # In this paper, the nine components of the velocity gradient tensor and the filter width  
        # are used as input features (nI = 10) in the ANN,
        #The output labels of the ANN are the six components of the SGS stress tensor
        self.fc1 = nn.Linear(10, 6)

        pytorch_layer_init_arange(self.fc1, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return x

# %%
# https://github.com/CrayLabs/NCAR_ML_EKE/blob/master/ml_eke/nn/nn_models.py
# pretty convoluted!

class TransBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        filters: int = 256,
        size: Tuple[int, int] = (3, 3)
    ) -> None:

        super().__init__()

        self.convTrans = nn.ConvTranspose2d(in_channels=in_channels,
                                            out_channels=filters,
                                            kernel_size=size,
                                            stride=1,
                                            padding=0,
                                            output_padding=0)

        self.bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.convTrans(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class EkeResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 20)
        self.conv2 = nn.Conv2d(5, 5, 32)
        self.conv3 = nn.Conv2d(7, 7, 32)
        self.fc1 = nn.Linear(4*4*128, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        # input shape: 1*1*4
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

        
        # x = torch.reshape(x, (-1,self.train_features, 1, 1))
        # x = self.transBlock1(x)
        # x = self.transBlock2(x)
        # x = self.transBlock3(x)

        # x = self.layer1(x)
        # x = self.maxpool(x)
        # x = self.layer2(x)
        # x = self.layer3(x)

        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)

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

        # unfortunately we don't have the true trained weights available, use aranges to validate instead
        pytorch_layer_init_arange(self.conv1, 23)
        pytorch_layer_init_arange(self.conv2, 5)
        pytorch_layer_init_arange(self.conv3, 7)
        pytorch_layer_init_arange(self.conv4, 13)

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

def create_torch_model_file(torch_module, forward_input):
    module = torch.jit.trace(torch_module, forward_input)

    # save the traced module to file
    trace_file_name = "traced_" + torch_module.__class__.__name__ + ".pt"
    torch.jit.save(module, trace_file_name)
    weights_file_name = "weights_" + torch_module.__class__.__name__ + ".safetensors"
    st.save_model(torch_module, weights_file_name)
    input_file_name = "input_" + torch_module.__class__.__name__ + ".safetensors"
    st.save_file({"random_input": forward_input}, input_file_name)

    return
    

# %%
if __name__ == "__main__":
    n = SubgridLESNet() # 
    n = DeepRLEddyNet() #DefaultNet()

    n.eval()

    torch.manual_seed(0) # reproducible "random" input
    if isinstance(n, DeepRLEddyNet):
        example_DeepRLEddyNet_forward_input = torch.rand(16**3, 3, 6, 6, 6)
        example_forward_input = example_DeepRLEddyNet_forward_input
    elif isinstance(n, SubgridLESNet):
        example_SubgridLESNet_forward_input = torch.rand(64**3, 10)
        example_forward_input = example_SubgridLESNet_forward_input

    print(example_forward_input)

    n.forward(example_forward_input)

    create_torch_model_file(n, example_forward_input)

# %%
