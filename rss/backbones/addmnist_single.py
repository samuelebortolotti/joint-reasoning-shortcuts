import torch.nn
from torch import nn
# from backbones.base.ops import *


class Flatten(nn.Module):
    def forward(self, input):
        return torch.flatten(input, start_dim=1)

class UnFlatten(nn.Module):
    def forward(self, input, hidden_channels, dim):
        return input.reshape(input.size(0), hidden_channels, dim[0], dim[1])


class MNISTSingleEncoder(nn.Module):
    def __init__(
        self, img_channels=1, hidden_channels=32, c_dim=10, latent_dim=16, dropout=0.5
    ):
        super(MNISTSingleEncoder, self).__init__()

        self.channels = 3
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.c_dim = c_dim
        self.latent_dim = latent_dim

        self.unflatten_dim = (3, 7)

        self.enc_block_1 = nn.Conv2d(
            in_channels=self.img_channels,
            out_channels=self.hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.enc_block_2 = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.enc_block_3 = nn.Conv2d(
            in_channels=self.hidden_channels * 2,
            out_channels=self.hidden_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.flatten = Flatten()

        self.dense_logvar = nn.Linear(
            in_features=int(
                4
                * self.hidden_channels
                * self.unflatten_dim[0]
                * self.unflatten_dim[1]
                * (3 / 7)
            ),
            out_features=self.latent_dim * self.c_dim,
        )

        self.dense_mu = nn.Linear(
            in_features=int(
                4
                * self.hidden_channels
                * self.unflatten_dim[0]
                * self.unflatten_dim[1]
                * (3 / 7)
            ),
            out_features=self.latent_dim * self.c_dim,
        )

        self.dense_c = nn.Linear(
            in_features=int(
                4
                * self.hidden_channels
                * self.unflatten_dim[0]
                * self.unflatten_dim[1]
                * (3 / 7)
            ),
            out_features=self.c_dim,
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # MNISTPairsEncoder block 1
        x = self.enc_block_1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # MNISTPairsEncoder block 2
        x = self.enc_block_2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # MNISTPairsEncoder block 3
        x = self.enc_block_3(x)
        x = nn.ReLU()(x)

        # mu and logvar
        x = self.flatten(
            x
        )  # batch_size, dim1, dim2, dim3 -> batch_size, dim1*dim2*dim3

        c, mu, logvar = self.dense_c(x), self.dense_mu(x), self.dense_logvar(x)

        # return encodings for each object involved
        c = torch.stack(torch.split(c, self.c_dim, dim=-1), dim=1)
        mu = torch.stack(torch.split(mu, self.latent_dim, dim=-1), dim=1)
        logvar = torch.stack(torch.split(logvar, self.latent_dim, dim=-1), dim=1)

        return c, mu, logvar


class MNISTSingleDecoder(nn.Module):
    def __init__(
        self,
        img_channels=1,
        hidden_channels=32,
        c_dim=10,
        latent_dim=160,
        dropout=0.5,
        **params,
    ):
        super(MNISTSingleDecoder, self).__init__()

        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.unflatten_dim = (4, 4)  # Adjusted for correct initial size

        # Linear layer to expand latent + class vector to hidden feature dimensions
        self.dense = nn.Linear(
            in_features=self.latent_dim + self.c_dim,
            out_features=hidden_channels * 4 * self.unflatten_dim[0] * self.unflatten_dim[1],
        )

        self.unflatten = UnFlatten()

        # Decoder blocks using ConvTranspose2d
        self.dec_block_1 = nn.ConvTranspose2d(
            in_channels=self.hidden_channels * 4,
            out_channels=self.hidden_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,  # Output: 8x8
        )

        self.dec_block_2 = nn.ConvTranspose2d(
            in_channels=self.hidden_channels * 2,
            out_channels=self.hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1,  # Output: 16x16
        )

        self.dec_block_3 = nn.ConvTranspose2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels // 2,
            kernel_size=4,
            stride=2,
            padding=1,  # Output: 32x32
        )

        self.dec_block_4 = nn.ConvTranspose2d(
            in_channels=self.hidden_channels // 2,
            out_channels=self.img_channels,
            kernel_size=3,
            stride=1,
            padding=3,  # Final adjustment for 28x28 output
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Forward pass through the decoder.

        Args:
            x: Input tensor of shape [batch_size, latent_dim + c_dim].

        Returns:
            torch.Tensor: Reconstructed image tensor of shape [batch_size, 1, 28, 28].
        """

        # Expand to feature dimensions
        x = self.dense(x)  # Shape: [batch_size, hidden_channels * 4 * H * W]
        x = self.unflatten(x, self.hidden_channels * 4, self.unflatten_dim)

        # Decoder block 1
        x = self.dec_block_1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # Decoder block 2
        x = self.dec_block_2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # Decoder block 3
        x = self.dec_block_3(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # Additional refinement block
        x = self.dec_block_4(x)
        x = torch.nn.Sigmoid()(x)  # Ensure output is in [0, 1] for pixel values

        return x


if __name__ == "__main__":
    model = MNISTSingleDecoder()

    input_tensor = torch.randn(1, 170)  # [batch_size, latent_dim + c_dim]
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Should print: torch.Size([1, 1, 28, 28])
