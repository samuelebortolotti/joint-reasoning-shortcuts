import torch
import torch.nn as nn

class CLEVRCNN(nn.Module):
    def __init__(self, num_classes, latent_dim=16):
        """
        Initializes a CNN-based classifier.

        Args:
            num_classes (int): Number of classes for classification.
        """
        super(CLEVRCNN, self).__init__()

        self.c_dim = num_classes
        self.latent_dim = latent_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Conv layer 1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample 1
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Conv layer 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Conv layer 3
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        )

        self.dense_logvar = nn.Linear(
            in_features=128,
            out_features=self.latent_dim * self.c_dim,
        )

        self.dense_mu = nn.Linear(
            in_features=128,
            out_features=self.latent_dim * self.c_dim,
        )

        self.dense_c = nn.Linear(
            in_features=128,
            out_features=self.c_dim,
        )

    def forward(self, x):
        """
        Forward pass through the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        features = self.cnn(x)
        features = features.view(features.size(0), -1)

        c, mu, logvar = self.dense_c(features), self.dense_mu(features), self.dense_logvar(features)

        # return encodings for each object involved
        c = torch.stack(torch.split(c, self.c_dim, dim=-1), dim=1)
        mu = torch.stack(torch.split(mu, self.latent_dim, dim=-1), dim=1)
        logvar = torch.stack(torch.split(logvar, self.latent_dim, dim=-1), dim=1)

        return c, mu, logvar


class CLEVRDECODER(nn.Module):
    def __init__(self, input_dim=255, output_channels=3, image_size=128):
        super(CLEVRDECODER, self).__init__()
        self.output_channels = output_channels
        self.image_size = image_size

        self.fc = nn.Linear(input_dim, 512 * 8 * 8)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 8, 8)
        x = self.deconv_layers(x)
        return x


if __name__ == "__main__":
    num_classes = 40
    model = CLEVRCNN(num_classes=num_classes)

    input_tensor = torch.randn(2, 3, 128, 128)
    output, mu, logvar = model(input_tensor)
    print("Output shape:", output.shape, mu.shape, logvar.shape)
