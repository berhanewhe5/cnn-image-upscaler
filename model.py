import tensorflow as tf


class PixelShuffle(tf.keras.layers.Layer):
    """
    Sub-pixel convolution layer (equivalent to PyTorch's nn.PixelShuffle).
    Rearranges elements in a tensor of shape (N, H, W, C * r^2)
    into (N, H*r, W*r, C) using tf.nn.depth_to_space.
    """

    def __init__(self, scale_factor, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, x):
        return tf.nn.depth_to_space(x, self.scale_factor)

    def get_config(self):
        config = super().get_config()
        config["scale_factor"] = self.scale_factor
        return config


def build_srcnn(num_channels=3):
    """
    Super-Resolution Convolutional Neural Network (SRCNN).

    Input:
        Bicubic-upsampled LR image with same spatial size as HR target.

    Output:
        Refined SR image of the same size.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(None, None, num_channels), name="input"),
            tf.keras.layers.Conv2D(
                64,
                kernel_size=9,
                padding="same",
                activation="relu",
                name="feature_extraction",
            ),
            tf.keras.layers.Conv2D(
                32,
                kernel_size=1,
                padding="same",
                activation="relu",
                name="non_linear_mapping",
            ),
            tf.keras.layers.Conv2D(
                num_channels,
                kernel_size=5,
                padding="same",
                activation="sigmoid",
                name="reconstruction",
            ),
        ],
        name="SRCNN",
    )
    return model


def build_espcn(scale_factor=2, num_channels=3):
    """
    Efficient Sub-Pixel Convolutional Neural Network (ESPCN).

    Input:
        Native low-resolution image.

    Output:
        High-resolution image after learned sub-pixel upscaling.

    Args:
        scale_factor: upscaling factor, e.g. 2, 3, 4, 8
        num_channels: usually 3 for RGB
    """
    inputs = tf.keras.Input(shape=(None, None, num_channels), name="lr_input")

    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="conv1",
    )(inputs)

    x = tf.keras.layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv2",
    )(x)

    x = tf.keras.layers.Conv2D(
        num_channels * (scale_factor ** 2),
        kernel_size=3,
        padding="same",
        name="conv3",
    )(x)

    x = PixelShuffle(scale_factor, name="pixel_shuffle")(x)
    outputs = tf.keras.layers.Activation("sigmoid", name="output")(x)

    model = tf.keras.Model(inputs, outputs, name=f"ESPCN_x{scale_factor}")
    return model


def build_model(architecture="espcn", scale_factor=2, num_channels=3):
    """
    Factory function that builds the requested model.

    Args:
        architecture: 'srcnn' or 'espcn'
        scale_factor: upscale factor for ESPCN
        num_channels: number of image channels
    """
    architecture = architecture.lower()

    if architecture == "srcnn":
        return build_srcnn(num_channels=num_channels)
    elif architecture == "espcn":
        return build_espcn(scale_factor=scale_factor, num_channels=num_channels)
    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Choose 'srcnn' or 'espcn'."
        )


if __name__ == "__main__":
    print("=== SRCNN ===")
    srcnn = build_model("srcnn")
    srcnn.summary()

    for scale in [2, 3, 4, 8]:
        print(f"\n=== ESPCN (x{scale}) ===")
        espcn = build_model("espcn", scale_factor=scale)
        espcn.summary()