class Config:
    """
    Configuration settings for the Adaptive Neural Network Weight Generator.
    """
    def __init__(self):
        self.input_size = 100
        self.output_size = 100
        self.num_layers = 4
        self.channels_per_layer = [128, 256, 512, 1024]
        self.latent_size = 256
        self.learning_rate = 1e-4
        self.batch_size = 64
        self.num_epochs = 100

    def validate_config(self):
        assert isinstance(self.num_layers, int) and self.num_layers > 0, "Number of layers must be a positive integer."
        assert isinstance(self.channels_per_layer, list) and len(self.channels_per_layer) == self.num_layers, "Mismatch in layers and channels per layer count."
        assert all(isinstance(ch, int) and ch > 0 for ch in self.channels_per_layer), "All channels must be positive integers."