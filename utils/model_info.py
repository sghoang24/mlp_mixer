"""Model info."""

MODEL_INFO = {
    's_16': {
        'num_blocks': 8,
        'patch_size': 16,
        'hidden_dim': 512,
        'tokens_mlp_dim': 256,
        'channels_mlp_dim': 2048,
    },
    's_32': {
        'num_blocks': 8,
        'patch_size': 32,
        'hidden_dim': 512,
        'tokens_mlp_dim': 256,
        'channels_mlp_dim': 2048,
    },
    'b_16': {
        'num_blocks': 12,
        'patch_size': 16,
        'hidden_dim': 768,
        'tokens_mlp_dim': 384,
        'channels_mlp_dim': 3072,
    },
    'b_32': {
        'num_blocks': 12,
        'patch_size': 32,
        'hidden_dim': 768,
        'tokens_mlp_dim': 384,
        'channels_mlp_dim': 3072,
    },
    'l_16': {
        'num_blocks': 24,
        'patch_size': 16,
        'hidden_dim': 1024,
        'tokens_mlp_dim': 512,
        'channels_mlp_dim': 4096,
    },
    'l_32': {
        'num_blocks': 24,
        'patch_size': 16,
        'hidden_dim': 1024,
        'tokens_mlp_dim': 512,
        'channels_mlp_dim': 4096,
    },
    'h_14': {
        'num_blocks': 32,
        'patch_size': 14,
        'hidden_dim': 1280,
        'tokens_mlp_dim': 640,
        'channels_mlp_dim': 5120,
    },
}
