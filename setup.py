from setuptools import setup

setup(
    name="RAVE",
    version="1.0",
    install_requires=[
        'accelerate', 
        'tqdm==4.62.3', 
        'effortless-config==0.7.0', 
        'einops==0.4.0', 
        'librosa', 
        'matplotlib==3.5.1', 
        'numpy', 
        'pytorch_lightning', 
        'scikit_learn==1.0.2', 
        'scipy==1.7.3', 
        'soundfile==0.10.3.post1', 
        'termcolor==1.1.0', 
        'torch', 
        'torchaudio', 
        'tensorboard==2.8.0', 
        'GPUtil==1.4.0', 
        'wandb==0.12.7', 
        'cached_conv @ git+https://github.com/caillonantoine/cached_conv.git#egg=cached_conv', 
        'UDLS @ git+https://github.com/caillonantoine/UDLS.git#egg=udls'],
    # ...
)
