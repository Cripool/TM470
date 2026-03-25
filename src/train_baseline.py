from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_datasets(data_fir: str, image_size=(180, 180), batch_size: init = 32):
    """
    
    _summary_

    Args:
        data_fir (str): _description_
        image_size (tuple, optional): _description_. Defaults to (180, 180).
        batch_size (init, optional): _description_. Defaults to 32.
    """