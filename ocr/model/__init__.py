from .block3_CNN_model import cnn2x3maxpool, cnn3x2maxpool
from .transferlearn_model import transfer_learn_ocr_model
from .vanilla_model import vanilla_ocr_model
__all__ = ['cnn2x3maxpool','cnn3x2maxpool','transfer_learn_ocr_model','vanilla_ocr_model']