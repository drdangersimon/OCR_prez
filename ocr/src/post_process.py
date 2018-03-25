"""
Conforms image to nrc specifications

"""
from . import ctc_decoding

def pipeline(ctc_out, max_str_length):
    # beam search decode
    nrc_array, prob = ctc_decoding.beamsearch_decode(ctc_out)
    return ''.join(nrc_array[0])

