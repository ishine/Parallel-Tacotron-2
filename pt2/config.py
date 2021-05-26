# this is a template config file from LJSpeech dataset


from typing import NamedTuple


class Config(NamedTuple):

    # model
    # encoder
    num_tokens = 256
    num_speakers = 1
    encoder_dim = 64
    # duration predictor
    duration_predictor_dim = 64
    # upsampling
    upsampling_dim = 64
    # decoder
    decoder_dim = 64
    decoder_num_blocks = 3

    # dsp
    n_mels = 80
    n_fft = 1024
    win_length = 1024
    hop_length = 256
    fmin  = 0
    fmax = 8000

    # dataset
    graphemes = """ !",.:;?abdefhijklmnoprstuvwxzæðŋɐɑɔəɚɛɜɡɪɹɾʃʊʌʒʔˈˌː̩θᵻ“”"""
    max_clip_length = 12 # seconds
    sample_rate = 22050 # samples per second

