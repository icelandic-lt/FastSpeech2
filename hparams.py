import os

# Dataset
dataset = "Talromur-b"
data_path = "/data/tts/talromur/published/alfur/"
data_split = None
data_split = {
    "train": "/data/tts/talromur/published/split/c_train.txt",
    "val": "/data/tts/talromur/published/split/c_val.txt",
    #"test": "/data/tts/talromur/published/split/c_test.txt",
}
#dataset = "LJSpeech"
#data_path = "/data/LJSpeech/LJSpeech-1.1"
#dataset = "Blizzard2013"
#data_path = "./Blizzard-2013/train/segmented/"


# Text
#text_cleaners = ['english_cleaners']
text_cleaners = ['transliteration_cleaners']


# Audio and mel
### for Talromur ###
#sampling_rate = 44100
#filter_length = 2048
#hop_length = 512
#win_length = 2048
### for LJSpeech ###
sampling_rate = 22050
filter_length = 1024
hop_length = 256
win_length = 1024
### for Blizzard2013 ###
#sampling_rate = 16000
#filter_length = 800
#hop_length = 200
#win_length = 800

max_wav_value = 32768.0
n_mel_channels = 80
mel_fmin = 0.0
mel_fmax = 8000.0


# FastSpeech 2
encoder_layer = 4
encoder_head = 2
encoder_hidden = 256
decoder_layer = 4
decoder_head = 2
decoder_hidden = 256
fft_conv1d_filter_size = 1024
fft_conv1d_kernel_size = (9, 1)
encoder_dropout = 0.2
decoder_dropout = 0.2

variance_predictor_filter_size = 256
variance_predictor_kernel_size = 3
variance_predictor_dropout = 0.5

max_seq_len = 1000


# Quantization for F0 and energy
### for Talromur ###
f0_min = 71.0
f0_max = 799.4
energy_min = 0.0
energy_max = 172.0
### for LJSpeech ###
#f0_min = 71.0
#f0_max = 795.8
#energy_min = 0.0
#energy_max = 315.0
### for Blizzard2013 ###
#f0_min = 71.0
#f0_max = 786.7
#energy_min = 21.23
#energy_max = 101.02

n_bins = 256


# Checkpoints and synthesis path
preprocessed_path = os.path.join("./preprocessed/", dataset)
checkpoint_path = os.path.join("./ckpt/", dataset)
synth_path = os.path.join("./synth/", dataset)
eval_path = os.path.join("./eval/", dataset)
test_path = os.path.join("./test/", dataset)
log_path = os.path.join("./log/", dataset)
#test_path = "./results"
test_path = "./sample_outputs"
#test_path = "/work/gunnar/herma_test_setningar"


# Optimizer
batch_size = 16
epochs = 1000
n_warm_up_step = 4000
grad_clip_thresh = 1.0
acc_steps = 1

betas = (0.9, 0.98)
eps = 1e-9
weight_decay = 0.


# Vocoder
vocoder = 'melgan'  # 'waveglow' or 'melgan'


# Log-scaled duration
log_offset = 1.


# Save, log and synthesis
save_step = 10000
synth_step = 1000
eval_step = 1000
eval_size = 256
log_step = 1000
clear_Time = 20
