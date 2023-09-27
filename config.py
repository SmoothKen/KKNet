
# exp_name = "dummy"
# file path
model_type = "dummy" # MCDNN, FTANet, MSNet, MLDRNet, dummy
data_path = "data"
train_file = "data/train_data.txt"
test_file = [
    "data/test_adc.txt",
    "data/test_mirex.txt",
    "data/test_melody.txt"
]
    
save_path = "model_backup"
resume_checkpoint = "model_backup/bestk_0.ckpt"
# resume_checkpoint = "model_backup/TO-FTANet_mirex_best.ckpt"
# "model_backup/TO-FTANet_adc_best.ckpt" # the model checkpoint

# train config
batch_size = 10
lr = 1e-4
epochs = 1000
n_workers = 4
save_period = 1
tone_class = 12 # 60
octave_class = 8 # 6
random_seed = 19961206
max_epoch = 500
freq_bin = 360

ablation_mode = "single" # single, tcfp, spl, spat, all, a parameter inherited from TONet's code, and remain single for our simplified model

include_model_tweak = False	# small tweak on vocal detection bin, unsure of its effectiveness
include_loss_component = False # loss component for prediction stability
include_adjusted_exp = False # z-transform
apply_median_filter = True # median filter baseline

startfreq = 32
stopfreq = 2050
cfp_dir = "cfp_360_new"

# feature config
fs = 44100.0
hop = 441.0
octave_res = 60
seg_dur = 1.28 # sec
seg_frame = int(seg_dur * fs // hop)
shift_dur = 1.28 # sec
shift_frame = int(shift_dur * fs // hop)

network_time_shrink_size = 8
