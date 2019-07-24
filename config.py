import numpy as np
import tensorflow as tf
import os

ikala_gt_fo_dir = '../datasets/iKala/PitchLabel/'
wav_dir = '../apps/datasets/import_files/nitech_jp_song_f001/'


voice_dir = './voice/'
backing_dir = './backing/'
# log_dir = './log_f0_npss/'
log_dir = './log/'
# 3 for just G1 output with uv, 2 for just G1 output without uv and 1 for a mixture (G1 after 1000), without numbers for both from ebeggining
val_dir = './'
feats_dir = './feats/'

phonemas = ['N', 'p', 'g', 'ch', 'd', 'w', 'o', 'a', 'b', 'ny', 'sh', 'h', 'u', 'z', 'i', 'py', 'f', 'ky', 'm', 'j', 'ry', 's', 'e', 'cl', 'n', 'r', 'sil', 't', 'k', 'y', 'ty', 'ts']

combs = ['s-a-N', 'g-u', 'w-a', 'ch-o-N', 'k-u', 'sh-i-t-e', 'ry-u', 'ch-u', 'o', 'i-cl', 'p-o', 'z-e', 'k-e', 'b-i-n-i', 'ch-i', 'd-a', 'a-cl', 'r-u', 'm-a', 'j-u', 'n-u', 'z-u', 'u-n-o', 'b-e-cl', 'y-u', 'sh-i-k-i', 'd-a-cl', 'k-o', 'z-o', 'k-u-cl', 'g-i', 'j-o', 'ny-a', 'z-u-i', 'ch-i-r-o', 'h-o', 'b-o', 'g-o', 's-a-n-i', 'sh-u', 'b-a', 'sh-i', 'm-e', 'm-i', 'd-o-r-i', 'u', 'p-u', 'j-a', 'N', 'n-e', 'f-u-cl', 'n-i-w-a', 'm-e-N', 'h-i', 'p-i-cl', 'g-e', 't-a', 'm-o', 'py-o', 'ty-u', 'j-i', 'o-N', 'z-u-cl', 'ch-i-N', 'n-o', 's-e', 'h-a', 'y-o', 'ry-a', 'm-u', 'n-a', 'z-a-i', 'ch-a-cl', 'b-e', 'a', 'ch-o', 'd-e', 's-u', 'y-a', 'sh-a-N', 'sil', 'sh-a', 'n-i', 'z-a', 'p-a', 'b-u', 'sil-sil', 'e', 'ch-a', 't-e', 'r-a-i', 'g-a', 't-o-N', 'r-o', 'b-i', 's-o', 'g-a-ch-a', 'p-i-N', 'k-a', 'u-cl', 'r-a', 'o-cl', 'ts-u', 'r-e', 'i-n-i', 't-o', 'k-i', 'j-i-N', 's-a', 'r-a-N', 'k-o-N', 'd-o', 'N-d-e', 'ky-o', 'i', 'r-i-N', 'r-i', 'f-u', 'e-cl', 't-o-r-i', 'p-o-cl', 'sh-o']

notes = [0, 64, 66, 67, 65, 69, 70, 71, 72, 73, 74, 68, 75, 76, 57, 59, 60, 61, 62, 63]

file_list = [x for x in os.listdir(feats_dir) if x.endswith('.hdf5')]
# train_list = file_list[:-1]
val_list = ['015.hdf5','029.hdf5', '040.hdf5']
train_list = [x for x in file_list if x not in val_list]

# singers = ['ADIZ', 'JLEE', 'JTAN', 'KENN', 'MCUR', 'MPOL', 'MPUR', 'NJAT', 'PMAR', 'SAMF', 'VKOW' ,'ZHIY', 'p255', 'p285', 'p260', 'p247', 'p266', 'p364', 'p265', 'p233', 'p341', 'p347', 'p243', 'p300', 'p284', 'p283', 'p239', 'p269', 'p236', 'p281', 'p293', 'p241', 'p240', 'p259', 'p244', 'p271', 'p294', 'p287', 'p263', 'p261', 'p334', 'p323', 'p227', 'p282', 'p313', 'p248', 'p277', 'p297', 'p314', 'p250', 'p335', 'p374', 'p315', 'p304', 'p298', 'p288', 'p234', 'p310', 'p262', 'p329', 'p251', 'p330', 'p339', 'p312', 'p256', 'p258', 'p231', 'p249', 'p317', 'p301', 'p292', 'p306', 'p360', 'p272', 'p316', 'p311', 'p308', 'p318', 'p229', 'p245', 'p361', 'p232', 'p257', 'p264', 'p237', 'p226', 'p246', 'p351', 'p270', 'p228', 'p286', 'p267', 'p376', 'p333', 'p252', 'p253', 'p345', 'p254', 'p278', 'p336', 'p268', 'p363', 'p326', 'p303', 'p362', 'p295', 'p274', 'p273', 'p305', 'p343', 'p276', 'p275', 'p225', 'p238', 'p302', 'p279', 'p307', 'p299', 'p340', 'p280', 'p230']
#
#
# vctk_speakers = ['p255', 'p285', 'p260', 'p247', 'p266', 'p364', 'p265', 'p233', 'p341', 'p347', 'p243', 'p300', 'p284', 'p283', 'p239', 'p269', 'p236', 'p281', 'p293', 'p241', 'p240', 'p259', 'p244', 'p271', 'p294', 'p287', 'p263', 'p261', 'p334', 'p323', 'p227', 'p282', 'p313', 'p248', 'p277', 'p297', 'p314', 'p250', 'p335', 'p374', 'p315', 'p304', 'p298', 'p288', 'p234', 'p310', 'p262', 'p329', 'p251', 'p330', 'p339', 'p312', 'p256', 'p258', 'p231', 'p249', 'p317', 'p301', 'p292', 'p306', 'p360', 'p272', 'p316', 'p311', 'p308', 'p318', 'p229', 'p245', 'p361', 'p232', 'p257', 'p264', 'p237', 'p226', 'p246', 'p351', 'p270', 'p228', 'p286', 'p267', 'p376', 'p333', 'p252', 'p253', 'p345', 'p254', 'p278', 'p336', 'p268', 'p363', 'p326', 'p303', 'p362', 'p295', 'p274', 'p273', 'p305', 'p343', 'p276', 'p275', 'p225', 'p238', 'p302', 'p279', 'p307', 'p299', 'p340', 'p280', 'p230']


#FFT Parameters
fs = 44100.0
nfft = 1024
hopsize = 256
hoptime = hopsize/fs
window = np.hanning(nfft)

max_models_to_keep = 10

#CQT Parameters
fmin = 32.70
bins_per_octave = 60
n_octaves = 6
cqt_bins = bins_per_octave*n_octaves
harmonics = [0.5, 1, 2, 3, 4, 5]

comp_mode = 'mfsc'

num_epochs = 2000

print_every = 1
validate_every = 1
save_every = 10

batches_per_epoch_train = 100
batches_per_epoch_val = 10

batch_size = 30
samples_per_file = 30

max_phr_len = 256

filter_len = 5
encoder_layers = 8
filters = 32

wavenet_layers = 7
rec_field = 2**wavenet_layers
wavenet_filters = 64
