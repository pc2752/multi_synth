import numpy as np
import tensorflow as tf

ikala_gt_fo_dir = '../datasets/iKala/PitchLabel/'
wav_dir = '../datasets/iKala/Wavfile/'
wav_dir_nus = '../datasets/nus-smc-corpus_48/'
wav_dir_mus = '../datasets/musdb18/train/'
wav_dir_mir = '../datasets/MIR1k/'
wav_dir_med = '../datasets/medleydB/'
wav_dir_vctk = '../datasets/VCTK/VCTK_files/VCTK-Corpus/wav48/'
wav_dir_vctk_lab = '../datasets/VCTK/VCTK_files/VCTK-Corpus/forPritish/'


voice_dir = './voice/'
backing_dir = './backing/'
log_dir = './log_feat_to_feat_just_GAN/'


phonemas = ['t', 'y', 'l', 'k', 'aa', 'jh', 'ae', 'ng', 'ah', 'hh', 'z', 'ey', 'f', 'uw', 'iy', 'ay', 'b', 's', 'd', 'sil', 'p', 'n', 'sh', 'ao', 'g', 'ch', 'ih', 'eh', 'aw', 'sp', 'oy', 'th', 'w', 'ow', 'v', 'uh', 'm', 'er', 'zh', 'r', 'dh', 'ax']


singers = ['ADIZ', 'JLEE', 'JTAN', 'KENN', 'MCUR', 'MPOL', 'MPUR', 'NJAT', 'PMAR', 'SAMF', 'VKOW' ,'ZHIY', 'p255', 'p285', 'p260', 'p247', 'p266', 'p364', 'p265', 'p233', 'p341', 'p347', 'p243', 'p300', 'p284', 'p283', 'p239', 'p269', 'p236', 'p281', 'p293', 'p241', 'p240', 'p259', 'p244', 'p271', 'p294', 'p287', 'p263', 'p261', 'p334', 'p323', 'p227', 'p282', 'p313', 'p248', 'p277', 'p297', 'p314', 'p250', 'p335', 'p374', 'p315', 'p304', 'p298', 'p288', 'p234', 'p310', 'p262', 'p329', 'p251', 'p330', 'p339', 'p312', 'p256', 'p258', 'p231', 'p249', 'p317', 'p301', 'p292', 'p306', 'p360', 'p272', 'p316', 'p311', 'p308', 'p318', 'p229', 'p245', 'p361', 'p232', 'p257', 'p264', 'p237', 'p226', 'p246', 'p351', 'p270', 'p228', 'p286', 'p267', 'p376', 'p333', 'p252', 'p253', 'p345', 'p254', 'p278', 'p336', 'p268', 'p363', 'p326', 'p303', 'p362', 'p295', 'p274', 'p273', 'p305', 'p343', 'p276', 'p275', 'p225', 'p238', 'p302', 'p279', 'p307', 'p299', 'p340', 'p280', 'p230']


vctk_speakers = ['p255', 'p285', 'p260', 'p247', 'p266', 'p364', 'p265', 'p233', 'p341', 'p347', 'p243', 'p300', 'p284', 'p283', 'p239', 'p269', 'p236', 'p281', 'p293', 'p241', 'p240', 'p259', 'p244', 'p271', 'p294', 'p287', 'p263', 'p261', 'p334', 'p323', 'p227', 'p282', 'p313', 'p248', 'p277', 'p297', 'p314', 'p250', 'p335', 'p374', 'p315', 'p304', 'p298', 'p288', 'p234', 'p310', 'p262', 'p329', 'p251', 'p330', 'p339', 'p312', 'p256', 'p258', 'p231', 'p249', 'p317', 'p301', 'p292', 'p306', 'p360', 'p272', 'p316', 'p311', 'p308', 'p318', 'p229', 'p245', 'p361', 'p232', 'p257', 'p264', 'p237', 'p226', 'p246', 'p351', 'p270', 'p228', 'p286', 'p267', 'p376', 'p333', 'p252', 'p253', 'p345', 'p254', 'p278', 'p336', 'p268', 'p363', 'p326', 'p303', 'p362', 'p295', 'p274', 'p273', 'p305', 'p343', 'p276', 'p275', 'p225', 'p238', 'p302', 'p279', 'p307', 'p299', 'p340', 'p280', 'p230']


#FFT Parameters
fs = 44100.0
nfft = 1024
hopsize = 512
hoptime = hopsize/fs
window = np.hanning(nfft)


#CQT Parameters
fmin = 32.70
bins_per_octave = 60
n_octaves = 6
cqt_bins = bins_per_octave*n_octaves
harmonics = [0.5, 1, 2, 3, 4, 5]

comp_mode = 'mfsc'