
import numpy as np
import matplotlib.pyplot as plt
from essentia import Pool
from essentia import run
import essentia.streaming as es
import essentia.standard as standard

from essentia.standard import *
import utils

kwargs = {
'inputSize': 4096,
'minFrequency': 65.41,
'maxFrequency': 6000,
'binsPerOctave': 48,
'sampleRate': 44100,
'rasterize': 'full',
'phaseMode': 'global',
'gamma': 0,
'normalize': 'none',
'window': 'hannnsgcq',
}

def get_cqt(audio):
    audio = essentia.array(audio)
    w = Windowing(type = 'hann')

    cqt = []
    dcs =[]
    nys =[]
    CQStand = standard.NSGConstantQ(**kwargs)
    for frame in FrameGenerator(audio, frameSize=4096, hopSize=2048, startFromZero=True):
        cqt_frame, dc_frame, ny_frame = CQStand(w(frame))
        h_size = cqt_frame.shape[1]
        cqt.append(cqt_frame.T)
        dcs.append(dc_frame)
        nys.append(ny_frame)
    return np.vstack(cqt), dcs, nys, h_size

def inver_cqt(cqt, dcs, nys, h_size):
    CQIStand = standard.NSGIConstantQ(**kwargs)
    recFrame = []

    for j, i in enumerate(range(0,cqt.shape[0], h_size)):
        cqt_frame = cqt[i:i+h_size]
        # import pdb;pdb.set_trace()
        inv_cqt_frame = CQIStand(cqt_frame.T, dcs[j], nys[j])
        recFrame.append(inv_cqt_frame)
        utils.progress(j, int(cqt.shape[0]/h_size), "Inverse Done")
    frameSize = kwargs['inputSize']

    y = recFrame[0]

    invWindow = Windowing(type='triangular',normalized=False, zeroPhase=False)(standard.essentia.array(np.ones(frameSize)))


    for i in range(1,len(recFrame)):
        y = np.hstack([y,np.zeros(int(frameSize/2))])
        y[-frameSize:] = y[-frameSize:] + recFrame[i] 
        utils.progress(i, len(recFrame), "Overlap Done")
        
    y = y[int(frameSize/2):]

    return y









