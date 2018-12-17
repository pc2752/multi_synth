import os,re
import numpy as np
import vamp

import config

def rock(audio):
    jojo = vamp.collect(audio, config.fs, "pyin:pyin", step_size=config.hopsize, output="notes")

    import pdb;pdb.set_trace()


def process_lab_file(filename, stft_len):

    lab_f = open(filename)
    # note_f=open(in_dir+lf[:-4]+'.notes')
    phos = lab_f.readlines()
    lab_f.close()

    phonemes=[]

    for pho in phos:
        st,end,phonote=pho.split()
        # import pdb;pdb.set_trace()
        st = int(np.round(float(st)/config.hoptime))
        en = int(np.round(float(end)/config.hoptime))
        if phonote=='pau' or phonote=='br':
            phonote='sil'
        phonemes.append([st,en,phonote])

    div_fac = float(end)/stft_len

    if abs(div_fac - config.hoptime) > 0.001 :

        # import pdb;pdb.set_trace()

        return [], False
    else:

        strings_p = np.zeros(phonemes[-1][1])

        for i in range(len(phonemes)):
            pho=phonemes[i]
            value = config.phonemas.index(pho[2])
            strings_p[pho[0]:pho[1]+1] = value

        return strings_p.reshape(-1,1), True
        # phonemas.add(phonote)

    



    # for i in range(len(phonemes)):
    #     phonemes[i][0] = int(float(phonemes[i][0])/div_fac)
    #     phonemes[i][1] = int(float(phonemes[i][1])/div_fac)

    import pdb;pdb.set_trace()

    # phonemes[-1][1] = stft_len