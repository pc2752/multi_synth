import os,re
import numpy as np
import vamp
import re
import matplotlib.pyplot as plt

import config

def note_str_to_num(note, base_octave=-1):
    """Convert note pitch as string to MIDI note number."""
    patt = re.match('^([CDEFGABcdefgab])([b#]*)(-?)(\d+)$', note)
    if patt is None:
        raise ValueError('invalid note string "{}"'.format(note))
    base_map = {'C': 0,
                'D': 2,
                'E': 4,
                'F': 5,
                'G': 7,
                'A': 9,
                'B': 11}
    base, modifiers, sign, octave = patt.groups()
    base_num = base_map[base.upper()]
    mod_num = -modifiers.count('b') + modifiers.count('#')
    sign_mul = -1 if sign == '-' else 1
    octave_num = 12*int(octave)*sign_mul - 12*base_octave
    note_num = base_num + mod_num + octave_num
    if note_num < 0 or note_num >= 128:
        raise ValueError('note string "{}" resulted in out-of-bounds note number {:d}'.format(note, note_num))
    return note_num


def note_num_to_str(note, base_octave=-1):
    """Convert MIDI note number to note pitch as string."""
    base = note % 12
    # XXX: base_map should probably depend on key
    base_map = ['C',
                'C#',
                'D',
                'D#',
                'E',
                'F',
                'F#',
                'G',
                'G#',
                'A',
                'A#',
                'B']
    base_note = note%12
    octave = int(np.floor(note/12)) + base_octave
    return '{}{:d}'.format(base_map[base_note], octave)

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
        st = int(np.round(float(st)/config.hoptime)/10000000)
        en = int(np.round(float(end)/config.hoptime)/10000000)
        if phonote=='pau' or phonote=='br':
            phonote='sil'
        phonemes.append([st,en,phonote])


    strings_p = np.zeros((phonemes[-1][1],2))

    for i in range(len(phonemes)):
        pho=phonemes[i]
        value = config.phonemas.index(pho[2])
        context = np.linspace(0.0,1.0, len(strings_p[pho[0]:pho[1]+1,0]))
        strings_p[pho[0]:pho[1]+1,0] = value
        strings_p[pho[0]:pho[1]+1,1] = context
    # import pdb;pdb.set_trace()

    return strings_p


def process_notes_file(filename, stft_len):

    lab_f = open(filename)
    # note_f=open(in_dir+lf[:-4]+'.notes')
    phos = lab_f.readlines()
    lab_f.close()

    phonemes=[]

    # prev = ['sil']

    for count, pho in enumerate(phos):
        st,end,phonote=pho.split()
        note, combo = phonote.split('/p:')
        # if combo != 'sil' and prev != 'sil':
        #     combo = combo.split('-')
        #     combo = prev + combo
        #     prev = [combo[-1]]
        #     if count >0:
        #         phonemes[-1][-1].append(combo[1])
        # else:
        #     combo = ['sil']
        #     prev = ['sil']
        #     if count >0:
        #         phonemes[-1][-1].append(['sil'])

        
        if note == 'xx':
            note_num = 0
        else:
            note_num = note_str_to_num(note)

        st = int(np.round(float(st)/config.hoptime)/10000000)
        en = int(np.round(float(end)/config.hoptime)/10000000)
        # if phonote=='pau' or phonote=='br':
        #     phonote='sil'
        phonemes.append([st,en,note_num, combo])
    # import pdb;pdb.set_trace()

    strings_p = np.zeros((phonemes[-1][1],2))
    strings_c = np.zeros((phonemes[-1][1],4))

    for i in range(len(phonemes)):
        pho=phonemes[i]
        value = config.notes.index(pho[2])
        
        context = np.linspace(0.0,1.0, len(strings_p[pho[0]:pho[1]+1,0]))
        strings_p[pho[0]:pho[1]+1] = value
        for j, p in enumerate(pho[3].split('-')):
            strings_c[pho[0]:pho[1] + 1, j] = config.phonemas.index(p)+1
        strings_p[pho[0]:pho[1]+1,1] = context
    return strings_p, strings_c.reshape(-1,4)
