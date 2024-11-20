import os
from matplotlib import pyplot as plt
import librosa
import numpy as np

def get_audio_paths(paths: list, file_extensions=['.wav', '.flac']):
    audio_paths = []
    if isinstance(paths, str):
        paths = [paths]
        
    for path in paths:  
        for root, dirs, files in os.walk(path):
            audio_paths += [os.path.join(root, file) for file in files if os.path.splitext(file)[-1].lower() in file_extensions]
                        
    audio_paths.sort(key=lambda x: os.path.split(x)[-1])
    
    return audio_paths

def draw_spec(x, figsize=(10, 6), title='', n_fft=2048,
              win_len=1024, hop_len=256, sr=48000, cmap='inferno',
              vmin=-50, vmax=40, colorbar=True,
            #   fmax=None,
              ylim=None,
              title_fontsize=10,
              label_fontsize=8):
    # if not fmax: fmax = sr/2
    fig = plt.figure(figsize=figsize)
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_len, win_length=win_len)
    stft = 20 * np.log10(np.clip(np.abs(stft), a_min=1e-8, a_max=None))


    plt.imshow(stft, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
               origin='lower', extent=[0, len(x) / sr, 0, sr / 2])

    if colorbar: plt.colorbar()
    plt.xlabel('Time (s)', fontsize=label_fontsize)
    plt.ylabel('Frequency (Hz)', fontsize=label_fontsize)
    if ylim is None:
        ylim = (0, sr / 2)
    plt.ylim(*ylim)
    plt.title(title, fontsize=title_fontsize)
    plt.show()
    