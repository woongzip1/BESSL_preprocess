from matplotlib import pyplot as plt
import soundfile
from utility import get_audio_paths
import os
import librosa
from tqdm import tqdm

""" import core extractor module
Extracts core & aligns the length of the GT dataset
! GT dataset length adjustment
! Core dataset length adjusted and saved

Bug files cannot decode
[9889:] index - /home/woongjib/Projects/Dataset_Crop/SBR_12/FSD50K/FSD50K.dev_audio/1767_mono.m4a

"""

import sys
sys.path.append("/home/woongjib/Projects/SBR/aac_analysis")
from utils import profile_decoding_output

# os dir
orig_dir = os.getcwd()
os.chdir("/home/woongjib/Projects/SBR/aac_analysis")

# path
gt_path = "/home/woongjib/Projects/Dataset_Crop/GT/FSD50K" 
# gt_path = "/home/woongjib/Projects/Dataset_Crop/GT/MUSDB18" 
# gt_path = "/home/woongjib/Projects/Dataset_Crop/GT/VCTK" 
gt_paths = get_audio_paths(gt_path)

print(gt_paths[0])

"""
Bug files cannot decode
[9889:] index - /home/woongjib/Projects/Dataset_Crop/SBR_12/FSD50K/FSD50K.dev_audio/1767_mono.m4a

"""

bar = tqdm(gt_paths[:])
for gt_path in bar:
    sbr_path = gt_path.replace("GT","SBRs/SBR_12").replace(".wav",".m4a")
    core_path = gt_path.replace("GT","SBR_12_Core")
    
    sbr_20_path = gt_path.replace("GT","SBRs/SBR_20").replace(".wav",".m4a")
    core_20_path = gt_path.replace("GT","SBR_20_Core")
    
    ## Core Extraction & Length Adjustment
    outdict = profile_decoding_output(sbr_path)
    core = outdict['core'] / 32768

    outdict_20 = profile_decoding_output(sbr_20_path)
    core_20 = outdict_20['core'] / 32768
    
    gt,_ = librosa.load(gt_path, sr=None)
    # sbr,_ = librosa.load(sbr_path, sr=None)    
    # sbr_20,_ = librosa.load(sbr_20_path, sr=None)

    print(gt_path)
    print(sbr_path)
    print(core_path)
    
    gt = gt[129:]
    # sbr = sbr[128:128+len(gt)]
    # sbr_20 = sbr_20[128:128+len(gt)]
    core = core[:len(gt)]
    core_20 = core_20[:len(gt)]
    assert gt.shape == core.shape == core_20.shape, f"Array lengths are not the same!:{gt_path}"    
    # print(gt.shape, core.shape, core_20.shape, end='\n')
    
    ## Plot
    # plt.plot(gt, label='gt')
    # plt.plot(sbr, label='sbr')
    # plt.plot(core, label='core')
    # plt.plot(core_20, label='core2')
    # start = 13500
    # plt.xlim(start, start+200)
    # plt.legend()
    # plt.show()
    
    ## Makedirs
    os.makedirs(os.path.dirname(core_path), exist_ok=True)
    os.makedirs(os.path.dirname(core_20_path), exist_ok=True)
    
    ## Save files
    soundfile.write(gt_path, gt, samplerate=48000)
    soundfile.write(core_path, core, samplerate=48000)
    soundfile.write(core_20_path, core_20, samplerate=48000) 

os.chdir(orig_dir)