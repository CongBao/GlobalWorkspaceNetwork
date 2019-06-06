# extract audio from video
# require ffmpeg

import os
import subprocess

from tqdm import tqdm

def extract(v_dir, a_dir):
    for v_name in tqdm(os.listdir(v_dir)):
        a_name = v_name.replace('.mp4', '.wav')
        i = os.path.join(v_dir, v_name)
        o = os.path.join(a_dir, a_name)
        cmd = 'ffmpeg -i ' + i + ' -f wav -ab 192000 -vn ' + o
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    v_dir = '/home/baocong/Data/FirstImpressionV2/video/test/'
    a_dir = '/home/baocong/Data/FirstImpressionV2/audio/test/'
    extract(v_dir, a_dir)
