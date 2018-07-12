import os 
from glob import glob 
import numpy as np
import cv2
path = './cam1/Test/'
dirs=sorted(glob(os.path.join(path,'*')))
print(dirs)
clips = np.empty([8,
                  128,
                    64,
                 (3 * (4 + 1))])
print(clips.size)
for num in range(len(dirs)/8+1):
    ep_dirs=dirs[num*(8):(num+1)*8]
    #print(ep_dirs)
    for clip_num,ep_dir in enumerate(ep_dirs):
        print(ep_dir)   #8
        
        ep_frame_paths=sorted(glob(os.path.join(ep_dir,'*')))
        #print(ep_frame_paths)
        start_index=len(ep_frame_paths)-(4)
        clip_frame_paths=ep_frame_paths[start_index:start_index+(4+1)]
        #print(clip_frame_paths)   #the last 4 frame
        for frame_num,frame_path in enumerate(clip_frame_paths):
            frame=imread(frame_path,mode='RGB')
            #print(frame_num)
            clips[clip_num, :, :, frame_num * 3:(frame_num + 1) * 3] = frame