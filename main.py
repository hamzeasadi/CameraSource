
import os
from pprint import pprint
import ffmpeg




if __name__ == "__main__":
    print(__file__)
    
    file_path:str = "/home/hasadi/project/Dataset/socraties/100/Eurecom_100_video_001.mp4"
    pprint(ffmpeg.probe(file_path))