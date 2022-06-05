import os
import glob
"""
under your file directory, there are two sub-directories: Glaucoma and Non-Glaucoma
"""


def file_name_walk(file_dir):
    with open("imgList.txt", "w") as f:
        files = sorted(glob.glob(file_dir + "/*/*.jpg"))
        files = list(map(lambda x: x+"\n", files))
        f.writelines(files)
        f.close()


def another_file_name_walk(file_dir):
    with open("segList.txt", "w") as f:
        files = sorted(glob.glob(file_dir + "/*/*.bmp"))
        files = list(map(lambda x: x + "\n", files))
        f.writelines(files)
        f.close()


if __name__ == '__main__':
    file_name_walk('../Backup/Training400')
    another_file_name_walk('../Backup/Disc_Cup_Masks')
