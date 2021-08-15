import os


def file_name_walk(file_dir):
    with open("imgList.txt", "w") as f:
        for root, dirs, files in os.walk(file_dir):
            if dirs:
                for dir in dirs:
                    for file in files:
                        if '.jpg' in file:
                            f.write(root+dir+file+'\n')
            else:
                for file in files:
                    if '.jpg' in file:
                        f.write(root+file+'\n')
        f.close()


if __name__ == '__main__':
    file_name_walk('Training400')
