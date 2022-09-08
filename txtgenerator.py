import os
import math
import numpy as np
import pandas as pd

train_path = "/home/sda/Users/YT/LCDnet/txtlabel/train_10.txt"
val_path = "/home/sda/Users/YT/LCDnet/txtlabel/val_10.txt"
homepath= "/home/sda/Users/YT/DRIVE"


def main():
    standard = pd.read_excel("/home/sda/Users/YT/standard.xlsx",index_col=0)
    imagespath = os.path.join(homepath,"training/images_10")
    images = os.listdir(imagespath)
    for image in images:
        verseid = image.split("_")[0]
        verseid = verseid.split("verse")[1]
        row = []
        prefix = image.split("_")[0]+"_"+image.split("_")[1]+"_"
        row.append(os.path.join(imagespath,image))
        row.append(os.path.join(homepath,"training/label_10/"+prefix+"manual1.png"))
        for index, val in standard.loc[verseid].iteritems():
            if type(val) != np.float64 and type(val) != float:
                if "C" in val:
                    row.append(2)
                elif "B" in val:
                    row.append(1)
                else:
                    row.append(0)
                break
        f = open(train_path,'a')
        f.write(row[0]+" "+row[1]+" "+str(row[2])+"\n")
        f.close()
        

if __name__== "__main__":
    main()

