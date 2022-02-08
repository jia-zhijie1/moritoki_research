import os
import datetime
import socket

T = 1
num_partition = 100
#path = "/Users/garammasala/sss21"

def mk_folder(T = T, num_partition = num_partition):
    host = socket.gethostname()
    if host == "garamumasarasannoMacBook-Air.local":
        path = "/Users/garammasala/reaserch"
    else:
        path = "/Users/mk042.DESKTOP-K9G0PKU/Desktop/sss21moritoki"
    now = datetime.datetime.now()
    new_dir_path = path + "/csvT{}N{}date{}".format(T, num_partition, now)
    os.mkdir(new_dir_path)
    return new_dir_path