import os
import datetime
import socket
import parameter

T = parameter.T
num_partition = parameter.num_partition
#path = "/Users/garammasala/sss21"

def mk_folder(T = T, num_partition = num_partition):
    new_dir_path = "experiment_T={}_N={}".format(T, num_partition)
    if not os.path.exists(new_dir_path):
        os.mkdir(new_dir_path)
    return new_dir_path