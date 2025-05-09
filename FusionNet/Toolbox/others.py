from datetime import datetime
import re

def dir_name(data_path, ckpt, time=False):
    ckpt_num = re.search(r"(\d+)", ckpt).group(1)
    if time:
        cur_time = "_" + datetime.now().strftime("%y%m%d-%H%M")
    else:
        cur_time = ""

    if "wv3" in data_path and "OrigScale" in data_path:
        return "wv3_full" + "_" + ckpt_num + cur_time
    elif "wv2" in data_path and "OrigScale" in data_path:
        return "wv2_full" + "_" + ckpt_num + cur_time
    elif "wv3" in data_path and "OrigScale" not in data_path:
        return "wv3_reduced" + "_" + ckpt_num + cur_time
    elif "wv2" in data_path and "OrigScale" not in data_path:
        return "wv2_reduced" + "_" + ckpt_num + cur_time