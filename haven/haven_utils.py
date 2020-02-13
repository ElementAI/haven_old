import hashlib 
import pickle
import os
import itertools
import torch
import numpy as np
import glob
from PIL import Image
import copy
import time
import scipy.io as io
import pandas as pd
from datetime import datetime
import pytz
import threading
import pylab as plt
import subprocess
import shlex 
import contextlib
import json
import sys


def get_longest_list(listOfLists):
    LL = listOfLists
    longest_list = []

    if LL is None:
        return longest_list

    for L in LL:
        if not isinstance(L, list):
            continue

        if not isinstance(L[0], list):
            L = [L]
        
        if len(L) > len(longest_list):
            longest_list = L

    #print(longest_list)
    return longest_list
    
def get_padding(kernel_size=1):
    return int((kernel_size - 1) / 2)


def cartesian_exp_group(exp_config):
    """Cartesian experiment config.

    It converts the exp_config into a list of experiment configuration by doing
    the cartesian product of the different configuration. It is equivalent to
    do a grid search.

    Args:
        exp_config: Dictionary with the experiment Configuration

    Returns:
        exp_list: List with an experiment dictionary for each individual
        experiment

    """
    # Create the cartesian product
    exp_config_copy = copy.deepcopy(exp_config)
    
    for k,v in exp_config_copy.items():
        if not isinstance(exp_config_copy[k], list):
            exp_config_copy[k] = [v]

    exp_list_raw = (dict(zip(exp_config_copy.keys(), values))
                    for values in itertools.product(*exp_config_copy.values()))

    # Convert into a list
    exp_list = []
    for exp_dict in exp_list_raw:
        exp_list += [exp_dict]
    return exp_list

def hash_dict(dictionary):
    """Create a hash for a dictionary."""
    dict2hash = ""

    for k in sorted(dictionary.keys()):
        if isinstance(dictionary[k], dict):
            v = hash_dict(dictionary[k])
        else:
            v = dictionary[k]

        dict2hash += os.path.join(str(k), str(v))

    return hashlib.md5(dict2hash.encode()).hexdigest()

def load_pkl(fname):
    """Load the content of a pkl file."""
    with open(fname, "rb") as f:
        return pickle.load(f)

def save_pkl(fname, data, with_rename=True, makedirs=True):
    """Save data in pkl format."""
    # Create folder
    if makedirs:
        os.makedirs(os.path.dirname(fname), exist_ok=True)

    # Save file
    if with_rename:
        fname_tmp = fname + "_tmp.pth"
        with open(fname_tmp, "wb") as f:
            pickle.dump(data, f)
        if os.path.exists(fname):
            os.remove(fname)
        os.rename(fname_tmp, fname)
    else:
        with open(fname, "wb") as f:
            pickle.dump(data, f)

def time_to_montreal():  # TODO: Remove commented code
    """Get time in Montreal zone."""
    # Get time
    ts = time.time()

    tz = pytz.timezone('America/Montreal')
    # dt = aware_utc_dt.astimezone(tz)
    dt = datetime.fromtimestamp(ts, tz)

    return dt.strftime("%I:%M %p (%b %d)")

def n2t(x, dtype="float"):
    if isinstance(x, (int, np.int64, float)):
        x = np.array([x])

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x

def t2n(x):
    if isinstance(x, (int, float)):
        return x

    if isinstance(x, (torch.cuda.FloatTensor, torch.cuda.IntTensor,
                      torch.cuda.LongTensor, torch.cuda.DoubleTensor)):
        x = x.cpu().numpy()

    if isinstance(x, (torch.FloatTensor, torch.IntTensor, torch.LongTensor,
                      torch.DoubleTensor)):
        x = x.numpy()

    return x
    
def l2f(X):
    if X.ndim == 3 and (X.shape[0] == 3 or X.shape[0] == 1):
        return X
    if X.ndim == 4 and (X.shape[1] == 3 or X.shape[1] == 1):
        return X

    if X.ndim == 4 and (X.shape[1] < X.shape[3]):
        return X

    # CHANNELS LAST
    if X.ndim == 3:
        return np.transpose(X, (2, 0, 1))
    if X.ndim == 4:
        return np.transpose(X, (0, 3, 1, 2))

    return X

def _denorm(image, mu, var, bgr2rgb=False):
    if image.ndim == 3:
        result = image * var[:, None, None] + mu[:, None, None]
        if bgr2rgb:
            result = result[::-1]
    else:
        result = image * var[None, :, None, None] + mu[None, :, None, None]
        if bgr2rgb:
            result = result[:, ::-1]
    return result


def denormalize(img, mode=0):
    # _img = t2n(img)
    # _img = _img.copy()
    image = t2n(img).copy().astype("float")

    if mode in [1, "rgb"]:
        mu = np.array([0.485, 0.456, 0.406])
        var = np.array([0.229, 0.224, 0.225])
        # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        image = _denorm(image, mu, var)

    elif mode in [2, "bgr"]:
        mu = np.array([102.9801, 115.9465, 122.7717])
        var = np.array([1, 1, 1])
        image = _denorm(image, mu, var, bgr2rgb=True).clip(0, 255).round()

    elif mode in [3, "basic"]:
        mu = np.array([0.5, 0.5, 0.5])
        var = np.array([0.5, 0.5, 0.5])

        image = _denorm(image, mu, var)

    return image

def get_image(imgs,
              mask=None,
              label=False,
              enlarge=0,
              gray=False,
              denorm=0,
              bbox_yxyx=None,
              annList=None,
              pretty=False,
              pointList=None,
              **options):

    imgs = denormalize(imgs, mode=denorm)
    if isinstance(imgs, Image.Image):
        imgs = np.array(imgs)
    if isinstance(mask, Image.Image):
        mask = np.array(mask)

    imgs = t2n(imgs).copy()
    imgs = l2f(imgs)

    if pointList is not None and len(pointList):
        h, w = pointList[0]["h"], pointList[0]["w"]
        mask_points = np.zeros((h, w))
        for p in pointList:
            y, x = p["y"], p["x"]
            mask_points[int(h*y), int(w*x)] = 1
        imgs = maskOnImage(imgs, mask_points, enlarge=1)

    if pretty or annList is not None:
        imgs = pretty_vis(imgs, annList, **options)
        imgs = l2f(imgs)

    if mask is not None and mask.sum() != 0:
        imgs = maskOnImage(imgs, mask, enlarge)

    if bbox_yxyx is not None:
        _, _, h, w = imgs.shape
        mask = bbox_yxyx_2_mask(bbox_yxyx, h, w)
        imgs = maskOnImage(imgs, mask, enlarge=1)

    # LABEL
    elif (not gray) and (label or imgs.ndim == 2 or
                         (imgs.ndim == 3 and imgs.shape[0] != 3) or
                         (imgs.ndim == 4 and imgs.shape[1] != 3)):

        imgs = label2Image(imgs)

        if enlarge:
            imgs = zoom(imgs, 11)

    # Make sure it is 4-dimensional
    if imgs.ndim == 3:
        imgs = imgs[np.newaxis]

    return imgs

def save_image(fname, img, makedirs=True):
    if img.dtype == 'uint8':
        img_pil = Image.fromarray(img)
        img_pil.save(fname)
    else:
        if makedirs:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
        images = f2l(t2n(img))
        imsave(fname , img)

def f2l(X):
    if X.ndim == 3 and (X.shape[2] == 3 or X.shape[2] == 1):
        return X
    if X.ndim == 4 and (X.shape[3] == 3 or X.shape[3] == 1):
        return X

    # CHANNELS FIRST
    if X.ndim == 3:
        return np.transpose(X, (1, 2, 0))
    if X.ndim == 4:
        return np.transpose(X, (0, 2, 3, 1))

    return X


def imsave(fname, arr, size=None):
    from PIL import Image
    arr = f2l(t2n(arr)).squeeze()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    #print(arr.shape)
    if size is not None:
        arr = Image.fromarray(arr)
        arr = arr.resize(size)
        arr = np.array(arr)
 
    img = Image.fromarray(np.uint8(arr * 255))
    img.save(fname)


# ========================================================
# results
import zipfile

def fname_parent(filepath, levels=1):
    common = filepath
    for i in range(levels + 1):
        common = os.path.dirname(common)
    return os.path.relpath(filepath, common)
    
def zipdir(src_dirname, out_fname, include_list=None):
    zipf = zipfile.ZipFile(out_fname, 'w', 
                           zipfile.ZIP_DEFLATED)
    # ziph is zipfile handle
    for root, dirs, files in os.walk(src_dirname):
        for file in files:
            
            if include_list is not None and file not in include_list:
                continue
            
            abs_path = os.path.join(root, file)
            rel_path = fname_parent(abs_path)
            print(rel_path)
            zipf.write(abs_path, rel_path)

    zipf.close()


def zip_score_list(exp_list, savedir_base, out_fname, include_list=None):
    for exp_dict in exp_list:
        exp_id = hash_dict(exp_dict)
        zipdir(os.path.join(savedir_base, exp_id), 
               out_fname, include_list=include_list)


def save_json(fname, data, makedirs=True):
    if makedirs:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)


def flatten_dict(exp_dict):
    result_dict = {}
    for k in exp_dict:
        # print(k, exp_dict)
        if isinstance(exp_dict[k], dict):
            for k2 in exp_dict[k]:
                result_dict[k2] = exp_dict[k][k2]
        else:
            result_dict[k] = exp_dict[k]
    return result_dict

def t2n(x):
    if isinstance(x, (int, float)):
        return x

    if isinstance(x, (torch.cuda.FloatTensor, torch.cuda.IntTensor,
                      torch.cuda.LongTensor, torch.cuda.DoubleTensor)):
        x = x.cpu().numpy()

    if isinstance(x, (torch.FloatTensor, torch.IntTensor, torch.LongTensor,
                      torch.DoubleTensor)):
        x = x.numpy()

    return x


def shrink2roi(img, roi):
    ind = np.where(roi != 0)

    y_min = min(ind[0])
    y_max = max(ind[0])

    x_min = min(ind[1])
    x_max = max(ind[1])

    return img[y_min:y_max, x_min:x_max]


def hash_dict(dictionary):
    """Create a hash for a dictionary."""
    dict2hash = ""
    if not isinstance(dictionary, dict):
        raise ValueError('dictionary is not a dict')
    for k in sorted(dictionary.keys()):
        if isinstance(dictionary[k], dict):
            v = hash_dict(dictionary[k])
        else:
            v = dictionary[k]

        dict2hash += os.path.join(str(k), str(v))

    return hashlib.md5(dict2hash.encode()).hexdigest()

def hash_str(str):
    return hashlib.md5(str.encode()).hexdigest()

def wait_until_safe2load(path, patience=10):
    """Wait until safe to load."""
    # Check if someone is writting in the file
    writing_flag = False
    if os.path.exists(path):
        writing_flag = load_json(path).get("writing")

    # Keep checking until noone is writting there
    waiting = 0
    while writing_flag and waiting < patience:
        time.sleep(.5)
        waiting += 1
        if os.path.exists(path):
            writing_flag = load_json(path).get("writing")

    return writing_flag


def wait_until_safe2save(path, patience=10):
    # Check if someone is reading the file
    reading_flag = False
    if os.path.exists(path):
        reading_flag = load_json(path).get("reading")

    # Keep trying until noone is reading the file
    reading = 0
    while reading_flag and reading < patience:
        time.sleep(.5)
        reading += 1
        if os.path.exists(path):
            reading_flag = load_json(path).get("reading")

    return reading_flag

def load_txt(fname):
    """Load the content of a txt file."""
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines

def torch_load(fname, map_location=None, safe_flag=False):
    """Load the content of a torch file."""
    fname_writing = fname + "_writing_dict.json.tmp"
    fname_reading = fname + "_reading_dict.json.tmp"

    if safe_flag:
        wait_until_safe2load(fname_writing)
        save_json(fname_reading, {"reading": 1})

    obj = torch.load(fname, map_location=map_location)

    if safe_flag:
        save_json(fname_reading, {"reading": 0})

    return obj


def torch_save(fname, obj, safe_flag=False):
    """"Save data in torch format."""
    # Create folder
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    # Define names of temporal files
    fname_tmp = fname + ".tmp"
    fname_writing = fname + "_writing_dict.json.tmp"
    fname_reading = fname + "_reading_dict.json.tmp"

    if safe_flag:
        wait_until_safe2save(fname_reading)
        save_json(os.path.dirname(fname_writing), {"writing": 1})

    torch.save(obj, fname_tmp)
    if os.path.exists(fname):
        os.remove(fname)
    os.rename(fname_tmp, fname)

    if safe_flag:
        save_json(os.path.dirname(fname_writing), {"writing": 0})



def time2mins(time_taken):
    """Convert time into minutes."""
    return time_taken / 60.


class Parallel:
    """Class for run a function in parallel."""

    def __init__(self):
        self.threadList = []
        self.count = 0

    def add(self, func,  *args):
        """Add a funtion."""
        self.threadList += [
            threading.Thread(target=func, name="thread-%d"%self.count,
                             args=args)]
        self.count += 1

    def run(self):
        for thread in self.threadList:
            thread.daemon = True
            print("  > Starting thread %s" % thread.name)
            thread.start()

    def close(self):
        for thread in self.threadList:
            print("  > Joining thread %s" % thread.name)
            thread.join()


def subprocess_call(cmd_string):
    return subprocess.check_output(
        shlex.split(cmd_string), shell=False).decode("utf-8")

def copy_code(src_path, dst_path, verbose=1):
    """Copy code."""
    time.sleep(.5)  # TODO: Why?

    if verbose:
        print("  > Copying code from %s to %s" % (src_path, dst_path))
    # Create destination folder
    os.makedirs(dst_path, exist_ok=True)

    rsync_code = "rsync -av -r -q  --delete-before --exclude='.git/' " \
                 " --exclude='*.pyc' --exclude='__pycache__/' %s %s" % (
                    src_path, dst_path)

    try:
        subprocess_call(rsync_code)
    except subprocess.CalledProcessError as e:
        raise ValueError("Ping stdout output:\n", e.output)  # TODO: Through an error?

    # print("  > Code copied\n")
    time.sleep(.5) 

@contextlib.contextmanager
def random_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)



def loadmat(fname):
    return io.loadmat(fname)

def join_df_list(df_list):
    result_df = df_list[0]
    for i in range(1, len(df_list)):
        result_df = result_df.join(df_list[i], how="outer", lsuffix='_%d'%i, rsuffix='')
    return result_df


def load_json(fname, decode=None):
    with open(fname, "r") as json_file:
        d = json.load(json_file)

    return d

def read_text(fname):
    # READS LINES
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # lines = [line.decode('utf-8').strip() for line in f.readlines()]
    return lines



