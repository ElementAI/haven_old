import shutil
import os 

from . import haven_utils as hu


def delete_and_backup_experiment(savedir):
    exp_id = os.path.split(savedir)[-1]
    assert(len(exp_id) == 32)
    hu.fname_parent(savedir)
    savedir_base = os.path.dirname(os.path.dirname(savedir))
    
    savedir = savedir_base + "/%s/" % exp_id
    dst = savedir_base + "/deleted/" + "/%s" % exp_id
    os.makedirs(dst, exist_ok=True)

    if os.path.exists(dst):
        shutil.rmtree(dst)

    if os.path.exists(savedir):
        shutil.move(savedir, dst)

def get_savedir(exp_dict, savedir_base):
    exp_id = hu.hash_dict(exp_dict)
    savedir = "%s/%s" % (savedir_base, exp_id)
    return savedir
