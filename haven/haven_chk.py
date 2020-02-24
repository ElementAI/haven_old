import shutil
import os 

from . import haven_utils as hu


def delete_experiment(savedir, backup_flag=False):
    """[summary]
    
    Parameters
    ----------
    savedir : [type]
        [description]
    backup_flag : bool, optional
        [description], by default False
    """
    # get experiment id
    exp_id = os.path.split(savedir)[-1]
    assert(len(exp_id) == 32)

    # get paths
    savedir_base = os.path.dirname(savedir)
    savedir = os.path.join(savedir_base, exp_id)

    if backup_flag:
        # create 'deleted' folder 
        dst = os.path.join(savedir_base, 'deleted', exp_id)
        os.makedirs(dst, exist_ok=True)

        if os.path.exists(dst):
            shutil.rmtree(dst)
    
    if os.path.exists(savedir):
        if backup_flag:
            # moves folder to 'deleted'
            shutil.move(savedir, dst)
        else:
            # delete experiment folder 
            shutil.rmtree(savedir)

    # make sure the experiment doesn't exist anymore
    assert(not os.path.exists(savedir))


def delete_and_backup_experiment(savedir):
    """[summary]
    
    Parameters
    ----------
    savedir : [type]
        [description]
    """
    # delete and backup experiment
    delete_experiment(savedir, backup_flag=True)

def get_savedir(exp_dict, savedir_base):
    """[summary]
    
    Parameters
    ----------
    exp_dict : [type]
        [description]
    savedir_base : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    # get experiment savedir
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)
    return savedir
