import shutil
import os 

from . import haven_utils as hu


def delete_experiment(savedir, backup_flag=False):
    """Delete an experiment. If the backup_flag is true it moves the experiment
    to the delete folder.
    
    Parameters
    ----------
    savedir : str
        Directory of the experiment
    backup_flag : bool, optional
        If true, instead of deleted is moved to delete folder, by default False
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
    """Delete an experiment and make a backup (Movo to the trash)
    
    Parameters
    ----------
    savedir : str
        Directory of the experiment
    """
    # delete and backup experiment
    delete_experiment(savedir, backup_flag=True)

def get_savedir(exp_dict, savedir_base):
    """[summary]
    
    Parameters
    ----------
    exp_dict : dict
        Dictionary describing the hyperparameters of an experiment
    savedir_base : str
        Directory where the experiments are saved
    
    Returns
    -------
    str
        Directory of the experiment
    """
    # get experiment savedir
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)
    return savedir


def copy_checkpoints_with_key_change(exp_list, savedir_base, filterby_list):
    """[summary]
    
    Parameters
    ----------
    exp_dict : dict
        Dictionary describing the hyperparameters of an experiment
    savedir_base : str
        Directory where the experiments are saved
    
    Returns
    -------
    str
        Directory of the experiment
    """
    # get experiment savedir
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)
    return savedir