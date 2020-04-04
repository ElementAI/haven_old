import os 
import tqdm
import shutil
from . import haven_utils as hu 
from . import haven_jupyter as hj


def to_dropbox(exp_list, savedir_base, dropbox_path, access_token, zipname):
    """[summary]
    
    Parameters
    ----------
    exp_list : [type]
        [description]
    savedir_base : [type]
        [description]
    dropbox_path : [type]
        [description]
    access_token : [type]
        [description]
    """
    # zip files 
    exp_id_list = [hu.hash_dict(exp_dict) for exp_dict in exp_list]
    src_fname = os.path.join(savedir_base, zipname)
    out_fname = os.path.join(dropbox_path, zipname)
    zipdir(exp_id_list, savedir_base, src_fname)

    upload_file_to_dropbox(src_fname, out_fname, access_token)
    print('saved: https://www.dropbox.com/home/%s' % out_fname)

    
def upload_file_to_dropbox(src_fname, out_fname, access_token):
    import dropbox
    dbx = dropbox.Dropbox(access_token)
    try:
        dbx.files_delete_v2(out_fname)
    except:
        pass
    with open(src_fname, 'rb') as f:
        dbx.files_upload(f.read(), out_fname)


def zipdir(exp_id_list, savedir_base, src_fname, add_jupyter=True, verbose=1):
    import zipfile
    zipf = zipfile.ZipFile(src_fname, 'w', zipfile.ZIP_DEFLATED)

    # ziph is zipfile handle
    if add_jupyter:
        abs_path = os.path.join(savedir_base, 'results.ipynb')
        hj.create_jupyter(fname=abs_path, 
                        savedir_base='results/', overwrite=False, print_url=False,
                        create_notebook=True)
        
        rel_path = 'results.ipynb'
        zipf.write(abs_path, rel_path)
        os.remove(abs_path)

    n_zipped = 0
    if verbose:
        tqdm_bar = tqdm.tqdm
    else:
        tqdm_bar = lambda x: x
        
    for exp_id in tqdm_bar(exp_id_list):
        if not os.path.isdir(os.path.join(savedir_base, exp_id)):
            continue
            
        for fname in ['score_list.pkl', "exp_dict.json"]:
            abs_path = os.path.join(savedir_base, exp_id, fname)
            rel_path = os.path.join( "results", exp_id, fname)
            if os.path.exists(abs_path):
                zipf.write(abs_path, rel_path)

        n_zipped += 1

    zipf.close()
    if verbose:
        print('Zipped: %d/%d exps in %s' % (n_zipped, len(exp_id_list), src_fname))