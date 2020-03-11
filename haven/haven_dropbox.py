import os 
import tqdm
from . import haven_utils as hu 


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

def zipdir(exp_id_list, savedir_base, src_fname):
    import zipfile
    zipf = zipfile.ZipFile(src_fname, 'w', zipfile.ZIP_DEFLATED)

    # ziph is zipfile handle
    n_zipped = 0
    for exp_id in tqdm.tqdm(exp_id_list):
        if not os.path.isdir(os.path.join(savedir_base, exp_id)):
            continue

        abs_path = os.path.join(savedir_base, exp_id, 'score_list.pkl')
        rel_path = os.path.join(exp_id, "score_list.pkl")
        if not os.path.exists(abs_path):
            continue

        zipf.write(abs_path, rel_path)

        abs_path = os.path.join(savedir_base, exp_id, "exp_dict.json")
        rel_path = os.path.join(exp_id, "exp_dict.json")
        zipf.write(abs_path, rel_path)
        
        n_zipped += 1

    
    zipf.close()
    print('zipped: %d/%d exps in %s' % (n_zipped, len(exp_id_list), src_fname))