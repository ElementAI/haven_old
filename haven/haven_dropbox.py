import os 
import tqdm

def generate_dropbox_script(dropbox_outdir_base, access_token):
    script = ("""
#import sys
#!{sys.executable} -m pip install dropbox --user

dropbox_outdir_base = '%s'
access_token = '%s'
out_fname = os.path.join(dropbox_outdir_base,results_fname)
hd.upload_file_to_dropbox(src_fname, out_fname, access_token)
print('saved: https://www.dropbox.com/home/%%s' %% out_fname)
          """ % (dropbox_outdir_base, access_token))
    return script

def upload_score_list_to_dropbox(exp_id_list, savedir_base, outdir_base, access_token):
    import dropbox
    dbx = dropbox.Dropbox(access_token)
      
    try:
        dbx.files_create_folder(outdir_base)
    except:
        pass
    # API v2
    for exp_id in exp_id_list:
        savedir = os.path.join(savedir_base, exp_id)
        outdir = os.path.join(outdir_base, exp_id)

        try:
            dbx.files_create_folder(outdir)
        except:
            pass

        with open(os.path.join(savedir, "exp_dict.json", 'rb')) as f:
            dbx.files_upload(f.read(), outdir+"/exp_dict.json")

        with open(os.path.join(savedir, "score_list.pkl", 'rb')) as f:
            dbx.files_upload(f.read(), outdir+"/score_list.pkl")

        print('saved: https://www.dropbox.com/home/%s' % outdir)

    
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

        abs_path = "%s/%s/score_list.pkl" % (savedir_base, exp_id)
        rel_path = "%s/score_list.pkl" % exp_id
        if not os.path.exists(abs_path):
            continue

        zipf.write(abs_path, rel_path)

        abs_path = "%s/%s/exp_dict.json" % (savedir_base, exp_id)
        rel_path = "%s/exp_dict.json" % exp_id
        zipf.write(abs_path, rel_path)
        
        n_zipped += 1

    
    zipf.close()
    print('zipped: %d/%d exps in %s' % (n_zipped, len(exp_id_list), src_fname))