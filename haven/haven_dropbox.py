def generate_dropbox_script(dropbox_outdir_base, access_token):
    script = ("""
#import sys
#!{sys.executable} -m pip install dropbox --user

dropbox_outdir_base = '%s'
access_token = '%s'
out_fname = os.path.join(dropbox_outdir_base,results_fname)
hr.upload_file_to_dropbox(src_fname, out_fname, access_token)
print('saved: https://www.dropbox.com/home/%%s' %% out_fname)
          """ % (dropbox_outdir_base, access_token))
    return script