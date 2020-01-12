import cv2 
import pylab as plt
import pandas as pd
from . import haven_utils as hu
import os
import pylab as plt 
import pandas as pd 
import numpy as np
import copy 
import glob 
from itertools import groupby 


def get_exp_list_with_regard_dict(savedir_base, regard_dict):
    exp_list = []
    dir_list = os.listdir(savedir_base)

    for exp_id in dir_list:
        savedir = os.path.join(savedir_base, exp_id)
        fname = savedir + "/exp_dict.json"
        if not os.path.exists(fname):
            continue
        exp_dict = hu.load_json(fname)
        exp_list += [exp_dict]

    exp_list_new = hu.filter_exp_list(exp_list, 
                                      regard_dict=regard_dict, 
                                      disregard_dict=None)
    return exp_list_new



def get_best_exp_dict(exp_list, savedir_base, score_key, lower_is_better=True, return_scores=False):
    scores_dict = []
    if lower_is_better:
        best_score = np.inf 
    else:
        best_score = 0.
    exp_dict_best = None
    
    for exp_dict in exp_list:
        exp_id = hu.hash_dict(exp_dict)
        
        savedir = savedir_base + "/%s/" % exp_id 
        if not os.path.exists(savedir + "/score_list.pkl"):
            continue
            
        score_list_fname = os.path.join(savedir, "score_list.pkl")
        if os.path.exists(score_list_fname):
            score_list = hu.load_pkl(score_list_fname)
            
        
#         print(len(score_list), score)
        # lower is better
        if lower_is_better:
            score = pd.DataFrame(score_list)[score_key].min()
            if best_score >= score:
                best_score = score
                exp_dict_best = exp_dict
        else:
            score = pd.DataFrame(score_list)[score_key].max()
            if best_score <= score:
                best_score = score
                exp_dict_best = exp_dict
        scores_dict += [{'score':score, 'epochs':len(score_list), 'exp_id':exp_id}]
#     print(best_score)
    if exp_dict_best is None:
        raise ValueError('exp_dict_best is None')
    scores_dict += [{'exp_id':hu.hash_dict(exp_dict_best), 'best_score':best_score}]
    if return_scores:
        return exp_dict_best, scores_dict
    return exp_dict_best

def filter_best_results(exp_list, savedir_base, groupby_key_list, score_key,
                        lower_is_better=True):
    exp_subsets = group_exp_list(exp_list, groupby_key_list)
    exp_list_new = [] 
    for exp_subset in exp_subsets:
        exp_dict_best = get_best_exp_dict(exp_subset, savedir_base, 
                    score_key=score_key, lower_is_better=lower_is_better)
        if exp_dict_best is None:
            continue
        exp_list_new += [exp_dict_best]
    return exp_list_new

def view_experiments(exp_list, savedir_base):
    df = get_dataframe_score_list(exp_list=exp_list,
                                     savedir_base=savedir_base)
    print(df)

def get_plot_across_runs(exp_list, score_list, savedir_base, 
                    title_list=None,
                     legend_list=None, avg_runs=0, 
                     s_epoch=None,e_epoch=None):
    nrows = 1
    # ncols = len(exp_configs)
    ncols = len(score_list)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(ncols*8, nrows*6))
    if nrows == 1 and ncols == 1:
        axs = [axs]
    
    for i, row in enumerate(score_list):
        # exp_list = cartesian_exp_config(EXP_GROUPS[exp_config_name])
    
        for exp_dict in exp_list:
            savedir = "%s/%s/" % (savedir_base,
                                  hu.hash_dict(exp_dict))
            path = savedir + "/score_list.pkl"
            if os.path.exists(path) and os.path.exists(savedir + "/exp_dict.json"):
                if exp_dict.get("runs") is None or not avg_runs:
                    mean_list = hu.load_pkl(path)
                    mean_df = pd.DataFrame(mean_list)
                    std_df = None

                elif exp_dict.get("runs") == 0:
                    # score_list = load_pkl(path)
                    mean_df, std_df = get_score_list_across_runs(exp_dict, savedir_base=savedir_base)

                else:
                    continue
                
                label_str = []
                for k in legend_list:
                    val = hu.flatten_dict(exp_dict).get(k)
                    label_str += [str(val)]
                label_str = "_".join(label_str)

                if s_epoch:
                    axs[i].plot(mean_df["epoch"][s_epoch:], 
                        mean_df[row][s_epoch:],
                                label=label_str, marker="*")

                elif e_epoch:
                    axs[i].plot(mean_df["epoch"][:e_epoch], mean_df[row][:e_epoch],
                                label=label_str, marker="*")
                else:
                    axs[i].plot(mean_df["epoch"], mean_df[row],
                                label=label_str, marker="*")
                if std_df is not None:
                    # do shading
                    offset = 0
                    # print(mean_df[row][offset:] - std_df[row][offset:])
                    # adsd
                    axs[i].fill_between(mean_df["epoch"][offset:], 
                            mean_df[row][offset:] - std_df[row][offset:],
                            mean_df[row][offset:] + std_df[row][offset:], 
                            # color = label2color[labels[i]],  
                            alpha=0.5)
                axs[i].grid(True)

        # prepare figure
        if "loss" in row:   
            axs[i].set_yscale("log")
            axs[i].set_ylabel(row + " (log)")
        else:
            axs[i].set_ylabel(row)
        axs[i].set_xlabel("epochs")
        axs[i].set_title("_".join([str(exp_dict.get(k)) for k in title_list]))
                            
        axs[i].legend( loc='best')  
        # axs[i].legend( loc='upper right', bbox_to_anchor=(0.5, -0.05))  
        # axs[i].set_ylim(.90, .94)  
    plt.grid(True)  
               
    return fig

def get_score_list_across_runs(exp_dict, savedir_base):    
    savedir = "%s/%s/" % (savedir_base,
                                 hu.hash_dict(exp_dict))
    score_list = hu.load_pkl(savedir + "/score_list.pkl")
    keys = score_list[0].keys()
    result_dict = {}
    for k in keys:
        result_dict[k] = np.ones((exp_dict["max_epoch"], 5))*-1
    

    bad_keys = set()
    for r in [0,1,2,3,4]:
        exp_dict_new = copy.deepcopy(exp_dict)
        exp_dict_new["runs"]  = r
        savedir_new = "%s/%s/" % (savedir_base,
                                  hu.hash_dict(exp_dict_new))
        
        
        if not os.path.exists(savedir_new + "/score_list.pkl"):
            continue
        score_list_new = hu.load_pkl(savedir_new + "/score_list.pkl")
        df = pd.DataFrame(score_list_new)
        for k in keys:
            values =  np.array(df[k])
            if values.dtype == "O":
                bad_keys.add(k)
                continue
            result_dict[k][:values.shape[0], r] = values

    for k in keys:
        if k in bad_keys:
                continue
        if -1 in result_dict[k]:
            return None, None

    mean_df = pd.DataFrame()
    std_df = pd.DataFrame()
    for k in keys:
        mean_df[k] = result_dict[k].mean(axis=1)
        std_df[k] = result_dict[k].std(axis=1)
    return mean_df, std_df

def get_plot(exp_list, 
             score_list, 
             savedir_base, 
             title_list=None,
             legend_list=None, 
             avg_runs=0, 
             s_epoch=None,
             e_epoch=None,
             axs=None,
             width=8,
             height=6):
    ncols = len(score_list)
    nrows = 1
    
    if axs is None:
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                                figsize=(ncols*width, nrows*height))
    if not hasattr(axs, 'size'):
        axs = [axs]

    for i, row in enumerate(score_list):
        for exp_dict in exp_list:
            exp_id = hu.hash_dict(exp_dict)
            savedir = savedir_base + "/%s/" % exp_id 

            path = savedir + "/score_list.pkl"
            if os.path.exists(path) and os.path.exists(savedir + "/exp_dict.json"):
                if exp_dict.get("runs") is None or not avg_runs:
                    mean_list = hu.load_pkl(path)
                    mean_df = pd.DataFrame(mean_list)
                    std_df = None

                elif exp_dict.get("runs") == 0:
                    mean_df, std_df = get_score_list_across_runs(exp_dict, savedir_base=savedir_base)
                    if mean_df is None:
                        mean_list = hu.load_pkl(path)
                        mean_df = pd.DataFrame(mean_list)
                else:
                    continue
                
                label = "_".join([str(exp_dict.get(k)) for 
                                k in legend_list])
                
                x_list = mean_df["epoch"]
                y_list = mean_df[row]
                if s_epoch:
                    x_list = x_list[s_epoch:]
                    y_list = y_list[s_epoch:]

                elif e_epoch:
                    x_list = x_list[:e_epoch]
                    y_list = y_list[:e_epoch]

                axs[i].plot(x_list, y_list,
                                label=label, marker="*")

                if std_df is not None:
                    offset = 0
                    axs[i].fill_between(x_list[offset:], 
                            y_list[offset:] - std_df[row][offset:],
                            y_list[offset:] + std_df[row][offset:], 
                            # color = label2color[labels[i]],  
                            alpha=0.5)
        if "loss" in row:   
            axs[i].set_yscale("log")
            axs[i].set_ylabel(row + " (log)")
        else:
            axs[i].set_ylabel(row)
        axs[i].set_xlabel("epochs")

        if title_list is not None:
            title = "_".join([str(exp_dict.get(k)) for k in title_list])
            axs[i].set_title(title)
                            
        axs[i].legend(loc="best")  

    plt.grid(True) 
    plt.tight_layout() 
               
    return fig

def get_dataframe_score_list(exp_list, col_list=None, savedir_base=None):
    score_list_list = []

    # aggregate results
    for exp_dict in exp_list:
        result_dict = {}

        exp_id = hu.hash_dict(exp_dict)
        result_dict["exp_id"] = exp_id
        savedir = savedir_base + "/%s/" % exp_id 
        if not os.path.exists(savedir + "/score_list.pkl"):
            score_list_list += [result_dict]
            continue

        for k in exp_dict:
            result_dict[k] = exp_dict[k]
            
        score_list_fname = os.path.join(savedir, "score_list.pkl")

        if os.path.exists(score_list_fname):
            score_list = hu.load_pkl(score_list_fname)
            score_df = pd.DataFrame(score_list)
            if len(score_list):
                score_dict_last = score_list[-1]
                for k, v in score_dict_last.items():
                    if "float" not  in str(score_df[k].dtype):
                        result_dict["*"+k] = v
                    else:
                        result_dict["*"+k] = "%.3f (%.3f-%.3f)" % (v, score_df[k].min(), score_df[k].max())

        score_list_list += [result_dict]

    df = pd.DataFrame(score_list_list).set_index("exp_id")
    
    # filter columns
    if col_list:
        df = df[[c for c in col_list if c in df.columns]]

    return df


def get_images(exp_list, savedir_base, n_exps=3, n_images=1, split="row",
               height=12, width=12):
    for k, exp_dict in enumerate(exp_list):
        if k >= n_exps:
            return
        result_dict = {}

        exp_id = hu.hash_dict(exp_dict)
        result_dict["exp_id"] = exp_id
        
        savedir = savedir_base + "/%s/" % exp_id 
        # img_list = glob.glob(savedir + "/*/*.jpg")[:n_images]
        img_list = glob.glob(savedir + "/images/images/*.jpg")[:n_images]
        if len(img_list) == 0:
            print('no images in %s' % savedir)
            return
        ncols = len(img_list)
        # ncols = len(exp_configs)
        nrows = 1
        if split == "row":
            fig, axs = plt.subplots(nrows=ncols, ncols=nrows, 
                                figsize=(ncols*width, nrows*height))
        else:
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                                figsize=(ncols*width, nrows*height))

        if not hasattr(axs, 'size'):
            axs = [[axs]]

        for i in range(ncols):
            img = plt.imread(img_list[i])
            axs[0][i].imshow(img)
            axs[0][i].set_axis_off()
            axs[0][i].set_title('%s:' % exp_id + hu.extract_fname(img_list[i]))
    #         fig.suptitle(exp_id)
        plt.axis('off')
        plt.tight_layout()
        
        plt.show()


def upload_score_list_to_dropbox(exp_id_list, savedir_base, outdir_base, access_token):
    import dropbox
    dbx = dropbox.Dropbox(access_token)
      
    try:
        dbx.files_create_folder(outdir_base)
    except:
        pass
    # API v2
    for exp_id in exp_id_list:
        savedir = savedir_base + "/%s" % exp_id
        outdir = outdir_base + "/%s" % exp_id

        try:
            dbx.files_create_folder(outdir)
        except:
            pass

        with open(savedir+"/exp_dict.json", 'rb') as f:
            dbx.files_upload(f.read(), outdir+"/exp_dict.json")

        with open(savedir+"/score_list.pkl", 'rb') as f:
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
    for exp_id in exp_id_list:
        if not os.path.isdir(os.path.join(savedir_base, exp_id)):
            continue
        abs_path = "%s/%s/exp_dict.json" % (savedir_base, exp_id)
        rel_path = "%s/exp_dict.json" % exp_id
        zipf.write(abs_path, rel_path)

        abs_path = "%s/%s/score_list.pkl" % (savedir_base, exp_id)
        rel_path = "%s/score_list.pkl" % exp_id
        zipf.write(abs_path, rel_path)

    zipf.close()


def group_exp_list(exp_list, groupby_key_list):
    def key_func(x):
        x_list = []
        for groupby_key in groupby_key_list:
            val = x[groupby_key]
            if isinstance(val, dict):
                val = hu.hash_dict(val)
            x_list += [val]

        return x_list
    
    exp_list.sort(key=key_func)

    exp_subsets = []
    for k,v in groupby(exp_list, key=key_func):
        v_list = list(v)
        exp_subsets += [v_list]
    return exp_subsets

def get_best_exp_list(exp_list, 
                      groupby_key=None):
    exp_list_new = []
    # aggregate results
    for exp_dict in exp_list:
        result_dict = {}

        exp_id = hu.hash_dict(exp_dict)
        result_dict["exp_id"] = exp_id
        savedir = savedir_base + "/%s/" % exp_id 
        if not os.path.exists(savedir + "/score_list.pkl"):
            score_list_list += [result_dict]
            continue

        for k in exp_dict:
            result_dict[k] = exp_dict[k]
            
        score_list_fname = os.path.join(savedir, "score_list.pkl")

        if os.path.exists(score_list_fname):
            score_list = hu.load_pkl(score_list_fname)
            score_df = pd.DataFrame(score_list)
            if len(score_list):
                score_dict_last = score_list[-1]
                for k, v in score_dict_last.items():
                    if "float" not  in str(score_df[k].dtype):
                        result_dict["*"+k] = v
                    else:
                        result_dict["*"+k] = "%.3f (%.3f-%.3f)" % (v, score_df[k].min(), score_df[k].max())

        score_list_list += [result_dict]


    return exp_list_new