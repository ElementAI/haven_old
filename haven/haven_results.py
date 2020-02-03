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
import os
import pylab as plt 
import pandas as pd 
import numpy as np
import copy 
import glob 
from itertools import groupby 

# =================================
# filtering
# =================================

import copy

class ResultManager:
    def __init__(self, savedir_base, exp_list=None):
        if exp_list is None:
            self.exp_list = get_exp_list_from_savedir_base(savedir_base)
        else:
            self.exp_list = exp_list

        self.savedir_base = savedir_base
        self.exp_sublists = [self.exp_list]
        
    def set_exp_sublists(self, regard_dict_list=None,
                          groupby_list=None,
                          order_groups=None):
        if regard_dict_list or groupby_list:
            self.exp_sublists = self.get_exp_sublists(regard_dict_list, 
                                                      groupby_list,
                                                      order_groups)
        else:
            self.exp_sublists = [self.exp_list]

    def get_scores(self, col_list=None, savedir_base=None):
        if not savedir_base:
            savedir_base = self.savedir_base

        df_list = []
        for exp_list in self.exp_sublists:
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
                        for k in score_df.columns:
                            v = np.array(score_df[k])
                            if 'float' in str(v.dtype):
                                v = v[~np.isnan(v)]

                            if "float"  in str(v.dtype):
                                result_dict[k] = ("%.3f (%.3f-%.3f)" % 
                                    (v[-1], v.min(), v.max()))
                            else:
                                result_dict[k] = v[-1]
                        #     else:
                        #         result_dict["*"+k] = ("%.3f (%.3f-%.3f)" % 
                        #             (score_df[k], score_df[k].min(), score_df[k].max()))

                score_list_list += [result_dict]

            df = pd.DataFrame(score_list_list).set_index("exp_id")
            
            # filter columns
            if col_list:
                df = df[[c for c in col_list if c in df.columns]]

            df_list += [df]

        return df_list

    def get_jobs(self, print_stats=True, print_errors=False, print_logs=False, n_lines=500, savedir_base=None):
        from haven_borgy import haven_borgy as hb
        if not savedir_base:
            savedir_base = self.savedir_base
        df_list = []
        for exp_list in self.exp_sublists:
            if print_stats:
                df = hb.print_job_stats(exp_list, 
                                    savedir_base=savedir_base)

            if "job_state" in df.columns:
                df["job_state"] = df["job_state"].fillna("NaN")
                stats = np.unique(df["job_state"])
                df_dict = {s:df[df["job_state"]==s] for s in stats}
                
                df_list += [df_dict]

            if print_errors or print_logs:
                hb.print_error_list(exp_list, savedir_base, logs_flag=print_logs, n_lines=n_lines)

        return df_list

    def to_zip(self, fname):
        from haven import haven_dropbox as hd

        exp_id_list = [hu.hash_dict(exp_dict) for exp_dict in self.exp_list]
        hd.zipdir(exp_id_list, self.savedir_base, fname)
        print('Zipped %d experiments in %s' % (len(exp_id_list), fname))

    def to_dropbox(self, fname, dropbox_path=None, access_token=None):
        from haven import haven_dropbox as hd

        dropbox_path = '/SLS_results/'
        access_token = 'Z61CnS89EjIAAAAAAABJ19VZt6nlqaw5PtWEBZYBhdLbW7zDyHOYP8GDU2vA2HAI'
        out_fname = os.path.join(dropbox_path, fname)
        src_fname = os.path.join(self.savedir_base, fname)
        self.to_zip(src_fname)
        hd.upload_file_to_dropbox(src_fname, out_fname, access_token)
        print('saved: https://www.dropbox.com/home/%s' % out_fname)

    def get_plots(self,
            transpose=False,
            y_list=None,
            title_list=None,
            legend_list=None, 
            avg_runs=0, 
            s_epoch=0,
            e_epoch=None,
            x_name='epoch',
            width=8,
            height=6,
            y_log_list=(None,),
            ylim_list=None,
            xlim_list=None,
            color_regard_dict=None,
            label_regard_dict=None,
                    legend_fontsize=None,
            y_fontsize=None, x_fontsize=None,
                        xtick_fontsize=None, ytick_fontsize=None,
                    title_fontsize=None,
                        linewidth=None,
                        markersize=None,
                    title_dict=None, y_only_first_flag=False, legend_kwargs=None,
                    markevery=1, bar_flag=None,
                    savedir=None, dropbox_dir=None, savedir_base=None, legend_only_first_flag=False,
                    legend_flag=True):
        if not savedir_base:
            savedir_base = self.savedir_base
        exp_sublists = self.exp_sublists

        if transpose:
            ncols = len(y_list)
            n_plots = len(exp_sublists)
        else:
            ncols = len(exp_sublists)
            n_plots = len(y_list)
        
        if ylim_list is not None:
            if isinstance(ylim_list[0], list):
                ylim_list_list = ylim_list 
            else:
                ylim_list_list = as_double_list(ylim_list) * n_plots

        if xlim_list is not None:
            if isinstance(xlim_list[0], list):
                xlim_list_list = xlim_list 
            else:
                xlim_list_list = as_double_list(xlim_list) * n_plots

        for j in range(n_plots):
            fig, axs = plt.subplots(nrows=1, ncols=ncols, 
                                    figsize=(ncols*width, 1*height))
            if not hasattr(axs, 'size'):
                axs = [axs]
            
            n_results_total = 0
            for i in range(ncols):
                if ylim_list is not None:
                    ylim = ylim_list_list[j][i]
                else:
                    ylim = None
                    
                if xlim_list is not None:
                    xlim = xlim_list[j][i]
                else:
                    xlim = None
                if y_only_first_flag and i > 0:
                    ylabel_flag = False
                else:
                    ylabel_flag = True

                if legend_only_first_flag:
                    legend_flag = False
                else:
                    legend_flag = legend_flag
                
                if transpose:
                    y_name = y_list[i]
                    exp_list = exp_sublists[j]
                else:
                    y_name = y_list[j]
                    exp_list = exp_sublists[i]
                    
                title, n_results = plot_exp_list(axs[i], exp_list, y_name=y_name, x_name=x_name, 
                            avg_runs=avg_runs, legend_list=legend_list, s_epoch=s_epoch, 
                            e_epoch=e_epoch, y_log_list=y_log_list, title_list=title_list, ylim=ylim,
                            xlim=xlim, color_regard_dict=color_regard_dict, label_regard_dict=label_regard_dict, legend_fontsize=legend_fontsize,
                            y_fontsize=y_fontsize, x_fontsize=x_fontsize, 
                            xtick_fontsize=xtick_fontsize, ytick_fontsize=ytick_fontsize,
                            title_fontsize=title_fontsize, linewidth=linewidth, markersize=markersize,
                            title_dict=title_dict, ylabel_flag=ylabel_flag,legend_kwargs=legend_kwargs,markevery=markevery,
                            savedir_base=savedir_base, bar_flag=bar_flag, legend_flag=legend_flag)
                n_results_total += n_results
                if n_results == 0:
                    print('no results for plot with title %s' % title)
                    continue
                else: 
                    if savedir:
                        plot_fname = os.path.join(savedir, '%s.png' % title.replace(' ', '')) 
                        plt.savefig(plot_fname)
                        print('saved: %s' % plot_fname)
                    if dropbox_dir:
                        from haven import haven_dropbox as hd
                        plot_fname = os.path.join(self.savedir_base, 'tmp.png')
                        plt.savefig(plot_fname)
                        # dropbox_dir = '/SLS_results/'
                        access_token = 'Z61CnS89EjIAAAAAAABJ19VZt6nlqaw5PtWEBZYBhdLbW7zDyHOYP8GDU2vA2HAI'
                        out_fname = os.path.join(dropbox_dir, '%s.png' % title.replace(' ', ''))
                        hd.upload_file_to_dropbox(plot_fname, out_fname, access_token)
                        print('saved: https://www.dropbox.com/home/%s' % out_fname)
            
            if n_results_total == 0:
                plt.close()
            else:
                if legend_only_first_flag:
                    if legend_kwargs is None:
                        legend_kwargs = {'loc': 'best'}
                    # plt.legend(fontsize=legend_fontsize, **legend_kwargs)  
                    # fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)  # create some space below the plots by increasing the bottom-value
                    # axs.flatten()[-2].legend(fontsize=legend_fontsize, **legend_kwargs)
                    plt.legend(fontsize=legend_fontsize, **legend_kwargs)  
                    
                  
                # plt.grid(True) 
                # plt.tight_layout()         
                plt.show()
                plt.close()

    def show_image(self, fname):
        ncols = 1
        nrows = 1
        height=12 
        width=12
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                                    figsize=(ncols*width, nrows*height))
        if not hasattr(axs, 'size'):
            axs = [[axs]]

        for i in range(ncols):
            img = plt.imread(os.path.join(self.savedir_base, fname))
            axs[0][i].imshow(img)
            axs[0][i].set_axis_off()
            axs[0][i].set_title('%s' % (fname))

        plt.axis('off')
        plt.tight_layout()
        
        plt.show()

    def get_images(self, n_exps=3, n_images=1, 
               height=12, width=12, legend_list=None):
        savedir_base = self.savedir_base
        for exp_list in self.exp_sublists:
            assert(legend_list is not None)
            for k, exp_dict in enumerate(exp_list):
                if k >= n_exps:
                    break
                result_dict = {}
                if legend_list is None:
                    label = ''
                else:
                    label = "_".join([str(exp_dict.get(k)) for 
                                            k in legend_list])

                exp_id = hu.hash_dict(exp_dict)
                result_dict["exp_id"] = exp_id
                print('Exp:', exp_id)
                savedir = savedir_base + "/%s/" % exp_id 
                # img_list = glob.glob(savedir + "/*/*.jpg")[:n_images]
                img_list = glob.glob(savedir + "/images/*.jpg")[:n_images]
                img_list += glob.glob(savedir + "/images/images/*.jpg")[:n_images]
                if len(img_list) == 0:
                    print('no images in %s' % savedir)
                    continue

                ncols = len(img_list)
                # ncols = len(exp_configs)
                nrows = 1
                fig, axs = plt.subplots(nrows=ncols, ncols=nrows, 
                                        figsize=(ncols*width, nrows*height))
                

                if not hasattr(axs, 'size'):
                    axs = [axs]

                for i in range(ncols):
                    img = plt.imread(img_list[i])
                    axs[i].imshow(img)
                    axs[i].set_axis_off()
                    axs[i].set_title('%s:%s' % (label, hu.extract_fname(img_list[i])))

                plt.axis('off')
                plt.tight_layout()
                
                plt.show()

    def get_exp_sublists(self, 
                        regard_dict_list=None, 
                        groupby_list=None,
                        order_groups=None):
        exp_list = self.exp_list
        # 1. group experiments
        if groupby_list is not None:
            exp_sublists, exp_group_dict = group_exp_list(exp_list, groupby_list)

            if order_groups:
                exp_sublists = [exp_group_dict[d] for d in order_groups]
        else:
            exp_sublists = [exp_list]

        # 2. filter each experiment group
        if regard_dict_list:
            regard_dict_list = as_double_list(regard_dict_list)

            exp_sublists_new = []
            for _sublist in exp_sublists:
                sublist = copy.deepcopy(_sublist)
                # apply filtration to sublist
                for i in range(len(regard_dict_list)):
                    sublist = filter_exp_list(sublist, regard_dict_list[i], 
                                            self.savedir_base)
                if len(sublist) == 0:
                    continue
                exp_sublists_new += [sublist]

            exp_sublists = exp_sublists_new
            
            print('%d sublists. Grouping by %s\n' % (len(exp_sublists), groupby_list))

            for i, exp_sublist in enumerate(exp_sublists):
                if groupby_list:
                    # print(exp_sublist)
                    group_name = '-'.join([str(exp_sublist[0].get(k)) for k in groupby_list])
                else:
                    group_name = 'all'
                print('Sublist %d: %d experiments - (%s)' % (i, len(exp_sublist), group_name ))

        return exp_sublists


def get_exp_list_from_savedir_base(savedir_base):
    import tqdm
    exp_list = []

    if '.zip' in savedir_base:
        import zipfile, json
        
        with zipfile.ZipFile(savedir_base) as z:
            for filename in tqdm.tqdm(z.namelist()):
                if not os.path.isdir(filename):
                    # read the file
                    with z.open(filename) as f:
                        if 'exp_dict.json' in filename:
                            data = f.read()
                            exp_list += [json.loads(data.decode("utf-8"))]
    else:
        
        dir_list = os.listdir(savedir_base)

        for exp_id in tqdm.tqdm(dir_list):
            savedir = os.path.join(savedir_base, exp_id)
            fname = savedir + "/exp_dict.json"
            if not os.path.exists(fname):
                continue
            exp_dict = hu.load_json(fname)
            exp_list += [exp_dict]

    return exp_list


def get_exp_list_zip(savedir_base):
    import os
    import zipfile

    with zipfile.ZipFile(savedir_base) as z:
        for filename in z.namelist():
            if not os.path.isdir(filename):
                # read the file
                with z.open(filename) as f:
                    for line in f:
                        print(line)

def get_best_exp_dict(exp_list, savedir_base, reduce_score, reduce_mode, return_scores=False):


    scores_dict = []
    if reduce_mode == 'min':
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
 
        if reduce_mode == 'min':
            score = pd.DataFrame(score_list)[reduce_score].min()
            if best_score >= score:
                best_score = score
                exp_dict_best = exp_dict
        else:
            score = pd.DataFrame(score_list)[reduce_score].max()
            if best_score <= score:
                best_score = score
                exp_dict_best = exp_dict
        scores_dict += [{'score':score, 'epochs':len(score_list), 'exp_id':exp_id}]
#     print(best_score)
    if exp_dict_best is None:
        return {}
    scores_dict += [{'exp_id':hu.hash_dict(exp_dict_best), 'best_score':best_score}]
    if return_scores:
        return exp_dict_best, scores_dict
    # print(reduce_score, scores_dict)
    return exp_dict_best


# def filter_best_results(exp_list, savedir_base, regard_dict_list, y_name,
#                         bar_flag):
#     exp_sublists = []

#     if bar_flag == 'min':
#         lower_is_better = True
#     elif bar_flag == 'max':
#         lower_is_better = False
    
#     for regard_dict in regard_dict_list:
#         exp_sublists += [filter_exp_list(exp_list, regard_dict=regard_dict)]
        
#     print('# exp subsets:', [len(es) for es in exp_sublists])
# #     stop
#     exp_list_new = [] 
#     for exp_subset in exp_sublists:
#         exp_dict_best = get_best_exp_dict(exp_subset, savedir_base, 
#                     score_key=y_name, lower_is_better=lower_is_better)
#         if exp_dict_best is None:
#             continue
#         exp_list_new += [exp_dict_best]
#     return exp_list_new

def view_experiments(exp_list, savedir_base):
    df = get_dataframe_score_list(exp_list=exp_list,
                                     savedir_base=savedir_base)
    print(df)




def get_score_list_across_runs(exp_dict, savedir_base, y_name, x_name):    
    savedir = "%s/%s/" % (savedir_base,
                                 hu.hash_dict(exp_dict))
    score_list = hu.load_pkl(savedir + "/score_list.pkl")
    keys = score_list[0].keys()
    result_dict = {}
    result_dict[y_name] = np.ones((exp_dict["max_epoch"], 5))*-1
    result_dict[x_name] = np.ones((exp_dict["max_epoch"], 5))*-1

    for r in [0,1,2,3,4]:
        exp_dict_new = copy.deepcopy(exp_dict)
        exp_dict_new["runs"]  = r
        savedir_new = "%s/%s/" % (savedir_base,
                                  hu.hash_dict(exp_dict_new))
        
        
        if not os.path.exists(savedir_new + "/score_list.pkl"):
            continue
        score_list_new = hu.load_pkl(savedir_new + "/score_list.pkl")
        df = pd.DataFrame(score_list_new)
        # print(df)
        
        values =  np.array(df[y_name])
        # print(values[0])

        try:
            float(values[0])
        except:
            continue

        result_dict[y_name][:values.shape[0], r] = values

        values =  np.array(df[x_name])
        result_dict[x_name][:values.shape[0], r] = values


    if -1 in result_dict[y_name]:
        return None, None

    mean_df = pd.DataFrame()
    std_df = pd.DataFrame()

    mean_df[y_name] = result_dict[y_name].mean(axis=1)
    std_df[y_name] = result_dict[y_name].std(axis=1)

    mean_df[x_name] = result_dict[x_name].mean(axis=1)
    std_df[x_name] = result_dict[x_name].std(axis=1)

    return mean_df, std_df


def plot_exp_list(axis, exp_list, y_name, x_name, avg_runs, legend_list, s_epoch, e_epoch, 
                  y_log_list, title_list, ylim=None, xlim=None, color_regard_dict=None, label_regard_dict=None, 
                  legend_fontsize=None, y_fontsize=None, x_fontsize=None, xtick_fontsize=None, ytick_fontsize=None,
                 title_fontsize=None, linewidth=None, markersize=None, title_dict=None, ylabel_flag=True, legend_kwargs=None, markevery=1,
                 savedir_base=None, bar_flag=None, legend_flag=True):
    bar_count = 0
    n_results = 0
    for exp_dict in exp_list:
        
        exp_id = hu.hash_dict(exp_dict)
        savedir = savedir_base + "/%s/" % exp_id 

        path = savedir + "/score_list.pkl"
        if not os.path.exists(savedir + "/exp_dict.json") or not os.path.exists(path):
            continue
        else:
            # average runs
            if exp_dict.get("runs") is None or not avg_runs:
                mean_list = hu.load_pkl(path)
                mean_df = pd.DataFrame(mean_list)
                std_df = None

            elif exp_dict.get("runs") == 0:
                mean_df, std_df = get_score_list_across_runs(exp_dict, savedir_base=savedir_base, y_name=y_name, x_name=x_name)
                if mean_df is None:
                    mean_list = hu.load_pkl(path)
                    mean_df = pd.DataFrame(mean_list)
            else:
                continue

            n_results += 1

            # get label
            label = "_".join([str(exp_dict.get(k)) for 
                            k in legend_list])

            if label_regard_dict is not None:
                for k in label_regard_dict:
                    if is_equal(label_regard_dict[k], exp_dict):
                        label = k
                        break
            
            # get color
            color = None
            if color_regard_dict is not None:
                for k in color_regard_dict:
                    if is_equal(color_regard_dict[k], exp_dict):
                        color = k
                        break
                
            x_list = np.array(mean_df[x_name])

            if y_name not in mean_df:
                continue
            y_list = np.array(mean_df[y_name])

            if 'float' in str(y_list.dtype):
                y_ind = ~np.isnan(y_list)

                y_list = y_list[y_ind]
                x_list = x_list[y_ind]

            if s_epoch:
                x_list = x_list[s_epoch:]
                y_list = y_list[s_epoch:]

            elif e_epoch:
                x_list = x_list[:e_epoch]
                y_list = y_list[:e_epoch]

            if y_list.dtype == 'object':
                continue
                
            if bar_flag:
                if bar_flag == 'max':
                    ix = np.argmax(y_list)    
                else:
                    ix = np.argmin(y_list)

                axis.bar([bar_count], [y_list[ix]],
                        color=color,
                        label='%s - (%s: %d, %s: %.3f)' % (label, x_name, x_list[-1], y_name, y_list[ix]))
                bar_count += 1
            else:
                axis.plot(x_list, y_list,color=color, linewidth=linewidth, markersize=markersize,
                        label=str(label), marker="*", markevery=markevery)

            if std_df is not None and not bar_flag:
#                 print(exp_dict)
                s_list = np.array(std_df[y_name])
                s_list = s_list[y_ind]
                s_list = s_list[s_epoch:]
                offset = 0
                axis.fill_between(x_list[offset:], 
                        y_list[offset:] - s_list[offset:],
                        y_list[offset:] + s_list[offset:], 
                        color = color,  
                        alpha=0.1)

    # get title
    if title_list is not None:
            title = "_".join([str(exp_dict.get(k)) for k in title_list])
            if title_dict is not None and title in title_dict:
                title = title_dict[title]
            axis.set_title(title, fontsize=title_fontsize)
    else:
        title = ''

    if n_results > 0:      
        if ylim is not None:
            axis.set_ylim(ylim)
        if xlim is not None:
            axis.set_xlim(xlim)
            
        if y_name in y_log_list:   
            axis.set_yscale("log")    
            y_name = y_name + " (log)"
        
        if ylabel_flag:
            axis.set_ylabel(y_name, fontsize=y_fontsize)
            
        if not bar_flag:
            axis.set_xlabel(x_name, fontsize=x_fontsize)

        axis.tick_params(axis='x', labelsize=xtick_fontsize)
        axis.tick_params(axis='y', labelsize=ytick_fontsize)
        
        axis.grid(True)
        if legend_flag:
            if legend_kwargs is None:
                legend_kwargs = {'loc': 'best'}
            axis.legend(fontsize=legend_fontsize, **legend_kwargs)  
            
    return title, n_results
    # axis.relim()
    # axis.autoscale_view()









def group_exp_list(exp_list, groupby_key_list):
    # # filter out nones 
    # exp_list_new = []
    # for exp_dict in exp_list:
    #     flag = True
    #     for k in groupby_key_list:
    #         if exp_dict.get(k) is None:
    #             flag = False
    #     if flag:
    #         exp_list_new += [exp_dict]
    # exp_list = exp_list_new

    def key_func(x):
        x_list = []
        for groupby_key in groupby_key_list:
            val = x.get(groupby_key)
            if isinstance(val, dict):
                val = val['name']
            x_list += [val]

        return x_list
    
    exp_list.sort(key=key_func)

    exp_sublists = []
    group_dict = groupby(exp_list, key=key_func)
    

    exp_group_dict = {}
    for k,v in group_dict:
        v_list = list(v)
        exp_sublists += [v_list]
        # print(k)
        exp_group_dict['_'.join(list(map(str, k)))] = v_list

    return exp_sublists, exp_group_dict


def is_equal(d1, d2):
    flag = True
    for k in d1:
        v1, v2 = d1.get(k), d2.get(k)

        # if both are values
        if not isinstance(v2, dict) and not isinstance(v1, dict):
            if v1 != v2:
                flag = False
            
        # if both are dicts
        elif isinstance(v2, dict) and isinstance(v1, dict):
            flag = is_equal(v1, v2)

        # if d1 is dict and not d2
        elif isinstance(v1, dict) and not isinstance(v2, dict):
            flag = False

        # if d1 is not and d2 is dict
        elif not isinstance(v1, dict) and isinstance(v2, dict):
            flag = False

        if flag is False:
            break
    
    return flag

def filter_regard_dict(exp_list, regard_dict, savedir_base):
    if isinstance(regard_dict, tuple):
        regard_dict, reduce_score, reduce_mode = regard_dict
    else:
        reduce_score = None

    tmp_list = []
    for exp_dict in exp_list:
        select_flag = False
        
        if is_equal(regard_dict, exp_dict):
            select_flag = True

        if select_flag:
            tmp_list += [exp_dict]

    if reduce_score is not None:
        tmp_list = [get_best_exp_dict(tmp_list, savedir_base, 
                                        reduce_score=reduce_score, reduce_mode=reduce_mode)]
    return tmp_list

def filter_exp_list(exp_list, regard_list, savedir_base):
    exp_list_new = []
    for regard_dict in regard_list:
        tmp_list = filter_regard_dict(exp_list, regard_dict, savedir_base)
        if len(tmp_list)==0 or len(tmp_list[0]) == 0:
            continue
        exp_list_new += tmp_list

    exp_list = exp_list_new


    return exp_list

def as_double_list(v):
    if not isinstance(v, list):
        v = [v]

    # double list for intersection
    if not isinstance(v[0], list):
        v = [v] 
    
    return v 

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




    


