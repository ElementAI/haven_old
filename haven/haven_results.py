import copy
import glob
import os
import sys
from itertools import groupby

import numpy as np
import pandas as pd
import pylab as plt
import tqdm

from . import haven_jobs as hjb
from . import haven_utils as hu


def group_exp_list(exp_list, groupby_list):
    """Split the experiment list into smaller lists where each
       is grouped by a set of hyper-parameters

    Parameters
    ----------
    exp_list : list
        A list of experiments, each defines a single set of hyper-parameters
    groupby_list : list
        The set of hyper-parameters to group the experiments

    Returns
    -------
    lists_of_exp_list : list
        Experiments grouped by a set of hyper-parameters

    Example
    -------
    >>>
    >>>
    >>>
    """
    groupby_list = hu.as_double_list(groupby_list)[0]
    def split_func(x):
        x_list = []
        for split_hparam in groupby_list:
            val = x.get(split_hparam)
            if isinstance(val, dict):
                val = val['name']
            x_list += [val]

        return x_list

    exp_list.sort(key=split_func)

    list_of_exp_list = []
    group_dict = groupby(exp_list, key=split_func)

    # exp_group_dict = {}
    for k, v in group_dict:
        v_list = list(v)
        list_of_exp_list += [v_list]
    #     # print(k)
    #     exp_group_dict['_'.join(list(map(str, k)))] = v_list

    return list_of_exp_list


def get_best_exp_dict(exp_list, savedir_base, metric, min_or_max='min', return_scores=False, verbose=True):
    """Obtain best the experiment for a specific metric.

    Parameters
    ----------
    exp_list : list
        A list of experiments, each defines a single set of hyper-parameters
    savedir_base : [type]
        A directory where experiments are saved
    metric : [type]
        [description]
    min_or_max : [type]
        [description]
    return_scores : bool, optional
        [description], by default False
    """
    scores_dict = []
    if min_or_max == 'min':
        best_score = np.inf
    else:
        best_score = 0.
    
    exp_dict_best = None
    for exp_dict in exp_list:
        exp_id = hu.hash_dict(exp_dict)
        savedir = os.path.join(savedir_base, exp_id)

        score_list_fname = os.path.join(savedir, 'score_list.pkl')
        if not os.path.exists(score_list_fname):
            if verbose:
                print('%s: missing score_list.pkl' % exp_id)
            continue
        
        score_list = hu.load_pkl(score_list_fname)

        if min_or_max == 'min':
            score = np.min([score_dict[metric] for score_dict in score_list])
            if best_score >= score:
                best_score = score
                exp_dict_best = exp_dict
        else:
            score = np.max([score_dict[metric] for score_dict in score_list])
            if best_score <= score:
                best_score = score
                exp_dict_best = exp_dict

        scores_dict += [{'score': score,
                         'epochs': len(score_list), 
                         'exp_id': exp_id}]

    if exp_dict_best is None:
        if verbose:
            print('no experiments with metric "%s"' % metric)
        return {}
        
    return exp_dict_best


def get_exp_list_from_exp_configs(exp_group_list, workdir, filterby_list=None, verbose=True):
    """[summary]
    
    Parameters
    ----------
    exp_group_list : [type]
        [description]
    workdir : [type]
        [description]
    filterby_list : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    assert workdir is not None
    
    from importlib import reload
    assert(workdir is not None)
    if workdir not in sys.path:
        sys.path.append(workdir)
    import exp_configs as ec
    reload(ec)

    exp_list = []
    for exp_group in exp_group_list:
        exp_list += ec.EXP_GROUPS[exp_group]
    if verbose:
        print('%d experiments' % len(exp_list))

    exp_list = filter_exp_list(exp_list, filterby_list, verbose=verbose)
    return exp_list

def get_exp_list(savedir_base, filterby_list=None, verbose=True):
    """[summary]
    
    Parameters
    ----------
    savedir_base : [type], optional
        A directory where experiments are saved, by default None
    filterby_list : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    exp_list = []
    dir_list = os.listdir(savedir_base)

    for exp_id in tqdm.tqdm(dir_list):
        savedir = os.path.join(savedir_base, exp_id)
        fname = os.path.join(savedir, 'exp_dict.json')
        if len(exp_id) != 32:
            if verbose:
                print('"%s/" is not an exp directory' % exp_id)
            continue 

        if not os.path.exists(fname):
            if verbose:
                print('%s: missing exp_dict.json' % exp_id)
            continue

        exp_dict = hu.load_json(fname)
        exp_list += [exp_dict]

    exp_list = filter_exp_list(exp_list, filterby_list)
    return exp_list


def zip_exp_list(savedir_base):
    """[summary]

    Parameters
    ----------
    savedir_base : [type]
        [description]
    """
    import zipfile

    with zipfile.ZipFile(savedir_base) as z:
        for filename in z.namelist():
            if not os.path.isdir(filename):
                # read the file
                with z.open(filename) as f:
                    for line in f:
                        print(line)


def filter_exp_list(exp_list, filterby_list, verbose=True):
    """[summary]
    
    Parameters
    ----------
    exp_list : [type]
        A list of experiments, each defines a single set of hyper-parameters
    filterby_list : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    if filterby_list is None:
        return exp_list

    filterby_list_list = hu.as_double_list(filterby_list)
    filterby_list = filterby_list_list[0]
    exp_list_new = []
    for exp_dict in exp_list:
        select_flag = False

        for filterby_dict in filterby_list:
            if hu.is_subset(filterby_dict, exp_dict):
                select_flag = True
                break

        if select_flag:
            exp_list_new += [exp_dict]
    if verbose:
        print('Filtered: %d/%d experiments gathered...' % (len(exp_list_new), len(exp_list)))
    return exp_list_new

def get_score_lists(exp_list, savedir_base, filterby_list=None, verbose=True):
    """[summary]
    
    Parameters
    ----------
    exp_list : [type]
        A list of experiments, each defines a single set of hyper-parameters
    savedir_base : [type]
        [description]
    filterby_list : [type], optional
        [description], by default None
    
    Returns
    -------
    [type]
        [description]

    Example
    -------
    >>> from haven import haven_results as hr
    >>> savedir_base='/mnt/datasets/public/issam/prototypes/isps/'
    >>> exp_list = hr.get_exp_list(savedir_base=savedir_base, 
    >>>                            filterby_list=[{'sampler':{'train':'basic'}}])
    >>> lists_of_score_lists = hr.get_score_lists(exp_list, savedir_base=savedir_base, columns=['train_loss', 'exp_id'])
    >>> print(lists_of_score_lists)
    """
    if len(exp_list) == 0:
        if verbose:
            print('exp_list is empty...')
        return

    exp_list = filter_exp_list(exp_list, filterby_list, verbose=verbose)
    score_lists = []

    # aggregate results
    for exp_dict in exp_list:
        exp_id = hu.hash_dict(exp_dict)
        savedir = os.path.join(savedir_base, exp_id)

        score_list_fname = os.path.join(savedir, 'score_list.pkl')
        if not os.path.exists(score_list_fname):
            if verbose:
                print('%s: missing score_list.pkl' % exp_id)
            continue
        
        else:
            score_lists += [hu.load_pkl(score_list_fname)]
    
    return score_lists

def get_score_df(exp_list, savedir_base, filterby_list=None, columns=None, verbose=True):
    """Get a table showing the scores for the given list of experiments 

    Parameters
    ----------
    exp_list : list
        A list of experiments, each defines a single set of hyper-parameters
    columns : list, optional
        a list of columns you would like to display, by default None
    savedir_base : str, optional
        A directory where experiments are saved
        
    Returns
    -------
    DataFrame
        a dataframe showing the scores obtained by the experiments

    Example
    -------
    >>> from haven import haven_results as hr
    >>> savedir_base='/mnt/datasets/public/issam/prototypes/isps/'
    >>> exp_list = hr.get_exp_list(savedir_base=savedir_base, 
    >>>                            filterby_list=[{'sampler':{'train':'basic'}}])
    >>> df = hr.get_score_df(exp_list, savedir_base=savedir_base, columns=['train_loss', 'exp_id'])
    >>> print(df)
    """
    if len(exp_list) == 0:
        if verbose:
            print('exp_list is empty...')
        return
    exp_list = filter_exp_list(exp_list, filterby_list, verbose=verbose)

    # aggregate results
    result_list = []
    for exp_dict in exp_list:
        result_dict = {}

        exp_id = hu.hash_dict(exp_dict)
        result_dict["exp_id"] = exp_id
        savedir = os.path.join(savedir_base, exp_id)
        score_list_fname = os.path.join(savedir, "score_list.pkl")

        for k in exp_dict:
            result_dict[k] = exp_dict[k]

        if not os.path.exists(score_list_fname):
            if verbose:
                print('%s: score_list.pkl is missing' % exp_id)
            continue
        else:
            score_list = hu.load_pkl(score_list_fname)
            score_df = pd.DataFrame(score_list)
            if len(score_list):
                for k in score_df.columns:
                    v = np.array(score_df[k])
                    if 'float' in str(v.dtype):
                        v = v[~np.isnan(v)]

                    if "float" in str(v.dtype):
                        result_dict[k] = ("%.3f (%.3f-%.3f)" %
                                          (v[-1], v.min(), v.max()))
                    else:
                        result_dict[k] = v[-1]

        result_list += [result_dict]

    df = pd.DataFrame(result_list)
    # df['*exp_id'] = df["exp_id"]
    # df = df.set_index('exp_id')
    # filter columns
    if columns:
        df = df[[c for c in columns if c in df.columns]]

    return df


def get_plot(exp_list, savedir_base, 
             x_metric, y_metric,
             mode='line',
             filterby_list=None,
             title_list=None,
             legend_list=None,
             log_metric_list=None,
             figsize=None,
             avg_across=None,
             fig=None,
             axis=None,
             ylim=None,
             xlim=None,
             legend_fontsize=None,
             y_fontsize=None,
             x_fontsize=None,
             ytick_fontsize=None,
             xtick_fontsize=None,
             title_fontsize=None,
             legend_kwargs=None,
             map_exp_list=tuple(),
             map_title_list=tuple(),
             map_xlabel_list=tuple(),
             map_ylabel_list=dict(),
             verbose=True):
    """Plots the experiment list in a single figure.
    
    Parameters
    ----------
    exp_list : list
        A list of experiments, each defines a single set of hyper-parameters
    savedir_base : str
        A directory where experiments are saved
    x_metric : str
        Specifies metric for the x-axis
    y_metric : str
        Specifies metric for the y-axis
    title_list : [type], optional
        [description], by default None
    legend_list : [type], optional
        [description], by default None
    meta_list : [type], optional
        [description], by default None
    log_metric_list : [type], optional
        [description], by default None
    figsize : tuple, optional
        [description], by default (8, 8)
    avg_metric : [type], optional
        [description], by default None
    axis : [type], optional
        [description], by default None
    ylim : [type], optional
        [description], by default None
    xlim : [type], optional
        [description], by default None
    legend_fontsize : [type], optional
        [description], by default None
    y_fontsize : [type], optional
        [description], by default None
    ytick_fontsize : [type], optional
        [description], by default None
    xtick_fontsize : [type], optional
        [description], by default None
    legend_kwargs : [type], optional
        [description], by default None
    
    Returns
    -------
    fig : [type]
        [description]
    axis : [type]
        [description]

    Examples
    --------
    >>> from haven import haven_results as hr
    >>> savedir_base='/mnt/datasets/public/issam/prototypes/isps/'
    >>> exp_list = hr.get_exp_list(savedir_base=savedir_base, 
    >>>                            filterby_list=[{'sampler':{'train':'basic'}}])
    >>> hr.get_plot(exp_list, savedir_base=savedir_base, x_metric='epoch', y_metric='train_loss', legend_list=['model'])
    """
    if axis is None:
        fig, axis = plt.subplots(nrows=1, ncols=1,
                                    figsize=figsize)
    # if len(exp_list) > 50:
    #     if verbose:
    #         raise ValueError('many experiments in one plot...use filterby_list to reduce them')
        
    exp_list = filter_exp_list(exp_list, filterby_list=filterby_list, verbose=verbose)
    visited = set()
    for exp_dict in exp_list:
        exp_id = hu.hash_dict(exp_dict)
        savedir = os.path.join(savedir_base, exp_id)
        score_list_fname = os.path.join(savedir, 'score_list.pkl')

        if not os.path.exists(score_list_fname):
            if verbose:
                print('%s: score_list.pkl does not exist...' % exp_id)
            continue

        else:
            # get scores
            if not avg_across:
                # get score list
                score_list = hu.load_pkl(score_list_fname)
                x_list = []
                y_list = []
                for score_dict in score_list:
                    if x_metric in score_dict and y_metric in score_dict:
                        x_list += [score_dict[x_metric]]
                        y_list += [score_dict[y_metric]]
            else:
                # average score list across an hparam
                if exp_id in visited:
                    # already used in averaging
                    continue
                
                filter_dict = {k:exp_dict[k] for k in exp_dict if k not in avg_across}
                exp_sublist = filter_exp_list(exp_list, filterby_list=[filter_dict], verbose=verbose)

                # get score list
                x_dict = {}
                for sub_dict in exp_sublist:
                    sub_id = hu.hash_dict(sub_dict)
                    sub_score_list_fname = os.path.join(savedir_base, sub_id, 'score_list.pkl')

                    if not os.path.exists(sub_score_list_fname):
                        if verbose:
                            print('%s: score_list.pkl does not exist...' % sub_id)
                        continue

                    visited.add(sub_id)

                    sub_score_list = hu.load_pkl(sub_score_list_fname)

                    for score_dict in sub_score_list:
                        if x_metric in score_dict and y_metric in score_dict:
                            x_val = score_dict[x_metric]
                            if not x_val in x_dict:
                                x_dict[x_val] = []

                            x_dict[x_val] += [score_dict[y_metric]]
                # import ipdb; ipdb.set_trace()
                if len(x_dict) == 0:
                    x_list = []
                    y_list = []
                else:
                    x_list = np.array(list(x_dict.keys()))
                    y_list_all = np.array(list(x_dict.values()))
                    y_std_list = np.std(y_list_all, axis=1)
                    y_list = np.mean(y_list_all, axis=1)            
    
            if len(x_list) == 0:
                x_list = np.NaN
                y_list = np.NaN
                if verbose:
                    print('%s: "(%s, %s)" not in score_list' % (exp_id, y_metric, x_metric))

            # map properties of exp
            if legend_list is not None:
                label = '_'.join([str(exp_dict.get(k)) for k in legend_list])
            else:
                label = exp_id

            color = None
            marker = '*'
            linewidth = None
            markevery = None
            markersize = None

            for filterby_dict, map_dict in map_exp_list:
                if hu.is_subset(filterby_dict, exp_dict):
                    marker = map_dict.get('marker', marker)
                    label = map_dict.get('label', label)
                    color = map_dict.get('color', color)
                    linewidth = map_dict.get('linewidth', linewidth)
                    markevery = map_dict.get('markevery', markevery)
                    break
        
            # plot
            if mode == 'line':
                # plot the mean in a line
                axis.plot(x_list, y_list, color=color, linewidth=linewidth, markersize=markersize,
                    label=str(label), marker=marker, markevery=markevery)

                if avg_across:
                    # add confidence interval
                    axis.fill_between(x_list, 
                            y_list - y_std_list,
                            y_list + y_std_list, 
                            color = color,  
                            alpha=0.1)

    # default properties
    if title_list is not None:
        title = '_'.join([str(exp_dict.get(k)) for k in title_list])
    else:
        title = ''

    ylabel = y_metric
    xlabel = x_metric

    # map properties
    for map_dict in map_title_list:
        if title in map_dict:
            title = map_dict[title]

    for map_dict in map_xlabel_list:
        if x_metric in map_dict:
            xlabel = map_dict[x_metric]

    for map_dict in map_ylabel_list:
        if y_metric in map_dict:
            ylabel = map_dict[y_metric]

    # set properties
    axis.set_title(title, title_fontsize)
    axis.set_ylim(ylim)
    axis.set_xlim(xlim)
    
    if log_metric_list and y_metric in log_metric_list:
        axis.set_yscale('log')
        ylabel = ylabel + ' (log)'

    axis.set_ylabel(ylabel, fontsize=y_fontsize)
    axis.set_xlabel(xlabel, fontsize=x_fontsize)

    axis.tick_params(axis='x', labelsize=xtick_fontsize)
    axis.tick_params(axis='y', labelsize=ytick_fontsize)

    axis.grid(True)

    legend_kwargs = legend_kwargs or {'loc': 'best'}
    axis.legend(fontsize=legend_fontsize, **legend_kwargs)

    plt.tight_layout()

    return fig, axis

def get_images(exp_list, savedir_base, n_exps=3, n_images=1,
                   height=12, width=12, legend_list=None,
                   dirname='images', verbose=True):
    """[summary]
    
    Parameters
    ----------
    exp_list : list
        A list of experiments, each defines a single set of hyper-parameters
    savedir_base : str
        A directory where experiments are saved
    n_exps : int, optional
        [description], by default 3
    n_images : int, optional
        [description], by default 1
    height : int, optional
        [description], by default 12
    width : int, optional
        [description], by default 12
    legend_list : [type], optional
        [description], by default None
    dirname : str, optional
        [description], by default 'images'

    Returns
    -------
    fig_list : list
        a list of pylab figures

    Example
    -------
    >>> from haven import haven_results as hr
    >>> savedir_base='/mnt/datasets/public/issam/prototypes/isps/'
    >>> exp_list = hr.get_exp_list(savedir_base=savedir_base, 
    >>>                            filterby_list=[{'sampler':{'train':'basic'}}])
    >>> hr.get_images(exp_list, savedir_base=savedir_base)
    """
    fig_list = []
    for k, exp_dict in enumerate(exp_list):
        if k >= n_exps:
            break
        result_dict = {}
        if legend_list is None:
            label = hu.hash_dict(exp_dict)
        else:
            label = '_'.join([str(exp_dict.get(k)) for
                                k in legend_list])

        exp_id = hu.hash_dict(exp_dict)
        result_dict['exp_id'] = exp_id
        if verbose:
            print('Exp:', exp_id)
        savedir = os.path.join(savedir_base, exp_id)

        base_dir = os.path.join(savedir, dirname)
        img_list = glob.glob(os.path.join(base_dir, '*.jpg'))
        img_list += glob.glob(os.path.join(base_dir, '*.png'))
        img_list.sort(key=os.path.getmtime)
        img_list = img_list[::-1]
        img_list = img_list[:n_images]

        if len(img_list) == 0:
            if verbose:
                print('no images in %s' % base_dir)
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
            axs[i].set_title('%s:%s' %
                                (label, os.path.split(img_list[i])[-1]))

        plt.axis('off')
        plt.tight_layout()
        fig_list += [fig]
    
    return fig_list


def get_job_stats(exp_list, savedir_base, username):
    """Gets the job stats
    
    Parameters
    ----------
    exp_list : list
        A list of experiments, each defines a single set of hyper-parameters
    savedir_base : str
        A directory where experiments are saved
    username : str
        Name of the user
    """
    hjb.add_job_utils()
    import haven_jobs_utils as hju

    hju.get_job_stats(
        exp_list, savedir_base=savedir_base, username=username)

def get_job_logs(exp_list, savedir_base, username):
    """Gets the job logs
    
    Parameters
    ----------
    exp_list : list
        A list of experiments, each defines a single set of hyper-parameters
    savedir_base : str
        A directory where experiments are saved
    username : str
        Name of the user
    """
    hjb.add_job_utils()
    import haven_jobs_utils as hju

    hju.get_job_logs(
        exp_list, savedir_base=savedir_base, username=username)
    hju.get_job_stats(
        exp_list, savedir_base=savedir_base, username=username)

def get_job_errors(exp_list, savedir_base, username):
    """Get the job errors.
    
    Parameters
    ----------
    exp_list : list
        A list of experiments, each defines a single set of hyper-parameters
    savedir_base : str
        A directory where experiments are saved
    username : str
        Name of the user
    """
    hjb.add_job_utils()
    import haven_jobs_utils as hju

    hju.get_job_errors(
        exp_list, savedir_base=savedir_base, username=username)
    hju.get_job_stats(
        exp_list, savedir_base=savedir_base, username=username)

class ResultManager:
    def __init__(self, 
                 savedir_base,
                 exp_list=None,
                 exp_group_list=None,
                 workdir=None,
                 username=None,
                 filterby_list=None,
                 groupby_list=None, 
                 verbose=True):
        """[summary]
        
        Parameters
        ----------
        savedir_base : [type]
            A directory where experiments are saved
        exp_list : [type], optional
            [description], by default None
        exp_group_list : [type], optional
            [description], by default None
        workdir : [type], optional
            [description], by default None
        username : [type], optional
            [description], by default None
        filterby_list : [type], optional
            [description], by default None
        groupby_list : [type], optional
            [description], by default None
        has_score_list : [type], optional
            [description], by default None
        
        Example
        -------
        >>> from haven import haven_results as hr
        >>> savedir_base='../results'
        >>> rm = hr.ResultManager(savedir_base=savedir_base,
                                filterby_list=[{'dataset':'mnist'}],
                                groupby_list=['dataset'],
                                verbose=1)
        >>> for df in rm.get_score_df():
        >>>     display(df)
        >>> fig_list = rm.get_plot_all(y_metric_list=['train_loss', 'val_acc'], 
                                    order='groups_by_metrics',
                                    x_metric='epoch', 
                                    figsize=(15,6),
                                    map_exp_list=[({'opt':{'name':'lenet'}}, {'label':'LeNet'})],
                                    title_list=['dataset'],
                                    legend_list=['model']) 
        """
        if exp_group_list is not None:
            exp_list = get_exp_list_from_exp_configs(exp_group_list=exp_group_list, workdir=workdir, verbose=verbose)

        elif exp_list is None:
            exp_list = get_exp_list(savedir_base=savedir_base, verbose=verbose)

        else:
            exp_list = exp_list

        self.savedir_base = savedir_base
        self.filterby_list = filterby_list
        self.username = username
        self.verbose = verbose

        exp_list = filter_exp_list(exp_list, filterby_list=filterby_list, verbose=verbose)
        self.exp_groups = group_exp_list(exp_list, groupby_list)
    
    def get_plot(self, **kwargs):
        fig_list = []
        for exp_list in self.exp_groups:
            fig, ax = get_plot(exp_list=exp_list, savedir_base=self.savedir_base, filterby_list=self.filterby_list, 
                        verbose=self.verbose,
                               **kwargs)
            fig_list += [fig]

        return fig_list

    def get_plot_all(self, y_metric_list, order='groups_by_metrics', **kwargs):
        """[summary]
        
        Parameters
        ----------
        y_metric_list : [type]
            [description]
        order : str, optional
            [description], by default 'groups_by_metrics'
        
        Returns
        -------
        [type]
            [description]
        
        """
        if order not in ['groups_by_metrics', 'metrics_by_groups']:
            raise ValueError('%s order is not defined, choose between %s' % (order, ['groups_by_metrics', 'metrics_by_groups']))
        
        figsize = kwargs.get('figsize') or None
        
        fig_list = []
        
        if order == 'groups_by_metrics':
            for exp_list in self.exp_groups:   
                fig, ax_list = plt.subplots(nrows=1, ncols=len(y_metric_list), figsize=figsize)

                for i, y_metric in enumerate(y_metric_list):
                    fig, _ = get_plot(exp_list=exp_list, savedir_base=self.savedir_base, y_metric=y_metric, 
                                    fig=fig, axis=ax_list[i], verbose=self.verbose, filterby_list=self.filterby_list,
                                    **kwargs)
                fig_list += [fig]

        elif order == 'metrics_by_groups':

            for y_metric in y_metric_list:   
                fig, ax_list = plt.subplots(nrows=1, ncols=len(self.exp_groups) , figsize=figsize)
                for i, exp_list in enumerate(self.exp_groups): 
                    fig, _ = get_plot(exp_list=exp_list, savedir_base=self.savedir_base, y_metric=y_metric, 
                                    fig=fig, axis=ax_list[i], verbose=self.verbose, filterby_list=self.filterby_list,
                                    **kwargs)
                fig_list += [fig]

        plt.tight_layout()

        return fig_list
    
    def get_score_df(self, **kwargs):
        """[summary]
        
        Returns
        -------
        [type]
            [description]
        """
        df_list = []
        for exp_list in self.exp_groups:
            df_list += [get_score_df(exp_list=exp_list, savedir_base=self.savedir_base, verbose=self.verbose, **kwargs)]
        return df_list 

    def get_score_lists(self, **kwargs):
        """[summary]
        
        Returns
        -------
        [type]
            [description]
        """
        score_groups = []
        for exp_list in self.exp_groups:
            score_groups += [get_score_lists(exp_list=exp_list, savedir_base=self.savedir_base, 
                             filterby_list=self.filterby_list, verbose=self.verbose, **kwargs)]
        return score_groups

    def get_images(self, **kwargs):
        """[summary]
        
        Returns
        -------
        [type]
            [description]
        """
        for exp_list in self.exp_groups:
            return get_images(exp_list=exp_list, savedir_base=self.savedir_base, verbose=self.verbose, **kwargs)

    def get_job_stats(self, **kwargs):
        """[summary]
        """
        for exp_list in self.exp_groups:
            get_job_stats(exp_list=exp_list, savedir_base=self.savedir_base, username=self.username)
    
    def get_job_errors(self, **kwargs):
        """[summary]
        """
        for exp_list in self.exp_groups:
            get_job_errors(exp_list=exp_list, savedir_base=self.savedir_base, username=self.username)
    
    def get_job_logs(self, **kwargs):
        """[summary]
        """
        for exp_list in self.exp_groups:
            get_job_errors(exp_list=exp_list, savedir_base=self.savedir_base, username=self.username)
   
    def to_zip(self, fname):
        """[summary]
        
        Parameters
        ----------
        fname : [type]
            [description]
        """
        from haven import haven_dropbox as hd

        exp_id_list = [hu.hash_dict(exp_dict) for exp_dict in self.exp_list]
        hd.zipdir(exp_id_list, self.savedir_base, fname)
        print('Zipped %d experiments in %s' % (len(exp_id_list), fname))

    def to_dropbox(self, fname, dropbox_path=None, access_token=None):
        """[summary]
        
        Parameters
        ----------
        fname : [type]
            [description]
        dropbox_path : [type], optional
            [description], by default None
        access_token : [type], optional
            [description], by default None
        """
        from haven import haven_dropbox as hd

        out_fname = os.path.join(dropbox_path, fname)
        src_fname = os.path.join(self.savedir_base, fname)
        self.to_zip(src_fname)
        hd.upload_file_to_dropbox(src_fname, out_fname, access_token)
        print('saved: https://www.dropbox.com/home/%s' % out_fname)
