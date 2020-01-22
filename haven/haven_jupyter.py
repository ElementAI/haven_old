import pandas as pd 
import pylab as plt 
from . import haven_utils
import glob 
import os


def create_jupyter(exp_group_list, savedir_base, 
                   workdir,
                   fname=None,
                   legend_list=None, 
                   score_list=None,
                   groupby_list=None,
                   regard_dict=None,
                   disregard_dict=None,
                   extra_cells=None
                   ):
    
    if fname is None:
        fname = workdir+"/view_results.ipynb"
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    script_header = generate_header_script(savedir_base, workdir)
    script_exp_list = generate_exp_list_script(
                            exp_group_list, 
                            regard_dict=regard_dict, 
                            disregard_dict=disregard_dict,
                            groupby_list=groupby_list)

    script_results = generate_results_script_basic(legend_list, score_list, groupby_list)
    
    cells = [script_header, 
                       script_exp_list,
                       script_results,
                       ]

    if extra_cells is not None:
        cells += extra_cells
    save_ipynb(fname, cells)
    print("saved: %s" % fname)
    
    # print('link: %s' % )

def generate_header_script(savedir_base, workdir):
    script = ("""
import itertools
import pprint
import argparse
import sys
import os
import pylab as plt
import pandas as pd
import sys
import numpy as np
import hashlib 
import pickle
import json
import glob
import copy
from itertools import groupby

savedir_base = '%s'
workdir = '%s'
sys.path.append(workdir)

from haven import haven_jupyter as hj
from haven import haven_results as hr

hj.init_datatable_mode()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100%% !important; }</style>"))

          """ % (savedir_base, workdir))
    return script

def generate_exp_list_script(exp_group_list, 
                             regard_dict=None, disregard_dict=None,
                             groupby_list=None):
    script = ("""
from haven import haven_utils as hu
import exp_configs

# get exp list
exp_list = []
for exp_group_name in %s:
    exp_list += exp_configs.EXP_GROUPS[exp_group_name]

# filter 
# For regard_dict, specify what key values you like, 
# for example, regard_dict={'dataset':'mnist'} only shows mnist experiments
exp_list = hu.filter_exp_list(exp_list, 
                            regard_dict=%s,
                            disregard_dict=%s) 
print("%%s experiments" %% len(exp_list))

# group experiments
groupby_key_list = %s

exp_subsets = hr.group_exp_list(exp_list, groupby_key_list)

print('Grouping by', groupby_key_list)
for exp_subset in exp_subsets:
    print('%%d experiments' %% (len(exp_subset)) )
          """ % (exp_group_list, regard_dict, disregard_dict, groupby_list))
    return script


def generate_results_script_basic(legend_list, score_list, groupby_list):
    script = ("""
for i, exp_subset in enumerate(exp_subsets):
    # exp_subset = hr.filter_best_results(exp_subset, savedir_base=savedir_base, groupby_key_list=['model'], score_key='val_mae')
    
    # score df
    df = hr.get_dataframe_score_list(exp_subset, savedir_base=savedir_base)
    display(df)

    # plot
    fig = hr.get_plot(exp_subset, %s, savedir_base, 
    title_list=%s,
    avg_runs=0,
    legend_list=%s,
    height=8,
    width=8
    )
    # plt.savefig('%%s/results/%%s_%%d.jpg' %% (workdir, exp_group_name, i))
    plt.show()

    # qualitative
    hr.get_images(exp_subset, savedir_base, n_exps=3, split="row",
                  height=12,
                  width=12, legend_list=%s)
          """% (score_list, groupby_list, legend_list, legend_list))
    return script

def generate_zip_script():
    script = ("""
exp_id_list = [hu.hash_dict(exp_dict) for exp_dict in exp_list]

results_fname = 'results.zip'
src_fname = savedir_base + '/%s' % results_fname
hr.zipdir(exp_id_list, savedir_base, src_fname)
print('Zipped %d experiments in %s' % (len(exp_id_list), src_fname))


          """)
    return script

def save_ipynb(fname, script_list):
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    nb['cells'] = [nbf.v4.new_code_cell(code) for code in
                   script_list]
    with open(fname, 'w') as f:
        nbf.write(nb, f)
    

def init_datatable_mode():
    """Initialize DataTable mode for pandas DataFrame represenation."""
    import pandas as pd
    from IPython.core.display import display, Javascript

    # configure path to the datatables library using requireJS
    # that way the library will become globally available
    display(Javascript("""
        require.config({
            paths: {
                DT: '//cdn.datatables.net/1.10.19/js/jquery.dataTables.min',
            }
        });
        $('head').append('<link rel="stylesheet" type="text/css" href="//cdn.datatables.net/1.10.19/css/jquery.dataTables.min.css">');
    """))

    def _repr_datatable_(self):
        """Return DataTable representation of pandas DataFrame."""
        # classes for dataframe table (optional)
        classes = ['table', 'table-striped', 'table-bordered']

        # create table DOM
        script = (
            f'$(element).html(`{self.to_html(index=False, classes=classes)}`);\n'
        )

        # execute jQuery to turn table into DataTable
        script += """
            require(["DT"], function(DT) {
                $(document).ready( () => {
                    // Turn existing table into datatable
                    $(element).find("table.dataframe").DataTable();
                })
            });
        """

        return script

    pd.DataFrame._repr_javascript_ = _repr_datatable_


def get_images(exp_list, savedir_base, n_exps=3):
    from IPython.core.display import display
    
    for k, exp_dict in enumerate(exp_list):
        if k >= n_exps:
            return
        result_dict = {}

        exp_id = haven_utils.hash_dict(exp_dict)
        result_dict["exp_id"] = exp_id
        savedir = savedir_base + "/%s/" % exp_id 
        img_list = glob.glob(savedir + "/images/images/*.jpg")
        
        ncols = len(img_list)
        # ncols = len(exp_configs)
        nrows = 1
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                                figsize=(ncols*12, nrows*3))
        
        if not isinstance(axs, list):
            axs = [axs]
   
        for i in range(ncols):
            img = plt.imread(img_list[i])
            axs[0][i].imshow(img)
            axs[0][i].set_axis_off()
            axs[0][i].set_title(haven_utils.extract_fname(img_list[i]))
#         fig.suptitle(exp_id)
        plt.axis('off')
        plt.tight_layout()
        display("Experiment %s %s" % (exp_id,"="*100))
        display(pd.DataFrame([exp_dict]))
        plt.show()
#         display("="*100)
        