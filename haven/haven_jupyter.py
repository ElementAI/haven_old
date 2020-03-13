import pandas as pd 
from . import haven_utils
import os

def launch_jupyter():
    """
    virtualenv -p python3 .
    source bin/activate
    pip install jupyter notebook
    jupyter notebook --ip 0.0.0.0 --port 2222 --NotebookApp.token='abcdefg'
    """
    print()
    

def view_jupyter(exp_group_list=[],
                 savedir_base='<savedir_base>', 
                 fname='results/example.ipynb', 
                 workdir='<workdir>',
                 install_flag=False,
                 run_flag=False,
                 job_flag=False):
    cells = [header_cell(), 
             exp_list_cell(savedir_base, workdir, exp_group_list)]

    if job_flag:
        cells += [job_cell()]
        
    if install_flag:
        cells += [install_cell()]

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    save_ipynb(fname, cells)

    if run_flag:
        run_notebook(fname)
        
    print('Saved Jupyter: %s' % fname)

def run_notebook(fname):
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    from nbconvert import PDFExporter

    with open(fname) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': 'results/'}})
    with open(fname, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    
def header_cell():
    script = ("""
from haven import haven_jupyter as hj
from haven import haven_results as hr
from haven import haven_dropbox as hd
from haven import haven_utils as hu

hj.init_datatable_mode()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
          """)
    return script

def install_cell():
    script = ("""
import sys
from importlib import reload

# !{sys.executable} -m pip install --upgrade --no-dependencies   git+https://github.com/ElementAI/haven --user
!{sys.executable} -m pip install --upgrade --no-dependencies  '/home/issam/Research_Ground/haven/' --user

reload(hj)
reload(hr)
reload(hd)
reload(hu)
          """)
    return script

def exp_list_cell(savedir_base, workdir, exp_group_list):
    script = ("""
# create result manager
rm = hr.ResultManager(savedir_base='%s', workdir='%s', exp_group_list=%s)

# filter experiments
rm.filter(regard_dict_list=None,
                    groupby_list=None,
                    has_score_list=False)

# get scores
df_list = rm.get_scores(columns=None)
for df in df_list:
    display(df)

# plot
rm.get_plots(y_list=['train_loss', 'val_acc'], transpose=True, 
             x_name='epoch', legend_list=['model'], 
             title_list=['dataset'])

# show images
rm.get_images(legend_list=['model'], dirname='images')
          """ % (savedir_base, workdir, exp_group_list))
    return script


def create_jupyter(fname='example.ipynb'):
    cells = [main_cell()]
    save_ipynb(fname, cells)        
    print('Saved Jupyter: %s' % fname)


def main_cell():
    script = ("""
# Specify variables
from haven import haven_jupyter as hj
from haven import haven_results as hr
from haven import haven_utils as hu

savedir_base = <path_to_saved_experiments>
exp_list = None
# exp_config_name = <exp_config_name>
# exp_list = hu.load_py(exp_config_name).EXP_GROUPS['mnist']

# get specific experiments, for example, {'model':'resnet34'}
filterby_list = None

# group the experiments based on a hyperparameter, for example, ['dataset']
groupby_list = None
verbose = 1

# table vars
columns = None

# plot vars
y_metric='train_loss'
x_metric='epoch'
log_metric_list = ['train_loss']
map_exp_list = []
figsize=(10,5)
title_list=['dataset']
legend_list=['model']
mode='line'

# image vars
image_legend_list = []
n_images=5

# job vars
username = 'anonymous'
columns = None

# dropbox vars
dropbox_path = ''
access_token =  ''
zipname = 'test.zip'

# get experiments
rm = hr.ResultManager(exp_list=exp_list, 
                      savedir_base=savedir_base, 
                      filterby_list=filterby_list,
                      groupby_list=groupby_list,
                      verbose=verbose
                     )

# launch dashboard
hj.get_dashboard(rm, vars())
          """)
    return script

def generate_zip_script(outdir):
    script = ("""
exp_id_list = [hu.hash_dict(exp_dict) for exp_dict in exp_list]
results_fname = '%%s_%%s.zip'%% (exp_group_name, len(exp_list))
src_fname = os.path.join('%s', results_fname)
print('save in:', src_fname)
stop
hd.zipdir(exp_id_list, savedir_base, src_fname)


          """ % outdir)
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

        
def get_dashboard(rm, vars, show_jobs=True):
    import pprint 

    from ipywidgets import Button, HBox, VBox
    from ipywidgets import widgets
    from ipywidgets.widgets.interaction import show_inline_matplotlib_plots

    from IPython.display import display
    from IPython.core.display import Javascript, display

    from haven import haven_results as hr
    from haven import haven_jupyter as hj
    from haven import haven_dropbox as hd

    hj.init_datatable_mode()


    dropbox = widgets.Output()
    tables = widgets.Output()
    plots = widgets.Output()
    images = widgets.Output()
    job_states = widgets.Output()
    job_logs = widgets.Output()
    job_failed = widgets.Output()

    # Display tabs
    tab = widgets.Tab(children = [tables, plots, images, job_states, job_logs, job_failed, dropbox])
    tab.set_title(0, 'Tables')
    tab.set_title(1, 'Plots')
    tab.set_title(2, 'Images')
    tab.set_title(3, 'Job States')
    tab.set_title(4, 'Job Logs')
    tab.set_title(5, 'Job Failed')
    tab.set_title(6, 'Dropbox')
    display(tab)

    # Display tables
    with tables:
        exp_table = rm.get_exp_table()
        # Get score table 
        score_table = rm.get_score_table()
        
        display(exp_table)
        display(score_table)

    # Display plots
    with plots:
        rm.get_plot_all(y_metric_list=vars['y_metric'], 
                x_metric=vars['x_metric'], 
                legend_list=vars['legend_list'], 
                map_exp_list=vars['map_exp_list'], 
                log_metric_list=vars['log_metric_list'],
                mode=vars['mode'],
    #                 xlim=(10, 100),
    #                 ylim=(0.5, 0.8),
                figsize=vars['figsize'],
                title_list=vars['title_list'])
        show_inline_matplotlib_plots()

    # Display images
    with images:
        rm.get_images(legend_list=vars['image_legend_list'], 
                      n_images=vars['n_images'])
        show_inline_matplotlib_plots()

    # Display job states
    with job_states:
        table_dict = rm.get_job_summary(username=vars['username'])[0]
        display(table_dict['status'])
        display(table_dict['table'])

    # Display job failed
    with job_logs:
        table_dict = rm.get_job_summary(username=vars['username'])[0]
    
        display(table_dict['status'])
        display(table_dict['table'])
        for logs in table_dict['logs']:
            pprint.pprint(logs)
                    
    # Display job failed
    with job_failed:
        table_dict = rm.get_job_summary(username=vars['username'])[0]
        if len(table_dict['failed']) == 0:
            display('no failed experiments')
        else:
            display(table_dict['failed'])
            for failed in table_dict['logs_failed']:
                pprint.pprint(failed)

    button = widgets.Button(description="Dropbox")

    with dropbox:
        display(button)

    def on_button_clicked(b):
        with dropbox:
            print("Button clicked.")
            exp_list_new = []
            for el in rm.exp_groups: 
                exp_list_new += el
            print(exp_list_new)
            hd.to_dropbox(vars['exp_list'], 
            vars['savedir_base'], 
            vars['dropbox_path'], 
            vars['access_token'], 
            vars['zipname'])

    button.on_click(on_button_clicked)

