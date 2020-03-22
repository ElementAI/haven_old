import pandas as pd 
from . import haven_utils
import os
import pprint, json
from haven import haven_utils as hu 
import copy


def launch_jupyter():
    """
    virtualenv -p python3 .
    source bin/activate
    pip install jupyter notebook
    jupyter notebook --ip 0.0.0.0 --port 2222 --NotebookApp.token='abcdefg'
    """
    print()
    

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

# please define the path to the experiments
savedir_base = <path_to_saved_experiments>
exp_list = None

# exp_config_name = <exp_config_name>
# exp_list = hu.load_py(exp_config_name).EXP_GROUPS['mnist']

# get specific experiments, for example, {'model':'resnet34'}
filterby_list = None

# group the experiments based on a hyperparameter, for example, ['dataset']
groupby_list = None
verbose = 0

# plot vars
y_metrics='train_loss'
x_metric='epoch'
log_metric_list = ['train_loss']
map_exp_list = []
title_list=['dataset']
legend_list=['model']

# get experiments
rm = hr.ResultManager(exp_list=exp_list, 
                      savedir_base=savedir_base, 
                      filterby_list=filterby_list,
                      verbose=verbose
                     )

# launch dashboard
hj.get_dashboard(rm, vars(), wide_display=True)
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
            f'$(element).html(`{self.to_html(index=True, classes=classes)}`);\n'
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

        
def get_dashboard(rm, vars, show_jobs=True, wide_display=False):
    dm = DashboardManager(rm, vars, show_jobs, wide_display)
    dm.display()

class DashboardManager:
    def __init__(self, rm, vars, show_jobs, wide_display):
        self.rm_old = rm
        self.vars = vars
        self.show_jobs = show_jobs
        self.wide_display = wide_display

    def display(self):
        import ast
        from ipywidgets import Button, HBox, VBox
        from ipywidgets import widgets

        from IPython.display import display
        from IPython.core.display import Javascript, display, HTML

        from haven import haven_results as hr
        from haven import haven_jupyter as hj

        t_savedir_base = widgets.Text(
            value=str(self.vars['savedir_base']),
            description='savedir_base:',
            disabled=False
                )

        t_filterby_list = widgets.Textarea(
            value=str(self.vars['filterby_list']),
            description='filterby_list:',
            disabled=False
                )
        bset = widgets.Button(description="set")
        display(widgets.HBox([t_savedir_base, t_filterby_list, bset]))

        hj.init_datatable_mode()

        dropbox = widgets.Output()
        tables = widgets.Output()
        plots = widgets.Output()
        images = widgets.Output()
        jobs = widgets.Output()

        main_out = widgets.Output()
        # Display tabs
        tab = widgets.Tab(children = [tables, plots, images, jobs,
                                    dropbox])
        tab.set_title(0, 'Tables')
        tab.set_title(1, 'Plots')
        tab.set_title(2, 'Images')
        tab.set_title(3, 'Jobs')
        tab.set_title(4, 'Dropbox')
            
        def on_button_clicked(sender):
            self.rm = hr.ResultManager(exp_list=self.rm_old.exp_list_all, 
                        savedir_base=str(t_savedir_base.value), 
                        filterby_list=ast.literal_eval(str(t_filterby_list.value)),
                        verbose=self.rm_old.verbose,
                        )
            
            main_out.clear_output()
            with main_out:
                if len(self.rm.exp_list) == 0:
                    if self.rm.n_exp_all > 0:
                        display('no experiments selected out of %d '
                            'for filtrby_list %s' % (self.rm.n_exp_all,
                                                    self.rm.filterby_list))
                        print('All exp_list.')
                        score_table = hr.get_score_df(exp_list=self.rm_old.exp_list_all,
                                                    savedir_base=self.rm_old.savedir_base)
                        display(score_table)
                    else:
                        print('no experiments exist...')
                    return
                
                display('Acquiring %d Experiments From: %s' % (len(self.rm.exp_list), 
                    self.rm.savedir_base))
                
                display(tab)

                tables.clear_output()
                plots.clear_output()
                images.clear_output()
                jobs.clear_output()
                dropbox.clear_output()

                self.table_tab(tables)
                self.plot_tab(plots)
                # Display images
                self.images_tab(images)
                # Display job states
                self.job_tab(jobs)
                # Dropbox tab
                self.dropbox_tab(dropbox)

                

        bset.on_click(on_button_clicked)
        display(main_out)
        if self.wide_display:
            display(HTML("<style>.container { width:100% !important; }</style>"))
            # display(HTML("<style>.output_result { max-width:100% !important; }</style>"))
            # display(HTML("<style>.prompt { display:none !important; }</style>"))

        # This makes cell show full height display
        style = """
        <style>
            .output_scroll {
                height: unset !important;
                border-radius: unset !important;
                -webkit-box-shadow: unset !important;
                box-shadow: unset !important;
            }
        </style>
        """
        display(HTML(style))

    def table_tab(self, output):
        from IPython.display import display
        from ipywidgets import widgets
        
        t_columns = widgets.Text(
            value=str(self.vars.get('columns')),
            description='columns:',
            disabled=False
                )
        t_score_columns = widgets.Text(
            value=str(self.vars.get('score_columns')),
            description='score_columns:',
            disabled=False
                )
        brefresh = widgets.Button(description="refresh")

        button = widgets.HBox([t_columns, t_score_columns, brefresh])
        output_plot = widgets.Output()

        with output:
            display(button)
            display(output_plot)

        def on_refresh_clicked(b):
            self.vars['columns'] = get_list_from_str(t_columns.value)
            self.vars['score_columns'] = get_list_from_str(t_score_columns.value)
            score_table = rm.get_score_table(columns=self.vars.get('columns'), 
                                            score_columns=self.vars.get('score_columns'))

            output_plot.clear_output()
            with output_plot:
                display(score_table) 

        brefresh.on_click(on_refresh_clicked)


    def job_tab(self, output):
        # plot tab
        from IPython.display import display
        from ipywidgets import widgets
        from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
        
        btable = widgets.Button(description="table")
        blogs = widgets.Button(description="logs")
        bfailed = widgets.Button(description="failed")

        button = widgets.HBox([btable, blogs, bfailed])
        output_plot = widgets.Output()

        with output:
            display(button)
            display(output_plot)
        
        def on_table_clicked(b):
            table_dict = self.rm.get_job_summary(verbose=self.rm.verbose,
                                            username=self.vars.get('username'))

            output_plot.clear_output()
            with output_plot:
                display(table_dict['status'])
                display(table_dict['table'])   

        btable.on_click(on_table_clicked)

        def on_logs_clicked(b):
            table_dict = self.rm.get_job_summary(verbose=self.rm.verbose,
                                            username=self.vars.get('username'))
            output_plot.clear_output()
            with output_plot:
                for logs in table_dict['logs']:
                    pprint.pprint(logs)        
        blogs.on_click(on_logs_clicked)

        def on_failed_clicked(b):
            table_dict = self.rm.get_job_summary(verbose=self.rm.verbose,
                                            username=self.vars.get('username'))
            output_plot.clear_output()
            with output_plot:
                if len(table_dict['failed']) == 0:
                    display('no failed experiments')
                else:
                    display(table_dict['failed'])
                    for failed in table_dict['logs_failed']:
                        pprint.pprint(failed)

        bfailed.on_click(on_failed_clicked)

    def plot_tab(self, output):
        # plot tab
        from IPython.display import display
        from ipywidgets import widgets
        from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
        
        ## add stuff
        tfigsize = widgets.Text(
            value=str(self.vars.get('figsize', '(10,5)')),
            description='figsize:',
            disabled=False
                )
        llegend_list = widgets.Text(
            value=str(self.vars.get('legend_list', '[model]')),
            description='legend_list:',
            disabled=False
                )
        
        t_y_metric = widgets.Text(
            value=str(self.vars.get('y_metrics', 'train_loss')),
            description='y_metrics:',
            disabled=False
                )

        t_x_metric = widgets.Text(
            value=str(self.vars.get('x_metric', 'epoch')),
            description='x_metric:',
            disabled=False
                )

        t_groupby_list = widgets.Text(
            value=str(self.vars.get('groupby_list')),
            description='groupby_list:',
            disabled=False
                )

        t_mode = widgets.Text(
            value=str(self.vars.get('mode', 'line')),
            description='mode:',
            disabled=False
                )

        t_bar_agg = widgets.Text(
            value=str(self.vars.get('bar_agg', 'mean')),
            description='bar_agg:',
            disabled=False
                )

        t_title_list = widgets.Text(
            value=str(self.vars.get('title_list', 'dataset')),
            description='title_list:',
            disabled=False
                )

        l_exp_params = widgets.Label(value="exp_params: %s" % str(self.rm.exp_params),
                )


        brefresh = widgets.Button(description="Refresh")
        button = widgets.VBox([widgets.HBox([brefresh, l_exp_params]),
                widgets.HBox([t_y_metric, t_x_metric,
                            t_groupby_list, llegend_list, tfigsize]),
                widgets.HBox([t_title_list, t_mode, t_bar_agg]) ])
        output_plot = widgets.Output()

        def on_clicked(b):
            output_plot.clear_output()
            with output_plot:
                w, h = tfigsize.value.strip('(').strip(')').split(',')

                self.vars['figsize'] = (int(w), int(h))
                self.vars['legend_list'] = get_list_from_str(llegend_list.value)
                self.vars['y_metrics'] = get_list_from_str(t_y_metric.value)
                self.vars['x_metric'] = t_x_metric.value
                self.vars['groupby_list'] = get_list_from_str(t_groupby_list.value)
                self.vars['mode'] = t_mode.value
                self.vars['title_list'] = get_list_from_str(t_title_list.value)
                self.vars['bar_agg'] = t_bar_agg.value

                self.rm.get_plot_all(y_metric_list=self.vars['y_metrics'], 
                    x_metric=self.vars['x_metric'], 
                    groupby_list=self.vars['groupby_list'],
                    legend_list=self.vars['legend_list'], 
                    map_exp_list=self.vars['map_exp_list'], 
                    log_metric_list=self.vars['log_metric_list'],
                    mode=self.vars['mode'],
                    bar_agg=self.vars['bar_agg'],
                    figsize=self.vars['figsize'],
                    title_list=self.vars['title_list'])

                show_inline_matplotlib_plots()
                
        brefresh.on_click(on_clicked)

        with output:
            display(button)
            display(output_plot)
            
    def images_tab(self, output):
        # plot tab
        from IPython.display import display
        from ipywidgets import widgets
        from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
        tfigsize = widgets.Text(
            value=str(self.vars.get('figsize', '(10,5)')),
            description='figsize:',
            disabled=False
                )
        llegend_list = widgets.Text(
            value=str(self.vars.get('legend_list', '[model]')),
            description='legend_list:',
            disabled=False
                )
        
        t_n_images = widgets.Text(
            value=str(self.vars.get('n_images', '5')),
            description='n_images:',
            disabled=False
                )

        t_n_exps = widgets.Text(
            value=str(self.vars.get('n_exps', '3')),
            description='n_exps:',
            disabled=False
                )

        brefresh = widgets.Button(description="Refresh")
        button = widgets.VBox([brefresh,
                widgets.HBox([t_n_images, t_n_exps,
                            llegend_list, tfigsize])])
        output_plot = widgets.Output()

        
        with output:
            display(button)
            display(output_plot)

        def on_clicked(b):
            output_plot.clear_output()
            with output_plot:
                w, h = tfigsize.value.strip('(').strip(')').split(',')
                self.vars['figsize'] = (int(w), int(h))
                self.vars['legend_list'] = get_list_from_str(llegend_list.value)
                self.vars['n_images'] = int(t_n_images.value)
                self.vars['n_exps'] = int(t_n_exps.value)

                self.rm.get_images(legend_list=self.vars['legend_list'], 
                        n_images=self.vars['n_images'],
                        n_exps=self.vars['n_exps'])
                show_inline_matplotlib_plots()
                
        brefresh.on_click(on_clicked)


    def dropbox_tab(self, output):
        from haven import haven_dropbox as hd
        from IPython.display import display
        from ipywidgets import widgets

        t_config = widgets.Text(
            value=self.vars.get('exp_config_fname','exp_config.py'),
            description='exp_config_fname:',
            disabled=False,
            width='auto'
                )
        t_groups = widgets.Text(
            value=str(self.vars.get('exp_groups','')),
            description='exp_groups:',
            disabled=False
                )

        t_dropbox_path = widgets.Text(
            value=self.vars.get('dropbox_path',''),
            description='dropbox_path:',
            disabled=False
                )

        t_access_token = widgets.Text(
            value=self.vars.get('access_token',''),
            description='access_token:',
            disabled=False
                )

        t_zip_name = widgets.Text(
            value=self.vars.get('zip_name',''),
            description='zip_name:',
            disabled=False
                )

        brefresh = widgets.Button(description="Upload to Dropbox")
        button = widgets.VBox([t_config,
                t_groups, t_dropbox_path, t_access_token, t_zip_name,
                brefresh])

        def on_clicked(b):
            with output:
                exp_config_fname = t_config.value
                exp_groups = get_list_from_str(t_groups.value)

                exp_list_new = []
                for group in exp_groups:
                    exp_list_new += hu.load_py(exp_config_fname).EXP_GROUPS[group]

                hd.to_dropbox(exp_list_new, 
                self.vars['savedir_base'], 
                t_dropbox_path.value, 
                t_access_token.value, 
                t_zip_name.value)

        brefresh.on_click(on_clicked)
        
        with output:
            display(button)


def get_list_from_str(string):
    
    if string is None:
        return string

    if string == 'None':
        return None

    return string.replace(' ','').strip(']').strip('[').replace('"', '').replace("'", "").split(',')
    
    # import ast
    # return ast.literal_eval(string)
    # if string is None:
    #     return "none"
    
    # if isinstance(string, str) and "[" not in string:
    #     return string

    # res = string.strip(" ").strip("[").strip("]").split(',')
    # return [s.strip('"').strip("'").replace(" ", "") for s in res]