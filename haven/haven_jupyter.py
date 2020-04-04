import pandas as pd 
from . import haven_utils
import os
import pprint, json
import copy
import pprint

try:
    import ast
    from ipywidgets import Button, HBox, VBox
    from ipywidgets import widgets

    from IPython.display import display
    from IPython.core.display import Javascript, display, HTML
    from IPython.display import FileLink, FileLinks
    from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
except:
    pass

from haven import haven_results as hr
from haven import haven_jupyter as hj
from haven import haven_utils as hu 
from haven import haven_dropbox as hd


def launch_jupyter():
    """
    virtualenv -p python3 .
    source bin/activate
    pip install jupyter notebook
    jupyter notebook --ip 0.0.0.0 --port 2222 --NotebookApp.token='abcdefg'
    """
    print()
    

def create_jupyter(fname='example.ipynb', 
                   savedir_base='<path_to_saved_experiments>', 
                   overwrite=False, print_url=False,
                   create_notebook=True):
    print('Jupyter') 

    if create_notebook and (overwrite or not os.path.exists(fname)):
        cells = [main_cell(savedir_base), install_cell()]
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        save_ipynb(fname, cells)  
        print('- saved:', fname)

    if print_url:
        from notebook import notebookapp
        servers = list(notebookapp.list_running_servers())
        hostname = os.uname().nodename
        
        flag = False
        for i, s in enumerate(servers):
            if s['hostname'] == 'localhost':
                continue
            flag = True
            url = 'http://%s:%s/' % (hostname, s['port'])
            print('- url:', url)

        if flag == False:
            print('a jupyter server was not found :(')
            print('a jupyter server can be started using the script in https://github.com/ElementAI/haven .')


def main_cell(savedir_base):
    script = ("""
from haven import haven_jupyter as hj
from haven import haven_results as hr
from haven import haven_utils as hu

# path to where the experiments got saved
savedir_base = '%s'
exp_list = None

# exp_list = hu.load_py(<exp_config_name>).EXP_GROUPS[<exp_group>]

# filter exps
# e.g. filterby_list =[{'dataset':'mnist'}] gets exps with mnist
filterby_list = None

# get experiments
rm = hr.ResultManager(exp_list=exp_list, 
                      savedir_base=savedir_base, 
                      filterby_list=filterby_list,
                      verbose=0
                     )

# launch dashboard
# make sure you have 'widgetsnbextension' enabled; 
# otherwise see README.md in https://github.com/ElementAI/haven

hj.get_dashboard(rm, vars(), wide_display=True)
          """ % savedir_base)
    return script

def install_cell():
    script = ("""
    !pip install --upgrade git+https://github.com/ElementAI/haven
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
                    $(element).find("table.dataframe").DataTable({"scrollX": true});

                    $('#container').css( 'display', 'block' );
                    table.columns.adjust().draw();
                    
                })
            });
        """
        

        return script

    pd.DataFrame._repr_javascript_ = _repr_datatable_

        
def get_dashboard(rm, vars=None, show_jobs=True, wide_display=False):
    dm = DashboardManager(rm, vars=vars, show_jobs=show_jobs, wide_display=wide_display)
    dm.display()

class DashboardManager:
    def __init__(self, rm, vars=None, show_jobs=True, wide_display=True):
        self.rm_old = rm
        
        if vars is None:
            fname = os.path.join(rm.savedir_base, '.dashboard_history.json')
            if os.path.exists(fname):
                self.vars = hu.load_json(fname)
            else:
                self.vars = {}

        self.vars = vars

        self.show_jobs = show_jobs
        self.wide_display = wide_display

    def display(self):
        self.t_savedir_base = widgets.Text(
            value=str(self.vars['savedir_base']),
            description='savedir_base:',
            layout=widgets.Layout(width='600px'),
            disabled=False
                )

        self.t_filterby_list = widgets.Textarea(
            value=str(self.vars.get('filterby_list')),
            description='filterby_list:',
            layout=widgets.Layout(width='600px'),
            disabled=False
                )

        self.update_rm()

        


        bdownload = widgets.Button(description="Zip to Download Experiments", 
                                   layout=widgets.Layout(width='300px'))
        bdownload_out = widgets.Output(layout=widgets.Layout(width='300px'))
        
        def on_download_clicked(b):
            fname = 'results.zip'
            bdownload_out.clear_output()
            with bdownload_out:
                self.rm.to_zip(savedir_base='', fname=fname)
            bdownload_out.clear_output()
            with bdownload_out:
                display('%d exps zipped.' % len(self.rm.exp_list))
                display(FileLink(fname, result_html_prefix="Download: "))

        bdownload.on_click(on_download_clicked)

        display(widgets.VBox([
                        widgets.HBox([self.t_savedir_base, bdownload]), 
                        widgets.HBox([self.t_filterby_list, bdownload_out ])
        ]))

        hj.init_datatable_mode()
        tables = widgets.Output()
        plots = widgets.Output()
        images = widgets.Output()

        main_out = widgets.Output()
        # Display tabs
        tab = widgets.Tab(children = [tables, plots, images])
        tab.set_title(0, 'Tables')
        tab.set_title(1, 'Plots')
        tab.set_title(2, 'Images')
            
        with main_out:
            display(tab)
            tables.clear_output()
            plots.clear_output()
            images.clear_output()

            # show tabs
            self.table_tab(tables)
            self.plot_tab(plots)
            self.images_tab(images)

        display(main_out)

        if self.wide_display:
            display(HTML("<style>.container { width:100% !important; }</style>"))

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
    
    def update_rm(self):
        self.rm = hr.ResultManager(exp_list=self.rm_old.exp_list_all, 
                    savedir_base=str(self.t_savedir_base.value), 
                    filterby_list=ast.literal_eval(str(self.t_filterby_list.value)),
                    verbose=self.rm_old.verbose,
                    )

        if len(self.rm.exp_list) == 0:
            if self.rm.n_exp_all > 0:
                display('No experiments selected out of %d '
                    'for filtrby_list %s' % (self.rm.n_exp_all,
                                            self.rm.filterby_list))
                display('Table below shows all experiments.')
                score_table = hr.get_score_df(exp_list=self.rm_old.exp_list_all,
                                              savedir_base=self.rm_old.savedir_base)
                display(score_table)
            else:
                display('No experiments exist...')
            return


    def table_tab(self, output):
        layout = {'width': '200px'}
        d_columns_txt = widgets.Label(value="Select Hyperparam column", 
                                      layout=layout,)
        d_columns = widgets.Dropdown(
                    options=['None'] + self.rm.exp_params,
                    value='None',
                    layout=layout,
                    disabled=False,
                )
        d_score_columns_txt = widgets.Label(value="Select Score column",
                                            layout=layout,)
        d_score_columns = widgets.Dropdown(
                options=['None', 'train_loss', 'val_score'],
                value='None',
                layout=layout,
                disabled=False,
            )

        bstatus = widgets.Button(description="Jobs Status")
        blogs = widgets.Button(description="Jobs Logs")
        bfailed = widgets.Button(description="Jobs Failed")

        b_meta = widgets.Button(description="Show exp_id")
        b_diff = widgets.Button(description="Filter Columns")

        brefresh = widgets.Button(description="Display")

        button = widgets.VBox([widgets.HBox([brefresh]),
                               widgets.HBox([b_meta, b_diff]),
                               widgets.HBox([bstatus, blogs, bfailed]),
                               widgets.HBox([d_columns_txt, d_score_columns_txt]),
                               widgets.HBox([d_columns, d_score_columns ]),
        ])
        output_plot = widgets.Output()

        with output:
            display(button)
            display(output_plot)

        def on_refresh_clicked(b):
            self.update_rm()

            self.vars['columns'] = get_list_from_str(d_columns.value)
            self.vars['score_columns'] = get_list_from_str(d_score_columns.value)
            score_table = self.rm.get_score_table(columns=self.vars.get('columns'), 
                                            score_columns=self.vars.get('score_columns'),
                                            hparam_diff=self.vars.get('hparam_diff', 0),
                                            show_meta=self.vars.get('show_meta', 0))

            output_plot.clear_output()
            with output_plot:
                display(score_table) 

        def on_table_clicked(b):
            table_dict = self.rm.get_job_summary(verbose=self.rm.verbose,
                                            username=self.vars.get('username'))

            output_plot.clear_output()
            with output_plot:
                display(table_dict['status'])
                display(table_dict['table'])   

        def on_logs_clicked(b):
            table_dict = self.rm.get_job_summary(verbose=self.rm.verbose,
                                            username=self.vars.get('username'))
            output_plot.clear_output()
            with output_plot:
                for logs in table_dict['logs']:
                    pprint.pprint(logs)        
        
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

        # Add call listeners
        brefresh.on_click(on_refresh_clicked)
        bstatus.on_click(on_table_clicked)
        blogs.on_click(on_logs_clicked)
        bfailed.on_click(on_failed_clicked)

        # meta stuff and column filtration
        def on_bmeta_clicked(b):
            self.vars['show_meta'] = 1 - self.vars.get('show_meta', 0)
            on_refresh_clicked(None)

        def on_hparam_diff_clicked(b):
            self.vars['hparam_diff'] = 2 - self.vars.get('hparam_diff', 0)
            on_refresh_clicked(None)

        b_meta.on_click(on_bmeta_clicked)
        b_diff.on_click(on_hparam_diff_clicked)

        
    def plot_tab(self, output):
        ## add stuff
        
        llegend_list = widgets.Text(
            value=str(self.vars.get('legend_list', '[model]')),
            description='legend_list:',
            disabled=False
                )
        llog_metric_list = widgets.Text(
            value=str(self.vars.get('log_metric_list', '[train_loss]')),
            description='log_metric_list:',
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

        brefresh = widgets.Button(description="Display")
        button = widgets.VBox([widgets.HBox([brefresh]),
                widgets.HBox([t_title_list]),
                widgets.HBox([t_y_metric, t_x_metric, ]),
                widgets.HBox([t_groupby_list, llegend_list, ]),
                widgets.HBox([t_mode, t_bar_agg]) ])

        output_plot = widgets.Output()

        def on_clicked(b):
            self.update_rm()

            output_plot.clear_output()
            with output_plot:
                self.vars['y_metrics'] = get_list_from_str(t_y_metric.value)
                self.vars['x_metric'] = t_x_metric.value
                
                w, h = 10, 5
                if len(self.vars['y_metrics']) > 1:
                    figsize = (2*int(w), int(h))
                    self.vars['figsize'] = figsize
                else:
                    self.vars['figsize'] = (int(w), int(h))

                self.vars['legend_list'] = get_list_from_str(llegend_list.value)
                
                self.vars['log_metric_list'] = get_list_from_str(llog_metric_list.value)
                self.vars['groupby_list'] = get_list_from_str(t_groupby_list.value)
                self.vars['mode'] = t_mode.value
                self.vars['title_list'] = get_list_from_str(t_title_list.value)
                self.vars['bar_agg'] = t_bar_agg.value

                self.rm.get_plot_all(y_metric_list=self.vars['y_metrics'], 
                    x_metric=self.vars['x_metric'], 
                    groupby_list=self.vars['groupby_list'],
                    legend_list=self.vars['legend_list'], 
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
        tfigsize = widgets.Text(
            value=str(self.vars.get('figsize_images', '(10,5)')),
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

        brefresh = widgets.Button(description="Display")
        button = widgets.VBox([brefresh,
                widgets.HBox([t_n_exps, t_n_images]),
                widgets.HBox([tfigsize, llegend_list, ]),
                            ])

        output_plot = widgets.Output()

        with output:
            display(button)
            display(output_plot)

        def on_clicked(b):
            self.update_rm()
            output_plot.clear_output()
            with output_plot:
                w, h = tfigsize.value.strip('(').strip(')').split(',')
                self.vars['figsize_images'] = (int(w), int(h))
                self.vars['legend_list'] = get_list_from_str(llegend_list.value)
                self.vars['n_images'] = int(t_n_images.value)
                self.vars['n_exps'] = int(t_n_exps.value)

                self.rm.get_images(legend_list=self.vars['legend_list'], 
                        n_images=self.vars['n_images'],
                        n_exps=self.vars['n_exps'],
                        figsize=self.vars['figsize_images'])
                show_inline_matplotlib_plots()
                
        brefresh.on_click(on_clicked)


def get_list_from_str(string):
    
    if string is None:
        return string

    if string == 'None':
        return None

    string = string.replace(' ','').replace(']', '').replace('[', '').replace('"', '').replace("'", "")
    if string == '':
        return None

    return string.split(',')
    
    # import ast
    # return ast.literal_eval(string)
    # if string is None:
    #     return "none"
    
    # if isinstance(string, str) and "[" not in string:
    #     return string

    # res = string.strip(" ").strip("[").strip("]").split(',')
    # return [s.strip('"').strip("'").replace(" ", "") for s in res]