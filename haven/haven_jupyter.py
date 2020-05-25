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

from . import haven_results as hr
from . import haven_jupyter as hj
from . import haven_utils as hu 
from . import haven_dropbox as hd


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

# filter exps
# e.g. filterby_list =[{'dataset':'mnist'}] gets exps with mnist
filterby_list = None

# get experiments
rm = hr.ResultManager(exp_list=exp_list, 
                      savedir_base=savedir_base, 
                      filterby_list=filterby_list,
                      verbose=0,
                      exp_groups=None
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
        self.rm_original = rm
        if vars is None:
            fname = os.path.join(rm.savedir_base, '.dashboard_history.json')
            if os.path.exists(fname):
                self.vars = hu.load_json(fname)
            else:
                self.vars = {}

        self.vars = vars

        self.show_jobs = show_jobs
        self.wide_display = wide_display

        self.layout=widgets.Layout(width='100px')
        self.layout_label=widgets.Layout(width='200px')
        self.layout_dropdown = widgets.Layout(width='200px')
        self.layout_button = widgets.Layout(width='200px')
        self.t_savedir_base = widgets.Text(
                value=str(self.vars['savedir_base']),
                layout=widgets.Layout(width='600px'),
                disabled=False
                    )
            
        self.t_filterby_list = widgets.Text(
                value=str(self.vars.get('filterby_list')),
                layout=widgets.Layout(width='1200px'),
                description='               filterby_list:',
                disabled=False
                    )


    def display(self):
        self.update_rm()

        # Select Exp Group
        l_exp_group = widgets.Label(value="Select exp_group", layout=self.layout_label,)

        exp_group_list = list(self.rm_original.exp_groups.keys())
        exp_group_selected = 'all'
        if self.vars.get('exp_group', 'all') in exp_group_list:
            exp_group_selected = self.vars.get('exp_group', 'all')

        d_exp_group = widgets.Dropdown(
            options=exp_group_list,
            value=exp_group_selected,
            layout=self.layout_dropdown,
        )
        self.rm_original.exp_list_all = self.rm_original.exp_groups.get(d_exp_group.value, 'all')
        l_n_exps = widgets.Label(value='Total Exps %d' % len(self.rm_original.exp_list_all), layout=self.layout,)
                
        def on_group_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.rm_original.exp_list_all = self.rm_original.exp_groups[change['new']]
                l_n_exps.value = 'Total Exps %d' % len(self.rm_original.exp_list_all)
        
        d_exp_group.observe(on_group_change)

        display(widgets.VBox([l_exp_group,
                        widgets.HBox([d_exp_group, l_n_exps, self.t_filterby_list]) 
        ]))

        hj.init_datatable_mode()
        tables = widgets.Output()
        plots = widgets.Output()
        images = widgets.Output()
        meta = widgets.Output()

        main_out = widgets.Output()
        # Display tabs
        tab = widgets.Tab(children = [tables, plots, images, meta])
        tab.set_title(0, 'Tables')
        tab.set_title(1, 'Plots')
        tab.set_title(2, 'Images')
        tab.set_title(3, 'Meta')
            
        with main_out:
            display(tab)
            tables.clear_output()
            plots.clear_output()
            images.clear_output()
            meta.clear_output()

            # show tabs
            self.table_tab(tables)
            self.plot_tab(plots)
            self.images_tab(images)
            self.meta_tab(meta)

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
        self.rm = hr.ResultManager(exp_list=self.rm_original.exp_list_all, 
                    savedir_base=str(self.t_savedir_base.value), 
                    filterby_list=get_dict_from_str(str(self.t_filterby_list.value)),
                    verbose=self.rm_original.verbose,
                    mode_key=self.rm_original.mode_key,
                    has_score_list=self.rm_original.has_score_list,
                    score_list_name=self.rm_original.score_list_name
                    )

        if len(self.rm.exp_list) == 0:
            if self.rm.n_exp_all > 0:
                display('No experiments selected out of %d '
                    'for filtrby_list %s' % (self.rm.n_exp_all,
                                            self.rm.filterby_list))
                display('Table below shows all experiments.')
                score_table = hr.get_score_df(exp_list=self.rm_original.exp_list_all,
                                              savedir_base=self.rm_original.savedir_base)
                display(score_table)
            else:
                display('No experiments exist...')
            return
        else:
            display('Selected %d/%d experiments using "filterby_list"' % (len(self.rm.selected_exp_list), len(self.rm.exp_list)))

    def meta_tab(self, output):
        with output:
            l_savedir_base = widgets.Label(value="savedir_base:", layout=self.layout_label,)
            l_filterby_list= widgets.Label(value="filterby_list:", layout=self.layout_label,)
            

            bdownload = widgets.Button(description="Zip to Download Experiments", 
                                    layout=self.layout_button)
                                    
            bdownload_out = widgets.Output(layout=self.layout_button)

            bdownload_dropbox = widgets.Button(description="Upload to Dropbox", 
                                    layout=self.layout_button)
                                    
            bdownload_out_dropbox  = widgets.Output(layout=self.layout_button)

            l_fname_list = widgets.Text(
                value=str(self.vars.get('fname_list', '')),
                layout=self.layout_dropdown,
                description='fname_list:',
                disabled=False
                    )

            l_dropbox_path = widgets.Text(
                value=str(self.vars.get('dropbox_path', '/shared')),
                description='dropbox_path:',
                layout=self.layout_dropdown,
                disabled=False
                    )
            l_access_token_path = widgets.Text(
                value=str(self.vars.get('access_token', '')),
                description='access_token:',
                layout=self.layout_dropdown,
                disabled=False
                    )
            def on_upload_clicked(b):
                fname = 'results.zip'
                bdownload_out_dropbox.clear_output()
                self.vars['fname_list'] = get_list_from_str(l_fname_list.value)
                self.vars['dropbox_path'] = l_dropbox_path.value
                self.vars['access_token'] = l_access_token_path.value
                with bdownload_out_dropbox:
                    self.rm.to_zip(savedir_base='', fname=fname, 
                                   fname_list=self.vars['fname_list'],
                                   dropbox_path=self.vars['dropbox_path'],
                                   access_token=self.vars['access_token'])

                os.remove('results.zip')
                display('result.zip sent to dropbox at %s.' % self.vars['dropbox_path'])


            def on_download_clicked(b):
                fname = 'results.zip'
                bdownload_out.clear_output()
                self.vars['fname_list'] = get_list_from_str(l_fname_list.value)

                with bdownload_out:
                    self.rm.to_zip(savedir_base='', fname=fname, 
                                   fname_list=self.vars['fname_list'])
                bdownload_out.clear_output()
                with bdownload_out:
                    display('%d exps zipped.' % len(self.rm.exp_list))
                display(FileLink(fname, result_html_prefix="Download: "))
                # bdownload_out.clear_output()
                # with bdownload_out:
                #     display('%d exps zipped.' % len(self.rm.exp_list))
                    
                

                

            bdownload.on_click(on_download_clicked)
            bdownload_zip = widgets.VBox([bdownload, bdownload_out])
            bdownload_dropbox.on_click(on_upload_clicked)
            bdownload_dropbox_vbox = widgets.VBox([ bdownload_dropbox, bdownload_out_dropbox])
            display(widgets.VBox([
                            widgets.HBox([l_savedir_base, self.t_savedir_base, ]), 
                            widgets.HBox([l_filterby_list, self.t_filterby_list,  ]),
                            widgets.HBox([l_fname_list, l_dropbox_path, l_access_token_path]),
                            widgets.HBox([bdownload_zip,
                                        bdownload_dropbox_vbox])
            ]) 
            )



    def table_tab(self, output):
        d_columns_txt = widgets.Label(value="Select Hyperparam column", 
                                      layout=self.layout_label,)
        d_columns = widgets.Dropdown(
                    options=['None'] + self.rm.exp_params,
                    value='None',
                    layout=self.layout_dropdown,
                    disabled=False,
                )
        d_score_columns_txt = widgets.Label(value="Select Score column",
                                            layout=self.layout_label,)
        d_score_columns = widgets.Dropdown(
                options=self.rm_original.score_keys,
                value='None',
                layout=self.layout_dropdown,
                disabled=False,
            )

        bstatus = widgets.Button(description="Jobs Status")
        blogs = widgets.Button(description="Jobs Logs")
        bfailed = widgets.Button(description="Jobs Failed")

        brefresh = widgets.Button(description="Display Table")
        b_meta = widgets.Button(description="Display Meta Table")
        b_diff = widgets.Button(description="Display Filtered Table")

        

        button = widgets.VBox([widgets.HBox([brefresh, b_diff, b_meta]),
                               widgets.HBox([bstatus, blogs, bfailed]),
                               widgets.HBox([d_columns_txt, d_score_columns_txt]),
                               widgets.HBox([d_columns, d_score_columns ]),
        ])
        output_plot = widgets.Output()

        with output:
            display(button)
            display(output_plot)

        def on_refresh_clicked(b):
            output_plot.clear_output()
            with output_plot:
                self.update_rm()

                self.vars['columns'] = get_list_from_str(d_columns.value)
                self.vars['score_columns'] = get_list_from_str(d_score_columns.value)
                score_table = self.rm.get_score_table(columns=self.vars.get('columns'), 
                                                score_columns=self.vars.get('score_columns'),
                                                hparam_diff=self.vars.get('hparam_diff', 0),
                                                show_meta=self.vars.get('show_meta', 0),
                                                add_prefix=True)
                display(score_table) 

        def on_table_clicked(b):
            output_plot.clear_output()
            with output_plot:
                self.update_rm()
                table_dict = self.rm.get_job_summary(verbose=self.rm.verbose,
                                                username=self.vars.get('username'), add_prefix=True)
                if "exp_dict" in table_dict['table'].columns:
                    del table_dict['table']['exp_dict']
            
            
           
                display(table_dict['status'])
                for state in ['succeeded', 'running', 'queuing', 'failed'][::-1]:
                    n_jobs = len(table_dict[state])
                    if n_jobs:
                        display('Experiments %s: %d' %(state, n_jobs))
                        del table_dict[state]['exp_dict']
                        display(table_dict[state].head())
                # display(table_dict['table'].head())   

        def on_logs_clicked(b):
            output_plot.clear_output()
            with output_plot:
                table_dict = self.rm.get_job_summary(verbose=self.rm.verbose,
                                                username=self.vars.get('username'))
                
                n_logs = len(table_dict['logs'])
           
                for i, logs in enumerate(table_dict['logs']):
                    print('\nLogs %d/%d' % (i+1, n_logs), '='*50)
                    print('exp_id:', logs['exp_id'])
                    print('job_id:', logs['job_id'])
                    print('job_state:', logs['job_state'])

                    print('\nexp_dict')
                    print('-'*50)
                    pprint.pprint(logs['exp_dict'])
                    
                    print('\nLogs')
                    print('-'*50)
                    pprint.pprint(logs['logs'])     
        
        def on_failed_clicked(b):
            output_plot.clear_output()
            with output_plot:
                self.update_rm()
                table_dict = self.rm.get_job_summary(verbose=self.rm.verbose,
                                                username=self.vars.get('username'))
                
                n_failed = len(table_dict['logs_failed'])
           
                if len(table_dict['failed']) == 0:
                    display('no failed experiments')
                else:
                    # display(table_dict['failed'])
                    for i, failed in enumerate(table_dict['logs_failed']):
                        print('\nFailed %d/%d' % (i+1, n_failed), '='*50)
                        print('exp_id:', failed['exp_id'])
                        print('job_id:', failed['job_id'])
                        print('job_state:', failed['job_state'])

                        print('\nexp_dict')
                        print('-'*50)
                        pprint.pprint(failed['exp_dict'])
                        
                        print('\nLogs')
                        print('-'*50)
                        pprint.pprint(failed['logs'])

                        

        # Add call listeners
        brefresh.on_click(on_refresh_clicked)
        bstatus.on_click(on_table_clicked)
        blogs.on_click(on_logs_clicked)
        bfailed.on_click(on_failed_clicked)

        d_columns.observe(on_refresh_clicked)
        d_score_columns.observe(on_refresh_clicked)

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
        llegend_format = widgets.Text(
            value=str(self.vars.get('legend_format', '')),
            description='legend_format:',
            disabled=False
                )
        ltitle_format = widgets.Text(
            value=str(self.vars.get('title_format', '')),
            description='title_format:',
            disabled=False
                )

        

        lcmap = widgets.Text(
            value=str(self.vars.get('cmap', 'jet')),
            description='cmap:',
            layout=self.layout_dropdown,
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

        # d_x_metric_txt = widgets.Label(value="x_metric:", 
        #                               layout=widgets.Layout(width='75px'),)
        # d_x_metric_columns = widgets.Dropdown(
        #             options=['None'] + self.rm_original.score_keys,
        #             value=str(self.vars.get('x_metric')),
        #             layout=self.layout_dropdown,
        #             disabled=False,
        #         )
        d_x_metric_columns = widgets.Text(
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

        d_style = widgets.Dropdown(
                    options=['False', 'True'],
                    value='False',
                    description='interactive:',
                    layout=self.layout_dropdown,
                    disabled=False,
                )
        d_avg_across_txt = widgets.Label(value="avg_across:", 
                                      layout=widgets.Layout(width='75px'),)

        d_avg_across_columns =  widgets.Text(
            value=str(self.vars.get('avg_across', 'None')),
            description='avg_across:',
            disabled=False
                )
        # d_avg_across_columns = widgets.Dropdown(
        #             options=['None'] + self.rm.exp_params,
        #             value=self.vars.get('avg_across', 'None'),
        #             layout=self.layout_dropdown,
        #             disabled=False,
        #         )
        bdownload = widgets.Button(description="Download Plots", 
                                    layout=self.layout_button)
        bdownload_out = widgets.Output(layout=self.layout_button)
        
        def on_download_clicked(b):
            fname = 'plots'
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt

            pp = PdfPages(fname)
            for fig in self.rm_original.fig_list:
                fig.savefig(pp, format='pdf')
            pp.close()

            bdownload_out.clear_output()
          
            with bdownload_out:
                display(FileLink(fname, result_html_prefix="Download: "))

        bdownload.on_click(on_download_clicked)

        brefresh = widgets.Button(description="Display Plot")
        button = widgets.VBox([widgets.HBox([brefresh, bdownload, bdownload_out]),
                widgets.HBox([t_title_list, d_style]),
                widgets.HBox([t_y_metric,  d_x_metric_columns]),
                widgets.HBox([t_groupby_list, llegend_list, ]),
                widgets.HBox([t_mode, t_bar_agg]),
                widgets.HBox([ltitle_format, llegend_format]),
                widgets.HBox([ d_avg_across_columns]),
                
                ])

        output_plot = widgets.Output()


        def on_clicked(b):
            if d_style.value == 'True':
                from IPython import get_ipython
                ipython = get_ipython()
                ipython.magic("matplotlib widget")
            output_plot.clear_output()
            with output_plot:
                self.update_rm()

            
            
                self.vars['y_metrics'] = get_list_from_str(t_y_metric.value)
                self.vars['x_metric'] = d_x_metric_columns.value
                
                w, h = 10, 5
                if len(self.vars['y_metrics']) > 1:
                    figsize = (2*int(w), int(h))
                    self.vars['figsize'] = figsize
                else:
                    self.vars['figsize'] = (int(w), int(h))

                self.vars['legend_list'] = get_list_from_str(llegend_list.value)
                self.vars['legend_format'] = llegend_format.value
                self.vars['log_metric_list'] = get_list_from_str(llog_metric_list.value)
                self.vars['groupby_list'] = get_list_from_str(t_groupby_list.value)
                self.vars['mode'] = t_mode.value
                self.vars['title_list'] = get_list_from_str(t_title_list.value)
                self.vars['bar_agg'] = t_bar_agg.value
                self.vars['title_format'] = ltitle_format.value
                self.vars['cmap'] = lcmap.value
                self.vars['avg_across'] = d_avg_across_columns.value

                avg_across_value = self.vars['avg_across']
                if avg_across_value== "None":
                    avg_across_value = None

                self.rm_original.fig_list = self.rm.get_plot_all(y_metric_list=self.vars['y_metrics'], 
                    x_metric=self.vars['x_metric'], 
                    groupby_list=self.vars['groupby_list'],
                    legend_list=self.vars['legend_list'], 
                    log_metric_list=self.vars['log_metric_list'],
                    mode=self.vars['mode'],
                    bar_agg=self.vars['bar_agg'],
                    figsize=self.vars['figsize'],
                    title_list=self.vars['title_list'],
                    legend_format=self.vars['legend_format'],
                    title_format=self.vars['title_format'],
                    cmap=self.vars['cmap'],
                    avg_across=avg_across_value)
        
           
                
                
                show_inline_matplotlib_plots()
        
        d_style.observe(on_clicked)
        brefresh.on_click(on_clicked)

        with output:
            display(button)
            display(output_plot)
            
    def images_tab(self, output):
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
        t_dirname = widgets.Text(
            value=str(self.vars.get('dirname', 'images')),
            description='dirname:',
            disabled=False
                )
        bdownload = widgets.Button(description="Download Images", 
                                    layout=self.layout_button)
        bdownload_out = widgets.Output(layout=self.layout_button)
        brefresh = widgets.Button(description="Display Images")
        button = widgets.VBox([widgets.HBox([brefresh, bdownload, bdownload_out]),
                widgets.HBox([t_n_exps, t_n_images]),
                widgets.HBox([tfigsize, llegend_list, ]),
                widgets.HBox([t_dirname, ]),
                            ])

        output_plot = widgets.Output()

        with output:
            display(button)
            display(output_plot)

        def on_clicked(b):
            output_plot.clear_output()
            with output_plot:
                self.update_rm()
            
            
                w, h = tfigsize.value.strip('(').strip(')').split(',')
                self.vars['figsize'] = (int(w), int(h))
                self.vars['legend_list'] = get_list_from_str(llegend_list.value)
                self.vars['n_images'] = int(t_n_images.value)
                self.vars['n_exps'] = int(t_n_exps.value)
                self.vars['dirname'] = t_dirname.value
                self.rm_original.fig_image_list =  self.rm.get_images(legend_list=self.vars['legend_list'], 
                        n_images=self.vars['n_images'],
                        n_exps=self.vars['n_exps'],
                        figsize=self.vars['figsize'],
                        dirname=self.vars['dirname'])
                show_inline_matplotlib_plots()
                
        brefresh.on_click(on_clicked)

        
        

        def on_download_clicked(b):
            fname = 'images'
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt

            pp = PdfPages(fname)
            for fig in self.rm_original.fig_image_list:
                fig.savefig(pp, format='pdf')
            pp.close()

            bdownload_out.clear_output()
          
            with bdownload_out:
                display(FileLink(fname, result_html_prefix="Download: "))

        bdownload.on_click(on_download_clicked)


def get_dict_from_str(string):
    if string is None:
        return string

    if string == 'None':
        return None

    if string == '':
        return None
        
    return ast.literal_eval(string)

# def multipage(filename, figs=None, dpi=200):
#     from matplotlib.backends.backend_pdf import PdfPages
#     import matplotlib.pyplot as plt

#     pp = PdfPages(filename)
#     for fig in figs:
#         fig.savefig(pp, format='pdf')
#     pp.close()

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