import pandas as pd 
from . import haven_utils
import os


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

def job_cell():
    script = ("""
# get job status
rm.get_job_logs()
rm.get_job_stats()
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

        