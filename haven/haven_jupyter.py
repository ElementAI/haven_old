import pandas as pd 
from . import haven_utils
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
    
def generate_header_script(savedir_base, workdir):
    script = ("""
from importlib import reload
from haven import haven_jupyter as hj
from haven import haven_results as hr
from haven import haven_dropbox as hd
from haven import haven_utils as hu

hj.init_datatable_mode()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
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

        