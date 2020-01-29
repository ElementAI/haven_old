
install by `pip install --upgrade git+https://github.com/ElementAI/haven`
install in jupyter by `!{sys.executable} -m pip install git+https://github.com/ElementAI/haven --user`

### 1. Train and validate experiments

| Command | Description |
| --- | --- |
| `python trainval.py -e MNIST -sb <savedir_base>`| run mnist experiments and save them at <savedir_base> |
| `python trainval.py -e MNIST -sb <savedir_base> -r 1` | reset the mnist experiments |

### 2. View experiments in command-line

| Command | Description |
| --- | --- |
| `python trainval.py -e MNIST -sb <savedir_base> -v 1` | view the mnist experiments |

### 3. View experiments in jupyter
| Command | Description |
| --- | --- |
| `python trainval.py -e MNIST -sb <savedir_base> -j 1` | create the jupyter file 'view_results.ipynb'|


## Haven structure

| File | Description |
| --- | --- |
| `haven_utils.py` | Global utility functions |
| `haven_results.py` | Functions related to viewing  and manipulating results |
