# # This starts a jupyter for visualizing results
# mkdir .jupyter_server
# cd .jupyter_server
# virtualenv -p python3 .
# source bin/activate
# pip install jupyter notebook
nohup jupyter notebook --ip 0.0.0.0 --port 9123 \
      --notebook-dir="/home/$USER" --NotebookApp.token='abcdefg'
cd ..