# Description
This is a repository for myself to learn general nlp techniques. The required packages are saved in the `requirements.txt` file.

To create a virtual environment with venv:

* Use commond prompt (not powershell) and change directory to the repo folder - `cd path_to_folder` 
* Run `python -m venv ./venv_name` - I have set up the `venv_name` to be `nlp_learning` and this folder is added to gitignore 
* Change directory to the venv folder `cd ./venv_name/Scripts`
* Run `activate` to activate the virtual environment, now you should see `(venv_name)` in front of your command promt
* Run `cd ..` twice to come back to the root folder
* Run `pip install -r requirements.txt` to install the required packages and now you are good to go

To set up the kernel of the virtual environment in your jupyter notebook:

* Run `python -m ipykernel install --user --name venv_name --display-name "NLP Learning"`


# Datasets
We will use the following datasets throughout the notebook, more datasets might be added into the list:
* https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection 
