# Intelligent-Intrusion-Detection-System
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/OWASP/wstg/issues)


OWASP IIDS is an open source software that leverages the benefits of Artificial Intelligence to detect the intrusion and alert the respective network administrator.


## Features
- Django Based:- It is a fully Djnago based application
- Supports Multiple ML and NN models:- As the accuracy of different datasets depends on different models, so we have provided multiple models for training



## Installation
> Note: We recommend installing IIDS in a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-virtualenv)
 to prevent conflicts with other projects or packages

The user must install all packages using PyPI using pip install
As we don't have a requirements.txt as of now, The user must manually install all packages.
The List of packages are:-
- Django  ```pip install django``` 
- restframework ```pip install djangorestframework```
- sklearn ```pip install sklearn```
- numpy ```pip install numpy```
- pandas ```pip install pandas```
- Keras ```pip install keras```
- PyTorch ```pip install torch```
- Cuda(if Nvidia GPU support is available)

## Usage
First the user needs to configure the json file named config.json by mentioning the model and all the hypermaters, which are to be used.


```bash
python manage.py get_data -d  data_path -c config_path -m ''
```
Fo Training:-
The above command needs to be run in the directory of ../iids/iids
For the flag -d u need to provide the path of dataset to be used for training.
For the flag -c u need to provide the path of config.json file.
For the flag -m u need to mention the type of model to be used, 'ml' if machine learning model is to be used and 'nn' for neural network.

For prediction:-
U need to provide the input data in a list format to optional flag -i, u don't need the -d flag, but u still need to provide arguments for flags -c and -m


## Suggestions and Feedback

To report and error or suggest an improvement, please create an [issue](https://github.com/OWASP/Intelligent-Intrusion-Detection-System/issues "Github issues") or create a Pull Request.

## How to Contribute
The IIDS still is in early development stages, so we need developers like you to contribute on this project. We suggest you to read the [Contributing guidelines](https://github.com/OWASP/Intelligent-Intrusion-Detection-System/blob/master/docs/CONTRIBUTING.md) before pulling a PR.

You can contact the maintainers and active developers on the [slack channel](https://join.slack.com/t/owasp-iids/shared_invite/zt-ee5uybw2-6Q92sWtUp~IvArd~~XQ9BQ) 

## License
OWASP's IIDS is licensed and distributed under the GNU General Public License v3.0.
