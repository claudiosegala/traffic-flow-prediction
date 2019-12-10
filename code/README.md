# Code

This folder contains the code and the structure of folder used to organize the plots that this implementation will generate. Be aware that:

+ You need to put the dataset in the dataset folder
+ Set the TCC_PATH variable in python code

## Usage

1. Initiate Virtual Environment

```bash
virtualenv --system-site-packages --python=/usr/bin/python3.7 ./venv
```

2. Activate Virtual Environment

```bash
source ./venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run 

```bash
python3 tcc.py
```
