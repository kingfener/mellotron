# python_syn.md


# python inser path
    print('sys.path=',sys.path)

    sys.path.append('/media/U1T/SProject/nvidia/mellotron')
    sys.path.insert(0, os.path.abspath('../mellotron'))   # ok
    sys.path.insert(0, os.path.abspath(__file__))  
    sys.path.insert(0, os.path.realpath(''))  # ok， 
    pwd=/media/U1T/SProject/nvidia/mellotron
    ├── config
    │   ├── hparams_LJ.py
    │   ├── hparams.py
    │   └── __pycache__
    │       ├── hparams_LJ.cpython-37.pyc
    │       └── hparams_LJ.cpython-38.pyc
    ├── text
    │   ├── cleaners.py
    │   ├── cmudict.py
    │   ├── __init__.py
    │   ├── LICENSE
    │   ├── numbers.py
    │   └── symbols.py








