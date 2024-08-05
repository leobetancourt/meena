import sys
import argparse
import importlib
import inspect
from pathlib import Path
from detail import Hydro

# code for taking Hydro config file, creating instance of class and passing all info to the run method in run.py

def main():
    parser = argparse.ArgumentParser(description='Read config file from command line.')
    parser.add_argument('argument', type=str, help='An argument to be passed to the script')
    config_file = parser.parse_args()
    config_path = Path(config_file.argument)
    print(str(config_path.parent.parent / 'configs'))
    sys.path.append(str(config_path.parent.parent / 'configs'))
    print(config_path.stem)
    config_module = importlib.import_module(config_path.stem)
    
    for name_local in dir(config_module):
        if inspect.isclass(getattr(config_module, name_local)):
            Cfg = getattr(config_module, name_local)
            print(Cfg)
            if issubclass(Cfg, Hydro):
                config = Cfg()
                print(config.resolution())


if __name__ == "__main__":
    main()