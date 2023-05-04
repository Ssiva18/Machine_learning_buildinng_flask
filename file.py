import os
import pandas as  pd
import numpy as np

# Manually create the directories ass well.
dirs_ = ['Data','save_models','src','components','config']

for i in dirs_:
    os.makedirs(i,exist_ok=True)

for i in dirs_:
    with open(os.path.join(i,'.gitkeep'),mode = 'wb') as gitfile:
         pass


# Manullay create files.

files = [os.path.join('components','logging.py'),
         os.path.join('components','exceptions.py'),
         os.path.join('src','Data_manager.py'),
         os.path.join('src','pipeline.py'),'params.yaml',
         os.path.join('src','features.py'),
         os.path.join('config','core.py')]

for i in files:
    with open(i,mode = 'wb')as file_:
        pass 