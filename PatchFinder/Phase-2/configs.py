"""
2024.08.13

This file contains the configurations for the model.

For convenience, we reused it for baselines as well (modified the configurations accordingly).
"""

import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
gpus = [0,1,2,3]
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# gpus = [0,1]

data_path='/mnt/local/Baselines_Bugs/PatchFinder/data' 
# data_path='/mnt/local/Baselines_Bugs/PatchFinder/data/top100_split' ### 04/10/2023 we reuse it to train new data (colbert)
# data_path='/mnt/local/Baselines_Bugs/ColBERT/data/cve_split/top100_split'
os.makedirs(data_path,exist_ok=True)

# train_filename    = 'train_data_top100.csv'
# validate_filename = 'validate_data_top100.csv'
test_filename     = 'test_data_top100.csv'

### 10/07
train_filename    = 'train_top100.csv'
validate_filename = 'validate_top100.csv'
# test_filename     = 'test_top100.csv'


train_file=os.path.join(data_path, train_filename)
valid_file=os.path.join(data_path, validate_filename)
test_file=os.path.join(data_path, test_filename)
batch_size=128

# save_path='/mnt/local/Baselines_Bugs/PatchFinder/output'
# os.makedirs(save_path,exist_ok=True)

debug=False
device = torch.device("cuda" if torch.cuda.is_available() and not debug else 'cpu')

import pytz
from datetime import datetime

### we define a function to get singapore time
def get_singapore_time():
    singaporeTz = pytz.timezone("Asia/Singapore") 
    timeInSingapore = datetime.now(singaporeTz)
    currentTimeInSinapore = timeInSingapore.strftime("%H:%M:%S")
    # print("Current time in Singapore is: ", currentTimeInSinapore)
    print(currentTimeInSinapore)