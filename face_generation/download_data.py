import helper
import os
from glob import glob
from matplotlib import pyplot


print("hello test!")

data_dir = './data'
helper.download_extract('mnist', data_dir)
helper.download_extract('celeba', data_dir)