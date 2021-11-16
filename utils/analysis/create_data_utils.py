import os
import shutil
import itertools
import numpy as np
import pandas as pd
from typing import Union
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from param import *

def erase_files(folder:str, verbose=True):
    """
    Erase files from a folder
    :param folder: name of the folder (e.g. '01_image')
    :param verbose: specifies whether to print descriptions (default=True)
    :return: empty
    """
    folder_path = str(Path(img_path, folder))
    for gender in [men_dir, women_dir, all_dir]:
        gender_root = str(Path(folder_path, gender))
        if os.path.exists(gender_root):
            if verbose:
                print(f"Erase directory tree from {gender_root}")
            shutil.rmtree(gender_root)
        else:
            if verbose:
                print(f"No directory tree to erase")


def train_val_split(X:str,y:str, gender:str, train_size=train_size, oversample=False):
    """
    Splits data into training and validation sets.
    :param X: input
    :param y: target
    :param gender: gender
    :param train_size: specifies the training/val split
    :return: a dictionary with data and a corresponding dataframe
    """
    df = pd.read_csv(df_path, index_col=[0])

    if gender == 'men':
        df = df.loc[df['gender'] == 1]
    elif gender == 'women':
        df = df.loc[df['gender'] == 0]

    df = shuffle(df)

    X_train, X_val, y_train, y_val = train_test_split(df[X], df[y], train_size=train_size, stratify=df[y])

    if oversample == True:
        oversampler = RandomOverSampler(sampling_strategy=.75)
        X_train = np.reshape(np.array(X_train), (len(X_train), 1))
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

        unique, count = np.unique(X_train, return_counts=True)
        duplicates = unique[count > 1]
        count = count[count > 1]

        for idx, img in enumerate(duplicates):
            file_df = df.loc[df[X] == img]
            for dup in range(0, count[idx] - 1):
                df = df.append(file_df, ignore_index=False)
        X_train = itertools.chain.from_iterable(X_train.tolist())

    return {'train': {'image': list(X_train)},
             'val'  : {'image': list(X_val)},
             'dataframe': df}


def copy_files(X:str, y:str, folder:str, dictionary:dict):
    """
    Copies files to the specified folder.
    :param X: input
    :param y: target
    :param folder: name of the folder (e.g. '01_image')
    :param dictionary: dictionary with dirs and dataframes
    :return: empty
    """
    folder_path = str(Path(img_path, folder))
    gender_root = dictionary['dir']
    df = dictionary['data']['dataframe']
    del dictionary['data']['dataframe']
    for key in dictionary['data'].keys():
        data_root = str(Path(gender_root, key))
        for idx, file in enumerate(dictionary['data'][key]['image']):
            dups = len(df.loc[df[X] == file, y].values)
            elected = df.loc[df[X] == file, y].values[0]
            if elected == 1:
                elected_root = str(Path(data_root, label1_dir))
                fpath = str(Path(folder_path, file))
                if os.path.exists(elected_root):
                    for img in range(dups):
                        new = file[:-4] + f'_{img}' + file[-4:]
                        shutil.copy(fpath, str(Path(elected_root, new)))
            else:
                elected_root = str(Path(data_root, label0_dir))
                fpath = str(Path(folder_path, file))
                if os.path.exists(elected_root):
                    for img in range(dups):
                        new = file[:-4] + f'_{img}' + file[-4:]
                        shutil.copy(fpath, str(Path(elected_root, new)))

def make_dirs(X:str, y:str, gender: Union[list, str], folder:str, mkdir=True, copy=True, oversample=False,verbose=True):
    """
    Creates directories and copies images to the relevant folders.
    :param X: input
    :param y: target
    :param gender: gender
    :param folder: name of the folder (e.g. '01_image')
    :param mkdir: specifies whether the function makes folder (default=True)
    :param copy: specifies whether the function copies the images (default=True).
    :param test_set: specifies whether we use a test set or not (default=None
    :param verbose: specifies whether to print descriptions (default=True)
    :return: a dictionary with dirs and data.
    """
    folder_path = str(Path(img_path, folder))
    data_dict = dict()

    if type(gender)==str:
        gender_list = [gender]
    else:
        gender_list = gender

    for g in gender_list:
        gender_root = str(Path(folder_path, g))
        data_dict[g] = {}
        data_dict[g]['dir'] = gender_root
        if os.path.exists(gender_root):
            print(f"Directory {gender_root} already exists")
            if not mkdir:
                if verbose:
                    print(f"Remove directory {gender_root}")
                shutil.rmtree(gender_root)
        else:
            for d in [train_dir, val_dir]:
                for label in [label0_dir, label1_dir]:
                    root = str(Path(folder_path, g, d, label))
                    if mkdir:
                        if verbose:
                            print(f"Make directory {root}")
                        os.makedirs(root)
                    else:
                        if os.path.exists(root):
                            if verbose:
                                print(f"Remove directory {root}")
                            shutil.rmtree(root)
                        else:
                            if verbose:
                                print(f"Can not remove {root} because it does not exists")
        data = train_val_split(X=X, y=y, gender=g, oversample=oversample)
        data_dict[g]['data'] = data

        if copy and mkdir:
            if verbose:
                print(f"Copy files to {g} directory")
            copy_files(X=X, y=y, folder=folder, dictionary=data_dict[g])
    return data_dict

