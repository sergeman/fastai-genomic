import sys
sys.path.append("..")
from faigen.data.sequence import Dna2VecList,regex_filter
import pandas as pd
import numpy as np
import os
from functools import partial
import configargparse
from pathlib import Path
from Bio.SeqRecord import SeqRecord
import yaml
from pathlib import Path
import os
from shutil import copy
from tqdm import tqdm

def filter_by_count(df:pd.DataFrame, min=1)->pd.DataFrame:
    res=df.copy()
    drop = res.index[res.index.values[np.asarray(res.seq_count.values) < min]]
    res.drop(drop, axis=0,inplace=True)
    return res.reset_index(drop=True)


def filter_by_label(df:pd.DataFrame, word:str)->pd.DataFrame:
    res,mask=df.copy(),[]
    for x in df.label.values: mask.append(False if word in x else True)
    drop = res.index[mask]
    res.drop(drop, axis=0,inplace=True)
    return res.reset_index(drop=True)

def main():
    argp = configargparse.get_argument_parser()
    argp.add_argument('-i', help='input label inventory csv', type=str)
    argp.add_argument('-o', help='output folder', type=str)
    argp.add_argument('-lsi', help='label selector (comma delimited numbers)', type=str)
    argp.add_argument('-lsr', help='regular expression for labeling', type=str)
    argp.add_argument('-rxkeep', help='keep if regular expression found', type=str)
    argp.add_argument('-rxdrop', help='drop if regular expression found', type=str)
    argp.add_argument('-d', help='label delimiter', type=str, default=" ")
    argp.add_argument('-split', help='split by folders, coma delimited string', type=str, default="train,valid,test")
    argp.add_argument('-portions', help='split by folders, coma delimited string', type=str, default="0.7,0.2,0.1")


    args = {k:v for k,v in vars(argp.parse_args()).items()}



    out = Path('/home/serge/database/data/genomes/ncbi-genomes-2019-04-07')
    folders = {
        'train': out / "Bacillus" / "train",
        'valid': out / "Bacillus" / "valid",
        'test': out / "Bacillus" / "test"
    }
    for k in folders:
        if not os.path.exists(folders[k]):
            os.makedirs(folders[k])

    for i in tqdm(range(short_list.shape[0])):
        cnt = short_list.loc[i, "seq_count"]
        train = int(0.75 * cnt)
        valid = cnt - train
        files = short_list.loc[i, "files"]
        for i in range(cnt):
            copy(files[i], folders["train"]) if i < train else copy(files[i], folders["valid"])