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

def filter_by_count(df:pd.DataFrame, min=1)->pd.DataFrame:
    res=df.copy()
    drop = res.index[res.index.values[np.asarray(res.seq_count.values) < min]]
    return res.drop(drop, axis=0)


def filter_by_label(df:pd.DataFrame, word:str)->pd.DataFrame:
    res,mask=df.copy(),[]
    for x in df.label.values: mask.append(False if word in x else True)
    drop = res.index[mask]
    return res.drop(drop, axis=0)


def main():
    argp = configargparse.get_argument_parser()
    argp.add_argument('-i', help='input folder with Fasta files', type=str, default='/data/genomes/GenSeq_fastas')
    argp.add_argument('-o', help='output file name', type=str)
    argp.add_argument('-g', choices=['folder', 'file'], help='granularity', type=str, default="file")
    argp.add_argument('-l', choices=['description', 'id', 'file'], help='source of labels', type=str, default="description")
    argp.add_argument('-lsi', help='label selector (comma delimited numbers)', type=str)
    argp.add_argument('-lsr', help='regular expression for labeling', type=str)
    argp.add_argument('-rxkeep', help='keep if regular expression found', type=str)
    argp.add_argument('-rxdrop', help='drop if regular expression found', type=str)
    argp.add_argument('-d', help='label delimiter', type=str, default=" ")
    argp.add_argument('-split', help='split by folders, coma delimited string', type=str, default="train,valid,test")
    argp.add_argument('-portions', help='split by folders, coma delimited string', type=str, default="0.7,0.2,0.1")


    args = {k:v for k,v in vars(argp.parse_args()).items()}
    input= Path(args["i"]) if args["i"] is not None else Path(".")
    filters=[]
    if args["rxkeep"] is not None: filters.append(partial(regex_filter, rx=args["rxkeep"]))
    if args["rxdrop"] is not None: filters.append(partial(regex_filter, rx=args["rxdrop"], keep=False))

    all_fastas = Dna2VecList.from_folder(input, filters=filters if len(filters) > 0 else None ).items

    output = input / "inventory"
    print("Creating Inventory in", str(output))

    if not os.path.exists(output):
        os.makedirs(output)

    fn = "sequences" if args["o"] is None else args["o"]

    with open(output / f'{fn}_inventory.yml', 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)


    inventory = pd.DataFrame(data=all_fastas)
    if args['lsi'] is not None:
        lsi = [int(x) for x in args["lsi"].split(",")]
        label_source = inventory.loc[:,args['l']].values
        tokens = [np.asarray(x.split(args["d"])) for x in list(label_source)]
        inventory["label"] = [" ".join(t[lsi]) for t in tokens]

    inventory.to_csv(Path(output / f"{fn}.csv"))

    if "label" not in inventory.columns.values: return

    if args["g"] == "file":
        files_df = inventory.groupby(["file"]).agg({"id": ['count', list],"label":set,
                                                    "len": [list, min, max, np.mean, np.std], "description": list})
        files_df.columns=["seq_count", "id", "label", 'len', 'min' , 'max', 'mean', 'std', "description"]
        files_df["label"] = [list(x)[0] for x in list(files_df.label.values)]

    files_df.to_csv( output / f"{fn}_by_file.csv")


    label_df = inventory.groupby("label").agg({ "id": ["count", list], "len": [list, min, max, np.median],"file": list})
    label_df.columns = [ "seq_count", 'id', "lengths", "min", "max", "median","files"]
    label_df.files = [list(set(x)) for x in list(label_df.files.values)]
    label_df["file_count"] = [len(x) for x in list(label_df.files.values)]

    label_df.to_csv(output / f"{fn}_by_label.csv")
    label_df.to_pickle( output / f"{fn}_by_label.pkl")



if __name__ == '__main__':
    main()
