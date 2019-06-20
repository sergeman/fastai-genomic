import os
import configargparse
from Bio.SeqRecord import SeqRecord

from faigen.data.sequence import Dna2VecList, GSFileProcessor


def preprocess_for_dna2vec_training(out_path, **kwargs):
    p = Path(out_path) if isinstance(out_path, str) else out_path
    data = Dna2VecList.from_folder(**kwargs)
    GSFileProcessor().process(data)
    if not os.path.exists(str(p)):
        os.makedirs(str(p))
    for i, seq in enumerate(iter(data.items)):
        record = SeqRecord(seq, id=data.ids[i], name=data.names[i], description=data.descriptions[i])
        with open(p / f"{data.ids[i]}.fasta", "w") as output:
            output.write(record.format("fasta"))


def main():
    argp = configargparse.get_argument_parser()
    argp.add_argument('-i', help='input folder with Fasta files', type=str, default='.')
    argp.add_argument('-o', help='output folder', type=str, default="../d2v_dataset")
    args = {k:v for k,v in vars(argp.parse_args()).items()}


    preprocess_for_dna2vec_training(path= args["i"], out_path=args["o"])

if __name__ == '__main__':
    main()
