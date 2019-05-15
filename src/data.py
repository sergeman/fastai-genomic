from fastai import *
from fastai.text import *
from Bio import Seq
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation, CompoundLocation


class GenomicBase(ItemBase):
    def __init__(self, data:Any): self.data=self.obj=data

class FASTA(GenomicBase):
    def __init__(self, data: Collection[SeqRecord]):
        self.data = self.obj = data

class FastaProcessor(PreProcessor):
    pass

class FastaDataBunch(DataBunch):
    pass

class FastaItemList(ItemList):
    "`ItemList` suitable for genomic sequence analysis."
    _bunch, _processor = FastaDataBunch, FastaProcessor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def open(self, fn):
        "Open Fasta in `fn`, subclass and overwrite for custom behavior."
        return SeqIO.parse(fn, 'fasta')

    def get(self, i):
        fn = super().get(i)
        res = self.open(fn)

    @classmethod
    def from_folder(cls, path: PathOrStr = '.', extensions: Collection[str] = None, **kwargs) -> ItemList:
        "Get the list of files in `path` that have an image suffix. `recurse` determines if we search subfolders."
        # fasta extensions bansed on https://en.wikipedia.org/wiki/FASTA_format
        extensions = ['fasta', 'fna', 'ffn', 'faa', 'frn']
        return super().from_folder(path=path, extensions=extensions, **kwargs)


if __name__ == '__main__':

    genome = SeqIO.parse("/data/genomes/fromKurt/genome_fastas/GCF_000005845.2_ASM584v2_genomic.fna", 'fasta')
