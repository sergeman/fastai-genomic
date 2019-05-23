from fastai import *
from fastai.text import *
from Bio import Seq
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation, CompoundLocation
import re


# fasta extensions bansed on https://en.wikipedia.org/wiki/FASTA_format
gen_seq_extensions = ['.fasta', '.fastq', '.fna', '.ffn', '.faa', '.frn']
gen_seq_formats = {"fasta": "fasta", "fna":"fasta","ffn":"fasta", "faa":"fasta", "frn":"fasta",
                   "fastq":"fastq"}

def get_fasta_files(c:PathOrStr, check_ext:bool=True, recurse=False)->FilePathList:
    "Return list of files in `c` that are fasta data files. `check_ext` will filter to `image_extensions`."
    return get_files(c, extensions=(fasta_extensions if check_ext else None), recurse=recurse)

def ifnone(a:Any,b:Any)->Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a

def download_fasta(url,dest, timeout=4):
    try: r = download_url(url, dest, overwrite=True, show_progress=False, timeout=timeout)
    except Exception as e: print(f"Error {url} {e}")

def _download_fasta_inner(dest, url, i, timeout=4):
    suffix = re.findall(r'\.\w+?(?=(?:\?|$))', url)
    suffix = suffix[0] if len(suffix)>0  else '.jpg'
    download_fasta(url, dest/f"{i:08d}{suffix}", timeout=timeout)

def download_fastas(urls:Collection[str], dest:PathOrStr, max_files:int=1000, max_workers:int=8, timeout=4):
    "Download fastas listed in text file `urls` to path `dest`, at most `max_pics`"
    urls = open(urls).read().strip().split("\n")[:max_files]
    dest = Path(dest)
    dest.mkdir(exist_ok=True)
    parallel(partial(_download_image_inner, dest, timeout=timeout), urls, max_workers=max_workers)

def gen_seq_reader(fn:PathOrStr):
    "Read the sequences in `fn`."
    ext=str(fn).split(".")[-1]
    return SeqIO.to_dict(SeqIO.parse(fn, gen_seq_formats[ext]))

##=====================================
## ItemBase
##=====================================

class GenomicItemBase(ItemBase):
    pass


class GenSeqFileProcessor(PreProcessor):
    "`PreProcessor` that opens the filenames and read the fastas."
    def process_one(self,item) -> Seq:
        content = gen_seq_reader(item['file'])
        for record in content:
            if content[record].id == item['id']:
                return content[record].seq
        return None

    def process(self, items:Collection) -> Collection[Seq]:
        df = pd.DataFrame(data=list(items), columns=['file', 'description', "id", "name"])
        multi_fastas = df.groupby("file").agg({"id": list})
        print(multi_fastas.head())
        res = []
        for row in multi_fastas.index.values:
            content = gen_seq_reader(str(row))
            for record in content:
                if content[record].id in multi_fastas.loc[row,'id']:
                    res.append(content[record].seq)
        return res



##=====================================
## DataBunch
##=====================================


class FastaDataBunch(DataBunch):
    "DataBunch suitable for generic sequence processing."


    @classmethod
    def from_folder(cls, path:PathOrStr, train:str='train', valid:str='valid', test:Optional[str]=None,
                    classes:Collection[Any]=None, tokenizer:Tokenizer=None, vocab:Vocab=None, chunksize:int=10000, max_vocab:int=60000,
                    min_freq:int=2, mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False, **kwargs):
        "Create a `FastaDataBunch` from text files in folders."
        path = Path(path).absolute()
        processor = [GenSeqFileProcessor()]
                    # + _get_processor(tokenizer=tokenizer, vocab=vocab, chunksize=chunksize, max_vocab=max_vocab,
                    # min_freq=min_freq, mark_fields=mark_fields, include_bos=include_bos, include_eos=include_eos)
        src = (GenSeqList.from_folder(path, processor=processor)
                       .split_by_folder(train=train, valid=valid))
        src = src.label_for_lm() if cls==TextLMDataBunch else src.label_from_folder(classes=classes)
        if test is not None: src.add_test_folder(path/test)
        return src.databunch(**kwargs)


##=====================================
## Item List
##=====================================

class GenSeqList(ItemList):
    "`ItemList` suitable for genomic sequence analysis."
    _bunch, _processor = FastaDataBunch, GenSeqFileProcessor

    @classmethod
    def from_folder(cls, path: PathOrStr = '.', extensions: Collection[str] = None, **kwargs) -> 'GenSeqList':
        "Get the list of files in `path` that have an image suffix. `recurse` determines if we search subfolders."
        extensions = ifnone(extensions, gen_seq_extensions)
        files=super().from_folder(path=path, extensions=extensions, **kwargs)
        res = []
        for file in files:
            content = gen_seq_reader(file)
            res += [{"file":str(file),'description':content[r].description, 'id':content[r].id, 'name':content[r].name}
                    for r in content.keys()]
        return cls(res)

    def by_regex(self, expr:str, attr='description') -> 'GenSeqList':
        """Select sequences matching regular expression over metadata attribute.
        Available attributes ```file, id,name, description```
        """
        p = re.compile(expr)
        return GenSeqList(list(filter(lambda x: p.search(x[attr]), self)))

if __name__ == '__main__':

    items = GenSeqList.from_folder("/data/genomes/GenSeq_fastas/valid")
    print(items.by_regex('v2_*',"file"))
    GenSeqFileProcessor().process(items)
    # print(GenSeqFileProcessor().process_one(items[0]))


    # bunch = FastaDataBunch.from_folder("/data/genomes/GenSeq_fastas")
    # print("DataBunch", bunch)
