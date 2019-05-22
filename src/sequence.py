from fastai import *
from fastai.text import *
from Bio import Seq
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation, CompoundLocation

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
    def process_one(self,item):
        return gen_seq_reader(item) if isinstance(item, Path) else item

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
    def from_folder(cls, path: PathOrStr = '.', extensions: Collection[str] = None, **kwargs) -> ItemList:
        "Get the list of files in `path` that have an image suffix. `recurse` determines if we search subfolders."
        extensions = ifnone(extensions, gen_seq_extensions)
        return super().from_folder(path=path, extensions=extensions, **kwargs)



if __name__ == '__main__':
    import time
    # fn = "/data/genomes/fromKurt/genome_fastas/GCF_000005845.2_ASM584v2_genomic.fna"
    # fn = '/data/genomes/fromKurt/genome_fastas/GCF_000156695.2_ASM15669v2_genomic.fna'
    # files = get_fasta_files('/data/genomes/fromKurt/genome_fastas/')

    # files = get_fasta_files('/data/genomes/LaBrock/20190502_output-20190512T025719Z-003/20190502_output/Basal-1-2016-A1_TAAGGCGA-GCGTAAGA_L008_R1_001.gz/')

    items = GenSeqList.from_folder("/data/genomes/GenSeq_fastas")
    print("ItemList: ", items)
    bunch = FastaDataBunch.from_folder("/data/genomes/GenSeq_fastas")
    print("DataBunch", bunch)

    # data = {}
    # for file in files:
    #     print(f'file={file}')
    #     parser = SeqIO.parse(file, 'fastq')
    #     # res=(rec for rec in parser)
    #     start_time = time.time()
    #     data = list(parser)
    #     print("--- %s seconds ---" % (time.time() - start_time))
    #     print(len(data))
    #     print(data[1:3])

    # for record in parser:
    #     print (record)
    #     data[record.description] = record
    #     print(f'data = id={record.description}, len={len(record.seq)}')
    # print(len(data))
    # print(data.keys())
    # seq = Fasta(fn)
    # print (len(seq))
    # print([s.id for s in seq])
