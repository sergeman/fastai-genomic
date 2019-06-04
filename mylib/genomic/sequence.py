from fastai import *
from fastai.text import *
from Bio import Seq
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation, CompoundLocation
import re
from dna2vec.multi_k_model import MultiKModel
import random


# fasta extensions bansed on https://en.wikipedia.org/wiki/FASTA_format
gen_seq_extensions = ['.fasta', '.fastq', '.fna', '.ffn', '.faa', '.frn']
gen_seq_formats = {"fasta": "fasta", "fna": "fasta", "ffn": "fasta", "faa": "fasta", "frn": "fasta",
                   "fastq": "fastq"}


def get_fasta_files(c: PathOrStr, check_ext: bool = True, recurse=False) -> FilePathList:
    "Return list of files in `c` that are fasta data files. `check_ext` will filter to `image_extensions`."
    return get_files(c, extensions=(gen_seq_extensions if check_ext else None), recurse=recurse)


def ifnone(a: Any, b: Any) -> Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


def download_fasta(url, dest, timeout=4):
    try:
        r = download_url(url, dest, overwrite=True, show_progress=False, timeout=timeout)
    except Exception as e:
        print(f"Error {url} {e}")


def _download_fasta_inner(dest, url, i, timeout=4):
    suffix = re.findall(r'\.\w+?(?=(?:\?|$))', url)
    suffix = suffix[0] if len(suffix) > 0 else '.jpg'
    download_fasta(url, dest / f"{i:08d}{suffix}", timeout=timeout)


def download_fastas(urls: Collection[str], dest: PathOrStr, max_files: int = 1000, max_workers: int = 8, timeout=4):
    "Download fastas listed in text file `urls` to path `dest`, at most `max_pics`"
    urls = open(urls).read().strip().split("\n")[:max_files]
    dest = Path(dest)
    dest.mkdir(exist_ok=True)
    parallel(partial(_download_image_inner, dest, timeout=timeout), urls, max_workers=max_workers)


def gen_seq_reader(fn: PathOrStr):
    "Read the sequences in `fn`."
    ext = str(fn).split(".")[-1]
    return SeqIO.to_dict(SeqIO.parse(fn, gen_seq_formats[ext]))



##=====================================
## Processors
##=====================================

class GSFileProcessor(PreProcessor):
    """`PreProcessor` Opens the fasta file listed in item,
    reads fasta and returns sequences with IDs provided by the item.
    """

    def __init__(self, ds: ItemList = None, filters=None):
        self.ds,self.filters = ds, filters

    def process_one(self, item) -> Seq:
        content = gen_seq_reader(item['file'])
        for record in content:
            if content[record].id == item['id']:
                return content[record].seq
        return None

    def process(self, ds: Collection) -> Collection[Seq]:
        df = pd.DataFrame(data=list(ds), columns=['file', 'description', "id", "name"])
        multi_fastas = df.groupby("file").agg({"id": list})
        res = []
        for row in multi_fastas.index.values:
            content = gen_seq_reader(str(row))
            for record in content:
                if content[record].id in multi_fastas.loc[row, 'id']:
                    res.append(content[record].seq)
        ds.items = apply_filters(res,self.filters)
        # return ds.items

class GSTokenizer():
    def __init__(self, ngram=8, skip=0, n_cpus=1):
        self.ngram, self.skip,self.n_cpus = ngram, skip,n_cpus

    def tokenizer(self, t):
        if self.ngram == 1:
            toks = list(t)
            if self.skip > 0:
                toks = toks[::2] if self.skip == 1 else toks[::self.skip]
        else:
            toks = [t[i:i + self.ngram] for i in range(0, len(t), self.ngram + self.skip) if i+self.ngram < len(t)]
        return toks

    def _process_all_1(self, texts:Collection[str]) -> List[List[str]]:
        "Process a list of `texts` in one process."
        return [self.tokenizer(str(t)) for t in texts]

    def process_all(self, texts:Collection[str]) -> List[List[str]]:
        "Process a list of `texts`."
        if self.n_cpus <= 1: return self._process_all_1(texts)
        with ProcessPoolExecutor(self.n_cpus) as e:
            return sum(e.map(self._process_all_1, partition_by_cores(texts, self.n_cpus)), [])



class GSTokenizeProcessor(PreProcessor):
    "`PreProcessor` that tokenizes the texts in `ds`."

    def __init__(self, ds: ItemList = None, tokenizer: Tokenizer = None, ngram:int=8, skip:int=0, chunksize: int = 10000,
                 mark_fields: bool = False):
        self.tokenizer, self.chunksize, self.mark_fields = ifnone(tokenizer, GSTokenizer(ngram=ngram, skip=skip)), chunksize, mark_fields

    def process_one(self, item):
        return self.tokenizer.tokenizer(item)

    def process(self, ds):
        tokens = []
        if len(ds.items) < self.chunksize: ds.items = self.tokenizer._process_all_1(ds.items); return
        for i in range(0, len(ds.items), self.chunksize):
            tokens += self.tokenizer.process_all(ds.items[i:i + self.chunksize])
        ds.items = tokens

class GSVocab(Vocab):
    def __init__(self, itos):
        self.itos = itos
        self.stoi = collections.defaultdict(int, {v: k for k, v in enumerate(self.itos)})

    @classmethod
    def create(cls, tokens, max_vocab, min_freq):
        freq = Counter(p for o in tokens for p in o)
        itos = [o for o, c in freq.most_common(max_vocab) if c >= min_freq]
        itos.insert(0, 'pad')
        return cls(itos)

class GSNumericalizeProcessor(PreProcessor):
    "`PreProcessor` that numericalizes the tokens in `ds`."

    def __init__(self, ds: ItemList = None, vocab: Vocab = None, max_vocab: int = 80000, min_freq: int = 3):
        vocab = ifnone(vocab, ds.vocab if ds is not None else None)
        self.vocab, self.max_vocab, self.min_freq = vocab, max_vocab, min_freq

    def process_one(self, item): return np.array(self.vocab.numericalize(item), dtype=np.int64)

    def process(self, ds):
        if self.vocab is None: self.vocab = GSVocab.create(ds.items, self.max_vocab, self.min_freq)
        ds.vocab = self.vocab
        super().process(ds)


class Dna2VecProcessor(PreProcessor):
    "`PreProcessor` that tokenizes the texts in `ds`."

    def __init__(self, ds: ItemList = None, agg:Callable=sum,
                 filepath:str='~/.fastai/models/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'):
        self.agg, self.embedding = agg, MultiKModel(filepath)

    def process_one(self, item):
        return embedding.vector(item)

    def process(self, ds):
        res=[]
        for item in ds.items:
            bases = list(filter(lambda x: set(x) == set('ATGC'), item))
            vectors = self.embedding.data[ds.ngram].model[bases] if len(bases) > 0 else np.asarray([[0.]*100,[0.]*100])

            aggregate=vectors if self.agg is None else self.agg(vectors)
            res.append(aggregate)
        ds.items = res



##=====================================
## DataBunch
##=====================================


class GSUDataBunch(DataBunch):
    "DataBunch suitable for unsupervised learning from fasta data"

    @classmethod
    def from_folder(cls, path: PathOrStr, train: str = 'train', valid: str = 'valid', test: Optional[str] = None,
                    classes: Collection[Any] = None, tokenizer: Tokenizer = None, vocab: Vocab = None,
                    chunksize: int = 10000,
                    max_vocab: int = 70000, min_freq: int = 2, mark_fields: bool = False, include_bos: bool = True,
                    filters:Collection[Callable] = None,
                    include_eos: bool = False, n_cpus: int = None, ngram: int = 8, skip: int = 0, **kwargs):
        "Create a unsupervised learning data bunch from fasta  files in folders."

        path = Path(path).absolute()
        tok = Tokenizer(tok_func=partial(GSTokenizer, ngram=ngram, skip=skip), n_cpus=n_cpus)
        processor = [GSFileProcessor(),
                      GSTokenizeProcessor(tokenizer=tok, chunksize=chunksize, mark_fields=mark_fields),
                      GSNumericalizeProcessor(vocab=vocab, max_vocab=max_vocab, min_freq=min_freq)]
        src = ItemLists(path, NumericalizedGSList.from_folder(path=path, filters = filters, processor=processor),
                        ItemList(items=[],ignore_empty=True))
        src=src.label_empty()
        if test is not None: src.add_test_folder(path / test)
        return src.databunch(**kwargs)

class Dna2VecDataBunch(DataBunch):
    "DataBunch of tokenized genomic sequences for use with dna2vec embedding"

    @classmethod
    def from_folder(cls, path: PathOrStr, train: str = 'train', valid: str = 'valid', test: Optional[str] = None,
                    classes: Collection[Any] = None, tokenizer: Tokenizer = None,
                    chunksize: int = 1000, mark_fields: bool = False,
                    filters:Collection[Callable] = None, n_cpus: int = 1,
                    ngram: int = 8, skip: int = 0, agg:Callable=None, **kwargs):
        "Create a unsupervised learning data bunch from fasta  files in folders."

        path = Path(path).absolute()
        tok = GSTokenizer(ngram=ngram, skip=skip, n_cpus=n_cpus)
        processor = [GSFileProcessor(),
                     GSTokenizeProcessor(tokenizer=tok, chunksize=chunksize, mark_fields=mark_fields),
                     Dna2VecProcessor(agg=agg)]
        src = ItemLists(path, Dna2VecList.from_folder(path=path, filters=filters, processor=processor),
                        ItemList(items=[],ignore_empty=True))
        src=src.label_empty()
        if test is not None: src.add_test_folder(path / test)
        return src.databunch(**kwargs)


##=====================================
## Item List
##=====================================


def regex_filter(items:Collection, rx:str= "", target:str= "description", search=True, keep=True) -> Collection:
    if rx== "": return items
    rx = re.compile(rx)
    if search: return list(filter(lambda x: rx.search(x[target]) if keep else  not rx.search(x[target]), items))
    return list(filter(lambda x: rx.match(x[target]) if keep else not rx.match(x[target]), items))


def id_filter(items:Collection, ids:Collection)->Collection:
    return [i for i in list(items) if i['id'] in ids]

def name_filter(items:Collection, names:Collection)->Collection:
    return [i for i in list(items) if i['name'] in names]

def count_filter(items:Collection, num_fastas:tuple=(1, None), keep:int=None, sample:str= "first") -> Collection:
    df = pd.DataFrame(data=list(items), columns=['file', 'description', "id", "name"])
    df_agg = df.groupby("file").agg({"id": list})
    selected_ids = []
    for file in iter(df_agg.index):
        ids = df_agg.loc[file,"id"]
        if len(ids) < num_fastas[0]: continue
        if num_fastas[1] is not None and len(ids) > num_fastas[1]: continue
        if keep is None:
            selected_ids += ids
        else:
            selected_ids += ids[:min([keep, len(ids)])] if sample == "first" else ids[-min([keep, len(ids)]):]
    res= id_filter(items=items, ids=selected_ids)
    return res

def total_count_filter(items:Collection, parser:Callable,num_fastas:tuple=(1, None), keep:int=None, sample:str= "first") -> Collection:

    df = pd.DataFrame(data=list(items), columns=['file', 'description', "id", "name"])
    df_agg = df.groupby("file").agg({"id": list})
    selected_ids = []
    for file in iter(df_agg.index):
        ids = df_agg.loc[file,"id"]
        if len(ids) < num_fastas[0]: continue
        if num_fastas[1] is not None and len(ids) > num_fastas[1]: continue
        if keep is None:
            selected_ids += ids
        else:
            selected_ids += ids[:min([keep, len(ids)])] if sample == "first" else ids[-min([keep, len(ids)]):]
    res= id_filter(items=items, ids=selected_ids)
    return res


def apply_filters(dicts:Collection, filters:Union[Callable, Collection[Callable]]=None) -> Collection:
    if filters is None: return dicts
    if callable(filters): return filters(items=dicts)
    for f in filters: dicts = f(items=dicts)
    return dicts





class NumericalizedGSList(ItemList):
    "`ItemList`of numericalised genomic sequences."
    _bunch, _processor = GSUDataBunch, [GSFileProcessor, GSTokenizeProcessor, GSNumericalizeProcessor]

    def __init__(self, items:Iterator, vocab:Vocab=None, pad_idx:int=1, **kwargs):
        super().__init__(items, **kwargs)
        self.vocab,self.pad_idx = vocab,pad_idx
        self.copy_new += ['vocab', 'pad_idx']


    @classmethod
    def from_folder(cls, path: PathOrStr = '.', extensions: Collection[str] = None,
                    filters:Union[Callable, Collection[Callable]]=None, vocab:GSVocab=None, **kwargs) -> ItemList:
        "Get the list of files in `path` that have an image suffix. `recurse` determines if we search subfolders."
        extensions = ifnone(extensions, gen_seq_extensions)
        this = super().from_folder(path=path, extensions=extensions, **kwargs)
        return cls(items=fasta_content(this,filters), path=path, vocab=vocab, **kwargs)


class Dna2VecList(ItemList):
    "`ItemList` of Kmer tokens vectorized by dna2vec embedding"
    _bunch, _processor = Dna2VecDataBunch, [GSFileProcessor, GSTokenizeProcessor,Dna2VecProcessor]

    def __init__(self, items:Iterator, path, ngram:int=8, agg:Callable=None, n_cpus=7, **kwargs):
        super().__init__(items, path, **kwargs)
        self.ngram,self.agg,self.n_cpus = ngram,agg,n_cpus
        self.descriptions=self.ids=self.names=self.files=None,None,None,None

    def fasta_content(self, filters):
        dicts = []
        for file in self.items:
            content = gen_seq_reader(file)
            dicts += [
                {"file": str(file), 'description': content[r].description, 'id': content[r].id, 'name': content[r].name}
                for r in content.keys()]
        self.items = apply_filters(dicts, filters)
        self.descriptions = [item['description'] for item in list(self.items)]
        self.ids = [item['id'] for item in list(self.items)]
        self.names = [item['name'] for item in list(self.items)]
        self.files = [item['file'] for item in list(self.items)]
        return self

    @classmethod
    def from_folder(cls, path: PathOrStr = '.', extensions: Collection[str] = None,
                    filters:Collection[Callable]=None, ngram:int=8, n_cpus=1, agg:Callable=None, **kwargs) -> ItemList:
        "Get the list of files in `path` that have an image suffix. `recurse` determines if we search subfolders."
        extensions = ifnone(extensions, gen_seq_extensions)
        this = super().from_folder(path=path, extensions=extensions, **kwargs)
        return this.fasta_content(filters)

    def store_by_genus(self,path):
        pass

if __name__ == '__main__':

    # gsu_bunch = GSUDataBunch.from_folder("/data/genomes/GenSeq_fastas/valid")
    bunch = Dna2VecDataBunch.from_folder("/data/genomes/GenSeq_fastas/test",
                                         filters=[partial(count_filter, keep=3, sample="last")],
                                         n_cpus=7,agg=partial(np.mean, axis=0))
    print(bunch)

