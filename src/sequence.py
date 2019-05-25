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
gen_seq_formats = {"fasta": "fasta", "fna": "fasta", "ffn": "fasta", "faa": "fasta", "frn": "fasta",
                   "fastq": "fastq"}


def get_fasta_files(c: PathOrStr, check_ext: bool = True, recurse=False) -> FilePathList:
    "Return list of files in `c` that are fasta data files. `check_ext` will filter to `image_extensions`."
    return get_files(c, extensions=(fasta_extensions if check_ext else None), recurse=recurse)


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

def _genomic_join_texts(texts: Collection[str], mark_fields: bool = False):
    if not isinstance(texts, np.ndarray): texts = np.array(texts)
    if is1d(texts): texts = texts[:, None]
    df = pd.DataFrame({i: texts[:, i] for i in range(texts.shape[1])})
    text_col = f'{BOS} {FLD} {1} ' + df[0].astype(str) if mark_fields else '' + df[0].astype(str)
    for i in range(1, len(df.columns)):
        text_col += (f' {FLD} {i + 1} ' if mark_fields else ' ') + df[i].astype(str)
    return text_col.values


##=====================================
## Processors
##=====================================
class GSFileProcessor(PreProcessor):
    """`PreProcessor` Opens the fasta file listed in item,
    reads fasta and returns sequences with IDs provided by the item.
    """

    def __init__(self, ds: ItemList = None):
        self.ds = ds

    def process_one(self, item) -> Seq:
        content = gen_seq_reader(item['file'])
        for record in content:
            if content[record].id == item['id']:
                return content[record].seq
        return None

    def process(self, items: Collection) -> Collection[Seq]:
        df = pd.DataFrame(data=list(items), columns=['file', 'description', "id", "name"])
        multi_fastas = df.groupby("file").agg({"id": list})
        res = []
        for row in multi_fastas.index.values:
            content = gen_seq_reader(str(row))
            for record in content:
                if content[record].id in multi_fastas.loc[row, 'id']:
                    res.append(content[record].seq)
        self.items = res
        return res

class GSTokenizer(BaseTokenizer):
    def __init__(self, lang='en', ngram=3, skip=0):
        self.lang, self.ngram, self.skip = lang, ngram, skip

    def tokenizer(self, t):
        if self.ngram == 1:
            toks = list(t)
            if self.skip > 0:
                toks = toks[::2] if self.skip == 1 else toks[::self.skip]
        else:
            toks = [t[i:i + self.ngram] for i in range(0, len(t), self.ngram + self.skip)]
        return toks

class GSTokenizeProcessor(PreProcessor):
    "`PreProcessor` that tokenizes the texts in `ds`."

    def __init__(self, ds: ItemList = None, tokenizer: Tokenizer = None, chunksize: int = 10000,
                 mark_fields: bool = False):
        self.tokenizer, self.chunksize, self.mark_fields = ifnone(tokenizer, Tokenizer()), chunksize, mark_fields

    def process_one(self, item):
        return self.tokenizer.tokenizer(item)

    def process(self, ds):
        ds.items = _genomic_join_texts(ds.items, self.mark_fields)
        tokens = []
        for i in range(0, len(ds), self.chunksize):
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
                    include_eos: bool = False,
                    regex:str = "", attr="description", n_cpus: int = None, ngram: int = 8, skip: int = 0, **kwargs):
        "Create a unsupervised learning data bunch from fasta  files in folders."

        path = Path(path).absolute()
        src = GSList.from_folder(path=path, regex=regex, attr=attr)

        tok = Tokenizer(tok_func=partial(GSTokenizer, ngram=ngram, skip=skip), n_cpus=n_cpus)
        GSFileProcessor().process(src)
        GSTokenizeProcessor(tokenizer=tok, chunksize=chunksize, mark_fields=mark_fields).process(src)
        GSNumericalizeProcessor(vocab=vocab, max_vocab=max_vocab, min_freq=min_freq).process(src)
        src = src.split_none()
        src = src.label_empty()
        if test is not None: src.add_test_folder(path / test)
        dl = src.databunch(**kwargs)
        # datasets = cls._init_ds(d1.train_ds, d1.valid_ds, d1.test_ds)
        # val_bs = bs
        # datasets = [
        #     LanguageModelPreLoader(ds, shuffle=(i == 0), bs=(bs if i == 0 else val_bs), bptt=bptt, backwards=False)
        #     for i, ds in enumerate(datasets)]
        # dls = [DataLoader(d, b, shuffle=False) for d, b in zip(datasets, (bs, val_bs, val_bs, val_bs)) if d is not None]
        #
        # return cls(*dls, path=path, collate_fn=collate_fn, no_check=False)

##=====================================
## Item List
##=====================================

class GSList(ItemList):
    "`ItemList` suitable for genomic sequence analysis."
    _bunch, _processor = GSUDataBunch, GSFileProcessor

    @classmethod
    def from_folder(cls, path: PathOrStr = '.', extensions: Collection[str] = None,
                    regex:str="", attr='description', **kwargs) -> ItemList:
        "Get the list of files in `path` that have an image suffix. `recurse` determines if we search subfolders."
        extensions = ifnone(extensions, gen_seq_extensions)
        files = super().from_folder(path=path, extensions=extensions, **kwargs)
        res = []
        for file in files:
            content = gen_seq_reader(file)
            res += [
                {"file": str(file), 'description': content[r].description, 'id': content[r].id, 'name': content[r].name}
                for r in content.keys()]
        return cls(items=list(filter(lambda x: re.compile(regex).search(x[attr]), res) if expr != "" else res, path=path)


if __name__ == '__main__':
    # items = GSList.from_folder("/data/genomes/GenSeq_fastas/valid")
    # # print(items.by_regex('v2_*',"file"))
    # fastas = GSFileProcessor().process(items)
    # # tokenizer = GenSeqTokenizer(ngram=6, skip=6)
    # tokens = [tokenizer.tokenizer(seq) for seq in fastas]

    # tok = Tokenizer(tok_func=partial(GSTokenizer, ngram=8, skip=8), pre_rules=[], post_rules=[], n_cpus=4)
    # GSTokenizeProcessor(tokenizer=tok).process(fastas)
    # print(GenSeqFileProcessor().process_one(items[0]))
    # src = GSList(NumericalizeProcessor( max_vocab=100000, min_freq=3).process(src))
    # print(src)
    bunch = GSUDataBunch.from_folder("/data/genomes/GenSeq_fastas/valid")
    a=10
    print(res)
