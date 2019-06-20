from fastai import *
from fastai.text import *
from Bio import Seq
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import re
from dna2vec.multi_k_model import MultiKModel
import random
from gensim.models import Word2Vec
from pathlib import Path
import os
from tqdm import tqdm
from  torch import tensor

# fasta extensions bansed on https://en.wikipedia.org/wiki/FASTA_format
gen_seq_extensions = ['.fasta', '.fastq', '.fna', '.ffn', '.faa', '.frn','.fa']
gen_seq_formats = {"fasta": "fasta", "fna": "fasta", "ffn": "fasta", "faa": "fasta", "frn": "fasta","fa":"fasta",
                   "fastq": "fastq"}

def ifnone(a: Any, b: Any) -> Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


def gen_seq_reader(fn: PathOrStr):
    "Read the sequences in `fn`."
    ext = str(fn).split(".")[-1]
    return SeqIO.to_dict(SeqIO.parse(fn, gen_seq_formats[ext]))

def seq_record(fn: PathOrStr, record_id:str):
    content = gen_seq_reader(fn)
    for record in content:
        if content[record].id == record_id:
            return content[record].seq
    return None


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
        return seq_record(item["file"], item["id"])

    def process(self, ds: Collection) -> Collection[Seq]:
        df = pd.DataFrame(data=list(ds.items), columns=['file', 'description', "id", "name"])
        multi_fastas = df.groupby("file").agg({"id": list})
        print ("Reading sequences")
        res = []
        for row in tqdm(multi_fastas.index.values):
            content = gen_seq_reader(str(row))
            for record in content:
                if content[record].id in multi_fastas.loc[row, 'id']:
                    res.append(content[record].seq)
        ds.items = apply_filters(res,self.filters)
        ds.state = "sequence"

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
            res = sum(e.map(self._process_all_1,
                             partition_by_cores(texts, self.n_cpus)), [])
        return res


class GSTokenizeProcessor(PreProcessor):
    "`PreProcessor` that tokenizes the texts in `ds`."

    def __init__(self, ds: ItemList = None, tokenizer: Tokenizer = None, ngram:int=8, skip:int=0, chunksize: int = 10000,
                 mark_fields: bool = False):
        self.tokenizer, self.chunksize, self.mark_fields = ifnone(tokenizer, GSTokenizer(ngram=ngram, skip=skip)), chunksize, mark_fields

    def process_one(self, sequence):
        return self.tokenizer.tokenizer(str(sequence))

    def process(self, ds):
        tokens = []
        print("Tokenizing")
        # if len(ds.items) < self.chunksize: ds.items = self.tokenizer._process_all_1(ds.items); return
        for i in range(0, len(ds.items), self.chunksize):
            advance = min((len(ds.items) - i * self.chunksize), self.chunksize )
            tokens += self.tokenizer.process_all(ds.items[i:i + advance])
        ds.items = tokens
        ds.state = "tokens"

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
        ds.state="numericalized"


class Dna2VecProcessor(PreProcessor):
    "`PreProcessor` that tokenizes the texts in `ds`."


    def __init__(self, ds: ItemList = None, agg:Callable=sum, emb=None, n_cpu=7):
        self.agg, self.n_cpu = agg, n_cpu
        self.emb = None if emb is None else emb if isinstance(emb, Word2Vec) else Word2Vec.load_word2vec_format(emb)


    def process_one(self, tokens):
        if self.emb is None: raise ValueError("Provide path to embedding or Word2Vec object using  ```emb``` instance variable ")
        tokens= list(filter(lambda x: set(x) == set('ATGC'), tokens))
        vectors = np.asarray([[0.] * 100, [0.] * 100])
        while len(tokens) > 0:
            try:
                vectors = self.emb[tokens]
                break
            except KeyError as e:
                tokens.remove(e.args[0])  # remove k-mer absent in the embedding
        return vectors if self.agg is None else self.agg(vectors)

    def _process_all_1(self, tokens:Collection[str]) -> List[List[str]]:
        return [self.process_one(t) for t in tokens]

    def process(self, ds):
        self.emb = ds.emb if (hasattr(ds, "emb") and ds.emb is not None) else self.emb
        res =[]

        print("Vectorizing")
        with ProcessPoolExecutor(self.n_cpu) as e:
            res = sum(e.map(self._process_all_1,
                             partition_by_cores(ds.items, self.n_cpu)), [])
        ds.items = res
        print (len(res))
        ds.state = "vector"



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
                    filters:Collection[Callable] = None, labeler:Callable=None, n_cpus: int = 1,
                    ngram: int = 8, skip: int = 0, agg:Callable=None, emb = None, **kwargs):

        path = Path(path).absolute()
        tok = ifnone(tokenizer, GSTokenizer(ngram=ngram, skip=skip, n_cpus=n_cpus))
        processors = [GSFileProcessor(),
                     GSTokenizeProcessor(tokenizer=tok, chunksize=chunksize, mark_fields=mark_fields),
                     Dna2VecProcessor(emb=emb, agg=agg)]
        train_items = Dna2VecList.from_folder(path=path / train, filters=filters, processor=processors)
        valid_items = Dna2VecList.from_folder(path=path / valid, filters=filters, processor=processors)
        src = ItemLists(path, train_items, valid_items)
        tl,cl = train_items.label_from_description(labeler)
        vl,_ = valid_items.label_from_description(labeler, labels=cl)

        src=src.label_from_lists(train_labels=tl, valid_labels=vl,label_cls=CategoryList, classes = cl)
        if test is not None: src.add_test_folder(path / test)
        return src.databunch(**kwargs)



def regex_filter(items:Collection, rx:str= "", target:str= "description", search=True, keep:bool=True) -> Collection:
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

def seq_len_filter(items:Collection, len:tuple=(1, None), keep:bool=True) -> Collection:
    """filters sequence by length. ```len``` tuple is (min,max) values for filtering, ```keep``` controls """
    selected_ids=[i["id"] for i in items if (i["len"] >= len[0] and (len[1] is None or i["len"] < len[1])) == keep]
    res= id_filter(items=items, ids=selected_ids)
    return res


def total_count_filter(items:Collection, parser:Callable,num_fastas:tuple=(1, None), balance:bool=True) -> Collection:
    """Counts items for category extracted by parser.
    Subsamples overrepresented categories to match max amount of samples in the least represented category  """
    pass


def describe(items:Collection) -> dict:
    """compute statistics for items in the list"""
    pass


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

    def __init__(self, items:Iterator, path, ngram:int=8, n_cpus:int=7,
                 emb:Union[Path,str,Word2Vec]=None,
                 agg:Callable=partial(np.mean, axis=0), #mean values of dna2vec embedding vectors for all k-mers in genome
                **kwargs):
        super().__init__(items, path, **kwargs)
        self.ngram,self.agg,self.n_cpus = ngram,agg,n_cpus
        self.emb = emb if isinstance(emb, Word2Vec) else Word2Vec.load_word2vec_format(emb) if emb is not None else None
        self.descriptions, self.ids, self.names, self.files, self.lengths= None, None, None, None, None
        self.state = "initial"

    def get_metadata(self, filters):
        dicts = []
        print ("Collecting sequence metadata")
        for file in tqdm(self.items):
            content = gen_seq_reader(file)
            dicts += [
                {"file": str(file),
                 'description': content[r].description,
                 'id': content[r].id,
                 'name': content[r].name,
                 "len":len(content[r].seq)}
                for r in content.keys()]
        self.items = apply_filters(dicts, filters)
        self.descriptions = [item['description'] for item in list(self.items)]
        self.ids = [item['id'] for item in list(self.items)]
        self.names = [item['name'] for item in list(self.items)]
        self.files = [item['file'] for item in list(self.items)]
        self.lengths = [item["len"] for item in list(self.items)]
        return self

    @property
    def c(self):
        return len(self.items[0])

    def get(self, i) ->Any:
        return self.items[i]

    def process_one(self, i):
        return self.items[i]

    def analyze_pred(self, pred):

        _, index = ensor.max()
        return index


    def label_from_description(self, labeler:Callable=None, labels:Collection=None):
        assert labeler is not None, "must provide labeler"
        lbls=list(map(labeler, self.descriptions))
        cl = list(set(lbls)) if labels is None else labels
        def _oh(i, cat_cnt):
            res=np.zeros(cat_cnt,dtype=int); res[i] = 1
            return res
        # return [_oh(cl.index(x), len(cl)) for x in lbls], cl
        # return [cl.index(x) for x in lbls],cl
        return lbls,cl

    @classmethod
    def from_folder(cls, path: PathOrStr = '.', extensions: Collection[str] = None,
                    filters:Collection[Callable]=None, ngram:int=8, n_cpus=1, agg:Callable=None, **kwargs) -> ItemList:
        "Get the list of files in `path` that have an image suffix. `recurse` determines if we search subfolders."
        extensions = ifnone(extensions, gen_seq_extensions)
        this = super().from_folder(path=path, extensions=extensions, **kwargs)
        return this.get_metadata(filters)


    @classmethod
    def store_by_label_class(self,path):
        """store fasta into multi-fasta files labeled by class """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__} {len(self.items)} with itmes, {self.ngram}-mer tokensation\n" \
            f" {self.emb}, reducer:{self.agg}" \
            f"\n Head: \n  {self.descriptions[0]}\n  {self.items[0]}" \
            f"\n Tail: \n  {self.descriptions[-1]}\n  {self.items[-1]}"

if __name__ == '__main__':

    #test DataBunch
    DB = "/home/serge/database/data/genomes/ncbi-genomes-2019-04-07/Bacillus"
    # DB="/data/genomes/GenSeq_fastas"
    bunch = Dna2VecDataBunch.from_folder(DB,
                                         filters=None,  #[partial(count_filter, keep=3, sample="last")],
                                         emb="../pretrained/embeddings/dna2vec-20190611-1940-k8to8-100d-10c-4870Mbp-sliding-LmP.w2v",
                                         ngram=8, skip=0,
                                         labeler=lambda x: " ".join(x.split()[1:3]),
                                         n_cpus=7,agg=partial(np.mean, axis=0),one_hot=True)
    print(bunch.train_ds.y)

    #test preprocessing for embedding training

    # Dna2VecList.preprocess_for_dna2vec_training(out_path="/data/genomes/dna2vec_train",
    #                                                    path="/data/genomes/GenSeq_fastas",
    #                                                    filters=[partial(regex_filter, rx="plasmid", keep=False),
    #                                                             partial(seq_len_filter, len=(100000,None))])

    #test labeling

    # data.label_from_description(lambda x: x.split()[1], from_item_lists=True)
    # print(data)

    #test get item
    # data = Dna2VecList.from_folder("/data/genomes/GenSeq_fastas/valid",filters=[partial(regex_filter, rx="plasmid")])
    # print(data.get(0))

    #test process all itmes
    # data = Dna2VecList.from_folder("/data/genomes/GenSeq_fastas", filters=[partial(regex_filter, rx="plasmid",keep=False)],
    #                                emb='/data/genomes/embeddings/dna2vec-20190614-1532-k11to11-100d-10c-4870Mbp-sliding-X6c.w2v')
    # # print(data.get(0)))
    # processors = [GSFileProcessor(),GSTokenizeProcessor(tokenizer=GSTokenizer(ngram=11, skip=0, n_cpus=7)),Dna2VecProcessor()]
    # for p in processors: p.process(data)
    # print(data)