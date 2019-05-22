data = (AlignmentsItemList.from_folder(bam_sam_folder)
        #Where are the sequences? -> in ```bam_sam_folder``` and its subfolders
        .generate_missing_index(),
        #Alignment files need indexes. Run samtools to generate index if missing
        .take_by_name(list_of_named_alignments),
        #select metachondria, chromosome, etc.
        .toFasta(),
        #generate fasta sequence from alignment
# .do_not_label()
        #create empty labels for unsupervised learning tasks
        .databunch(bs=16, collate_fn=bb_pad_collate))
        #Finally we convert to a DataBunch, use a batch size of 16,
        # and we use bb_pad_collate to collate the data into a mini-batch

##=====================================
## ItemBase classes
##=====================================

class AlignmentIndexBase(ItemBase)

class AlignmentItemBase(ItemBase):
    def __init__(self, file:path, generate_index=False):
        self.alignments = self.load(file)
        self.seq_index = seq, seq_index

    def loadAlighment(self, file):
        pass

    def loadAlignmentIndex(self, aignmentFileName):
        with open(aignmentFileName) as
        if self.generate_index

    def _getNamedPart(self, part:str):
    ```select a named parts e.g chromosome, metachondria etc.```

    def  toFasta(self, item:Collection[str], position:slice) -> str:
        pass

    def toVariants(self, item:Collection[str], position:slice) -> Collection[str]:
        pass


class AlignmentFileProcessor(PreProcessor):
    "`PreProcessor` that opens the filenames and read the fastas."
    def process_one(self,item):
        return gen_seq_reader(item) if isinstance(item, Path) else item


class AlignmentItemList(ItemList):
    "Special `ItemList` for BAM/SAM alignment files"
    _bunch = AlignmentDataBunch

    def do_not_label(self, **kwargs):
        "A special labelling method for unsupervised learning"
        self.__class__ = UnlabeledAlignemtList
        kwargs['label_cls'] = UnlabeledAlignemtList
        return self.label_empty(**kwargs)

    def named_parts(self):
        """Retreave named parts from alignment index"""
        pass





##=====================================
## DataBunch
##=====================================


class AlignmentDataBunch(DataBunch):
    "DataBunch suitable for generic sequence processing."


    @classmethod
    def from_folder(cls, path:PathOrStr, train:str='train', valid:str='valid', test:Optional[str]=None,
                    classes:Collection[Any]=None, tokenizer:Tokenizer=None, vocab:Vocab=None, chunksize:int=10000, max_vocab:int=60000,
                    min_freq:int=2, mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False, **kwargs):
        "Create a `AlignmentDataBunch` from text files in folders."
        path = Path(path).absolute()
        processor = [AlignmentFileProcessor()]  +
                    _get_processor(tokenizer=tokenizer, vocab=vocab, chunksize=chunksize, max_vocab=max_vocab,
                    min_freq=min_freq, mark_fields=mark_fields, include_bos=include_bos, include_eos=include_eos)
        src = (AlignmentItemList.from_folder(path, processor=processor)
                       .split_by_folder(train=train, valid=valid))
        src = src.label_for_clustering() if cls==UnlabeledAlignementDataBunch else src.label_from_folder(classes=classes)
        if test is not None: src.add_test_folder(path/test)
        return src.databunch(**kwargs)


##=====================================
## Unlabeled Alignment Data Bunch
##=====================================

class UnlabeledAlignemtList(AlignementItemList):
    "Special `ItemList` for a language model."
    _bunch = AlignmentDataBunch


class UnlabeledAlignementDataBunch(AlignmentDataBunch):
    "DataBunch suitable for unsupervised learning over alignment data"

    def label_for_clustering(self, **kwargs):
            "A special labelling method for unsupervised learning"
            self.__class__ = UnlabeledAlignemtList
            kwargs['label_cls'] = UnlabeledAlignemtList
            return self.label_empty(**kwargs)
