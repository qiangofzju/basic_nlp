# -*- coding: utf-8 -*-
from model.data_utils import load_vocab, get_processing_word
from model.general_utils import get_logger


class Config(object):
    def __init__(self, load=True, args=None):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        self.path_log = args.model_path + '/log.txt'

        # embeddings
        self.dim_char = 100
        self.dim_word = 300
        self.embeddings = None

        # dataset
        self.filename_test = args.input_path

        self.min_count = 1
        dataset_dic = args.data_path + '/'
        self.filename_words = dataset_dic + "words.txt"
        self.filename_tags = dataset_dic + "tags.txt"
        self.filename_chars = dataset_dic + "chars.txt"

        self.dir_output = args.output_path + '/'

        self.dir_model = args.model_path + '/model.weights/'

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()

    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords = len(self.vocab_words)
        self.ntags = len(self.vocab_tags)
        self.nchars = len(self.vocab_chars)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                                                   self.vocab_chars, lowercase=True, chars=self.use_chars)

        self.processing_tag = get_processing_word(self.vocab_tags, lowercase=False, allow_unk=False)

    # embeddings
    dim_char = 100
    dim_word = 300
    embeddings = None

    max_iter = None  # if not None, max number of examples in Dataset

    min_count = 1

    # training
    train_embeddings = False
    # nepochs = 15
    # dropout = 0.5
    batch_size = 32
    lr_method = "adam"
    lr = 0.001
    lr_decay = 0.9
    clip = -1  # if negative, no clipping
    nepoch_no_imprv = 3

    # model hyperparameters
    hidden_size_char = 100  # lstm on chars
    hidden_size_lstm = 300  # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True  # if crf, training is 1.7x slower on CPU
    use_chars = True  # if char embedding, training is 3.5x slower on CPU
