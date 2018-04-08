from dataset import pad_sequence, word2vec_mapped_size
from gensim.models import Word2Vec
from konlpy.tag import Twitter
from os import path

class Preprocessor(object):
    def __init__(self):
        self.twitter = Twitter()
        self.model = Word2Vec(None, min_count=3, size=50, iter=10, sg=0)
        self.model_initialized = False

        if path.isfile("./results/word2vec"):
            self.model = Word2Vec.load("./results/word2vec")
            self.model_initialized = True

            print("[Preprocess] Initialized from existing word2vec vocabulary!")

    def make_dict(self, dataset):
        online_sentences = [y for x in dataset.loaded_data for y in x[0]]
        dataset_len = len(dataset.loaded_data) * 2
        self.update_model(online_sentences, dataset_len)

        print("[Preprocess] Added %d sentences to dict." % dataset_len)

    def save_to(self, file):
        self.model.save(file)

    def update_model(self, sentences, total_examples):
        self.model.build_vocab(sentences, update=self.model_initialized)
        self.model.train(sentences, epochs=self.model.iter, total_examples=total_examples)

    def load_from(self, file):
        self.model = Word2Vec.load(file)
        self.model_initialized = True

    def parse_split(self, split):
        return list(map(
            lambda word: "/".join(word),
            self.twitter.pos(split, norm=True, stem=True)
        ))

    def parse_sentence(self, sentence):
        sentence_split = sentence.split('\t')

        return [
            self.parse_split(sentence_split[0]),
            self.parse_split(sentence_split[1])
        ]

    def map_vector(self, splits):
        def map_handler(x):
            if x in self.model.wv.vocab:
                return self.model[x]

            return self.model["./Punctuation"]

        # Updating Word2Vec model re-defines word vector and it is not suitable for preprocessing
        # Please refer to RaRe-Technologies/gensim/issues/1131, gensim/test/test_word2vec.py#L189

        return [
            list(map(map_handler, splits[0])),
            list(map(map_handler, splits[1]))
        ]

    def preprocess_test(self, sentence):
        splits = self.parse_sentence(sentence)
        mapped_vectors = self.map_vector(splits)
        pad_len = max([len(mapped_vectors[0]), len(mapped_vectors[1])])
        padded_vectors = [
            pad_sequence(mapped_vectors[0], pad_len, word2vec_mapped_size),
            pad_sequence(mapped_vectors[1], pad_len, word2vec_mapped_size)
        ]

        return padded_vectors
