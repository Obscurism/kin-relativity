from konlpy.tag import Twitter
from os import path

import gensim

class Preprocessor(object):
    def __init__(self):
        self.twitter = Twitter()
        self.model = None

        if path.isfile("./results/word2vec"):
            self.model = gensim.models.Word2Vec.load("./results/word2vec")

    def make_dict(self, dataset):
        online_sentences = [y for x in dataset.loaded_data for y in x[0]]
        dataset_len = len(dataset.loaded_data) * 2
        self.update_model(online_sentences, dataset_len)

        print("Added %d sentences to dict." % dataset_len)

    def save_to(self, file):
        self.model.save(file)

    def update_model(self, sentences, total_examples):
        self.model.build_vocab(sentences, update=True)
        self.model.train(sentences, epochs=self.model.iter, total_examples=total_examples)

    def load_from(self, file):
        self.model = gensim.models.Word2Vec.load(file)

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
            if x in self.model.wv_vocab:
                return self.model.wv.vocab[x]

            return self.model.wv.vocab["./Punctuation"]

        # Updating Word2Vec model re-defines word vector and it is not suitable for preprocessing
        # Please refer to RaRe-Technologies/gensim/issues/1131, gensim/test/test_word2vec.py#L189

        return [
            list(map(map_handler, splits[0])),
            list(map(map_handler, splits[1]))
        ]

    def preprocess_test(self, sentence):
        splits = self.parse_sentence(sentence)

        return self.map_vector(splits)
