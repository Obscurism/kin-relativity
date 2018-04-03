from konlpy.tag import Twitter
import gensim

class Preprocessor(object):
    def __init__(self):
        self.twitter = Twitter()
        self.model = None

    def make_dict(self, dataset):
        self.model = gensim.models.Word2Vec(
            [x[0] for x in dataset.loaded_data], min_count=1, size=50, iter=10, sg=0
        )

    def save_to(self, file):
        self.model.save(file)

    def load_from(self, file):
        pass

    def parse_sentence(self, sentence):
        return map(
            lambda word: "/".join(word),
            self.twitter.pos(sentences, norm=True, stem=True)
        )

    def preprocess_test(self, sentence):
        pass

    def preprocess_train(self, sentence):
        pass
