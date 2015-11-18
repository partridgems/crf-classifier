from corpus import Character, Character2, CharacterTest, ThaiWordCorpus
from crf import CRF, sequence_accuracy
from maxent import MaxEnt
from unittest import TestCase, main

class TestSegmenting(TestCase):

    def setUp(self):
        self.corpus = ThaiWordCorpus('orchid97_features.bio', CharacterTest)
        # self.corpus = ThaiWordCorpus('orchid97_features.bio.small', Character)
        crf = CRF(self.corpus.label_codebook, self.corpus.feature_codebook)
        self.crf =crf

    def test_segmenting(self):
        train = self.corpus[0:20000]
        dev = self.corpus[20000:21000]
        test = self.corpus[21000:24000]
        # train = self.corpus[0:350]
        # dev = self.corpus[350:375]
        # test = self.corpus[375:436]
        self.crf.train(train, dev)

        accuracy = sequence_accuracy(self.crf, test)
        print '%2.1f%%' % (accuracy*100)
        self.assertGreaterEqual(accuracy, 0.80)


# class TestSegmentingMaxEnt(TestCase):
#
#     def setUp(self):
#         self.corpus = ThaiWordCorpus('orchid97_features.bio', Character)
#         # self.corpus = ThaiWordCorpus('orchid97_features.bio.small', Character)
#         self.corpus.documents = [char for seq in self.corpus for char in seq]
#         me = MaxEnt(self.corpus.label_codebook, self.corpus.feature_codebook)
#         self.me = me
#
#     def test_segmenting(self):
#         train = self.corpus[0:50000]
#         dev = self.corpus[50000:55000]
#         test = self.corpus[55000:70000]
#         # train = self.corpus[0:8000]
#         # dev = self.corpus[8000:8500]
#         # test = self.corpus[8500:9518]
#         self.me.train(train, dev)
#
#         accuracy = self.me.sequence_accuracy(test)
#         print '%2.1f%%' % (accuracy)
#         self.assertGreaterEqual(accuracy, 0.80)


if __name__ == '__main__':
    main(verbosity=2)
