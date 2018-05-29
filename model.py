"""
@author: Fangshu Gao <gaofangshu@foxmail.com>
@brief: prepare data and features
"""

import csv
import random
import numpy as np
import pandas as pd
import nltk
import sys
import time
import re
from sklearn import svm
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from utils import ngram_utils, dist_utils, np_utils

DIR_TRIAN = "social_train.txt"
DIR_TEST = "social_test.txt"
DIR_NODEINFO = "node_information.csv"

# nltk.download('punkt')  # for tokenization
# nltk.download('stopwords')
STPWDS = set(nltk.corpus.stopwords.words("english"))


class Data():
    def __init__(self, sample):
        assert type(sample) == bool
        self.stemmer = nltk.stem.PorterStemmer()
        # the columns of the node data frame below are:
        # (0) paper unique ID (integer)
        # (1) publication year (integer)
        # (2) paper title (string)
        # (3) authors (strings separated by ,)
        # (4) name of journal (optional) (string)
        # (5) abstract (string) - lowercased, free of punctuation except intra-word dashes

        self.data_trian = None
        self.data_test = None
        self.data_node_info = None

        self.node_dict = None


        # self.data_tkzd_title = []
        # self.data_tkzd_title_rm_stpwds = []
        # self.data_tkzd_title_rm_stpwds_stem = []
        # self.data_tkzd_abstract = []
        # self.data_tkzd_abstract_rm_stpwds = []
        # self.data_tkzd_abstract_rm_stpwds_stem = []
        #
        # valid_ids = self.get_valid_ids(self.training_set)
        #
        # self.testing_set = self.split_to_list(data_test)
        # # self.node_info_set = [element for element in data_node_info if element[0] in valid_ids]
        # self.node_info_set = data_node_info
        # self.node_position = self.add_position(self.node_info_set)  # {paperID : rowID in self.node_info_set}
        #
        # self.prepare_data()
        # self.feature = None

    def sample(self, prop):
        # to test code we select sample
        to_keep = random.sample(range(self.data_train.shape[0]), k=int(round(self.data_train.shape[0] * prop)))
        self.data_train = self.data_train.iloc[to_keep]

    def get_batch(self, from_iloc, to_iloc, data_set):
        # ids from self.data_train
        if data_set == "train":
            batch_data = self.data_train.iloc[from_iloc: to_iloc]
        if data_set == "test":
            batch_data = self.data_test.iloc[from_iloc: to_iloc]
        result = batch_data[["id_source","id_target"]].apply(self.get_features, axis=1)
        return result

    def get_features(self, ids):
        # ids = [source_id, target_id]

        # get base data
        # self.data_train.merge(self.data_node_info, left_on='id_source', right_on='rkey', how='outer')

        # features from self.data_tkzd_title
        obs_tkzd_title_source = self.node_dict[ids[0]]["tkzd_title"]
        obs_tkzd_title_target = self.node_dict[ids[1]]["tkzd_title"]
        # bigrams_tkzd_title_source = ngram_utils._ngrams(obs_tkzd_title_source, 2)
        # bigrams_tkzd_title_target = ngram_utils._ngrams(obs_tkzd_title_target, 2)

        jaccard_tkzd_title = dist_utils._jaccard_coef(obs_tkzd_title_source, obs_tkzd_title_target)
        dice_tkzd_title = dist_utils._dice_dist(obs_tkzd_title_source, obs_tkzd_title_target)
        # jaccard_bigr_tkzd_title = dist_utils._jaccard_coef(bigrams_tkzd_title_source, bigrams_tkzd_title_target)
        # dice_bigr_tkzd_title = dist_utils._dice_dist(bigrams_tkzd_title_source, bigrams_tkzd_title_target)

        # features from self.data_tkzd_title_rm_stpwds
        # obs_tkzd_title_rm_stpwds_source = self.node_dict[source_id]["tkzd_title_rm_stpwds"]
        # obs_tkzd_title_rm_stpwds_target = self.node_dict[target_id]["tkzd_title_rm_stpwds"]
        # bigrams_tkzd_title_rm_stpwds_source = ngram_utils._ngrams(obs_tkzd_title_rm_stpwds_source, 2)
        # bigrams_tkzd_title_rm_stpwds_target = ngram_utils._ngrams(obs_tkzd_title_rm_stpwds_target, 2)

        # jaccard_tkzd_title_rm_stpwds = dist_utils._jaccard_coef(obs_tkzd_title_rm_stpwds_source,
        #                                                         obs_tkzd_title_rm_stpwds_target)
        # dice_tkzd_title_rm_stpwds = dist_utils._dice_dist(obs_tkzd_title_rm_stpwds_source,
        #                                                   obs_tkzd_title_rm_stpwds_target)
        # jaccard_bigr_tkzd_title_rm_stpwds = dist_utils._jaccard_coef(bigrams_tkzd_title_rm_stpwds_source,
        #                                                              bigrams_tkzd_title_rm_stpwds_target)
        # dice_bigr_tkzd_title_rm_stpwds = dist_utils._dice_dist(bigrams_tkzd_title_rm_stpwds_source,
        #                                                        bigrams_tkzd_title_rm_stpwds_target)

        # TODO: # features from self.tkzd_title_rm_stpwds_stem

        # features from self.data_tkzd_abstract
        obs_tkzd_abstract_source = self.node_dict[ids[0]]["tkzd_abstract"]
        obs_tkzd_abstract_target = self.node_dict[ids[1]]["tkzd_abstract"]
        bigrams_tkzd_abstract_source = list(nltk.bigrams(obs_tkzd_abstract_source))
        bigrams_tkzd_abstract_target = list(nltk.bigrams(obs_tkzd_abstract_target))
        trigrams_tkzd_abstract_source = list(nltk.trigrams(obs_tkzd_abstract_source))
        trigrams_tkzd_abstract_target = list(nltk.trigrams(obs_tkzd_abstract_target))
        # fourgrams_tkzd_abstract_source = ngram_utils._ngrams(obs_tkzd_abstract_source, 4)
        # fourgrams_tkzd_abstract_target = ngram_utils._ngrams(obs_tkzd_abstract_target, 4)

        # biterms_tkzd_abstract_source = list(nltk.bigrams(obs_tkzd_abstract_source))
        # biterms_tkzd_abstract_target = list(nltk.bigrams(obs_tkzd_abstract_target))
        # triterms_tkzd_abstract_source = ngram_utils._nterms(obs_tkzd_abstract_source, 3)
        # triterms_tkzd_abstract_target = ngram_utils._nterms(obs_tkzd_abstract_target, 3)
        # fourterms_tkzd_abstract_source = ngram_utils._nterms(obs_tkzd_abstract_source, 4)
        # fourterms_tkzd_abstract_target = ngram_utils._nterms(obs_tkzd_abstract_target, 4)

        jaccard_tkzd_abstract = dist_utils._jaccard_coef(obs_tkzd_abstract_source, obs_tkzd_abstract_target)
        jaccard_bigr_tkzd_abstract = dist_utils._jaccard_coef(bigrams_tkzd_abstract_source,
                                                              bigrams_tkzd_abstract_target)
        jaccard_trigr_tkzd_abstract = dist_utils._jaccard_coef(trigrams_tkzd_abstract_source,
                                                               trigrams_tkzd_abstract_target)
        # jaccard_fourgr_tkzd_abstract = dist_utils._jaccard_coef(fourgrams_tkzd_abstract_source,
        #                                                         fourgrams_tkzd_abstract_target)
        # jaccard_bitm_tkzd_abstract = dist_utils._jaccard_coef(biterms_tkzd_abstract_source,
        #                                                       biterms_tkzd_abstract_target)
        # jaccard_tritm_tkzd_abstract = dist_utils._jaccard_coef(triterms_tkzd_abstract_source,
        #                                                        triterms_tkzd_abstract_target)
        # jaccard_fourtm_tkzd_abstract = dist_utils._jaccard_coef(fourterms_tkzd_abstract_source,
        #                                                         fourterms_tkzd_abstract_target)

        dice_tkzd_abstract = dist_utils._dice_dist(obs_tkzd_abstract_source, obs_tkzd_abstract_target)
        dice_bigr_tkzd_abstract = dist_utils._dice_dist(bigrams_tkzd_abstract_source, bigrams_tkzd_abstract_target)
        dice_trigr_tkzd_abstract = dist_utils._dice_dist(trigrams_tkzd_abstract_source,
                                                         trigrams_tkzd_abstract_target)
        # dice_fourgr_tkzd_abstract = dist_utils._dice_dist(fourgrams_tkzd_abstract_source,
        #                                                   fourgrams_tkzd_abstract_target)
        # dice_bitm_tkzd_abstract = dist_utils._dice_dist(biterms_tkzd_abstract_source, biterms_tkzd_abstract_target)
        # dice_tritm_tkzd_abstract = dist_utils._dice_dist(triterms_tkzd_abstract_source,
        #                                                  triterms_tkzd_abstract_target)
        # dice_fourtm_tkzd_abstract = dist_utils._dice_dist(fourterms_tkzd_abstract_source,
        #                                                   fourterms_tkzd_abstract_target)

        # features from self.data_tkzd_abstract_rm_stpwds
        obs_tkzd_abstract_rm_stpwds_source = self.node_dict[ids[0]]["tkzd_abstract_rm_stpwds"]
        obs_tkzd_abstract_rm_stpwds_target = self.node_dict[ids[1]]["tkzd_abstract_rm_stpwds"]
        bigrams_tkzd_abstract_rm_stpwds_source = list(nltk.bigrams(obs_tkzd_abstract_rm_stpwds_source))
        bigrams_tkzd_abstract_rm_stpwds_target = list(nltk.bigrams(obs_tkzd_abstract_rm_stpwds_target))
        trigrams_tkzd_abstract_rm_stpwds_source = list(nltk.trigrams(obs_tkzd_abstract_rm_stpwds_source))
        trigrams_tkzd_abstract_rm_stpwds_target = list(nltk.trigrams(obs_tkzd_abstract_rm_stpwds_target))
        # fourgrams_tkzd_abstract_rm_stpwds_source = ngram_utils._ngrams(obs_tkzd_abstract_rm_stpwds_source, 4)
        # fourgrams_tkzd_abstract_rm_stpwds_target = ngram_utils._ngrams(obs_tkzd_abstract_rm_stpwds_target, 4)

        # biterms_tkzd_abstract_rm_stpwds_source = ngram_utils._nterms(obs_tkzd_abstract_rm_stpwds_source, 2)
        # biterms_tkzd_abstract_rm_stpwds_target = ngram_utils._nterms(obs_tkzd_abstract_rm_stpwds_target, 2)
        # triterms_tkzd_abstract_rm_stpwds_source = ngram_utils._nterms(obs_tkzd_abstract_rm_stpwds_source, 3)
        # triterms_tkzd_abstract_rm_stpwds_target = ngram_utils._nterms(obs_tkzd_abstract_rm_stpwds_target, 3)
        # fourterms_tkzd_abstract_rm_stpwds_source = ngram_utils._nterms(obs_tkzd_abstract_rm_stpwds_source, 4)
        # fourterms_tkzd_abstract_rm_stpwds_target = ngram_utils._nterms(obs_tkzd_abstract_rm_stpwds_target, 4)

        jaccard_tkzd_abstract_rm_stpwds = dist_utils._jaccard_coef(obs_tkzd_abstract_rm_stpwds_source,
                                                                   obs_tkzd_abstract_rm_stpwds_target)
        jaccard_bigr_tkzd_abstract_rm_stpwds = dist_utils._jaccard_coef(bigrams_tkzd_abstract_rm_stpwds_source,
                                                                        bigrams_tkzd_abstract_rm_stpwds_target)
        jaccard_trigr_tkzd_abstract_rm_stpwds = dist_utils._jaccard_coef(trigrams_tkzd_abstract_rm_stpwds_source,
                                                                         trigrams_tkzd_abstract_rm_stpwds_target)
        # jaccard_fourgr_tkzd_abstract_rm_stpwds = dist_utils._jaccard_coef(fourgrams_tkzd_abstract_rm_stpwds_source,
        #                                                                   fourgrams_tkzd_abstract_rm_stpwds_target)
        # jaccard_bitm_tkzd_abstract_rm_stpwds = dist_utils._jaccard_coef(biterms_tkzd_abstract_rm_stpwds_source,
        #                                                                 biterms_tkzd_abstract_rm_stpwds_target)
        # jaccard_tritm_tkzd_abstract_rm_stpwds = dist_utils._jaccard_coef(triterms_tkzd_abstract_rm_stpwds_source,
        #                                                                  triterms_tkzd_abstract_rm_stpwds_target)
        # jaccard_fourtm_tkzd_abstract_rm_stpwds = dist_utils._jaccard_coef(fourterms_tkzd_abstract_rm_stpwds_source,
        #                                                                   fourterms_tkzd_abstract_rm_stpwds_target)

        dice_tkzd_abstract_rm_stpwds = dist_utils._dice_dist(obs_tkzd_abstract_rm_stpwds_source,
                                                             obs_tkzd_abstract_rm_stpwds_target)
        dice_bigr_tkzd_abstract_rm_stpwds = dist_utils._dice_dist(bigrams_tkzd_abstract_rm_stpwds_source,
                                                                  bigrams_tkzd_abstract_rm_stpwds_target)
        dice_trigr_tkzd_abstract_rm_stpwds = dist_utils._dice_dist(trigrams_tkzd_abstract_rm_stpwds_source,
                                                                   trigrams_tkzd_abstract_rm_stpwds_target)
        # dice_fourgr_tkzd_abstract_rm_stpwds = dist_utils._dice_dist(fourgrams_tkzd_abstract_rm_stpwds_source,
        #                                                             fourgrams_tkzd_abstract_rm_stpwds_target)
        # dice_bitm_tkzd_abstract_rm_stpwds = dist_utils._dice_dist(biterms_tkzd_abstract_rm_stpwds_source,
        #                                                           biterms_tkzd_abstract_rm_stpwds_target)
        # dice_tritm_tkzd_abstract_rm_stpwds = dist_utils._dice_dist(triterms_tkzd_abstract_rm_stpwds_source,
        #                                                            triterms_tkzd_abstract_rm_stpwds_target)
        # dice_fourtm_tkzd_abstract_rm_stpwds = dist_utils._dice_dist(fourterms_tkzd_abstract_rm_stpwds_source,
        #                                                             fourterms_tkzd_abstract_rm_stpwds_target)

        # TODO: # features from self.data_tkzd_abstract_rm_stpwds_stem

        result = pd.Series([jaccard_tkzd_title,
                            dice_tkzd_title,
                            jaccard_tkzd_abstract,
                            jaccard_bigr_tkzd_abstract,
                            jaccard_trigr_tkzd_abstract,
                            dice_tkzd_abstract,
                            dice_bigr_tkzd_abstract,
                            dice_trigr_tkzd_abstract,
                            jaccard_tkzd_abstract_rm_stpwds,
                            jaccard_bigr_tkzd_abstract_rm_stpwds,
                            jaccard_trigr_tkzd_abstract_rm_stpwds,
                            dice_tkzd_abstract_rm_stpwds,
                            dice_bigr_tkzd_abstract_rm_stpwds,
                            dice_trigr_tkzd_abstract_rm_stpwds,
                            ])

        return result
            # file = open("/media/gaofangshu/Windows/GaoFangshu/RUC/project/feature_nlp.txt", "a+")
            # file.write(output_list)
            # file.write("\n")
            # file.close
            #
            # if counter % 10 == 0:
            #     sys.stdout.write("\rPreparing features: %.1f%%" % (100 * counter / size))
            #     sys.stdout.flush()

    def load_data(self):
        # (0) paper unique ID (integer)
        # (1) publication year (integer)
        # (2) paper title (string)
        # (3) authors (strings separated by ,)
        # (4) name of journal (optional) (string)
        # (5) abstract (string) - lowercased, free of punctuation except intra-word dashes
        self.data_train = pd.read_csv(DIR_TRIAN, names=["id_source", "id_target", "predict"], header=None, sep=" ")
        self.data_test = pd.read_csv(DIR_TEST, names=["id_source", "id_target"], header=None, sep=" ")
        self.data_node_info = pd.read_csv(DIR_NODEINFO, names=["id", "year", "title", "author", "journal", "abstract"], header=None)

    def get_valid_ids(self, data):
        assert type(dir) == list
        valid_ids = set()
        for element in data:
            valid_ids.add(element[0])
            valid_ids.add(element[1])
        return valid_ids

    def split_to_list(self, data, by=" "):
        assert type(data) == list
        return [element[0].split(by) for element in data]

    def add_position(self, data):
        assert type(data) == list
        ids = []
        position = {}
        for element in data:
            position[element[0]] = len(ids)
            ids.append(element[0])
        return position

    def add_feature(self, feature):
        pass

    def prepare_data(self):
        # title
        # convert to lowercase and tokenize
        tkzd_title = self.data_node_info["title"].apply(lambda x: x.lower().split(" "))
        self.data_node_info["tkzd_title"] = tkzd_title
        # remove stopwords
        tkzd_title_rm_stpwds = self.data_node_info["tkzd_title"].apply(lambda x: [token for token in x if token not in STPWDS])
        self.data_node_info["tkzd_title_rm_stpwds"] = tkzd_title_rm_stpwds
        # convert to root or original word
        tkzd_title_rm_stpwds_stem = self.data_node_info["tkzd_title_rm_stpwds"].apply(lambda x: [self.stemmer.stem(token) for token in x])
        self.data_node_info["tkzd_title_rm_stpwds_stem"] = tkzd_title_rm_stpwds_stem

        # authors
        re_author = self.data_node_info["author"].apply(lambda x: re.sub(r'\(.*\)|\s', "", x) if x is not np.nan else np.nan)    # delete contents in brackets and useless space
        tkzd_author = re_author.apply(lambda x: x.lower().split(",") if x is not np.nan else np.nan)
        self.data_node_info["tkzd_author"] = tkzd_author
        # TODO: handle (School) (number)

        # journal name
        # TODO: self.data_node_info["journal"]


        # abstract
        # convert to lowercase and tokenize
        tkzd_abstract = self.data_node_info["abstract"].apply(lambda x: x.lower().split(" "))
        self.data_node_info["tkzd_abstract"] = tkzd_abstract

        # remove stopwords
        tkzd_abstract_rm_stpwds = self.data_node_info["tkzd_abstract"].apply(lambda x: [token for token in x if token not in STPWDS])
        self.data_node_info["tkzd_abstract_rm_stpwds"] = tkzd_abstract_rm_stpwds

        # convert to root or original word
        tkzd_abstract_rm_stpwds_stem = self.data_node_info["tkzd_abstract_rm_stpwds"].apply(lambda x: [self.stemmer.stem(token) for token in x])
        self.data_node_info["tkzd_abstract_rm_stpwds_stem"] = tkzd_abstract_rm_stpwds_stem

        # save node data to dictionary, index is "id"
        self.node_dict = self.data_node_info.set_index('id').T.to_dict('series')

        del(self.data_node_info)

        print("data prepared")



    def get_observation(self):
        # get observation from self.node_info_set
        # TODO: delete or not?
        pass


if __name__ == '__main__':
    data = Data(sample=True)
    data.load_data()
    data.sample(prop=1)
    data.prepare_data()
    # data.get_features()
    training_features = data.get_batch(0, data.data_train.shape[0], "train")
    training_index = training_features.index
    # scale
    training_features = preprocessing.scale(training_features)

    labels_array = data.data_train["predict"][training_index]
    print("evaluating")

    # evaluation
    kf = KFold(training_features.shape[0], n_folds=10)
    sumf1 = 0
    for train_index, test_index in kf:
        X_train, X_test = training_features[train_index], training_features[test_index]
        y_train, y_test = labels_array.iloc[train_index], labels_array.iloc[test_index]
        # initialize basic SVM
        classifier = svm.LinearSVC()
        # train
        classifier.fit(X_train, y_train)
        pred = classifier.predict(X_test)
        sumf1 += f1_score(pred, y_test)

    print("\n\n")
    print(sumf1 / 10.0)
