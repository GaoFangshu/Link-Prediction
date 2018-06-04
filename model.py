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
import igraph
from sklearn import svm
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from utils import ngram_utils, dist_utils, np_utils
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt


DIR_TRIAN = "social_train.txt"
DIR_TEST = "social_test.txt"
DIR_NODEINFO = "node_information.csv"
PREDICT = "randomprediction.csv"

RUN_FOR_FIRST_TIME = False
SUBMIT = True
LOAD_SAMPLE = True

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
        self.data_train_positive = None
        self.data_test = None
        self.data_node_info = None

        self.node_dict = None

        # graph
        self.graph_paper = igraph.Graph(directed=True)
        self.id_graphid_paper = {}


    def sample(self, prop, load = False):
        # to test code we select sample
        if load:
            features_node = pd.read_csv("features_node", header=None, index_col=0)
            features_index = features_node.index
            self.data_train = self.data_train.ix[features_index]
            self.data_train_positive = self.data_train[self.data_train["predict"] == 1]
        else:
            to_keep = random.sample(range(self.data_train.shape[0]), k=int(round(self.data_train.shape[0] * prop)))
            self.data_train = self.data_train.iloc[to_keep]
            self.data_train_positive = self.data_train[self.data_train["predict"] == 1]

    def get_batch(self, from_iloc, to_iloc, data_set, get_item):
        # ids from self.data_train
        if data_set == "train":
            batch_data = self.data_train.iloc[from_iloc: to_iloc]
        if data_set == "test":
            batch_data = self.data_test.iloc[from_iloc: to_iloc]
        if get_item == "node":
            print("getting node features")
            features_node = batch_data[["id_source","id_target"]].apply(self.get_features, axis=1)
            return features_node
        if get_item == "network_jaccard":
            print("getting network features")
            features_network = batch_data[["id_source", "id_target"]].apply(self.get_graph_simi, axis=1)
            return features_network
        if get_item == "page_rank":
            self.pagerank = self.graph_paper.pagerank()
            features_pagerank_from = batch_data[["id_source", "id_target"]].apply(self.get_pagerank, args=("from",), axis=1)
            features_pagerank_to = batch_data[["id_source", "id_target"]].apply(self.get_pagerank, args=("to",), axis=1)
            features_pagerank = pd.concat([features_pagerank_from, features_pagerank_to], axis=1)
            return features_pagerank

    def lookup_graph_id(self, ids):
        # ids is from data.data_train[["id_source", "id_target"]].apply(..., axis=1)
        graphids = self.get_direct(ids)
        return graphids[0], graphids[1]

    def get_graph_simi(self, ids):
        # ids is from data.data_train[["id_source", "id_target"]].apply(..., axis=1)
        graphid_from, graphid_to = self.lookup_graph_id(ids)

        graphid_in = pd.Series(self.graph_paper.neighbors(graphid_to, mode="IN"))
        simi_dice_in = graphid_in.apply(self.simi_dice, args=(graphid_from,))
        return np.mean(simi_dice_in)


    def simi_dice(self, graphid_in, graphid_from):
        # calculate
        simi_dice = self.graph_paper.similarity_jaccard(vertices=(graphid_in, graphid_from), mode="OUT", loops=False)[0][1]
        return simi_dice

    def get_pagerank(self, ids, direct):
        # self.pagerank = self.graph.pagerank() before doing this
        graphid_from, graphid_to = self.lookup_graph_id(ids)
        if direct == "from":
            return self.pagerank[graphid_from]
        if direct == "to":
            return self.pagerank[graphid_to]

    def get_features(self, ids):
        # ids = [source_id, target_id]

        # features from self.data_tkzd_title
        obs_tkzd_title_source = self.node_dict[ids[0]]["tkzd_title"]
        obs_tkzd_title_target = self.node_dict[ids[1]]["tkzd_title"]

        jaccard_tkzd_title = dist_utils._jaccard_coef(obs_tkzd_title_source, obs_tkzd_title_target)
        dice_tkzd_title = dist_utils._dice_dist(obs_tkzd_title_source, obs_tkzd_title_target)

        # TODO: # features from self.tkzd_title_rm_stpwds_stem

        # features from self.data_tkzd_abstract
        obs_tkzd_abstract_source = self.node_dict[ids[0]]["tkzd_abstract"]
        obs_tkzd_abstract_target = self.node_dict[ids[1]]["tkzd_abstract"]
        bigrams_tkzd_abstract_source = list(nltk.bigrams(obs_tkzd_abstract_source))
        bigrams_tkzd_abstract_target = list(nltk.bigrams(obs_tkzd_abstract_target))
        trigrams_tkzd_abstract_source = list(nltk.trigrams(obs_tkzd_abstract_source))
        trigrams_tkzd_abstract_target = list(nltk.trigrams(obs_tkzd_abstract_target))

        jaccard_tkzd_abstract = dist_utils._jaccard_coef(obs_tkzd_abstract_source, obs_tkzd_abstract_target)
        jaccard_bigr_tkzd_abstract = dist_utils._jaccard_coef(bigrams_tkzd_abstract_source,
                                                              bigrams_tkzd_abstract_target)
        jaccard_trigr_tkzd_abstract = dist_utils._jaccard_coef(trigrams_tkzd_abstract_source,
                                                               trigrams_tkzd_abstract_target)

        dice_tkzd_abstract = dist_utils._dice_dist(obs_tkzd_abstract_source, obs_tkzd_abstract_target)
        dice_bigr_tkzd_abstract = dist_utils._dice_dist(bigrams_tkzd_abstract_source, bigrams_tkzd_abstract_target)
        dice_trigr_tkzd_abstract = dist_utils._dice_dist(trigrams_tkzd_abstract_source,
                                                         trigrams_tkzd_abstract_target)

        # features from self.data_tkzd_abstract_rm_stpwds
        obs_tkzd_abstract_rm_stpwds_source = self.node_dict[ids[0]]["tkzd_abstract_rm_stpwds"]
        obs_tkzd_abstract_rm_stpwds_target = self.node_dict[ids[1]]["tkzd_abstract_rm_stpwds"]
        bigrams_tkzd_abstract_rm_stpwds_source = list(nltk.bigrams(obs_tkzd_abstract_rm_stpwds_source))
        bigrams_tkzd_abstract_rm_stpwds_target = list(nltk.bigrams(obs_tkzd_abstract_rm_stpwds_target))
        trigrams_tkzd_abstract_rm_stpwds_source = list(nltk.trigrams(obs_tkzd_abstract_rm_stpwds_source))
        trigrams_tkzd_abstract_rm_stpwds_target = list(nltk.trigrams(obs_tkzd_abstract_rm_stpwds_target))

        jaccard_tkzd_abstract_rm_stpwds = dist_utils._jaccard_coef(obs_tkzd_abstract_rm_stpwds_source,
                                                                   obs_tkzd_abstract_rm_stpwds_target)
        jaccard_bigr_tkzd_abstract_rm_stpwds = dist_utils._jaccard_coef(bigrams_tkzd_abstract_rm_stpwds_source,
                                                                        bigrams_tkzd_abstract_rm_stpwds_target)
        jaccard_trigr_tkzd_abstract_rm_stpwds = dist_utils._jaccard_coef(trigrams_tkzd_abstract_rm_stpwds_source,
                                                                         trigrams_tkzd_abstract_rm_stpwds_target)

        dice_tkzd_abstract_rm_stpwds = dist_utils._dice_dist(obs_tkzd_abstract_rm_stpwds_source,
                                                             obs_tkzd_abstract_rm_stpwds_target)
        dice_bigr_tkzd_abstract_rm_stpwds = dist_utils._dice_dist(bigrams_tkzd_abstract_rm_stpwds_source,
                                                                  bigrams_tkzd_abstract_rm_stpwds_target)
        dice_trigr_tkzd_abstract_rm_stpwds = dist_utils._dice_dist(trigrams_tkzd_abstract_rm_stpwds_source,
                                                                   trigrams_tkzd_abstract_rm_stpwds_target)

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

    def init_graph(self):
        # run after `prepare_data`, need self.node_dict
        self.graph_paper.add_vertices(list(self.node_dict.keys()))

        for i in range(self.node_dict.__len__()):
            self.id_graphid_paper[self.graph_paper.vs["name"][i]] = i
        edges = self.data_train_positive[["id_source", "id_target"]].apply(self.get_direct, axis=1)
        self.graph_paper.add_edges(edges.tolist())


    def get_direct(self, ids, return_type="graph_id"):
        # need self.id_graphid
        # input: int: id1 and id2
        # output: tuple: (from_graph_id, to_graph_id) or (from_id, to_id)
        # year(from_id) >= year(to_id)

        year_id1 = self.node_dict[ids[0]]["year"]
        year_id2 = self.node_dict[ids[1]]["year"]
        if return_type == "graph_id":
            if year_id1 >= year_id2:  # TODO: how to deal with papers in same year, I ignore it now
                return (self.id_graphid_paper[ids[0]], self.id_graphid_paper[ids[1]])
            else:
                return (self.id_graphid_paper[ids[1]], self.id_graphid_paper[ids[0]])
        if return_type == "id":
            if year_id1 >= year_id2:  # TODO: how to deal with papers in same year, I ignore it now
                return (ids[0], ids[1])
            else:
                return (ids[1], ids[0])

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

    def get_node_dict(self):
        # save node data to dictionary, index is "id"
        self.node_dict = self.data_node_info.set_index('id').T.to_dict('series')

    def prepare_data(self, delete=True):
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
        re_author = self.data_node_info["author"].apply(lambda x: re.sub(r'\(.*?\)|\s|\\\"\"\{|\\\"\"\{\\|\\\\\'|\\\'|\\\"\"|\\\"|\\|\}|\'', "", x) if x is not np.nan else np.nan)    # delete contents in brackets and useless space
        re_author = re_author.apply(lambda x: re.sub(r'\(.*', "", x) if x is not np.nan else np.nan)
        re_author = re_author.apply(lambda x: re.sub(r'\&', ",", x) if x is not np.nan else np.nan)
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
        if delete:
            del(self.data_node_info)

        print("data prepared")

    def predict(self):
        # TODO
        pass

    def get_observation(self):
        # get observation from self.node_info_set
        # TODO: delete or not?
        pass


if __name__ == '__main__':
    data = Data(sample=True)
    data.load_data()

    data.sample(prop=1, load=LOAD_SAMPLE)

    data.get_node_dict()
    data.prepare_data()
    data.init_graph()

    if RUN_FOR_FIRST_TIME:
        t0 = time.clock()
        features_node = data.get_batch(0, data.data_train.shape[0], "train", get_item="node")
        features_network = data.get_batch(0, data.data_train.shape[0], "train", get_item="network_jaccard")
        features_pagerank = data.get_batch(0, data.data_train.shape[0], "train", get_item="page_rank")
        print(time.clock() - t0)

        features_node.to_csv("features_node")
        features_network.to_csv("features_network")
        features_pagerank.to_csv("features_pagerank")

        t0 = time.clock()
        test_features_node = data.get_batch(0, data.data_test.shape[0], "test", get_item="node")
        test_features_network = data.get_batch(0, data.data_test.shape[0], "test", get_item="network_jaccard")
        test_features_pagerank = data.get_batch(0, data.data_test.shape[0], "test", get_item="page_rank")
        print(time.clock() - t0)

        test_features_node.to_csv("test_features_node")
        test_features_network.to_csv("test_features_network")
        test_features_pagerank.to_csv("test_features_pagerank")

    else:
        features_node = pd.read_csv("features_node", header=0, index_col=0)
        features_network = pd.read_csv("features_network", header=None, index_col=0)
        features_pagerank = pd.read_csv("features_pagerank", header=0, index_col=0)
        test_features_node = pd.read_csv("test_features_node", header=0, index_col=0)
        test_features_network = pd.read_csv("test_features_network", header=None, index_col=0)
        test_features_pagerank = pd.read_csv("test_features_pagerank", header=0, index_col=0)

    features_network[np.isnan(features_network)] = 0
    test_features_network[np.isnan(test_features_network)] = 0

    training_features = pd.concat([features_node, features_network, features_pagerank], axis=1)
    training_index = training_features.index
    # scale
    training_features = preprocessing.scale(training_features)
    labels_array = data.data_train["predict"][training_index]

    if SUBMIT:
        X_train = training_features
        y_train = labels_array

        testing_features = pd.concat([test_features_node, test_features_network, test_features_pagerank], axis=1)
        testing_features = preprocessing.scale(testing_features)
        X_test = testing_features
    else:
        X_train, X_test, y_train, y_test = train_test_split(training_features, labels_array, test_size=0.2, random_state=0)

    # train model
    model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective="binary:logistic")
    model.fit(X_train, y_train)

    # test
    ans = model.predict(X_test)

    if SUBMIT:
        predict = pd.read_csv(PREDICT, sep=",")
        predict["prediction"] = ans
        predict.to_csv("prediction", index=False)
    else:
        # calculate f1
        f1 = f1_score(ans, y_test)
        print("F1 Accuracy: %.2f" % f1)

        # show importance
        plot_importance(model)
        plt.show()