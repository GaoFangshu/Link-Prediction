"""
@author: Fangshu Gao <gaofangshu@foxmail.com>
@brief: main
"""

import igraph
import math
import nltk
import numpy as np
import pandas as pd
import random
import re
import time
import xgboost as xgb

from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from utils import dist_utils
from xgboost import plot_importance

nltk.download('punkt')  # for tokenization
nltk.download('stopwords')
STPWDS = set(nltk.corpus.stopwords.words("english"))

DIR_TRIAN = "social_train.txt"
DIR_TEST = "social_test.txt"
DIR_NODEINFO = "node_information.csv"
PREDICT = "randomprediction.csv"


# ------- Control -------
RUN_FOR_FIRST_TIME = False
SUBMIT = False
LOAD_SAMPLE = True
TUNING = False
TUNING_PARMS = "max_depth & min_child_weight"
ENSEMBLE = True
# -----------------------


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
        self.graph_author = igraph.Graph(directed=True)
        self.id_graphid_author = {}

    def AciteB(self, edge):
        neighbors = self.graph_author.neighbors(self.graph_author.vs[edge[0]], mode="OUT")  # get citaion of from_author
        length = neighbors.count(edge[1])  # how many times did from_author cite to_author
        return (length)

    def AciteB_all(self, edge):
        neighbors = self.graph_author.neighbors(self.graph_author.vs[edge[0]],
                                                mode="ALL")  # get neighbots of from_author
        length = neighbors.count(edge[1])  # how many times did from_author cite to_author
        return (length)

    def add_position(self, data):
        assert type(data) == list
        ids = []
        position = {}
        for element in data:
            position[element[0]] = len(ids)
            ids.append(element[0])
        return position

    def apply_pagerank(self, input_list, pageranktype):
        # input_list: e.g. [(123,456), (234,252)]
        if type(input_list) == list:
            if pageranktype == "mean_from":
                author_pagerank_from = list(map(self.lookup_author_pagerank_from, input_list))
                return np.mean(author_pagerank_from)
            elif pageranktype == "mean_to":
                author_pagerank_from = list(map(self.lookup_author_pagerank_to, input_list))
                return np.mean(author_pagerank_from)
            elif pageranktype == "max_from":
                author_pagerank_from = list(map(self.lookup_author_pagerank_from, input_list))
                return np.max(author_pagerank_from)
            elif pageranktype == "max_to":
                author_pagerank_from = list(map(self.lookup_author_pagerank_to, input_list))
                return np.max(author_pagerank_from)
        else:
            return 0

    def author_citation_edge(self, ids):
        citation_list = []
        graphids = self.get_direct(ids, return_type="id")
        from_authors = self.node_dict[graphids[0]]['tkzd_author']
        to_authors = self.node_dict[graphids[1]]['tkzd_author']
        if type(from_authors) == list and type(to_authors) == list:
            for from_a in from_authors:
                for to_a in to_authors:
                    if from_a != '' and to_a != '':
                        citation_list.append((self.id_graphid_author[from_a], self.id_graphid_author[to_a]))
            return citation_list
        else:
            return np.nan

    def BciteA(self, edge):
        neighbors = self.graph_author.neighbors(self.graph_author.vs[edge[1]], mode="OUT")  # get citaion of to_author
        length = neighbors.count(edge[1])  # how many times did to_author cite  rom_author
        return (length)

    def get_authors_list(self):
        authors_list = []

        for key, value in self.node_dict.items():
            if type(value['tkzd_author']) == list:
                authors_list.extend(value['tkzd_author'])

        authors_list = list(set(authors_list))
        authors_list.remove('')
        return authors_list

    def get_author_overlap(self, ids):
        from_author = self.node_dict[ids[0]]["tkzd_author"]
        to_author = self.node_dict[ids[1]]["tkzd_author"]
        if (type(from_author) == list) and (type(to_author) == list):
            return len(set(from_author).intersection(set(to_author)))
        else:
            return 0

    def get_batch(self, from_iloc, to_iloc, data_set, get_item):
        # ids from self.data_train
        if data_set == "train":
            batch_data = self.data_train.iloc[from_iloc: to_iloc]
        elif data_set == "test":
            batch_data = self.data_test.iloc[from_iloc: to_iloc]
        if get_item == "node":
            print("getting node features")
            features_node = batch_data[["id_source", "id_target"]].apply(self.get_features, axis=1)
            return features_node
        elif get_item == "network_jaccard_from":
            print("getting network features")
            network_from = batch_data[["id_source", "id_target"]].apply(self.get_graph_simi, args=("from",), axis=1)
            features_network_from_mean = network_from.apply(lambda x: x[0], axis=1)
            features_network_from_sum = network_from.apply(lambda x: x[1], axis=1)
            features_network_from = pd.concat([features_network_from_mean, features_network_from_sum], axis=1)
            return features_network_from
        elif get_item == "network_jaccard_to":
            print("getting network features")
            network_to = batch_data[["id_source", "id_target"]].apply(self.get_graph_simi, args=("to",), axis=1)
            features_network_to_mean = network_to.apply(lambda x: x[0], axis=1)
            features_network_to_sum = network_to.apply(lambda x: x[1], axis=1)
            features_network_to = pd.concat([features_network_to_mean, features_network_to_sum], axis=1)
            return features_network_to
        elif get_item == "pagerank_paper":
            self.pagerank_paper = self.graph_paper.pagerank()
            features_pagerank_from = batch_data[["id_source", "id_target"]].apply(self.get_pagerank, args=("from",),
                                                                                  axis=1)
            features_pagerank_to = batch_data[["id_source", "id_target"]].apply(self.get_pagerank, args=("to",), axis=1)
            features_pagerank_paper = pd.concat([features_pagerank_from, features_pagerank_to], axis=1)
            return features_pagerank_paper
        elif get_item == "mean_aciteb":
            if data_set == "train":
                modified_data = pd.concat([pd.DataFrame([{"id_source": 201080, "id_target": 9905149}]),
                                           batch_data[["id_source", "id_target"]]])  # TODO: need automation
            elif data_set == "test":
                modified_data = pd.concat([pd.DataFrame([{"id_source": 9912290, "id_target": 7120}]),
                                           batch_data[["id_source", "id_target"]]])  # TODO: need automation
            citation_edges = modified_data.apply(self.author_citation_edge, axis=1)
            citation_edges = citation_edges.iloc[1:]

            features_AciteB = citation_edges.apply(self.meanAciteB)
            features_meanAciteB = features_AciteB.apply(lambda x: x[0])
            features_maxAciteB = features_AciteB.apply(lambda x: x[1])
            features_sumAciteB = features_AciteB.apply(lambda x: x[2])

            features_BciteA = citation_edges.apply(self.meanBciteA)
            features_meanBciteA = features_BciteA.apply(lambda x: x[0])
            features_maxBciteA = features_BciteA.apply(lambda x: x[1])
            features_sumBciteA = features_BciteA.apply(lambda x: x[2])

            features_AciteB_all = citation_edges.apply(self.meanAciteB, args=("all",))
            features_meanAciteB_all = features_AciteB_all.apply(lambda x: x[0])
            features_maxAciteB_all = features_AciteB_all.apply(lambda x: x[1])
            features_sumAciteB_all = features_AciteB_all.apply(lambda x: x[2])

            if data_set == "train":
                features_meanAciteB[batch_data["predict"] == 1] -= 1
                features_meanAciteB[features_meanAciteB == -1] = 0
                features_meanBciteA[batch_data["predict"] == 1] -= 1
                features_meanBciteA[features_meanBciteA == -1] = 0

                features_maxAciteB[batch_data["predict"] == 1] -= 1
                features_maxAciteB[features_maxAciteB == -1] = 0
                features_maxBciteA[batch_data["predict"] == 1] -= 1
                features_maxBciteA[features_maxBciteA == -1] = 0

                features_meanAciteB_all[batch_data["predict"] == 1] -= 1
                features_meanAciteB_all[features_meanAciteB_all == -1] = 0
                features_maxAciteB_all[batch_data["predict"] == 1] -= 1
                features_maxAciteB_all[features_maxAciteB_all == -1] = 0

                features_maxmeancite = np.max(pd.concat([features_meanAciteB, features_meanBciteA], axis=1), axis=1)
                features_maxmaxcite = np.max(pd.concat([features_maxAciteB, features_maxBciteA], axis=1), axis=1)
                features_maxsumcite = np.max(pd.concat([features_sumAciteB, features_sumBciteA], axis=1), axis=1)

                return pd.concat([features_meanAciteB, features_maxAciteB, features_sumAciteB, features_meanBciteA,
                                  features_maxBciteA, features_sumBciteA, features_maxmeancite, features_maxmaxcite,
                                  features_maxsumcite, features_meanAciteB_all, features_maxAciteB_all,
                                  features_sumAciteB_all], axis=1)
            elif data_set == "test":

                features_maxmeancite = np.max(pd.concat([features_meanAciteB, features_meanBciteA], axis=1), axis=1)
                features_maxmaxcite = np.max(pd.concat([features_maxAciteB, features_maxBciteA], axis=1), axis=1)
                features_maxsumcite = np.max(pd.concat([features_sumAciteB, features_sumBciteA], axis=1), axis=1)

                return pd.concat([features_meanAciteB, features_maxAciteB, features_sumAciteB, features_meanBciteA,
                                  features_maxBciteA, features_sumBciteA, features_maxmeancite, features_maxmaxcite,
                                  features_maxsumcite, features_meanAciteB_all, features_maxAciteB_all,
                                  features_sumAciteB_all], axis=1)


        elif get_item == "pagerank_author":
            self.pagerank_author = self.graph_author.pagerank()  # ids are vertice ids in graph_author
            if data_set == "train":
                modified_data = pd.concat([pd.DataFrame([{"id_source": 201080, "id_target": 9905149}]),
                                           batch_data[["id_source", "id_target"]]])  # TODO: need automation
            elif data_set == "test":
                modified_data = pd.concat([pd.DataFrame([{"id_source": 9912290, "id_target": 7120}]),
                                           batch_data[["id_source", "id_target"]]])  # TODO: need automation
            citation_edges = modified_data.apply(self.author_citation_edge, axis=1)
            citation_edges = citation_edges.iloc[1:]

            features_author_pagerank_mean_from = citation_edges.apply(self.apply_pagerank, args=("mean_from",))
            features_author_pagerank_mean_to = citation_edges.apply(self.apply_pagerank, args=("mean_to",))
            features_author_pagerank_max_from = citation_edges.apply(self.apply_pagerank, args=("max_from",))
            features_author_pagerank_max_to = citation_edges.apply(self.apply_pagerank, args=("max_to",))

            features_pagerank_author = pd.concat([features_author_pagerank_mean_from, features_author_pagerank_mean_to,
                                                  features_author_pagerank_max_from, features_author_pagerank_max_to],
                                                 axis=1)
            return features_pagerank_author
        elif get_item == "adamic_adar_paper":
            features_adamic_adar_paper = batch_data[["id_source", "id_target"]].apply(self.similarity,
                                                                                      args=(self.graph_paper,), axis=1)
            return features_adamic_adar_paper
        elif get_item == "dyear":
            features_dyear = batch_data[["id_source", "id_target"]].apply(self.get_year, axis=1)
            return features_dyear
        elif get_item == "author_overlap":
            features_author_overlap = batch_data[["id_source", "id_target"]].apply(self.get_author_overlap, axis=1)
            return features_author_overlap

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
        elif return_type == "id":
            if year_id1 >= year_id2:  # TODO: how to deal with papers in same year, I ignore it now
                return (ids[0], ids[1])
            else:
                return (ids[1], ids[0])

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

    def get_graph_simi(self, ids, mode):
        # ids is from data.data_train[["id_source", "id_target"]].apply(..., axis=1)
        graphid_from, graphid_to = self.lookup_graph_id(ids)

        if mode == "from":
            graphid_in = self.graph_paper.neighbors(graphid_to, mode="IN")
            try:
                graphid_in.remove(graphid_from)
            except:
                pass
            graphid_in = pd.Series(graphid_in)

            simi_jaccard_in = graphid_in.apply(self.simi_jaccard, args=(graphid_from, graphid_to, "OUT",))
            return [np.mean(simi_jaccard_in), np.sum(simi_jaccard_in)]
        elif mode == "to":
            graphid_out = self.graph_paper.neighbors(graphid_from, mode="OUT")
            try:
                graphid_out.remove(graphid_to)
            except:
                pass
            graphid_out = pd.Series(graphid_out)

            simi_jaccard_out = graphid_out.apply(self.simi_jaccard, args=(graphid_to, graphid_from, "IN"))
            return [np.mean(simi_jaccard_out), np.sum(simi_jaccard_out)]

    def get_node_dict(self):
        # save node data to dictionary, index is "id"
        self.node_dict = self.data_node_info.set_index('id').T.to_dict('series')

    def get_pagerank(self, ids, direct):
        # self.pagerank = self.graph.pagerank() before doing this
        graphid_from, graphid_to = self.lookup_graph_id(ids)
        if direct == "from":
            return self.pagerank_paper[graphid_from]
        if direct == "to":
            return self.pagerank_paper[graphid_to]

    def get_valid_ids(self, data):
        assert type(dir) == list
        valid_ids = set()
        for element in data:
            valid_ids.add(element[0])
            valid_ids.add(element[1])
        return valid_ids

    def get_year(self, ids, return_type="graph_id"):
        # need self.id_graphid
        # input: int: id1 and id2

        year_id1 = self.node_dict[ids[0]]["year"]
        year_id2 = self.node_dict[ids[1]]["year"]
        if return_type == "graph_id":
            if year_id1 >= year_id2:  # TODO: how to deal with papers in same year, I ignore it now
                return year_id1 - year_id2
            else:
                return year_id1 - year_id1

    def init_graph_author(self):
        authors_list = self.get_authors_list()
        self.graph_author.add_vertices(authors_list)
        # add author vertice
        for i in range(self.graph_author.vcount()):
            self.id_graphid_author[self.graph_author.vs["name"][i]] = i
        # add citation edges
        author_citation_edges = self.data_train_positive[["id_source", "id_target"]].apply(self.author_citation_edge,
                                                                                           axis=1)
        list_citation_edges = author_citation_edges.tolist()
        list_citation_edges = [x for x in list_citation_edges if str(x) != "nan"]
        list_citation_edges = [y for x in list_citation_edges for y in x]

        self.graph_author.add_edges(list_citation_edges)

    def init_graph_paper(self):
        # run after `prepare_data`, need self.node_dict
        self.graph_paper.add_vertices(list(self.node_dict.keys()))

        for i in range(self.node_dict.__len__()):
            self.id_graphid_paper[self.graph_paper.vs["name"][i]] = i
        edges = self.data_train_positive[["id_source", "id_target"]].apply(self.get_direct, axis=1)
        self.graph_paper.add_edges(edges.tolist())

    def load_data(self):
        # (0) paper unique ID (integer)
        # (1) publication year (integer)
        # (2) paper title (string)
        # (3) authors (strings separated by ,)
        # (4) name of journal (optional) (string)
        # (5) abstract (string) - lowercased, free of punctuation except intra-word dashes
        self.data_train = pd.read_csv(DIR_TRIAN, names=["id_source", "id_target", "predict"], header=None, sep=" ")
        self.data_test = pd.read_csv(DIR_TEST, names=["id_source", "id_target"], header=None, sep=" ")
        self.data_node_info = pd.read_csv(DIR_NODEINFO, names=["id", "year", "title", "author", "journal", "abstract"],
                                          header=None)

    def lookup_author_pagerank_from(self, edge):
        return self.pagerank_author[edge[0]]

    def lookup_author_pagerank_to(self, edge):
        return self.pagerank_author[edge[1]]

    def lookup_graph_id(self, ids):
        # ids is from data.data_train[["id_source", "id_target"]].apply(..., axis=1)
        graphids = self.get_direct(ids)
        return graphids[0], graphids[1]

    def meanAciteB(self, input_list, mode="out"):
        # input_list: e.g. [(123,456), (234,252)]
        if mode == "out":
            if type(input_list) == list:
                acitebs = list(map(self.AciteB, input_list))
                return [np.mean(acitebs), np.max(acitebs), np.sum(acitebs)]
            else:
                return [0, 0, 0]
        elif mode == "all":
            if type(input_list) == list:
                acitebs = list(map(self.AciteB_all, input_list))
                return [np.mean(acitebs), np.max(acitebs), np.sum(acitebs)]
            else:
                return [0, 0, 0]

    def meanBciteA(self, input_list):
        # input_list: e.g. [(123,456), (234,252)]
        if type(input_list) == list:
            bciteas = list(map(self.BciteA, input_list))
            return [np.mean(bciteas), np.max(bciteas), np.sum(bciteas)]
        else:
            return [0, 0, 0]

    def prepare_data(self, delete=True):
        # title
        # convert to lowercase and tokenize
        tkzd_title = self.data_node_info["title"].apply(lambda x: x.lower().split(" "))
        self.data_node_info["tkzd_title"] = tkzd_title
        # remove stopwords
        tkzd_title_rm_stpwds = self.data_node_info["tkzd_title"].apply(
            lambda x: [token for token in x if token not in STPWDS])
        self.data_node_info["tkzd_title_rm_stpwds"] = tkzd_title_rm_stpwds
        # convert to root or original word
        tkzd_title_rm_stpwds_stem = self.data_node_info["tkzd_title_rm_stpwds"].apply(
            lambda x: [self.stemmer.stem(token) for token in x])
        self.data_node_info["tkzd_title_rm_stpwds_stem"] = tkzd_title_rm_stpwds_stem

        # authors
        re_author = self.data_node_info["author"].apply(
            lambda x: re.sub(r'\(.*?\)|\s|\\\"\"\{|\\\"\"\{\\|\\\\\'|\\\'|\\\"\"|\\\"|\\|\}|\'', "",
                             x) if x is not np.nan else np.nan)  # delete contents in brackets and useless space
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
        tkzd_abstract_rm_stpwds = self.data_node_info["tkzd_abstract"].apply(
            lambda x: [token for token in x if token not in STPWDS])
        self.data_node_info["tkzd_abstract_rm_stpwds"] = tkzd_abstract_rm_stpwds

        # convert to root or original word
        tkzd_abstract_rm_stpwds_stem = self.data_node_info["tkzd_abstract_rm_stpwds"].apply(
            lambda x: [self.stemmer.stem(token) for token in x])
        self.data_node_info["tkzd_abstract_rm_stpwds_stem"] = tkzd_abstract_rm_stpwds_stem

        # save node data to dictionary, index is "id"
        self.node_dict = self.data_node_info.set_index('id').T.to_dict('series')
        if delete:
            del (self.data_node_info)

        print("data prepared")

    def sample(self, prop, load=False):
        # to test code we select sample
        if load:
            features_node = pd.read_csv("features_node", header=0, index_col=0)
            features_index = features_node.index
            self.data_train = self.data_train.ix[features_index]
            self.data_train_positive = self.data_train[self.data_train["predict"] == 1]
        else:
            to_keep = random.sample(range(self.data_train.shape[0]), k=int(round(self.data_train.shape[0] * prop)))
            self.data_train = self.data_train.iloc[to_keep]
            self.data_train_positive = self.data_train[self.data_train["predict"] == 1]

    def simi_jaccard(self, graphid_in, graphid_from, graphid_to, mode):

        graphid_from_neighbors = self.graph_paper.neighbors(graphid_from, mode=mode)
        graphid_in_neighbors = self.graph_paper.neighbors(graphid_in, mode=mode)
        try:
            graphid_from_neighbors.remove(graphid_to)
        except:
            pass
        graphid_in_neighbors.remove(graphid_to)
        simi_jaccard = dist_utils._jaccard_coef(graphid_from_neighbors, graphid_in_neighbors)
        return simi_jaccard

    def similarity(self, ids, graph, method="adamic_adar", direct=False):
        if direct:
            pass
        else:
            i = self.id_graphid_paper[ids[0]]
            j = self.id_graphid_paper[ids[1]]
        if method == "adamic_adar":
            return sum([1.0 / math.log(graph.degree(v)) for v in
                        set(graph.neighbors(i)).intersection(set(graph.neighbors(j)))])

    def split_to_list(self, data, by=" "):
        assert type(data) == list
        return [element[0].split(by) for element in data]


class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=7))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))

            for j, (train_idx, test_idx) in enumerate(folds):
                t0 = time.clock()
                print("\nTraining model %i, fold %i" % ((i + 1), (j + 1)))
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]

                print("    mean of fold prediction: ", np.mean(S_train[test_idx, i], axis=0))
                print("    time for this fold: %.2f sec" % (time.clock() - t0))

            print("mean of model prediction: ", np.mean(S_train[:, i], axis=0))

            S_test[:, i] = S_test_i.mean(1)

        print("\nFirst layer finished\n")
        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred, S_train, S_test


if __name__ == '__main__':
    data = Data(sample=True)
    data.load_data()
    data.sample(prop=1, load=LOAD_SAMPLE)

    if RUN_FOR_FIRST_TIME:

        data.get_node_dict()
        data.prepare_data()
        data.init_graph_paper()
        data.init_graph_author()

        features_network_to = pd.read_csv("features_network_to", header=0, index_col=0)
        features_network_from = pd.read_csv("features_network_from", header=0, index_col=0)
        test_features_network_to = pd.read_csv("test_features_network_to", header=0, index_col=0)
        test_features_network_from = pd.read_csv("test_features_network_from", header=0, index_col=0)

        t0 = time.clock()
        features_node = data.get_batch(0, data.data_train.shape[0], "train", get_item="node")
        features_network_from = data.get_batch(0, data.data_train.shape[0], "train", get_item="network_jaccard_from")
        features_network_to = data.get_batch(0, data.data_train.shape[0], "train", get_item="network_jaccard_to")
        features_network = pd.concat([features_network_from, features_network_to, np.max(
            pd.concat([features_network_from.iloc[:, 0], features_network_to.iloc[:, 0]], axis=1), axis=1), np.max(
            pd.concat([features_network_from.iloc[:, 1], features_network_to.iloc[:, 1]], axis=1), axis=1)], axis=1)
        features_pagerank_paper = data.get_batch(0, data.data_train.shape[0], "train", get_item="pagerank_paper")
        features_pagerank_paper = pd.concat([features_pagerank_paper, np.max(features_pagerank_paper, axis=1)], axis=1)
        features_meanAciteB = data.get_batch(0, data.data_train.shape[0], "train", get_item="mean_aciteb")
        features_pagerank_author = data.get_batch(0, data.data_train.shape[0], "train", get_item="pagerank_author")
        features_pagerank_author = pd.concat([features_pagerank_author, np.max(features_pagerank_author, axis=1)],
                                             axis=1)
        features_adamic_adar_paper = data.get_batch(0, data.data_train.shape[0], "train", get_item="adamic_adar_paper")
        features_dyear = data.get_batch(0, data.data_train.shape[0], "train", get_item="dyear")
        features_author_overlap = data.get_batch(0, data.data_train.shape[0], "train", get_item="author_overlap")
        print(time.clock() - t0)

        features_node.to_csv("features_node")
        features_network_from.to_csv("features_network_from")
        features_network_to.to_csv("features_network_to")
        features_network.to_csv("features_network")
        features_pagerank_paper.to_csv("features_pagerank_paper")
        features_meanAciteB.to_csv("features_meanAciteB")
        features_pagerank_author.to_csv("features_pagerank_author")
        features_adamic_adar_paper.to_csv("features_adamic_adar_paper")
        features_dyear.to_csv("features_dyear")
        features_author_overlap.to_csv("features_author_overlap")

        t0 = time.clock()
        test_features_node = data.get_batch(0, data.data_test.shape[0], "test", get_item="node")
        test_features_network_from = data.get_batch(0, data.data_test.shape[0], "test", get_item="network_jaccard_from")
        test_features_network_to = data.get_batch(0, data.data_test.shape[0], "test", get_item="network_jaccard_to")
        test_features_network = pd.concat([test_features_network_from, test_features_network_to, np.max(
            pd.concat([test_features_network_from.iloc[:, 0], test_features_network_to.iloc[:, 0]], axis=1), axis=1),
                                           np.max(pd.concat([test_features_network_from.iloc[:, 1],
                                                             test_features_network_to.iloc[:, 1]], axis=1), axis=1)],
                                          axis=1)
        test_features_pagerank_paper = data.get_batch(0, data.data_test.shape[0], "test", get_item="pagerank_paper")
        test_features_pagerank_paper = pd.concat(
            [test_features_pagerank_paper, np.max(test_features_pagerank_paper, axis=1)], axis=1)
        test_features_meanAciteB = data.get_batch(0, data.data_test.shape[0], "test", get_item="mean_aciteb")
        test_features_pagerank_author = data.get_batch(0, data.data_test.shape[0], "test", get_item="pagerank_author")
        test_features_pagerank_author = pd.concat(
            [test_features_pagerank_author, np.max(test_features_pagerank_author, axis=1)], axis=1)
        test_features_adamic_adar_paper = data.get_batch(0, data.data_test.shape[0], "test",
                                                         get_item="adamic_adar_paper")
        test_features_dyear = data.get_batch(0, data.data_test.shape[0], "test", get_item="dyear")
        test_features_author_overlap = data.get_batch(0, data.data_test.shape[0], "test", get_item="author_overlap")
        print(time.clock() - t0)

        test_features_node.to_csv("test_features_node")
        test_features_network_from.to_csv("test_features_network_from")
        test_features_network_to.to_csv("test_features_network_to")
        test_features_network.to_csv("test_features_network")
        test_features_pagerank_paper.to_csv("test_features_pagerank_paper")
        test_features_meanAciteB.to_csv("test_features_meanAciteB")
        test_features_pagerank_author.to_csv("test_features_pagerank_author")
        test_features_adamic_adar_paper.to_csv("test_features_adamic_adar_paper")
        test_features_dyear.to_csv("test_features_dyear")
        test_features_author_overlap.to_csv("test_features_author_overlap")

    else:
        t0 = time.clock()
        # -------- features_node --------
        # 0.  jaccard_tkzd_title
        # 1.  dice_tkzd_title
        # 2.  jaccard_tkzd_abstract
        # 3.  jaccard_bigr_tkzd_abstract
        # 4.  jaccard_trigr_tkzd_abstract
        # 5.  dice_tkzd_abstract
        # 6.  dice_bigr_tkzd_abstract
        # 7.  dice_trigr_tkzd_abstract
        # 8.  jaccard_tkzd_abstract_rm_stpwds
        # 9.  jaccard_bigr_tkzd_abstract_rm_stpwds
        # 10. jaccard_trigr_tkzd_abstract_rm_stpwds
        # 11. dice_tkzd_abstract_rm_stpwds
        # 12. dice_bigr_tkzd_abstract_rm_stpwds
        # 13. dice_trigr_tkzd_abstract_rm_stpwds
        features_node = pd.read_csv("features_node", header=0, index_col=0)  # 14 columns

        # -------- features_pagerank_paper --------
        # 14. pagerank_paper_from
        # 15. pagerank_paper_to
        # 16. pagerank_paper_max
        features_pagerank_paper = pd.read_csv("features_pagerank_paper", header=0, index_col=0)  # 3 columns

        # -------- features_meanAciteB --------
        # 17. meanAciteB
        # 18. maxAciteB
        # 19. sumAciteB
        # 20. meanBciteA
        # 21. maxBciteA
        # 22. sumBciteA
        # 23. maxmeancite
        # 24. maxmaxcite
        # 25. maxsumcite
        # 26. meanAciteB_all
        # 27. maxAciteB_all
        # 28. sumAciteB_all
        features_meanAciteB = pd.read_csv("features_meanAciteB", header=0, index_col=0)  # 12 columns

        # -------- features_pagerank_author --------
        # 29. author_pagerank_mean_from
        # 30. author_pagerank_mean_to
        # 31. author_pagerank_max_from
        # 32. author_pagerank_max_to
        # 33. author_pagerank_max_max
        features_pagerank_author = pd.read_csv("features_pagerank_author", header=0, index_col=0)  # 5 columns

        # -------- features_adamic_adar_paper --------
        # 34. adamic_adar_paper
        features_adamic_adar_paper = pd.read_csv("features_adamic_adar_paper", header=None, index_col=0)  # 1 column

        # -------- features_dyear --------
        # 35. dyear
        features_dyear = pd.read_csv("features_dyear", header=None, index_col=0)  # 1 column

        # -------- features_author_overlap --------
        # 36. author_overlap
        features_author_overlap = pd.read_csv("features_author_overlap", header=None, index_col=0)  # 1 column

        # -------- features_network --------
        # 37. network_from_mean
        # 38. network_from_sum
        # 39. network_to_mean
        # 40. network_to_sum
        # 41. network_to_mean_max
        # 42. network_to_sum_max
        features_network = pd.read_csv("features_network", header=0, index_col=0)  # 6 columns

        test_features_node = pd.read_csv("test_features_node", header=0, index_col=0)
        test_features_pagerank_paper = pd.read_csv("test_features_pagerank_paper", header=0, index_col=0)
        test_features_meanAciteB = pd.read_csv("test_features_meanAciteB", header=0, index_col=0)
        test_features_pagerank_author = pd.read_csv("test_features_pagerank_author", header=0, index_col=0)
        test_features_adamic_adar_paper = pd.read_csv("test_features_adamic_adar_paper", header=None, index_col=0)
        test_features_dyear = pd.read_csv("test_features_dyear", header=None, index_col=0)
        test_features_author_overlap = pd.read_csv("test_features_author_overlap", header=None, index_col=0)
        test_features_network = pd.read_csv("test_features_network", header=0, index_col=0)
        print(time.clock() - t0)

    features_network[np.isnan(features_network)] = 0
    test_features_network[np.isnan(test_features_network)] = 0

    training_features = pd.concat(
        [features_node, features_pagerank_paper, features_meanAciteB, features_pagerank_author,
         features_adamic_adar_paper, features_dyear, features_author_overlap, features_network], axis=1)

    # training_features = training_features.iloc[:, [i for i in range(43) if i not in (
    # 19, 22, 25, 28, 37, 38, 39, 40, 41, 42)]]  # weights 1 (n_estimators=50)
    # training_features = training_features.iloc[:, [i for i in range(43) if i not in (
    # 19, 22, 25, 28, 37, 38, 39, 40, 41, 42, 5, 6, 7, 10, 13, 18, 32)]]  # weights 2 (n_estimators=7)

    training_features = training_features.iloc[:,
                        [0, 2, 3, 8, 9, 14, 15, 16, 17, 18, 20, 21, 23, 24, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 40,
                         41, 42]]  # select important features

    training_index = training_features.index
    training_features = preprocessing.scale(training_features)
    labels_array = data.data_train["predict"][training_index]

    testing_features = pd.concat(
        [test_features_node, test_features_pagerank_paper, test_features_meanAciteB, test_features_pagerank_author,
         test_features_adamic_adar_paper, test_features_dyear, test_features_author_overlap, test_features_network],
        axis=1)

    # testing_features = testing_features.iloc[:, [i for i in range(43) if i not in (
    # 19, 22, 25, 28, 37, 38, 39, 40, 41, 42)]]  # weights 1 (n_estimators=50)
    # testing_features = testing_features.iloc[:, [i for i in range(43) if i not in (
    # 19, 22, 25, 28, 37, 38, 39, 40, 41, 42, 5, 6, 7, 10, 13, 18, 32)]]  # weights 2 (n_estimators=7)

    testing_features = testing_features.iloc[:,
                       [0, 2, 3, 8, 9, 14, 15, 16, 17, 18, 20, 21, 23, 24, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 40,
                        41, 42]]  # select important features

    testing_features = preprocessing.scale(testing_features)

    basemodel_1 = xgb.XGBClassifier(learning_rate=0.1, n_estimators=7, max_depth=5, min_child_weight=1, seed=0,
                                    subsample=0.8, colsample_bytree=0.8, gamma=0, reg_alpha=0, reg_lambda=1,
                                    objective="binary:logistic", silent=True, random_state=1, n_jobs=-1)

    basemodel_2 = GradientBoostingClassifier(n_estimators=6, learning_rate=0.1, max_depth=3, random_state=2)
    basemodel_3 = RandomForestClassifier(n_estimators=5, max_features=5, random_state=3, n_jobs=-1)
    basemodel_4 = ExtraTreesRegressor(n_jobs=-1)
    basemodel_5 = LogisticRegression(solver="sag", n_jobs=-1)

    if ENSEMBLE:
        if SUBMIT:
            X_train = training_features
            y_train = labels_array
            X_test = testing_features
        else:
            X_train, X_test, y_train, y_test = train_test_split(training_features, labels_array, test_size=0.2,
                                                                random_state=0)

        stacker = xgb.XGBClassifier(n_estimators=2, n_jobs=-1, subsample=0.8)
        ensemble = Ensemble(n_folds=5, stacker=stacker,
                            base_models=[basemodel_1, basemodel_2, basemodel_3, basemodel_4, basemodel_5])
        ans, s_train, s_test = ensemble.fit_predict(X_train, y_train, X_test)

        if SUBMIT:
            predict = pd.read_csv(PREDICT, sep=",")
            predict["prediction"] = ans
            predict.to_csv("prediction", index=False)
            pd.DataFrame(s_train).to_csv("s_train", index=True)
            pd.DataFrame(s_test).to_csv("s_test", index=True)
        else:
            # show importance
            plot_importance(ensemble.stacker)
            plt.show()

            ans_train = ensemble.stacker.predict(s_train)
            f1_train = f1_score(ans_train, y_train)
            print("F1 accuracy of training: %.10f" % f1_train)
            # calculate f1
            f1 = f1_score(ans, y_test)
            print("F1 accuracy of testing: %.10f" % f1)

    else:
        if SUBMIT:
            X_train = training_features
            y_train = labels_array
            X_test = testing_features
        else:
            X_train, X_test, y_train, y_test = train_test_split(training_features, labels_array, test_size=0.2,
                                                                random_state=0)

        if TUNING:
            if TUNING_PARMS == "n_estimators":
                cv_params = {'n_estimators': [175, 200, 225, 250, 275]}  # best: 225 or 250
                other_params = {'learning_rate': 0.1, 'n_estimators': 150, 'max_depth': 5, 'min_child_weight': 1,
                                'seed': 0,
                                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                                'objective': "binary:logistic"}
            elif TUNING_PARMS == "max_depth & min_child_weight":
                cv_params = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9],
                             'min_child_weight': [1, 2, 3, 4, 5, 6]}  # best: 6, 1
                other_params = {'learning_rate': 0.1, 'n_estimators': 225, 'max_depth': 5, 'min_child_weight': 1,
                                'seed': 0,
                                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                                'objective': "binary:logistic"}

            model = xgb.XGBClassifier(**other_params)
            optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1,
                                         n_jobs=-1)
            optimized_GBM.fit(X_train, y_train)
            evalute_result = optimized_GBM.grid_scores_
            print('evalute result:{0}'.format(evalute_result))
            print('best params: {0}'.format(optimized_GBM.best_params_))
            print('best score: {0}'.format(optimized_GBM.best_score_))
        else:
            # train model
            model = basemodel_1

            if SUBMIT:
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric="error")

            # show importance
            plot_importance(model)
            plt.show()

            # test
            ans = model.predict(X_test)

            if SUBMIT:
                predict = pd.read_csv(PREDICT, sep=",")
                predict["prediction"] = ans
                predict.to_csv("prediction", index=False)

            else:
                ans_train = model.predict(X_train)
                f1_train = f1_score(ans_train, y_train)
                print("F1 accuracy of training: %.10f" % f1_train)
                # calculate f1
                f1 = f1_score(ans, y_test)
                print("F1 accuracy of testing: %.10f" % f1)
