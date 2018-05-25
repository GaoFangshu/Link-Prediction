import csv
import random
import numpy as np
import nltk

DIR_TRIAN = "social_train.txt"
DIR_TEST = "social_test.txt"
DIR_NODEINFO = "node_information.csv"

#nltk.download('punkt')  # for tokenization
#nltk.download('stopwords')
STPWDS = set(nltk.corpus.stopwords.words("english"))


class Data():
    def __init__(self, sample):
        self.stemmer = nltk.stem.PorterStemmer()
        # the columns of the node data frame below are:
        # (0) paper unique ID (integer)
        # (1) publication year (integer)
        # (2) paper title (string)
        # (3) authors (strings separated by ,)
        # (4) name of journal (optional) (string)
        # (5) abstract (string) - lowercased, free of punctuation except intra-word dashes

        data_train = self.load_data(DIR_TRIAN)
        data_test = self.load_data(DIR_TEST)
        data_node_info = self.load_data(DIR_NODEINFO)

        self.data_tkzd_title = []
        self.data_tkzd_title_rm_stpwds = []
        self.data_tkzd_title_rm_stpwds_stem = []
        self.tkzd_title = []
        self.tkzd_title = []
        self.tkzd_title = []


        self.token_abstract = []


        if sample:
            # to test code we select sample
            to_keep = random.sample(range(len(data_train)), k=int(round(len(data_train) * 0.05)))
            data_train_keep = [data_train[i] for i in to_keep]
            self.training_set = self.split_to_list(data_train_keep)
        else:
            self.training_set = self.split_to_list(data_train)

        #valid_ids = self.get_valid_ids(self.training_set)

        self.testing_set = self.split_to_list(data_test)
        #self.node_info_set = [element for element in data_node_info if element[0] in valid_ids]
        self.node_info_set = data_node_info
        self.node_position = self.add_position(self.node_info_set)

        self.add_tokenized_data()


    def load_data(self, dir):
        with open(dir, "r") as f:
            data = list(csv.reader(f))
        return data

    def get_valid_ids(self, data):
        valid_ids = set()
        for element in data:
            valid_ids.add(element[0])
            valid_ids.add(element[1])
        return valid_ids

    def split_to_list(self, data, by=" "):
        return [element[0].split(by) for element in data]

    def add_position(self, data):
        ids = []
        position = {}
        for element in data:
            position[element[0]] = len(ids)
            ids.append(element[0])
        return position

    def add_feature(self, feature):
        pass

    def add_tokenized_data(self):
        # add lowercase and tokenized title and abstract
        for i in range(len(self.node_info_set)):

            # title
            # convert to lowercase and tokenize
            tkzd_title = self.node_info_set[i][2].lower().split(" ")
            self.data_tkzd_title.append(tkzd_title)
            # remove stopwords
            tkzd_title_rm_stpwds = [token for token in tkzd_title if token not in STPWDS]
            self.data_tkzd_title_rm_stpwds.append(tkzd_title_rm_stpwds)
            # convert to root or original word
            tkzd_title_rm_stpwds_stem = [self.stemmer.stem(token) for token in tkzd_title_rm_stpwds]
            self.data_tkzd_title_rm_stpwds_stem.append(tkzd_title_rm_stpwds_stem)

            # authors
            authors = self.node_info_set[3].split(",")
            self.data_authors = authors

            # journal name


            # (6) abstract (string) - lowercased, free of punctuation except intra-word dashes
            # abstract


            print('')
            #target_title = target_info[2].lower().split(" ")
            #target_title = [token for token in target_title if token not in STPWDS]
            #self.token_target_title = [self.stemmer.stem(token) for token in target_title]



            #source_auth = source_info[3].split(",")
            #target_auth = target_info[3].split(",")

            #overlap_title.append(len(set(source_title).intersection(set(target_title))))
            #temp_diff.append(int(source_info[1]) - int(target_info[1]))
            #comm_auth.append(len(set(source_auth).intersection(set(target_auth))))

    def remove_stpwds(self):
        pass

    def feature_overlap_title(self, observation):
        # number of overlapping words in title
        # `observation` is an observation of training data, e.g. self.training_set[i]
        overlap_title = []
        source = observation[0]    # id of source paper
        target = observation[1]    # id of target paper

        source_info = self.node_info_set[self.node_position[source]]
        target_info = self.node_info_set[self.node_position[target]]

        # convert to lowercase and tokenize
        source_title = source_info[2].lower().split(" ")
        # remove stopwords
        source_title = [token for token in source_title if token not in STPWDS]
        source_title = [self.stemmer.stem(token) for token in source_title]

        target_title = target_info[2].lower().split(" ")
        target_title = [token for token in target_title if token not in STPWDS]
        target_title = [self.stemmer.stem(token) for token in target_title]

        source_auth = source_info[3].split(",")
        target_auth = target_info[3].split(",")

        counter = 0
        for i in range(len(self.training_set)):
            source = self.training_set[i][0]
            target = self.training_set[i][1]

            source_info = self.node_info_set[self.node_position[source]]
            target_info = self.node_info_set[self.node_position[target]]

            # convert to lowercase and tokenize
            source_title = source_info[2].lower().split(" ")
            # remove stopwords
            source_title = [token for token in source_title if token not in STPWDS]
            source_title = [self.stemmer.stem(token) for token in source_title]

            target_title = target_info[2].lower().split(" ")
            target_title = [token for token in target_title if token not in STPWDS]
            target_title = [self.stemmer.stem(token) for token in target_title]

            source_auth = source_info[3].split(",")
            target_auth = target_info[3].split(",")

            overlap_title.append(len(set(source_title).intersection(set(target_title))))
            #temp_diff.append(int(source_info[1]) - int(target_info[1]))
            #comm_auth.append(len(set(source_auth).intersection(set(target_auth))))

            if counter % 10000 == 0:
                print(counter, "training examples processsed")
            counter += 1

    def feature_time_diff(self):
        # temporal distance between the papers
        time_diff = []

    def feature_comm_auth(self):
        # number of common authors
        comm_auth = []

if __name__ == '__main__':


    data = Data(sample=True)

    print('')
