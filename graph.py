import igraph
import pandas as pd
import numpy as np
import time
from model import Data

data = Data(sample=True)
data.load_data()
data.sample(prop=1, load=True)
# data.get_node_dict()
data.prepare_data(delete=False)
data.data_node_info = data.data_node_info[['id', 'year', 'title', 'author', 'tkzd_author']]

authors_list = []

for key, value in data.node_dict.items():
    if type(value['tkzd_author']) == list:
        authors_list.extend(value['tkzd_author'])

print(len(authors_list))

authors_list = list(set(authors_list))

print(len(authors_list))

graph_author = igraph.Graph(directed=True)
graph_author.add_vertices(authors_list)




data.data_node_info['tkzd_author'][0:3].apply(lambda x: authors_list.extend(x))

data.init_graph()

data.graph_paper

data.graph_paper.pagerank()

data.graph_paper.similarity_dice(vertices=(10327, 25399), mode="OUT", loops=False)


#def



def get_graph_simi(ids, train=True):
    # ids is from data.data_train[["id_source", "id_target"]].apply(..., axis=1)
    graphids = data.get_direct(ids)
    # graphid_from = graphids[0]    # from
    graphid_to = graphids[1]    # to
    # if the training observation has connection, we need delete it first.
    if train:
        # num_edge = data.graph.get_eid(v1=graphid_from, v2=graphid_to, directed=False, error=False)
        # if num_edge != -1:
        #     # delete edge
        #     data.graph.delete_edges([(graphid_from, graphid_to)])
        graphid_in = pd.Series(data.graph_paper.neighbors(graphid_to, mode="IN"))
        simi_dice_in = graphid_in.apply(simi_dice, args=(graphid_to,))
        # if num_edge != -1:
        #     # add edge back
        #     data.graph.add_edges([(graphid_from, graphid_to)])
        return np.mean(simi_dice_in)
    else:
        graphid_in = pd.Series(data.graph_paper.neighbors(graphid_to, mode="IN"))
        simi_dice_in = graphid_in.apply(simi_dice, args=(graphid_to,))
        return np.mean(simi_dice_in)


def get_graph_simi(ids, train=True):
    # ids is from data.data_train[["id_source", "id_target"]].apply(..., axis=1)
    graphids = data.get_direct(ids)
    graphid_from = graphids[0]
    graphid_to = graphids[1]
    # if the training observation has connection, we need delete it first.
    if train:
        # num_edge = data.graph.get_eid(v1=graphid_from, v2=graphid_to, directed=False, error=False)
        # if num_edge != -1:
        #     # delete edge
        #     data.graph.delete_edges([(graphid_from, graphid_to)])
        graphid_in = pd.Series(data.graph_paper.neighbors(graphid_to, mode="IN"))
        simi_dice_in = graphid_in.apply(simi_dice, args=(graphid_from,))
        # if num_edge != -1:
        #     # add edge back
        #     data.graph.add_edges([(graphid_from, graphid_to)])
        return np.mean(simi_dice_in)
    else:
        graphid_in = pd.Series(data.graph_paper.neighbors(graphid_to, mode="IN"))
        simi_dice_in = graphid_in.apply(simi_dice, args=(graphid_from,))
        return np.mean(simi_dice_in)

def get_graph_simi(ids):
    # ids is from data.data_train[["id_source", "id_target"]].apply(..., axis=1)
    graphids = data.get_direct(ids)
    graphid_from = graphids[0]
    graphid_to = graphids[1]

    graphid_in = pd.Series(data.graph_paper.neighbors(graphid_to, mode="IN"))
    simi_dice_in = graphid_in.apply(simi_dice, args=(graphid_from,))
    return np.mean(simi_dice_in)

def simi_dice(graphid_in, graphid_from):
    # calculate
    simi_dice = data.graph_paper.similarity_jaccard(vertices=(graphid_in, graphid_from), mode="OUT", loops=False)[0][1]
    return simi_dice

data.data_train[["id_source", "id_target"]].iloc[0:1000].apply(get_graph_simi, axis=1)


t0 = time.clock()
test = data.data_train[["id_source", "id_target"]].apply(get_graph_simi, axis=1)
print(time.clock() - t0)    # 1.160415999999998 sec

#
# edges.tolist()
#
# g=igraph.Graph(directed=True)
#
#
# data.graph.add_edges([(9905020, 9310158)])
#
# g.add_vertices(list(data.node_dict.keys()))
# g.add_edges(edges.tolist())
# g.add_edges([('204235', '9812066'),('9905020', '9310158')])
#
# id_graphid= {}

# for i in range(data.node_dict.__len__()):
#     id_graphid[g.vs["name"][i]] = i
