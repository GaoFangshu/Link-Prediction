import igraph
from model import Data

data = Data(sample=True)
data.load_data()
data.sample(prop=1)
data.prepare_data()
data.init_graph()

data.graph

data.graph.pagerank()

data.graph.similarity_dice(mode="OUT", loops=False)


edges = data.data_train[["id_source", "id_target"]].iloc[0:3].apply(data.get_direct, axis=1)


edges.tolist()

g=igraph.Graph(directed=True)




g.add_vertices(list(data.node_dict.keys()))
g.add_edges(edges.tolist())
g.add_edges([('204235', '9812066'),('9905020', '9310158')])

id_graphid= {}

for i in range(data.node_dict.__len__()):
    id_graphid[g.vs["name"][i]] = i
