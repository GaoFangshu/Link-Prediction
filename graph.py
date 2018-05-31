import igraph
from model import Data

data = Data(sample=True)
data.load_data()
data.sample(prop=0.01)


#
# def get_ddirect(ids):
#     # input: int: id1 and id2
#     # output: tuple: (from_id, to_id)
#     # year(from_id) >= year(to_id)
#     print(ids)
#     year_id1 = data.data_node_info["year"][data.data_node_info["id"] == ids[0]].iloc[0]   # TODO: risky because id may not unique
#     year_id2 = data.data_node_info["year"][data.data_node_info["id"] == ids[1]].iloc[0]
#     print(year_id1)
#     print(year_id2)
#     if year_id1 >= year_id2:  # TODO: how to deal with papers in same year, I ignore it now
#         return (ids[0], ids[1])
#     else:
#         return (ids[1], ids[0])

data.data_train[["id_source", "id_target"]].iloc[0:2].apply(data.get_direct, axis=1)


g=igraph.Graph(directed=True)

g.add_vertices(["1","3","5","7"])
g.add_edges([("1","3"), ("7","5"), ("3","1")])