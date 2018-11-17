import osmnx as ox
G = ox.graph_from_place("Verdun, Montreal, Canada", network_type="all")
ox.plot_graph(G)
