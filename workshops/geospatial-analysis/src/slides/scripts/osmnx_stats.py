import osmnx as ox 
G = ox.graph_from_address("Arc de Triomphe, Paris")
stats = ox.basic_stats(G)