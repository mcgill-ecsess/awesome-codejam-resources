import osmnx as ox
S = ox.gdf_from_place("Island of Montreal, Canada")
ox.plot_shape(S)
