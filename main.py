import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


cmap = ["#8dd3c7", "#80b1d3", "#fb8072",
        "#b3de69",
        #"#ffffb3",
        "#fdb462",
        "#bebada",
        "#fccde5",
        "#d9d9d9",
        "#bc80bd",
        "#ccebc5",
        "#b15928",
        "#1f78b4"]

cmap_border = [adjust_lightness(c, 0.4) for c in cmap]

def draw_network(G, pos, **kwargs):
    modules = {node: module - 1
               for node, module in nx.get_node_attributes(G, "modules").items()}
    num_modules = max(modules.values()) + 1

    #cmap = sns.color_palette("deep", num_modules)
    #cmap_border = sns.color_palette("dark", num_modules)

    edge_color = []
    for source, target in G.edges:
        source_module = modules[source]
        target_module = modules[target]
        
        if source_module == target_module:
            edge_color.append(adjust_lightness(cmap[source_module], 0.5))
        else:
            edge_color.append("#cccccc")
        
    nx.draw_networkx_edges(G, pos, alpha=1, edge_color=edge_color, **kwargs)

    for i in (1, 2):
        nodes = [node for node, type_ in G.nodes.data("type") if type_ == i]

        colors = [cmap[modules[node]] for node in nodes]
        edge_colors = [cmap_border[modules[node]] for node in nodes]

        nx.draw_networkx_nodes(G,
                               pos,
                               nodelist=nodes,
                               node_color=colors,
                               edgecolors=edge_colors,
                               node_shape="s" if i == 1 else "o",
                               **kwargs)

