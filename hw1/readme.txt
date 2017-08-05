generate_graph():
    This function generates a complete graph with 10 nodes and all edges are assigned weights randomly. The function returns the graph.
balance_graph(g,num_b):
    This function check a random triad and make it balance if it is unbalanced. Also it will call a function to recalculate the number of balanced triads in a graph if the graph is modified. It returns the new graph and new number of balanced triads.
balance_triads(g):
    This function calculates how many triads are balanced. It returns the number of balanced triads.
main:
    generates 100 graph and calculate the number of balanced triads for 1000000 iteration. Calculate the average and use matplotlib to plot the graph.
Results:
    The average fraction of balanced triads starts at about 0.5 (pretty close to the expected value), and after about 10^5 iteration it goes to 1, which means the graph is balanced.There’s lots of ups and downs in the graph. Because when changing one edge, it will affect (n-2) triads, where n is the total number of nodes. Some of the triads will change from balanced to unbalanced and some will do the opposite.