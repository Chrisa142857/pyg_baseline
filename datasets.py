from torch_geometric.datasets import Planetoid, CoraFull, WebKB, Reddit, WikipediaNetwork, Flickr, Actor, ZINC, AQSOL, WikiCS, GNNBenchmarkDataset, HeterophilousGraphDataset, Amazon, Coauthor

def get_data(name, splitid):
    path = f'../data/{name}' 
    if name in ['cora', 'pubmed', 'citeseer']:
        graph = Planetoid(root=path, name=name, split='geom-gcn')
    elif name in ['cornell', 'texas', 'wisconsin']:
        graph = WebKB(path ,name=name)
    elif name == 'corafull':
        graph = CoraFull(root=path)
    elif name == 'reddit':
        graph = Reddit(root=path)
    elif name in ["chameleon", "crocodile", "squirrel"]:
        graph = WikipediaNetwork(root=path, name=name)
    elif name == 'flickr':
        graph = Flickr(root=path)
    elif name in ["roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"]:
        graph = HeterophilousGraphDataset(root=path, name=name)
    graph = graph[0]
    if len(graph.train_mask.shape) == 2:
        train_mask = graph.train_mask.T[splitid].bool()
        val_mask = graph.val_mask.T[splitid].bool()
        test_mask = graph.test_mask.T[splitid].bool()
    elif len(graph.train_mask.shape) == 1:
        train_mask = graph.train_mask.bool()
        val_mask = graph.val_mask.bool()
        test_mask = graph.test_mask.bool()

    graph.train_mask = train_mask
    graph.val_mask = val_mask
    graph.test_mask = test_mask

    return graph