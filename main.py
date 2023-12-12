from datasets import get_data
from models import ToyMPNN
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, SAGEConv, TransformerConv
import torch
from tqdm import trange
import numpy as np

lr = 1e-3
splits = 10
epoch = 1000
verb_inter = 100
dname = 'cora'
device = 'cuda:0'
nlayer, hidch = 4, 768

def main_small_graph(CONV_OP):
    accs = []
    for spliti in trange(splits):
        data = get_data(dname, spliti).to(device)
        inch, outch = data.x.shape[1], int(data.y.max()+1)
        gnn = ToyMPNN(CONV_OP, nlayer, inch, outch, hidch).to(device)
        optimizer = torch.optim.AdamW(gnn.parameters(), lr=lr)
        lr_scheduler = get_lr_scheduler_with_warmup(optimizer=optimizer, num_warmup_steps=None, num_steps=epoch, warmup_proportion=0.005)
        loss_fn = torch.nn.CrossEntropyLoss()
        best_val = 0
        for e in range(epoch):
            gnn.train()
            train_loss, train_correct_num = one_step(gnn, data, data.train_mask, optimizer, lr_scheduler, loss_fn, istrain=True)
            train_acc = (train_correct_num / data.train_mask.sum()).item()
            gnn.eval()
            with torch.no_grad():
                val_loss, val_correct_num = one_step(gnn, data, data.val_mask, optimizer, lr_scheduler, loss_fn)
                val_acc = (val_correct_num / data.val_mask.sum()).item()
                test_loss, test_correct_num = one_step(gnn, data, data.test_mask, optimizer, lr_scheduler, loss_fn)
                test_acc = (test_correct_num / data.test_mask.sum()).item()
            # if e % verb_inter == 0:
            #     print(f"Split {spliti+1:02d} Epoch {e+1:04d} \t loss:\t({train_loss:.5f}, {val_loss:.5f}, {test_loss:.5f}) \t acc:\t({train_acc*100:.2f}, {val_acc*100:.2f}, {test_acc*100:.2f}) (train, val, test)")
            if val_acc > best_val: 
                best_val = val_acc
                best_acc = test_acc
        accs.append(best_acc)
    return accs

def one_step(model, graph, mask, optimizer, scheduler, loss_fn, istrain=False):
    logits = model(graph.x, graph.edge_index)
    loss = loss_fn(input=logits[mask], target=graph.y[mask])
    if istrain:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return loss.detach().cpu().item(), (logits.argmax(1)[mask]==graph.y[mask]).detach().cpu().sum()


def get_lr_scheduler_with_warmup(optimizer, num_warmup_steps=None, num_steps=None, warmup_proportion=None,
                                 last_step=-1):

    if num_warmup_steps is None and (num_steps is None or warmup_proportion is None):
        raise ValueError('Either num_warmup_steps or num_steps and warmup_proportion should be provided.')

    if num_warmup_steps is None:
        num_warmup_steps = int(num_steps * warmup_proportion)

    def get_lr_multiplier(step):
        if step < num_warmup_steps:
            return (step + 1) / (num_warmup_steps + 1)
        else:
            return 1

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=last_step)

    return lr_scheduler


if __name__ == "__main__":
    accs = main_small_graph(GCNConv)
    print(f"ACC: {np.mean(accs)} +/- {np.std(accs)} (GCN)")
    accs = main_small_graph(GATConv)
    print(f"ACC: {np.mean(accs)} +/- {np.std(accs)} (GAT)")
    accs = main_small_graph(SAGEConv)
    print(f"ACC: {np.mean(accs)} +/- {np.std(accs)} (SAGE)")
    accs = main_small_graph(TransformerConv)
    print(f"ACC: {np.mean(accs)} +/- {np.std(accs)} (TransformerConv)")
