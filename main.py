import argparse
from surrogate_model import *
from scipy import sparse
import deeprobust.graph.defense
from deeprobust.graph.defense import *
from deeprobust.graph.data import Dataset
import json
import time
from noisy_gcn import Noisy_GCN

MODIFIED_PATH = r'ModifiedGraph'

def get_gpu_mem_info(gpu_id=0):
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free

def test_defence(args, features_tensor, ori_adj_tensor, mod_adj_tensor,
                 labels_tensor, model_name, idx_train, idx_test):

    # convert data type
    labels = labels_tensor.detach().cpu().numpy()
    features = features_tensor.detach().cpu().to_sparse()
    mod_adj = mod_adj_tensor.detach().cpu().to_sparse()
    ori_adj = ori_adj_tensor.detach().cpu().to_sparse()
    row, col, data, shape = features._indices()[0], features._indices()[1], features._values(), features.size()
    features = sp.csr_matrix((data, (row, col)), shape=shape)
    row, col, data, shape = mod_adj._indices()[0], mod_adj._indices()[1], mod_adj._values(), mod_adj.size()
    mod_adj = sp.csr_matrix((data, (row, col)), shape=shape)
    row, col, data, shape = ori_adj._indices()[0], ori_adj._indices()[1], ori_adj._values(), ori_adj.size()
    ori_adj = sp.csr_matrix((data, (row, col)), shape=shape)

    memo = 0
    if model_name in ['GCN', 'SGC', 'APPNP', 'GCNII']:
        model = BasicGNN(args, nnodes=mod_adj_tensor.shape[0], nfeat=features_tensor.shape[1], nclass=labels.max() + 1,
                         nhid=args.hidden, dropout=args.dropout, device=device, finetune_epoch=0, use_mp=True,
                         basic_model=model_name)
        memo = 0#_, memo, _ = get_gpu_mem_info(int(args.gpu))
        if args.attacker in ['pgd']:
            model.fit(features_tensor, ori_adj_tensor, labels_tensor, idx_train, idx_test)
        else:
            model.fit(features_tensor, mod_adj_tensor, labels_tensor, idx_train, idx_test)
        output, acc_global = model.test(features_tensor, mod_adj_tensor, labels_tensor, idx_test)
        acc_target = 0
    elif model_name in ['GCN_SF', 'SGC_SF', 'APPNP_SF', 'GCNII_SF',
                        'GCN_PST', 'SGC_PST', 'APPNP_PST', 'GCNII_PST',
                        'GCN_PMLP', 'SGC_PMLP', 'APPNP_PMLP', 'GCNII_PMLP',
                        'GCN-Jacard_PST', 'GCN-Jacard_SF', 'NoisyGCN_PST',
                        'NoisyGCN_SF']:
        basic_model_name, trick_name = model_name.split('_')[0], model_name.split('_')[1]
        model = BasicGNN(args, nnodes=mod_adj_tensor.shape[0], nfeat=features_tensor.shape[1], nclass=labels.max() + 1,
                         nhid=args.hidden, dropout=args.dropout, device=device, finetune_epoch=3, use_mp=False,
                         basic_model=basic_model_name, trick=trick_name)
        memo = 0#_, memo, _ = get_gpu_mem_info(int(args.gpu))
        if args.attacker in ['pgd']:
            model.fit(features_tensor, ori_adj_tensor, labels_tensor, idx_train, idx_test)
        else:
            model.fit(features_tensor, mod_adj_tensor, labels_tensor, idx_train, idx_test)
        output, acc_global = model.test(features_tensor, mod_adj_tensor, labels_tensor, idx_test)
        acc_target = 0
    elif model_name == 'ReconGNN':
        model = ReconGNN(args, nnodes=mod_adj_tensor.shape[0], nfeat=features_tensor.shape[1], nclass=labels.max() + 1,
                       nhid=args.hidden, dropout=args.dropout, device=device, finetune_epoch=3)
        model.fit(features_tensor, mod_adj_tensor, labels_tensor, idx_train, idx_val)
        #_, memo, _ = get_gpu_mem_info(int(args.gpu))
        memo = 0
        output, acc_global = model.test(features_tensor, mod_adj_tensor, labels_tensor, idx_test)
        acc_target = 0
    elif model_name == 'Mir-GNN':
        model = BasicGNN(args, nnodes=mod_adj_tensor.shape[0], nfeat=features_tensor.shape[1], nclass=labels.max() + 1,
                         nhid=args.hidden, dropout=args.dropout, device=device, finetune_epoch=3)
        model.fit(features_tensor, mod_adj_tensor, labels_tensor, idx_train, idx_val)
        #_, memo, _ = get_gpu_mem_info(int(args.gpu))
        memo = 0
        output, acc_global = model.test(features_tensor, mod_adj_tensor, labels_tensor, idx_test)
        acc_target = 0
        #Visualization().embedding_visualization(model_name, args.dataset, output[idx_test], labels[idx_test])
    elif model_name == 'NoisyGCN':
        best_acc_val = 0
        for beta in np.arange(0.1, args.beta_max, args.beta_min):
            time_start = time.time()
            classifier = Noisy_GCN(nfeat=features.shape[1], nhid=args.hidden,
                                   nclass=labels.max().item() + 1, dropout=args.dropout,
                                   device=device, noise_ratio_1=beta)

            classifier = classifier.to(device)
            classifier.fit(features, mod_adj, labels, idx_train, train_iters=200,
                           idx_val=idx_val,
                           idx_test=idx_test,
                           verbose=False, attention=False)
            # _, memo, _ = get_gpu_mem_info(int(args.gpu))
            classifier.eval()

            # Validation Acc
            acc_val, _ = classifier.test(idx_val)
            acc_global, acc_target = 0, 0
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                acc_global, _ = classifier.test(idx_test)
                acc_global = acc_global.item()
            time_during = time.time() - time_start
            model = classifier

    elif model_name == 'RGCN':
        time_start = time.time()
        model = RGCN(nnodes=mod_adj.shape[0], nfeat=features.shape[1], nclass=labels.max() + 1,
                     nhid=args.hidden, dropout=args.dropout, device=device)
        model.fit(features, mod_adj, labels, idx_train, idx_val, train_iters=args.epochs, verbose=False)
        #_, memo, _ = get_gpu_mem_info(int(args.gpu))
        time_during = time.time() - time_start
        print(time_during)
        acc_global = model.test(idx_test)
        acc_target = 0

    elif model_name == 'Simp-GCN':
        model = SimPGCN(nnodes=features.shape[0], nfeat=features.shape[1], nclass=labels.max() + 1,
                    nhid=args.hidden, dropout=args.dropout, device=device)
        model.to(device)
        model.fit(features, mod_adj, labels, idx_train, idx_val, train_iters=args.epochs, verbose=False)
        #_, memo, _ = get_gpu_mem_info(int(args.gpu))
        acc_global = model.test(idx_test)
        acc_target = 0

    elif model_name == 'ProGNN':
        labels = torch.tensor(labels,dtype=torch.int64).to(device)
        gnn = deeprobust.graph.defense.GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=int(labels.max().item() + 1),
            dropout=args.dropout, device=device)
        gnn.to(device)
        model = ProGNN(gnn, args, device)
        model.fit(features_tensor, mod_adj_tensor, labels, idx_train, idx_val)
        #_, memo, _ = get_gpu_mem_info(int(args.gpu))
        acc_global = model.test(features_tensor, labels, idx_test)
        acc_target = 0

    elif model_name == 'GCN-Jacard':
        model = GCNJaccard(nfeat=features.shape[1], nclass=labels.max() + 1,
                           nhid=args.hidden, dropout=args.dropout, device=device)
        model = model.to(device)
        model.fit(features, mod_adj, labels, idx_train, idx_val, train_iters=args.epochs, threshold=0.01, verbose=False)
        #_, memo, _ = get_gpu_mem_info(int(args.gpu))
        model.eval()
        acc_global = model.test(idx_test)
        acc_target = 0

    elif model_name == 'GCN_SVD':
        # Setup Defense Model
        model = GCNSVD(nfeat=features.shape[1], nclass=labels.max() + 1,
                       nhid=args.hidden, dropout=args.dropout, device=device)

        model = model.to(device)
        model.fit(features, mod_adj, labels, idx_train, idx_val, k=args.k, verbose=False)
        #_, memo, _ = get_gpu_mem_info(int(args.gpu))
        model.eval()
        acc_global = model.test(idx_test)
        acc_target = 0

    return acc_global, acc_target, memo, model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fold_num', type=int, default=10,
                        help='Number of fold to test.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--ptb_rate', type=float, default=0.2, help='pertubation rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    # parameters for ProGNN
    parser.add_argument('--symmetric', action='store_true', default=False,
                        help='whether use symmetric matrix in ProGNN')
    parser.add_argument('--lr_adj', type=float, default=0.01,
                        help='lr for ProGNN')
    parser.add_argument('--alpha', type=float, default=5e-4,
                        help='weight of l1 norm in ProGNN')
    parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
    parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
    parser.add_argument('--lambda_', type=float, default=0, help='weight of feature smoothing')
    parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
    parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
    parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
    parser.add_argument('--only_gcn', action='store_true',
                        default=False, help='test the performance of gcn without other components')
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug mode')
    # parameters for GCN_SVD
    parser.add_argument('--k', type=int, default=42, help='Truncated Components.')
    # parameters for NoisyGCN
    parser.add_argument('--beta_max', type=float, default=0.11)
    parser.add_argument('--beta_min', type=float, default=0.01)
    # other parameters
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--step', type=int, default=100)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--attacker', type=str, default='meta', choices=['grad','meta'])
    parser.add_argument('--dataset', type=str, default='citeseer',
                        choices=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('-target_class', type=int, default=0, help='target class')

    args = parser.parse_args()
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:%s' % args.gpu if torch.cuda.is_available() else "cpu")

    data = Dataset(root='./data/', name=args.dataset, seed=args.seed, setting='nettack')#nettack #prognn
    adj, features, labels = data.adj, data.features, data.labels
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    if device != 'cpu':
        adj = adj.to(device)
        features = features.to(device)
        labels = labels.to(device)

    # test robust performance
    vanilla_models = ['GCN', 'SGC', 'APPNP', 'GCNII','GCN_SF', 'SGC_SF', 'APPNP_SF', 'GCNII_SF',
                   'GCN_PST', 'SGC_PST', 'APPNP_PST', 'GCNII_PST']
    robust_models = ['GCN-Jacard', 'GCN-Jacard_PST', 'GCN-Jacard_SF', 'NoisyGCN','NoisyGCN_PST','NoisyGCN_SF',
                     'EvenNet', 'EvenNet_PST', 'EvenNet_SF', 'Simp-GCN', 'Simp-GCN_PST', 'Simp-GCN_SF']
    test_models = ['GCN_PST', 'GCN_SF']
    ptb_rates = ['0%', '5%', '10%', '20%']

    for pr in ptb_rates:
        pr = pr.split('%')[:-1][0]
        pr = int(pr)/100.
        if pr<=0:
            mod_adj = adj
        else:
            mod_adj = sparse.load_npz("ModifiedGraph/%s_%s_adj_%s.npz" % (args.dataset,args.attacker,pr)) 
            mod_adj,_,_ = preprocess(mod_adj, features, labels, preprocess_adj=False)
        mod_adj = mod_adj.to(device)
        model_acc_global_mean, model_acc_target_mean, model_acc_global_std, model_acc_target_std, \
            model_running_time, model_memo = [], [], [], [], [], []
        for model_name in test_models:
            acc_global_mean, acc_target_mean, memory, out, time_cost = [], [], 0, 0, 0
            for r in range(args.fold_num):
                time_start = time.time()
                acc_global, acc_target, memo, model = test_defence(args, features, adj, mod_adj, labels,
                                                                   model_name, idx_train, idx_test)
                acc_global_mean.append(acc_global)
                acc_target_mean.append(acc_target)
                print(model_name, acc_global, acc_target)
                time_during = time.time() - time_start
                time_cost += float(time_during)
                memory += memo
            time_cost/=args.fold_num
            memory/=args.fold_num
            model_acc_global_mean.append(np.mean(np.array(acc_global_mean)))
            model_acc_target_mean.append(np.mean(np.array(acc_target_mean)))
            model_acc_global_std.append(np.std(np.array(acc_global_mean)))
            model_acc_target_std.append(np.std(np.array(acc_target_mean)))
            model_running_time.append(time_cost)
            model_memo.append(memory)
        print(test_models)
        print("Avg. global accuracy: ", model_acc_global_mean)
        print("Std. global accuracy: ", model_acc_global_std)
        print("Avg. runing time: ", model_running_time)

        for model_idx in range(len(test_models)):
            model_name = test_models[model_idx]
            log_json = {}
            log_json['model_acc'] = model_acc_global_mean[model_idx]
            log_json['model_std'] = model_acc_global_std[model_idx]
            log_json['model_running_time'] = model_running_time[model_idx]

            with open('./checkpoints/'+model_name+'_'+args.attacker+'_'+args.dataset+
                      '_'+str(pr)+'.json', 'w') as f:
                f.write(json.dumps(log_json))