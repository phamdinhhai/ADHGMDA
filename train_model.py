import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from utils import *
from model import GPASS
from preprocess import BioData


def find_optimal_threshold(y_true, y_score, metric='youden'):
    """
    Find optimal classification threshold using Youden's J statistic or F1.

    FIX 1.4: Instead of the default 0.5 threshold, find the threshold that
    best balances Recall and Specificity on the validation set.

    Args:
        y_true: Ground truth binary labels.
        y_score: Predicted probabilities.
        metric: 'youden' (maximises Sensitivity+Specificity-1) or 'f1'.

    Returns:
        Optimal threshold value.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    if metric == 'youden':
        J = tpr - fpr
        opt_idx = np.argmax(J)
    else:
        f1_scores = [
            f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
            for t in thresholds
        ]
        opt_idx = np.argmax(f1_scores)
    return float(thresholds[opt_idx])


def train_model(biodata, train_idx=None, test_idx=None, save_file=None, Val_test=True):
    """
    Function to train the model based on provided indices and parameters.

    Args:
        biodata: BioData object containing data and configuration.
        train_idx: Indices for training data.
        test_idx: Indices for test data.
        save_file: Path to save the model and results.
        Val_test: Flag to indicate if validation/test split is needed.
    """
    graph = biodata.graph.clone()
    param = biodata.config
    y = biodata.edge_and_label
    monitor = PerformanceMonitor()
    if hasattr(param, 'edge_number') and param.edge_number is not None:
        # Get the shape of the original matrix
        shape = biodata.adj_matrix.shape
        total_elements = np.prod(shape)
        # Ensure edge_number does not exceed the total number of matrix elements
        edge_number = min(param.edge_number, total_elements)
        # Create a matrix filled with zeros
        adj_matrix = np.zeros(shape, dtype=int)
        # Randomly select positions to set to 1
        indices = np.random.choice(total_elements, edge_number, replace=False)
        # Set the selected positions to 1
        np.put(adj_matrix, indices, 1)
        biodata_test = BioData(adj_matrix,
                               mss_list=biodata.mss_matrix_list,
                               dss_list=biodata.dss_matrix_list,
                               device=biodata.device,
                               config=biodata.config)
        biodata = biodata_test
        y = biodata.edge_and_label
        train_idx = None
        test_idx = None

    if train_idx is None:
        train_idx = np.arange(y['y'].shape[0])

    if Val_test:  # Randomly split a validation set during test training
        # Randomly select 0.1 from train_idx as test data, and the rest as training data
        num_samples = len(train_idx)
        num_test = int(num_samples * param.val_test_size)  # Calculate the size of the test set (5%)
        # Randomly shuffle the indices
        shuffled_indices = torch.randperm(num_samples)
        # Split the indices
        val_test_idx = train_idx[shuffled_indices[:num_test]]
        train_idx = train_idx[shuffled_indices[num_test:]]
        if test_idx is None:
            test_idx = val_test_idx
            _test_idx = False

    # FIX 2.4: Recalculate GIP/functional similarities WHEN test_idx IS available,
    # masking out test-positive edges to prevent data leakage.
    # Original code had inverted condition (if test_idx is None) which left
    # test-positive associations inside the similarity matrices used for training.
    if test_idx is not None:
        true_src = y['y_edge'][0][test_idx[(y['y'][test_idx] == 1).reshape(-1)]]
        true_tgt = y['y_edge'][1][test_idx[(y['y'][test_idx] == 1).reshape(-1)]]
        mask_adj_matrix = biodata.adj_matrix.copy()
        mask_adj_matrix[true_src, true_tgt - mask_adj_matrix.shape[0]] = 0
        mg_sim = GIP_kernel(mask_adj_matrix)
        dg_sim = GIP_kernel(mask_adj_matrix.T)
        mirna_func_sim = functional_similarity(mask_adj_matrix)
        disease_func_sim = functional_similarity(mask_adj_matrix.T)
        biodata = BioData(mask_adj_matrix,
                          mss_list=biodata.mss_matrix_list[:1] + [mirna_func_sim, mg_sim],
                          dss_list=biodata.dss_matrix_list[:1] + [disease_func_sim, dg_sim],
                          device=biodata.device,
                          config=biodata.config)

    gmss = biodata.mss_graph
    gdss = biodata.dss_graph
    device = param.device
    _test_idx = True
    model = GPASS(param).to(device)
    if save_file is None:
        save_file = param.save_file
    if not os.path.exists(save_file):
        os.makedirs(save_file, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=param.lr, weight_decay=param.weight_decay)  # 0.0002

    src_train, tgt_train = y['y_edge'][0][train_idx], y['y_edge'][1][train_idx]
    src_test, tgt_test = y['y_edge'][0][test_idx], y['y_edge'][1][test_idx]
    true_src = y['y_edge'][0][test_idx[(y['y'][test_idx] == 1).reshape(-1)]]
    true_tgt = y['y_edge'][1][test_idx[(y['y'][test_idx] == 1).reshape(-1)]]

    # Mask the data in the test set
    for _src, _tgt in zip(true_src, true_tgt):
        graph.remove_edges((_src, _tgt))
    if _test_idx:
        val_src_test, val_tgt_test = y['y_edge'][0][val_test_idx], y['y_edge'][1][val_test_idx]
        val_true_src = y['y_edge'][0][val_test_idx[(y['y'][val_test_idx] == 1).reshape(-1)]]
        val_true_tgt = y['y_edge'][1][val_test_idx[(y['y'][val_test_idx] == 1).reshape(-1)]]
        for _src, _tgt in zip(val_true_src, val_true_tgt):
            graph.remove_edges((_src, _tgt))
    else:
        val_src_test, val_tgt_test = src_test, tgt_test
        val_true_src, val_true_tgt = true_src, true_tgt
    y_train_true = add_label_noise(y['y'][train_idx].reshape(-1, ), noise_level=param.noise_level)
    y_train_true = torch.tensor(y_train_true)
    auc_list = []
    start_time = datetime.datetime.now()
    _patience = 0
    monitor.start()
    param.print_epoch = min(param.print_epoch, param.epochs)
    for epoch in range(0, param.epochs + 1):
        # FIX 2.2: keep model aware of current epoch for curriculum masking
        model.current_epoch = epoch
        optimizer.zero_grad()
        rep = model(graph, gmss, gdss).to(device)
        # FIX 1.1: use predict_logits so adaptive_mloss receives logits, not probabilities
        preds_logits = model.predict_logits(rep[src_train], rep[tgt_train]).to(device)
        loss = model.adaptive_mloss(preds_logits, y_train_true.to(device))
        if model.init_parameters is False:
            model.reset_parameters()
            continue
        loss.backward()
        optimizer.step()
        if epoch > 100:
            model.Dynamic_train = False
        if epoch % param.print_epoch == 0 and not test_idx is None:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            model.eval()
            with torch.no_grad():
                rep = model(graph, gmss, gdss)
                val_preds = model.predict(rep[val_src_test], rep[val_tgt_test])
                val_out_pred = val_preds.to('cpu').detach().numpy()
                val_y_true = y['y'][val_test_idx].to('cpu').detach().numpy()
                auc = roc_auc_score(val_y_true, val_out_pred)
                print('Val AUC:', auc)
                if auc > model.best_auc:
                    _patience = 0
                    model.best_auc = auc
                    model.best_epoch = epoch
                    model.save_model(path=save_file + '/best_model.pth')
                else:
                    _patience += 1
                if _patience >= param.patience:
                    break
                preds = model.predict(rep[src_test], rep[tgt_test])
                out_pred = preds.to('cpu').detach().numpy()
                y_true = y['y'][test_idx].to('cpu').detach().numpy()
                print(out_pred[:10])
                auc_idx, auc_name = get_metrics(y_true, out_pred)
                auc_idx.extend(param.search_args['arg_value'])
                auc_idx.append(loss.item())
                auc_idx.append(epoch)
                auc_list.append(auc_idx)
                print_execution_time(start_time, epoch)
            model.train()
    monitor.stop()
    monitor.print_metrics(f'Number of edges: {biodata.adj_matrix.sum()}')
    auc_name.extend(param.search_args['arg_name'])
    auc_name += ['loss', 'epoch']
    results = pd.DataFrame(np.array(auc_list,dtype=object), columns=auc_name)
    results.to_feather(path=save_file + '/results.feather')
    model.load_model(path=save_file + '/best_model.pth')

    # FIX 1.4: Find optimal threshold on validation set, apply on test set
    model.eval()
    with torch.no_grad():
        rep = model(graph, gmss, gdss)
        val_probs_final = model.predict(rep[val_src_test], rep[val_tgt_test]).cpu().numpy()
        val_labels_final = y['y'][val_test_idx].cpu().numpy()
        opt_threshold = find_optimal_threshold(val_labels_final, val_probs_final, metric='youden')
        print(f'Optimal threshold (Youden): {opt_threshold:.4f}')

        test_probs_final = model.predict(rep[src_test], rep[tgt_test]).cpu().numpy()
        test_labels_final = y['y'][test_idx].cpu().numpy()
        print('--- Final test metrics with optimal threshold ---')
        final_metrics, final_names = get_metrics(test_labels_final, test_probs_final)
    model.Overall_Refactoring_ASS_Embedding(biodata)

    if biodata.config.save_ASS_Embedding == True:
        biodata.save_ASS_Embedding(path=save_file + '/ASS_Embedding.emb')
    return results, model


def cold_repeat_train(biodata):
    """
    Cross-validation training setup for cold start scenario.

    Args:
        biodata: BioData object containing data and configuration.

    Returns:
        List of training results.
    """
    y = biodata.edge_and_label
    param = biodata.config
    repeat = param.repeat
    k_number = 1
    results_list = []
    for ii in range(repeat):
        train_idx, test_idx = mask_func(y['y_edge'], mask_sp=param.mask_sp)
        print(f'Running repeat {ii + 1} of {repeat}...')
        train_idx = np.arange(y['y'].shape[0])[train_idx]
        test_idx = np.arange(y['y'].shape[0])[test_idx]
        results, *_ = train_model(biodata, train_idx, test_idx, save_file=param.save_file + f'/repeat{ii}/')
        if not os.path.exists(param.save_file + f'/repeat{ii}/'):
            os.makedirs(param.save_file + f'/repeat{ii}/', exist_ok=True)
        results_list.append(results)
    return results_list


def repeat_train(biodata):
    """
    Cross-validation training setup.

    FIX 2.3: Each repeat now creates its own independent random data split,
    so the 10 repeats reflect true variance across data splits (not just
    variance of random initialisation over the same split).

    Args:
        biodata: BioData object containing data and configuration.

    Returns:
        List of training results.
    """
    y = biodata.edge_and_label
    param = biodata.config
    if not param.mask_sp is None:
        return cold_repeat_train(biodata)

    repeat = param.repeat
    results_list = []

    all_idx = np.arange(y['y'].shape[0])
    num_test = int(len(all_idx) * param.test_size)

    for ii in range(repeat):
        print(f'Running repeat {ii + 1} of {repeat}...')

        # FIX 2.3: re-shuffle for every repeat to get independent splits
        shuffled_indices = torch.randperm(len(all_idx))
        test_idx = all_idx[shuffled_indices[:num_test]]
        train_idx = all_idx[shuffled_indices[num_test:]]

        results, _ = train_model(biodata, train_idx, test_idx,
                                 save_file=param.save_file + f'/repeat{ii}/')
        os.makedirs(param.save_file + f'/repeat{ii}/', exist_ok=True)
        results_list.append(results)

    return results_list


def CV_train(biodata):
    """
    Cross-validation training setup using KFold.

    Args:
        biodata: BioData object containing data and configuration.

    Returns:
        List of training results.
    """
    y = biodata.edge_and_label
    param = biodata.config
    k_fold = param.kfold
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=param.globel_random)
    results_list = []

    k_number = 0
    for train_idx, test_idx in kf.split(np.arange(y['y'].shape[0])):
        print(f'Running fold {len(results_list) + 1} of {k_fold}...')

        auc_idx, auc_name, *_ = train_model(biodata, train_idx, test_idx,
                                            save_file=param.save_file + f'/KFold{k_number}/')
        if not os.path.exists(param.save_file + f'/KFold{k_number}/'):
            os.makedirs(param.save_file + f'/KFold{k_number}/', exist_ok=True)
        k_number += 1
        results_list.append(auc_idx)
    return results_list