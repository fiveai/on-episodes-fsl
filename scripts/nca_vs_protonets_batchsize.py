import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------------------------
latex_style = True
small_picture = False
tfevent_extension = '/ADD/'
base_log_path = '../runs'
expm_path = 'nca_vs_protonets'
split = 'val'
n_seed = 3
batch_sizes = [128, 256, 512]
datasets = ['CIFARFS', 'miniimagenet']
accuracy_ids = ['val/1-shot/Best-CL2N', 'val/5-shot/Best-CL2N', 'test/1-shot/CL2N', 'test/5-shot/CL2N']
expm_to_color = {'nca': 'tab:green',
                 'proto-1shot-8x': 'tab:red',
                 'proto-5shot-8x': 'tab:purple',
                 'proto-5shot-16x': 'tab:blue',
                 'proto-5shot-32x': 'tab:orange'}

expm_to_legend = {'nca': 'NCA',
                 'proto-1shot-8x': 'Proto-nets 1-shot a=8',
                 'proto-5shot-8x': 'Proto-nets 5-shot a=8',
                 'proto-5shot-16x': 'Proto-nets 5-shot a=16',
                 'proto-5shot-32x': 'Proto-nets 5-shot a=32'}

# ordered_legend = [expm_to_text[key] for key in expm_to_text]


def _args_from_expm(expm):

    dataset_mask = [i in expm for i in datasets]
    dataset_matches = [i for idx, i in enumerate(datasets) if dataset_mask[idx]]
    assert len(dataset_matches) == 1, 'ERROR: multiple dataset match in the experiment id'

    batch_mask = ['batch'+str(i) in expm for i in batch_sizes]
    batch_match = [i for idx, i in enumerate(batch_sizes) if batch_mask[idx]]
    assert len(batch_match) == 1, 'ERROR: multiple batch sizes match in the experiment id'

    seed_mask = ['seed'+str(i) in expm for i in range(n_seed)]
    seed_match = [idx for idx in range(n_seed) if seed_mask[idx]]
    assert len(seed_match) == 1, 'ERROR: multiple seeds match in the experiment id'

    # extract expm memorable id from full id string
    splits = expm.split('-')[5:10]
    if 'nca' in '-'.join(splits):
        splits = 'nca'
    else:
        assert 'proto' in splits[0]
        assert '1shot' in splits[1] or '5shot' in splits[1]
        assert 'w' in splits[3]
        n_way = int(splits[3].replace('w', ''))
        proto_id = str(int(batch_match[0]/n_way)) + 'x'
        splits = splits[0] + '-' + splits[1] + '-' + proto_id

    return dataset_matches[0], batch_match[0], seed_match[0], splits

# ---------------------------------------------------------------------------------------------------------------------------------


if latex_style:
    params = {
        "text.usetex": True,
        "font.family": "lmodern",
        "text.latex.unicode": True,
    }
    plt.rcParams.update(params)

expm_args = {}
expm_types = set()

# Extract metadata abut experiments in the folder
expm_list = next(os.walk(os.path.join(base_log_path, expm_path)))[1]
for expm in expm_list:
    dataset, batch_size, seed, expm_id = _args_from_expm(expm)
    expm_args[expm] = {'dataset': dataset, 'batch_size': batch_size, 'seed': seed, 'expm_id': expm_id}
    expm_types.add(expm_id)

# Create data structure to store all the accuracies
all_accuracies = {}
for d in datasets:
    all_accuracies[d] = {}
    for expm in expm_types:
        all_accuracies[d][expm] = {}
        for bs in batch_sizes:
            all_accuracies[d][expm][bs] = {}
            for a_id in accuracy_ids:
                all_accuracies[d][expm][bs][a_id] = []


for expm in expm_list:
    dataset, batch_size, seed = expm_args[expm]['dataset'], expm_args[expm]['batch_size'], expm_args[expm]['seed']
    full_path = os.path.join(base_log_path, expm_path, expm, split)
    tfevent_file = os.listdir(full_path)
    assert len(tfevent_file) == 1 and tfevent_extension in tfevent_file[0]
    ea = event_accumulator.EventAccumulator(os.path.join(full_path, tfevent_file[0]))
    ea.Reload()
    tags = ea.Tags()['scalars']
    if len(tags) > 0:
        for a_id in accuracy_ids:
            if a_id in tags:
                acc = ea.Scalars(a_id)[0][2]
                e_id = expm_args[expm]['expm_id']
                all_accuracies[dataset][e_id][batch_size][a_id].append(acc)
            else:
                print('\n> WARNING - skipping ' + expm)

# ---------------------------------------------------------------------------------------------------------------------------------

# These are the experiments we want to plot
expm_types_plots = ['proto-1shot-8x', 'proto-5shot-8x', 'proto-5shot-16x', 'proto-5shot-32x']


assert set(expm_types_plots) <= set(expm_types)

for d in datasets:
    for acc in accuracy_ids:
        FIG_SIZE = (5, 3.1)
        # FIG_SIZE = (6, 3.7)
        fig = plt.figure(figsize=FIG_SIZE)
        min_y, max_y = 100, 0
        for expm in expm_types_plots:
            y_mu, y_sigma = np.zeros(len(batch_sizes)), np.zeros(len(batch_sizes))
            for idx, b in enumerate(batch_sizes):
                y_mu[idx] = np.mean(all_accuracies[d][expm][b][acc])
                y_sigma[idx] = np.std(all_accuracies[d][expm][b][acc])

            plt.plot(batch_sizes, y_mu, lw=1.5, marker='d', color=expm_to_color[expm])
            plt.fill_between(batch_sizes, y_mu+y_sigma, y_mu-y_sigma, alpha=0.2, color=expm_to_color[expm])

            if min(y_mu) < min_y:
                min_y = min(y_mu)
            if max(y_mu) > max_y:
                max_y = max(y_mu)

        ylabel = '1-shot accuracy (\%)' if '1-shot' in acc else '5-shot accuracy (\%)'
        dataset_title = 'CIFAR-FS' if d == 'CIFARFS' else 'miniImageNet'
        split_title = '(validation set)' if 'val' in acc else ' (test set)'

        axes = plt.axes()
        axes.set_ylim([np.floor(min_y)-1, np.ceil(max_y)+1])

        plt.xscale('log', basex=2)
        plt.xticks(batch_sizes, [str(i) for i in batch_sizes], fontsize=9)
        locs, labels = plt.yticks(fontsize=9)
        plt.yticks(np.arange(min(locs), max(locs), 2.0))
        plt.legend([expm_to_legend[e] for e in expm_types_plots], facecolor="white", framealpha=1.0, edgecolor="k", fancybox=True)
        plt.title(dataset_title + ' ' + split_title, fontsize=16)
        plt.xlabel('Batch size', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)

        splits = acc.split('/')
        plt.savefig(os.path.join('../../paper-softnn/art', d + '_' + splits[0] + '_' + splits[1] + '.pdf'), bbox_inches='tight')
        #plt.show()
        plt.close()
