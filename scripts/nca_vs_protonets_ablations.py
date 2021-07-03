
#todo: this needs to be updated as it uses hardcoded results

import numpy as np
from src.utils.ablations_results import *
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------------------------
latex_style = True
# FIG_SIZE = (5, 3.1)
FIG_SIZE = (15, 5)
title_size = 16
xticks_size = 16
yticks_size = 21
bar_size = 0.4
bar_color = 'tab:purple'
bar_edgecolor = 'black'
bar_lw = 1.5

# ---------------------------------------------------------------------------------------------------------------------------------
if latex_style:
    params = {
        "text.usetex": True,
        "font.family": "lmodern",
        "text.latex.unicode": True,
    }
    plt.rcParams.update(params)


bars = ('Proto-nets 5-shot no S/Q (7)', 'Proto-nets 5-shot no proto (6)', 'Proto-nets 5-shot (5)', 'Proto-nets 1-shot (4)', 'NCA fixed batch composition (3)', 'NCA replacement sampling (2)', 'NCA (1)')

labels_order = [5, 6, 4, 2, 7, 1, 0]

y_pos = np.arange(len(bars))

f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey='row', figsize=FIG_SIZE)

plt.yticks(y_pos, bars)
ax1.barh(y_pos, [mini_val_1shot_avg[i] for i in labels_order], xerr=[mini_val_1shot_std[j] for j in labels_order], height=bar_size, linewidth=bar_lw, color=bar_color, edgecolor=bar_edgecolor)
ax1.set_xlim([58, np.ceil(max(mini_val_1shot_avg))+1])
ax1.set_title('miniImageNet (1-shot)', size=title_size)
for t in ax1.xaxis.get_major_ticks():
    t.label.set_fontsize(xticks_size)
ax1.set_xticklabels([str(int(x))+'\%' if (x).is_integer() else str(x)+'\%' for x in ax1.get_xticks()])

ax2.barh(y_pos, [mini_val_5shot_avg[i] for i in labels_order], xerr=[mini_val_5shot_std[j] for j in labels_order], height=bar_size, linewidth=bar_lw, color=bar_color, edgecolor=bar_edgecolor)
ax2.set_xlim([72, np.ceil(max(mini_val_5shot_avg))+1])
ax2.set_title('miniImageNet (5-shot)', size=title_size)
for t in ax2.xaxis.get_major_ticks():
    t.label.set_fontsize(xticks_size)
ax2.set_xticklabels([str(int(x))+'\%' if (x).is_integer() else str(x)+'\%' for x in ax2.get_xticks()])

ax3.barh(y_pos, [cifar_val_1shot_avg[i] for i in labels_order], xerr=[cifar_val_1shot_std[j] for j in labels_order], height=bar_size, linewidth=bar_lw, color=bar_color, edgecolor=bar_edgecolor)
ax3.set_xlim([56, np.ceil(max(cifar_val_1shot_avg))+1])
ax3.set_title('CIFAR-FS (1-shot)', size=title_size)
for t in ax3.xaxis.get_major_ticks():
    t.label.set_fontsize(xticks_size)
ax3.set_xticklabels([str(int(x))+'\%' if (x).is_integer() else str(x)+'\%' for x in ax3.get_xticks()])

ax4.barh(y_pos, [cifar_val_5shot_avg[i] for i in labels_order], xerr=[cifar_val_5shot_std[j] for j in labels_order], height=bar_size, linewidth=bar_lw, color=bar_color, edgecolor=bar_edgecolor)
ax4.set_xlim([68, np.ceil(max(cifar_val_5shot_avg))+1])
ax4.set_title('CIFAR-FS (5-shot)', size=title_size)
for t in ax4.xaxis.get_major_ticks():
    t.label.set_fontsize(xticks_size)
ax4.set_xticklabels([str(int(x))+'\%' if (x).is_integer() else str(x)+'\%' for x in ax4.get_xticks()])

for t in ax1.yaxis.get_major_ticks():
    t.label.set_fontsize(yticks_size)

plt.show()
# plt.savefig('../../paper-softnn/art/ablations_val.pdf', bbox_inches='tight')
