import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import container
import itertools
from sklearn import metrics

from helpers import calc_toc, calc_tob_pt, get_dwh_collection

d = 3
models = ['Xgb', 'MLP', 'ResNet']


def plot_confusion_matrix(cm, ax, classes=('BkGd', 'signal'), d=3, title='Test Confusion matrix', cmap='Set3', th=0.5,
                          model='model'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    title = title + ' th=' + str(th)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 4),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(model + ' 2x' + str(d) + 'by' + str(d))
    return


def plot_results(y_test, predictions, model_name, d=3):
    th = 0.5
    fig, ax = plt.subplots()
    y = np.asarray(np.greater_equal(np.array(predictions), th), int)
    cm = metrics.confusion_matrix(np.array(y_test, dtype=int), y)
    plot_confusion_matrix(cm, ax, classes=['BG', 'Signal'], d=d, th=th, model=model_name)

    fpr, tpr, thresh = metrics.roc_curve(np.array(y_test, dtype=int), np.array(predictions))
    tpr_sig = tpr
    fpr_bg, tpr_bg, thresh = metrics.roc_curve(np.array(y_test, dtype=int), np.array(predictions), pos_label=0)
    precision, recall, thresholds = metrics.precision_recall_curve(np.array(y_test, dtype=int), np.array(predictions))
    auc = metrics.roc_auc_score(np.array(y_test, dtype=int), np.array(predictions))
    pr_auc = metrics.auc(recall, precision)
    f1 = 2 * (precision * recall) / (precision + recall)
    label_roc = model_name + ' auc_roc=' + str(round(auc, 3))
    label_pr = model_name + ' auc_pr=' + str(round(pr_auc, 3))
    f1_max_score = max(f1)
    label_f1 = model_name + ' f1 max score=' + str(round(f1_max_score, 3))
    ax1.plot(fpr, tpr, label=label_roc)
    ax2.plot(recall, precision, label=label_pr)
    ax3.plot(np.array([-0.001] + list(thresholds)), f1, label=label_f1)
    ax1.legend(loc="lower right")
    ax2.legend(loc="lower right")
    ax1.set_xlabel('FPR')
    ax1.set_ylabel('TPR')
    ax2.set_xlabel('recall')
    ax2.set_ylabel('precision')
    ax3.set_xlabel('th')
    ax3.set_ylabel('f1')

    t1 = 'roc_auc ' + '2x' + str(d) + 'by' + str(d)
    t2 = 'pr_auc ' + '2x' + str(d) + 'by' + str(d)
    t3 = 'f1 ' + '2x' + str(d) + 'by' + str(d)

    ax1.title.set_text(t1)
    ax2.title.set_text(t2)
    ax3.title.set_text(t3)

    ax3.legend(loc="lower right")
    return


# plot conf matrix, roc_auc,pr_auc,f1 for all models
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
for model in models:
    model_name = model + '_' + str(d) + 'by' + str(d) + '.csv'
    model_results = pd.read_csv(model_name)
    plot_results(model_results['signal'], model_results['score'], model, d)
plt.show()
plt.savefig("AUCs"+'_'+str(d)+'by'+str(d)+".png",bbox_inches = 'tight')

# plot model score dist for all models
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
for model, ax in [(models[0], ax1), (models[1], ax2), (models[2], ax3)]:
    model_name = model + '_' + str(d) + 'by' + str(d) + '.csv'
    df_test = pd.read_csv(model_name)
    total_y = df_test.signal
    total_predictions = df_test.score
    hist, bins = np.histogram(df_test.loc[df_test['signal'] == 0, 'score'])
    ax.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1] - bins[0]), color='blue', align='edge',
           alpha=0.6)
    hist, bins = np.histogram(df_test.loc[df_test['signal'] == 1, 'score'])
    ax.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1] - bins[0]), color='orange', align='edge',
           alpha=0.6)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylabel('Fraction of TOBs', fontsize=15)
    ax.set_xlabel('Score distribution', fontsize=15)
    ax.set_ylim((0, 1))
    ax.set_title(model + ' 2x' + str(d) + 'by' + str(d))
    ax.grid()
    legend = ['Background']
    legend += ['Signal']
    ax.legend(legend, loc='upper center', prop={'size': 20})
plt.savefig("Score_Dist"+'_'+str(d)+'by'+str(d)+".png", bbox_inches = 'tight')

# plot TOC
low_x_limit = 20
high_x_limit = 50
toc_bins = np.concatenate((np.arange(0, low_x_limit + 2, 2), np.arange(low_x_limit + 2, high_x_limit + 2, 4)))
rate = 20 / 100
shape_collection = get_dwh_collection(2, d, d)
f, ax = plt.subplots()
legend = []

df_test = pd.read_csv('test_set_cells' + '_' + str(d) + 'by' + str(d) + ".csv")
data_sig = df_test[df_test.signal == 1].copy()
data_bkg = df_test[df_test.signal == 0].copy()
lolims = np.ones(shape=toc_bins.shape, dtype=bool)
uplims = np.ones(shape=toc_bins.shape, dtype=bool)

for i in range(len(shape_collection)):
    # calculate shape energy
    data_sig['tob_pt'] = data_sig.apply(lambda x: calc_tob_pt(x, shape_collection[i], 2, d, d), axis=1)
    data_bkg['tob_pt'] = data_bkg.apply(lambda x: calc_tob_pt(x, shape_collection[i], 2, d, d), axis=1)
    bl_toc_curve, bl_toc_curve_err = calc_toc(data_sig, data_bkg, 'tau_pt_vis', 'tob_pt', toc_bins, rate)
    auc_bl = (bl_toc_curve[:-1] * (toc_bins[1:] - toc_bins[:-1])).sum() / high_x_limit
    name = 'Baseline, AUC=' + str(round(auc_bl, 3))
    ax.errorbar(toc_bins, bl_toc_curve, yerr=bl_toc_curve_err, drawstyle='steps-mid', label=name, color='#984ea3',
                uplims=uplims, lolims=lolims)

for model, ax, color, ls in [(models[0], ax, '#377eb8', ':'), (models[1], ax, '#ff7f00', '--'),
                             (models[2], ax, '#4daf4a', '-.')]:
    model_name = model + '_' + str(d) + 'by' + str(d) + '.csv'
    df_test = pd.read_csv(model_name)
    data_sig = df_test[df_test.signal == 1].copy()
    data_bkg = df_test[df_test.signal == 0].copy()

    toc_curve, toc_curve_err = calc_toc(data_sig, data_bkg, 'tau_pt_vis', 'score', toc_bins, rate)
    auc = (toc_curve[:-1] * (toc_bins[1:] - toc_bins[:-1])).sum() / high_x_limit
    name = model + ', AUC=' + str(round(auc, 3))
    ax.errorbar(toc_bins, toc_curve, yerr=toc_curve_err, color=color, drawstyle='steps-mid', label=name, uplims=uplims,
                lolims=lolims, ls=ls)

ax.set_ylim((0, 1.1))
ax.set_xlim((0, 50))
ax.set_xlabel(r'$p^{\tau}_{T}$ [GeV]', fontsize=20)
ax.set_ylabel('Efficiency', fontsize=20)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.grid(True)
ax.set_title('Turn-on Curve ' + ' 2x' + str(d) + 'by' + str(d), fontsize=20)

handles, labels = ax.get_legend_handles_labels()
handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
ax.legend(handles, labels, loc='lower right', prop={'size': 12})
plt.savefig("TOC" + '_' + str(d) + 'by' + str(d) + "_tau.png", bbox_inches='tight')
