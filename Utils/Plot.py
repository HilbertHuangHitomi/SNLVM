import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

direction_colors = plt.cm.nipy_spectral(np.arange(8)/8)


def plot_embeddings(mean, direction_index, modeldir, heldout=False):
    def plot_ax(i, proj, title):
        proj = (proj - np.mean(proj)) / np.std(proj)
        if heldout:
            for j in range(len(direction_index)):
                if direction_index[j] == 0:
                    axs[i].scatter(proj[j,0], proj[j,1], alpha=1.0, color=direction_colors[direction_index[j]])
                else:
                    axs[i].scatter(proj[j,0], proj[j,1], alpha=0.25, color=direction_colors[direction_index[j]])
        else:
            axs[i].scatter(proj[:,0], proj[:,1], alpha=.8, color=direction_colors[direction_index])
        axs[i].axis('equal')
        axs[i].set_xlabel('dim 1')
        axs[i].set_ylabel('dim 2')
        axs[i].set_title(title)
        axs[i].axes.set_xlim(-2.5, 2.5)
        axs[i].axes.set_ylim(-2.5, 2.5)
        axs[i].set_xticks([-2, 0, 2])
        axs[i].set_yticks([-2, 0, 2])
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    proj = PCA(n_components=2, whiten=True).fit_transform(mean)
    plot_ax(0, proj, 'PCA')
    proj = TSNE(n_components=2).fit_transform(mean)
    plot_ax(1, proj, 't-SNE')
    fig.savefig(os.path.join(modeldir, 'embedding.svg'), dpi=300)
    fig.clear()
    plt.close()


def plot_behaviour(b, b_gt, di, heldout):
    for i in range(b.shape[0]):
        plt.plot(b_gt[i,:,0], b_gt[i,:,1], '--', alpha=.1, color=direction_colors[di[i]], lw=1.5)
        if heldout:
            if di[i] == 0:
                plt.plot(b[i,:,0],b[i,:,1], color=direction_colors[di[i]])
            else:
                plt.plot(b[i,:,0],b[i,:,1], alpha=.25, color=direction_colors[di[i]])
        else:
            plt.plot(b[i,:,0],b[i,:,1], color=direction_colors[di[i]])
    plt.axis('equal')


def plot_behaviour_reconstruction(z, behavioural_data, direction_index, modeldir, heldout=False):
    X = np.transpose(z.numpy(), (0,2,1)).reshape((z.shape[0], -1))
    y = np.transpose(behavioural_data, (0,2,1)).reshape((behavioural_data.shape[0], -1))
    reg = Ridge(normalize=True, fit_intercept=True).fit(X, y)
    beh_fit = reg.predict(X)
    b = np.zeros_like(behavioural_data)
    b[:,:,0] = beh_fit[:,:behavioural_data.shape[1]]
    b[:,:,1] = beh_fit[:,behavioural_data.shape[1]:]
    Xt = np.transpose(b, (0,2,1)).reshape((b.shape[0], -1))
    yt = np.transpose(behavioural_data, (0,2,1)).reshape((behavioural_data.shape[0], -1))
    fig = plt.figure(figsize=(4,4));
    plot_behaviour(b, behavioural_data, direction_index, heldout)
    R2 = r2_score(yt, Xt)
    plt.title('R2: {:.3%}'.format(R2))
    fig.savefig(os.path.join(modeldir, 'behavior.svg'), dpi=300)
    fig.clear()
    plt.close()
    return R2


def plot_2factor(f, i1, i2, di, n_show = None, ax=None, labels=None, title=None):
    if labels is None:
        labels = ['Factor '+str(i+1) for i in (i1,i2)]
    if ax is None:
        ax = plt.subplot(111)
    if n_show is None:
        n_show = f.shape[0]
    for t in range(n_show):
        plt.plot(f[t,:,i1],f[t,:,i2],color=direction_colors[np.array(di)[t]],alpha=.2, lw=1)
    for i in range(8):
        if np.sum(di==i)>0:
            plt.plot(np.mean(f.numpy()[di==i,:,i1],axis=0),
                     np.mean(f.numpy()[di==i,:,i2],axis=0),color=direction_colors[i],alpha=1, lw=2)
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.xticks(())
    plt.yticks(())
    plt.axis('equal')


def plot_all_2factors(f, di, b=None, b_gt=None, modeldir=None):
    n_factors = f.shape[-1]
    fig = plt.figure(figsize=(2*(n_factors-1), 2*(n_factors-1)))
    for f1 in range(n_factors):
        for f2 in range(f1+1,n_factors):
            ax = plt.subplot2grid((n_factors-1, n_factors-1),(f1,f2-1))
            plot_2factor(f, f1, f2, di, ax=ax)
    if (b is not None) and (b_gt is not None):
        ax = plt.subplot2grid((n_factors-1, n_factors-1),(n_factors-2, 0))
        plot_behaviour(b_gt, b, di, ax=ax)
    fig.savefig(os.path.join(modeldir, 'adjacency.svg'), dpi=300)
    fig.clear()
    plt.close()


def plot_1factor(f, i, di, n_show = 40, ax=None, labels=None, title=None):
    if labels is None:
        labels = 'Factor '+str(i+1)
    if ax is None:
        ax = plt.subplot(111)
    n_show = f.shape[0]
    x = np.arange(f.shape[1])*10.
    for t in range(n_show):
        plt.plot(x, (f[t,:,i]),color=direction_colors[di[t]],alpha=.2, lw=1)
    for d in range(8):
        if np.sum(di==d)>1:
            plt.plot(x, np.mean(f.numpy()[di==d,:,i],axis=0), color=direction_colors[d],alpha=1, lw=2)
    plt.title(labels)
    plt.xlabel('Time (ms)')


def plot_all_1factors(f, di, b=None, b_gt=None, modeldir=None):
    n_factors = f.shape[-1]
    fig = plt.figure(figsize=(3*n_factors, 2))
    for f1 in range(n_factors):
        ax = plt.subplot2grid((1, n_factors),(0,f1))
        plot_1factor(f, f1, di, ax=ax, n_show=np.min((80, f.shape[0])))
    fig.savefig(os.path.join(modeldir, 'factors.svg'), dpi=300)
    fig.clear()
    plt.close()


def plot_firing_rates(log_f, spikes, label, neurons=4, step=0.01, modeldir=None):
    fig, axs = plt.subplots(8, neurons, figsize=(neurons, 8))
    metric = []
    for i in range(8):
        data = spikes[label==i,:,:]
        for j in range(neurons):
            ax = axs[i,j]
            pred = (np.exp(log_f)*step)[np.array(label)==i,:,j]
            x = np.arange(data.shape[1])
            ax.plot(x,np.mean(data,axis=0)[:,j],'k',alpha=0.5)
            ax.plot(x,np.mean(pred,axis=0),lw=2, color=direction_colors[i])
            metric.append(np.sqrt(mean_squared_error(np.mean(data,axis=0)[:,j], np.mean(pred,axis=0))))
            ax.set_xticks(())
            ax.set_yticks(())
            plt.ylim(0, 1.5)
    rmse = np.mean(metric)
    plt.suptitle('{:.3}'.format(rmse))
    fig.savefig(os.path.join(modeldir, 'firing.svg'), dpi=300)
    fig.clear()
    plt.close()
    return rmse
