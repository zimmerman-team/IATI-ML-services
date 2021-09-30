import numpy as np
import matplotlib.pyplot as plt
import seaborn
import tempfile
import mlflow
import utils
import pandas as pd

def barplots(npa,rel=None,type_=None): # FIXME duplicated code
    seaborn.set(rc={'figure.figsize': (27, 9)})
    if type_ == 'fields':
        assert rel is not None
        sections = rel.divide(npa)
    elif type_ == 'losses':
        sections = np.hsplit(npa, npa.shape[1])
    elif type_ == 'latent':
        sections = [npa]
    else:
        raise Exception("unknown type_ "+str(type_))

    vmin = utils.min_across(sections)
    vmax = utils.max_across(sections)
    print("vmin",vmin,"vmax",vmax)
    width_ratios = [
                       0.05+float(section.shape[1])/float(npa.shape[1])
                       for section
                       in sections
                   ] + [0.02]
    fig, axs = plt.subplots(
        ncols=len(sections)+1,
        gridspec_kw=dict(width_ratios=width_ratios)
    )

    for section_i,section in enumerate(sections):
        ax = seaborn.barplot(
            data=section,
            ax=axs[section_i],
            estimator=np.mean,
            ci=None,#'sd',
            #capsize=.2,
            color='lightblue'
        )
        ax.set_ylim(vmin*1.05,vmax*1.05)
        axs[section_i].set(xlabel=rel.fields[section_i].name)

    #fig.colorbar(axs[len(sections)-1].collections[0], cax=axs[len(sections)])
    return fig

def heatmaps(npa,rel=None,type_=None):
    seaborn.set(rc={'figure.figsize': (27, 9)})
    if type_ == 'fields':
        assert rel is not None
        sections = rel.divide(npa)
    elif type_ == 'losses':
        sections = np.hsplit(npa, npa.shape[1])
    elif type_ == 'latent':
        sections = [npa]
    else:
        raise Exception("unknown type_ "+str(type_))

    vmin = utils.min_across(sections)
    vmax = utils.max_across(sections)
    print("vmin",vmin,"vmax",vmax)
    width_ratios = [
        0.05+float(section.shape[1])/float(npa.shape[1])
        for section
        in sections
    ] + [0.02]
    fig, axs = plt.subplots(
        ncols=len(sections)+1,
        gridspec_kw=dict(width_ratios=width_ratios)
    )

    for section_i,section in enumerate(sections):
        seaborn.heatmap(
            section,
            ax=axs[section_i],
            cbar=False,
            vmin=vmin,
            vmax=vmax
        )
        axs[section_i].set(xlabel=rel.fields[section_i].name)

    fig.colorbar(axs[len(sections)-1].collections[0], cax=axs[len(sections)])
    return fig

def correlation(data):
    d = pd.DataFrame(data=data)
    # Compute the correlation matrix
    corr = d.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_metric = np.mean(np.abs(np.array(corr.fillna(0))))
    return corr, corr_metric, mask

def correlation_heatmap(corr, corr_metric, mask, epoch_nr):
    seaborn.set_theme(style="white")
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))
    # Generate a custom diverging colormap
    cmap = seaborn.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    seaborn.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5}
    )
    ax.set(xlabel="epoch "+str(epoch_nr)+" avg abs corr="+str(corr_metric))
    return fig

def log_heatmaps_artifact(name, npa, which_tset,rel=None, type_=None):
    fig = heatmaps(npa, rel=rel, type_=type_)
    filename = tempfile.mktemp(prefix=f"heatmaps_{name}_{type_}_{which_tset}",suffix=".png")
    fig.savefig(filename)
    mlflow.log_artifact(filename)
    return filename

def log_barplots_artifact(name, npa, which_tset,rel=None, type_=None):
    fig = barplots(npa, rel=rel, type_=type_)
    filename = tempfile.mktemp(prefix=f"barplots_{name}_{type_}_{which_tset}",suffix=".png")
    fig.savefig(filename)
    mlflow.log_artifact(filename)
    return filename

def log_correlation_heatmap_artifact(name, corr, corr_metric, mask, which_tset, epoch_nr):
    print(f"creating and logging correlation heatmap for {name} {which_tset} epoch {epoch_nr}..")
    fig = correlation_heatmap(corr, corr_metric, mask, epoch_nr)
    filename = tempfile.mktemp(prefix=f"correlation_heatmap_{name}_{which_tset}_{epoch_nr:04}_",suffix=".png")
    fig.savefig(filename)
    mlflow.log_artifact(filename)
    print("done creating and logging correlation heatmap")
    return filename
