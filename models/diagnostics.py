import numpy as np
import matplotlib.pyplot as plt
import seaborn
import tempfile
import mlflow
import pandas as pd
import hiddenlayer

from common import utils


def barplots(npa, rel=None, type_=None):  # FIXME duplicated code
    """
    For each field of the relation it generates a barplot
    with the data of that field found in the glued-together
    numpy array.
    The result is a figure in which many plots are shown
    and they have a common y-axis scale.
    :param npa: the glued-together numpy array of relation data
    :param rel: the relation
    :param type_: the type of plot. Can be: `fields` if the `npa`
        data array represents glued-together fields with per-feature
        information ; `losses` in which provided data is supposed to
        have as many columns as the number of fields in the relation
        (hence it's not per-feature information, but some sort of aggregation) ;
        `latent` is not going to split the numpy array
        # FIXME NAMINGS!! : losses is not a generic enough name, as well as latent,
        #                   and fields does not convey the per-faeature aspect of it
    :return: the figure
    """
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
    width_ratios = [
                       0.05+float(section.shape[1])/float(npa.shape[1])
                       for section
                       in sections
                   ] + [0.02]
    fig, axs = plt.subplots(
        ncols=len(sections)+1,
        gridspec_kw=dict(width_ratios=width_ratios)
    )

    for section_i, section in enumerate(sections):
        ax = seaborn.barplot(
            data=section,
            ax=axs[section_i],
            estimator=np.mean,
            ci=None,  # 'sd',
            # capsize=.2,
            color='lightblue'
        )
        ax.set_ylim(vmin*1.05, vmax*1.05)
        axs[section_i].set(xlabel=rel.fields[section_i].name)

    # fig.colorbar(axs[len(sections)-1].collections[0], cax=axs[len(sections)])
    return fig


def heatmaps(npa, rel=None, type_=None):
    # FIXME: a lot of duplication from `barplots`
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
    width_ratios = [
        0.05+float(section.shape[1])/float(npa.shape[1])
        for section
        in sections
    ] + [0.02]
    fig, axs = plt.subplots(
        ncols=len(sections)+1,
        gridspec_kw=dict(width_ratios=width_ratios)
    )

    for section_i, section in enumerate(sections):
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
    """
    Generates a correlation matrix and a correlation measure.
    :param data: data whose feature-feature correlations need to be computed
    :return: a correlation matrix, a scalar of average rectified correlations,
        a triangular matrix mask that has the same shape as the correlation matrix
    """
    d = pd.DataFrame(data=data)
    # Compute the correlation matrix
    corr = d.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_metric = np.mean(np.abs(np.array(corr.fillna(0))))
    return corr, corr_metric, mask


def correlation_heatmap(corr, corr_metric, mask, epoch_nr):
    """
    Plots a correlation matrix heatmap given the correlation matrix
    :param corr: the correlation matrix
    :param corr_metric: the aggregated correlation metric
    :param mask: the triangular mask that will be used to plot only
        the lower triangular matrix of the correlation matrix
    :param epoch_nr: which epoch number was this correlation heatmap
        calculated fom
    :return: the figure
    """
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


def log_heatmaps_artifact(name, npa, which_tset, rel=None, type_=None):
    """
    Plots a heatmap and logs it as a mlflow artifact
    :param name: name of the plot
    :param npa: numpy array of data
    :param which_tset: is this the training set or the validation/test set?
    :param rel: the relation the data data is about
    :param type_: type of plotting # FIXME: expand
    :return:
    """
    print(f"log_heatmaps_artifact name:{name} npa.shape="+str(npa.shape))
    fig = heatmaps(npa, rel=rel, type_=type_)
    filename = tempfile.mktemp(prefix=f"heatmaps_{name}_{type_}_{which_tset}", suffix=".png")
    fig.savefig(filename)
    mlflow.log_artifact(filename)
    return filename


def log_barplots_artifact(name, npa, which_tset, rel=None, type_=None):
    """
    Makes a barplots plot and logs it as a mlflow artifact
    :param name: name of the barplots
    :param npa: numpy array of data
    :param which_tset: is this the training set or the validation/test set?
    :param rel: the relation this data is about
    :param type_: type of plotting # FIXME: expand
    :return: the plotted image filename
    """
    fig = barplots(npa, rel=rel, type_=type_)
    filename = tempfile.mktemp(prefix=f"barplots_{name}_{type_}_{which_tset}", suffix=".png")
    fig.savefig(filename)
    mlflow.log_artifact(filename)
    return filename


def log_correlation_heatmap_artifact(name, corr, corr_metric, mask, which_tset, epoch_nr):
    """
    Makes a correlation matrix plot and logs it into a mlflow artifact
    :param name: name of the correlation heatmap plot
    :param corr: correlation matrix
    :param corr_metric: aggregated correlation value
    :param mask: triangular mask to the correlation matrix
    :param which_tset: is this the training or the validation/test set?
    :param epoch_nr: which epoch number is this plot about?
    :return: the plotted image filename
    """
    print(f"creating and logging correlation heatmap for {name} {which_tset} epoch {epoch_nr}..")
    fig = correlation_heatmap(corr, corr_metric, mask, epoch_nr)
    filename = tempfile.mktemp(prefix=f"correlation_heatmap_{name}_{which_tset}_{epoch_nr:04}_", suffix=".png")
    fig.savefig(filename)
    mlflow.log_artifact(filename)
    print("done creating and logging correlation heatmap")
    return filename


def log_net_visualization(model, features):
    """
    Makes a plot of the network structure and logs it into a mlflow artifact
    :param model: the pytorch_lightning.LightningModule
    :param features:
    :return:
    """
    hl_graph = hiddenlayer.build_graph(model, features)
    hl_graph.theme = hiddenlayer.graph.THEMES["blue"].copy()
    filename = tempfile.mktemp(suffix=".png")
    hl_graph.save(filename, format="png")
    mlflow.log_artifact(filename)
