import numpy as np
import matplotlib.pyplot as plt
import seaborn
import tempfile
import mlflow
import utils

def heatmaps(npa,rel=None,type_=None):
    assert type_ in ('fields','losses')
    if type_ == 'fields':
        assert rel is not None
        sections = rel.divide(npa)
    elif type_ == 'losses':
        sections = np.hsplit(npa, npa.shape[1])

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

def log_heatmaps_artifact(name, npa, which_tset,rel=None, type_=None):
    fig = heatmaps(npa, rel=rel, type_=type_)
    filename = tempfile.mktemp(prefix=f"heatmaps_{name}_{type_}_{which_tset}",suffix=".png")
    fig.savefig(filename)
    mlflow.log_artifact(filename)
    return filename
