import numpy as np
import matplotlib.pyplot as plt
import seaborn
import tempfile
import mlflow

def split_heatmap(npa, rel):
    seaborn.set(rc={'figure.figsize': (27, 9)})
    sections = []
    pos = 0
    vmin = np.min(npa)
    vmax = np.max(npa)
    for field in rel.fields:
        field.name
        till = pos+field.n_features
        section = npa[:,pos:till]
        sections.append(section)
        pos = till

    width_ratios = [max(0.1,float(curr.n_features)/float(rel.n_features)) for curr in rel.fields] + [0.02]
    fig, axs = plt.subplots(ncols=rel.n_fields+1, gridspec_kw=dict(width_ratios=width_ratios))

    for section_i,section in enumerate(sections):
        #print("section",section)
        seaborn.heatmap(section,ax=axs[section_i],cbar=False,vmin=vmin,vmax=vmax)
        axs[section_i].set(xlabel=rel.fields[section_i].name)
        pos = till
    fig.colorbar(axs[rel.n_fields-1].collections[0], cax=axs[rel.n_fields])
    return fig

def log_split_heatmap_artifact(npa, rel):
    fig = split_heatmap(npa, rel)
    filename = tempfile.mktemp(prefix="split_heatmap_",suffix=".png")
    fig.savefig(filename)
    mlflow.log_artifact(filename)
    return filename