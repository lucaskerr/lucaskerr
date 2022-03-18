#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os as os
import numpy as np
import scanpy as sc
import anndata
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import cell2location
import scvi

from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text
import seaborn as sns

import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule


# In[20]:


results_folder = f'/wynton/home/bivona/dkerr1/cell2location/take3/'
sp_data_folder = f'/wynton/home/bivona/dkerr1/cell2location/batch1/'

print(os.listdir(results_folder))

# create paths and names to results folders for reference regression and cell2location models
ref_run_name = f'{results_folder}reference_signatures_031722'
run_name = f'{results_folder}cell2location_map_031722'


# In[21]:


#Define read and load function
def read_and_qc(sample_name, path=sp_data_folder + 'rawdata/'):
    r""" This function reads the data for one 10X spatial experiment into the anndata object.
    It also calculates QC metrics. Modify this function if required by your workflow.

    :param sample_name: Name of the sample
    :param path: path to data
    """

    adata = sc.read_visium(path + str(sample_name) + '/outs/',
                           count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.obs['sample'] = sample_name
    adata.var['SYMBOL'] = adata.var_names
    adata.var.rename(columns={'gene_ids': 'ENSEMBL'}, inplace=True)
    adata.var_names = adata.var['ENSEMBL']
    adata.var.drop(columns='ENSEMBL', inplace=True)
    
    # Calculate QC metrics
    from scipy.sparse import csr_matrix
    adata.X = adata.X.toarray()
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata.X = csr_matrix(adata.X)
    adata.var['mt'] = [gene.startswith('mt-') for gene in adata.var['SYMBOL']]
    adata.obs['mt_frac'] = adata[:, adata.var['mt'].tolist()].X.sum(1).A.squeeze()/adata.obs['total_counts']

    # add sample name to obs names
    adata.obs["sample"] = [str(i) for i in adata.obs['sample']]
    adata.obs_names = adata.obs["sample"]                           + '_' + adata.obs_names
    adata.obs.index.name = 'spot_id'

    return adata
    
def select_slide(adata, s, s_col='sample'):
    r""" This function selects the data for one slide from the spatial anndata object.

    :param adata: Anndata object with multiple spatial experiments
    :param s: name of selected experiment
    :param s_col: column in adata.obs listing experiment name for each location
    """

    slide = adata[adata.obs[s_col].isin([s]), :]
    s_keys = list(slide.uns['spatial'].keys())
    #s_keys_mod = [re.sub('-Tumor','T', i) for i in s_keys]
    #s_keys_mod = [re.sub('-Normal','N', i) for i in s_keys_mod] 
    #s_keys_mod = list([str(s[-1] + s[:-1]) for s in s_keys_mod])
    s_spatial = np.array(s_keys)[[s in k for k in s_keys]][0]

    slide.uns['spatial'] = {s_spatial: slide.uns['spatial'][s_spatial]}

    return slide


# In[22]:


# load sample name .csvs
sample_data = pd.read_csv(sp_data_folder + 'reactions.csv')


# Read the data into anndata objects
slides = []
for i in sample_data['Reactions']:
    slides.append(read_and_qc(i, path=sp_data_folder))


# Combine anndata objects together
adata_vis = slides[0].concatenate(
    slides[1:],
    batch_key="sample",
    uns_merge="unique",
    batch_categories=sample_data['Reactions'],
    index_unique=None
)


# mitochondria-encoded (MT) genes should be removed for spatial mapping
adata_vis.var['SYMBOL'] = adata_vis.var_names

# find mitochondria-encoded (MT) genes
adata_vis.var['MT_gene'] = [gene.startswith('MT-') for gene in adata_vis.var['SYMBOL']]
    
# remove MT genes for spatial mapping (keeping their counts in the object)
adata_vis.obsm['MT'] = adata_vis[:, adata_vis.var['MT_gene'].values].X.toarray()
adata_vis = adata_vis[:, ~adata_vis.var['MT_gene'].values]


# In[23]:


# Read reference single-cell data
adata_ref = sc.read(f'/wynton/home/bivona/dkerr1/cell2location/reference_SS03/SS03_Merged_Sc_Object.h5ad')


# In[24]:


#Prepare single-cell data
# Use ENSEMBL as gene IDs to make sure IDs are unique and correctly matched
adata_ref.var['Gene_IDs'] = adata_ref.var.index
adata_ref.var.index = adata_ref.var['Gene_IDs_ENSEMBL'].copy()
adata_ref.var_names = adata_ref.var['Gene_IDs_ENSEMBL'].copy()
adata_ref.var.index.name = None
adata_ref.raw.var['Gene_IDs'] = adata_ref.raw.var.index
adata_ref.raw.var.index = adata_ref.raw.var['Gene_IDs_ENSEMBL'].copy()
adata_ref.raw.var.index.name = None

#Make unique gene ids (may need to reactivate this)
adata_ref.var_names_make_unique()


# In[25]:


#Check excluded genes
from cell2location.utils.filtering import filter_genes
selected = filter_genes(adata_ref, cell_count_cutoff=1, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)


# In[26]:


# filter the object
adata_ref = adata_ref[:, selected].copy()


# In[27]:


#Calculate averages of cell location (prefered for smart-seq2)
inf_aver = cell2location.cluster_averages.cluster_averages.compute_cluster_averages(adata_ref, 'Cell_Minor_Short', use_raw=True, layer=None)
#inf_aver.index = adata_ref.var['Gene_IDs_ENSEMBL']
inf_aver.index = anndata.utils.make_index_unique(inf_aver.index.astype(str), '-')

#Find intersects between ENSEMBL-IDs of smart-seq2 and Visium data sets
intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
adata_vis = adata_vis[:, intersect].copy()
inf_aver = inf_aver.loc[intersect, :].copy()


# In[11]:





# In[28]:


# prepare anndata for cell2location model
scvi.data.setup_anndata(adata=adata_vis, batch_key="sample")
scvi.data.view_anndata_setup(adata_vis)


#Train the model
# create and train the model
mod = cell2location.models.Cell2location(
    adata_vis, cell_state_df=inf_aver, 
    # the expected average cell abundance: tissue-dependent 
    # hyper-prior which can be estimated from paired histology:
    N_cells_per_location=10,
    # hyperparameter controlling normalisation of
    # within-experiment variation in RNA detection (using default here):
    detection_alpha=200
) 
    
    
mod.train(max_epochs=30000, 
          batch_size=None, 
          train_size=1,
          plan_kwargs={'optim': pyro.optim.ClippedAdam(optim_args={'lr': 0.002, 'clip_norm': 10})},
          use_gpu=True)


# In[13]:


# In this section, we export the estimated cell abundance (summary of the posterior distribution).
adata_vis = mod.export_posterior(
    adata_vis, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs, 'use_gpu': True}
)


# In[ ]:





# In[30]:


# Save model
mod.save(f"{run_name}_031922", overwrite=True)

#mod = cell2location.models.Cell2location.load(f"{run_name}", adata_vis)

# Save anndata object with results
adata_file = f"{run_name}sp_031922.h5ad"
adata_vis.write(adata_file)
adata_file


# In[17]:


mod.plot_QC()


# In[25]:


fig = mod.plot_spatial_QC_across_batches()


# In[42]:


adata_vis.obs[adata_vis.uns['mod']['factor_names']] = adata_vis.obsm['q05_cell_abundance_w_sf']

# plot in spatial coordinates
with mpl.rc_context({'axes.facecolor':  'black',
                     'figure.figsize': [4.5, 5]}):

    sc.pl.spatial(adata_vis, cmap='magma',
                  # show first 8 cell types
                  color=['Cilitated Airway Cell', 'Inflammatory Fibroblast - THY1+', 'Plasma B-cell', 'Naive T-cell', 
                  'Airway Smooth Muscle', 'AT2 Pneumocyte - Damage Associated', 'AT2 Pneumocyte', 'Capillary Aerocyte'],
                  ncols=4, size=1.3,
                  img_key='hires',
                  # limit color scale at 99.2% quantile of cell abundance
                  vmin=0, vmax='p99.2'
                 )


# In[29]:


adata_vis.obsm['q05_cell_abundance_w_sf']


# In[34]:


from cell2location import run_colocation
res_dict, adata_vis = run_colocation(
    adata_vis,
    model_name='CoLocatedGroupsSklearnNMF',
    train_args={
      'n_fact': np.arange(11, 13), # IMPORTANT: use a wider range of the number of factors (5-30)
      'sample_name_col': 'sample', # columns in adata_vis.obs that identifies sample
      'n_restarts': 3 # number of training restarts
    },
    export_args={'path': f'{run_name}/CoLocatedComb/'}
)


# In[35]:


# Here we plot the NMF weights (Same as saved to `cell_type_fractions_heatmap`)
res_dict['n_fact12']['mod'].plot_cell_type_loadings()


# In[ ]:




