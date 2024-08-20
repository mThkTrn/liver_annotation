import unittest

import joblib, torch, sklearn, scipy

import liver_annotation as la

import pandas as pd

import anndata as ad

import numpy as np

import scanpy as sc

from packaging import version

class TestLibrary(unittest.TestCase):

    
    def test_joblib_version(self):
        required_version = '1.4.2'
        self.assertTrue(version.parse(joblib.__version__) >= version.parse(required_version),
                        f"joblib version should be at least {required_version}")

    def test_torch_version(self):
        required_version = '2.1.2+cpu'
        self.assertTrue(version.parse(torch.__version__) >= version.parse(required_version),
                        f"torch version should be at least {required_version}")

    def test_sklearn_version(self):
        required_version = '1.2.2'
        self.assertTrue(version.parse(sklearn.__version__) >= version.parse(required_version),
                        f"scikit-learn version should be at least {required_version}")

    def test_scipy_version(self):
        required_version = '1.11.4'
        self.assertTrue(version.parse(scipy.__version__) >= version.parse(required_version),
                        f"scipy version should be at least {required_version}")

    def test_testdata_load(self):
        # import counts matrix
        matrix = scipy.io.mmread('test_data/coassolo_control/matrix.mtx.gz').tocsr()

        # import barcodes file with cell names
        barcodes = pd.read_csv('test_data/coassolo_control/barcodes.tsv.gz', header=None, sep='\t')
        barcodes.columns = ['barcode']

        # import features file with gene names
        features = pd.read_csv('test_data/coassolo_control/features.tsv.gz', header=None, sep='\t')
        features.columns = ['gene_id', 'gene_name', 'feature_type']
    def test_predict(self):
        # import counts matrix
        matrix = scipy.io.mmread('test_data/coassolo_control/matrix.mtx.gz').tocsr()

        # import barcodes file with cell names
        barcodes = pd.read_csv('test_data/coassolo_control/barcodes.tsv.gz', header=None, sep='\t')
        barcodes.columns = ['barcode']

        # import features file with gene names
        features = pd.read_csv('test_data/coassolo_control/features.tsv.gz', header=None, sep='\t')
        features.columns = ['gene_id', 'gene_name', 'feature_type']

        # Create the AnnData object
        adata = ad.AnnData(X=matrix.T, obs=barcodes, var=features)

        # Set the index for obs and var
        adata.obs_names = adata.obs['barcode']
        adata.var_names = adata.var['gene_name']

        # Preprocessing
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata = adata[:, adata.var.highly_variable]
        sc.pp.scale(adata, max_value=10)

        # Dimensionality Reduction
        sc.tl.pca(adata, svd_solver='arpack')

        # Compute Neighborhood Graph
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

        # Clustering
        sc.tl.leiden(adata, resolution=0.5)

        # Plotting the results
        # sc.tl.umap(adata)
        # sc.pl.umap(adata, color='leiden')

        la.classify_cells(adata, species = "mouse")

        return la.cluster_annotations(adata, species = "mouse", clusters="leiden")

if __name__ == '__main__':
    unittest.main()