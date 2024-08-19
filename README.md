# Liver Annotation

Liver Annotation is a Python package designed to annotate clusters in single-cell RNA sequencing (scRNA-seq) data from liver samples. This package provides a machine learning model that is specifically trained on liver cells, enabling out-of-the-box functionality without the need for pre-existing expert-annotated data.

## Features

- Machine learning model trained specifically on liver cells.
- Supports both neural network and random forest classifier models.
- Annotates clusters using either the most common annotation or probability-based methods.

## Installation

To install the package, use pip:

```bash
pip install liver_annotation
```

## Usage

### Classification of Cells

You can classify cells by cell type using the classify_cells function. The function requires an input in_data which is a standard scanpy/anndata object with gene expression data.

```python
from liver_annotation import classify_cells

# Example usage
classify_cells(ann_data_obj, species="human", model_type="nn")
```

- `species`: Choose between `"human"` or `"mouse"`.
- `model_type`: Choose between `"rfc"` (random forest classifier) or `"nn"` (neural network).

### Cluster Annotation

Annotate clusters using the cluster_annotations function. This function requires an input in_data and allows you to specify the clustering algorithm and model type.

```python
from liver_annotation import cluster_annotations

# Example usage
cluster_annotations(in_data, species="human", clusters="louvain", algorithm="mode", model_type="nn")
```

- `clusters`: The column in `in_data.obs` to use for cluster data.
- `algorithm`: Choose between `"mode"` or `"prob"` for cluster annotation.
- `model_type`: Choose between `"rfc"` or `"nn"`.

## Dependencies

- `torch`
- `joblib`
- `scipy`
- `numpy`
- `scanpy`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any questions or issues, please contact Madhavendra Thakur at madhavendra.thakur@gmail.com.