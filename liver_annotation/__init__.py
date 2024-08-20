import torch
import joblib
from joblib import load
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import scipy
import numpy as np
from joblib import load
import sys
import os
import sklearn

# Load the objects using joblib
models_dir = os.path.join(os.path.dirname(__file__), 'models')
print(f"models_dir is {models_dir}")
# Load the objects using joblib
human_gene_cols = load(os.path.join(models_dir, 'human_gene_cols.joblib'))
hle = load(os.path.join(models_dir, 'hle.joblib'))
mouse_gene_cols = load(os.path.join(models_dir, 'mouse_gene_cols.joblib'))
mle = load(os.path.join(models_dir, 'mle.joblib'))

# Define the neural network class again to match the saved architecture
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters (these should match the ones used during training)
h_hidden_size = 256
m_hidden_size = 256

# Load PyTorch models
# Initialize the models with the same architecture
human_model_nn = Net(input_size=20007, hidden_size=h_hidden_size, num_classes=20)
mouse_model_nn = Net(input_size=31053, hidden_size=m_hidden_size, num_classes=18)

# Load the state dictionaries

human_state_dict = torch.load(os.path.join(models_dir, 'human_model_nn.pth'), map_location=torch.device('cpu'))

new_human_state_dict = {k.replace('module.', ''): v for k, v in human_state_dict.items()}

human_model_nn.load_state_dict(new_human_state_dict)

mouse_state_dict = torch.load(os.path.join(models_dir, 'mouse_model_nn.pth'), map_location=torch.device('cpu'))

new_mouse_state_dict = {k.replace('module.', ''): v for k, v in mouse_state_dict.items()}

mouse_model_nn.load_state_dict(new_mouse_state_dict)

# Move models to the appropriate device and set them to evaluation mode
human_model_nn.to(device)
mouse_model_nn.to(device)
human_model_nn.eval()
mouse_model_nn.eval()

# Load scikit-learn models using joblib
mouse_model = load(os.path.join(models_dir, 'mouse_model.joblib'))
human_model = load(os.path.join(models_dir, 'human_model.joblib'))

print("All models loaded successfully.")

# Classification of cells by cell type
def classify_cells(in_data, species = "human", model_type = "nn"):
    '''
    in_data: a standard scanpy/anndata object with gene expression data stored in in_data.var_names and metadata at in_data.obs
    
    species: `human` or `mouse`. The species of the sample whose data is stored in `in_data`.
    
    model_type: `rfc` or `nn` depnding on whether you want to run a random forest classifier classification or neural network classification respectively
    '''

    if species == "human":
        gene_cols = human_gene_cols
        le = hle
        if model_type == "rfc":
            model = human_model
        elif model_type == "nn":
            model = human_model_nn
        else: raise ValueError("model_type must be either 'rfc' or 'nn'")

    elif species == "mouse":
        gene_cols = mouse_gene_cols
        le = mle
        if model_type == "rfc":
            model = mouse_model
        elif model_type == "nn":
            model = mouse_model_nn
        else: raise ValueError("model_type must be either 'rfc' or 'nn'")

    else:
        raise ValueError("species must be either 'human' or 'mouse'")

    shared_genes = [gene for gene in gene_cols if gene in in_data.var_names]

    if len(shared_genes) != len(gene_cols):
        print(f"Warning: {len(gene_cols) - len(shared_genes)} genes from the training data are not in this dataset.")
    
    mat = in_data[:, shared_genes].X

    if scipy.sparse.issparse(mat):
        mat = mat.toarray()

    #if genes not present in data, set value to 0. Since this is a sparsey matrix, 0 is the expected value.
    mat_with_missing = np.zeros((mat.shape[0], len(gene_cols)))
    existing_data = [gene_cols.index(gene) for gene in shared_genes]
    mat_with_missing[:, existing_data] = mat


    
    if model_type == "rfc":
        
        preds = model.predict(mat_with_missing)
        pred_names = le.inverse_transform(preds)
        
        prob_preds = model.predict_proba(mat_with_missing)

        in_data.obs["rfc_model_predictions"] =  pred_names

        in_data.obs["rfc_model_probabilities"] = [list(k) for k in prob_preds]
    elif model_type == "nn":
        # Convert data to PyTorch tensor
        tensor_data = torch.FloatTensor(mat_with_missing)
        
        # Create DataLoader
        batch_size = 128  # You can adjust this
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Set model to evaluation mode
        model.eval()

        # Make predictions
        all_preds = []
        all_probs = []
        
        device = next(model.parameters()).device  # Get the device of the model

        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to(device)  # Move batch to the same device as the model
                outputs = model(batch)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Convert predictions to cell type names
        pred_names = le.inverse_transform(all_preds)
        
        in_data.obs["nn_model_predictions"] = pred_names
        in_data.obs["nn_model_probabilities"] = [list(k) for k in all_probs]



def most_frequent(List):
    return max(set(List), key = List.count)

#Cluster annotation
def cluster_annotations(in_data, species, clusters="louvain", algorithm = "mode", model_type = "nn"):
    
    """
    
    in_data: a standard scanpy/anndata object with gene expression data stored in in_data.var_names and metadata at in_data.obs
    
    clusters: column in in_data.obs to use for cluster data
    
    algorithm: type of algorithm used to do cluster annotation. "mode" annotates a cluster with the most common annotation of the cells in that cluster, whereas "prob" annotates a cluster by summing the probabilities of each cell in the cluster to be of each cell type, and taking the highest sum.
    
    model_type: "rfc" or "nn", depnding on whether you want to run a random forest classifier classification or neural network classification respectively
    
    species: "mouse" or "human", only required if running model with "prob"
    """
    
    if model_type == "rfc":
        target_slot = "rfc_model_predictions"
        target_prob_slot = "rfc_model_probabilities"
    elif model_type == "nn":
        target_slot = "nn_model_predictions"
        target_prob_slot = "nn_model_probabilities"
    else:
        raise ValueError("model_type must be either 'rfc' or 'nn'")
    if algorithm == "mode":
        out = {}
        cluster_names = np.unique(in_data.obs[clusters])
        for name in cluster_names:
            filtered_data = in_data[in_data.obs[clusters]==name]
            model_predictions = filtered_data.obs[target_slot].tolist()
            out[name] = most_frequent(model_predictions)
        return out
    elif algorithm == "prob":
        if species == "human":
            le = hle
        elif species == "mouse":
            le = mle

        out = {}
        cluster_names = np.unique(in_data.obs[clusters])
        for name in cluster_names:
            filtered_data = in_data[in_data.obs[clusters]==name]
            model_probabilities = filtered_data.obs[target_prob_slot].tolist()
            sum_list = model_probabilities[0]
            for i in range(1, len(model_probabilities[0])):
                for cell in model_probabilities:
                    sum_list[i]+=cell[i]
            pred = sum_list.index(max(sum_list))
            pred_name = le.inverse_transform([pred])[0]
            out[name] = pred_name
        return out
    else:
        raise ValueError("algorithm parameter must be either 'mode' or 'prob'")