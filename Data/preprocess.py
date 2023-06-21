import os
import pickle
import numpy as np
from scipy.sparse import coo_matrix

def load_amazon_data(data_name,
                    data_path):

    assert data_name in ["amazon1", "amazon3"]
    
    if data_name == 'amazon1': 

        domain_names = ["books", "dvd", "electronics", "kitchen"]
        input_dim = 5000  # Number of features to be used in the experiment

        # Load & process the amazon dataset
        amazon = np.load(os.path.join(data_path, "Amazon1.npz"))
        amazon_xx = coo_matrix((amazon['xx_data'], (amazon['xx_col'], amazon['xx_row'])),
                                shape=amazon['xx_shape'][::-1]).tocsc()
        amazon_xx = amazon_xx[:, :input_dim]
        amazon_yy = amazon['yy']
        amazon_yy = (amazon_yy + 1) / 2  # from {-1, 1} to {0, 1}
        amazon_offset = amazon['offset'].flatten()  # starting indices of the four domains

        # Partition the data into four domains and for each domain partition the data set into training and test set
        data_insts, data_labels, num_insts = [], [], []
        for i in range(len(domain_names)):
            data_insts.append(amazon_xx[amazon_offset[i]: amazon_offset[i+1], :])
            data_labels.append(amazon_yy[amazon_offset[i]: amazon_offset[i+1], :])
        
    elif data_name == 'amazon3':
        with open(os.path.join(data_path, 'Amazon3.pkl'), 'rb') as f:
            amazon = pickle.load(f)
            
        domain_names, data_insts, data_labels = amazon['Domain_names'], amazon['Embeddings'], amazon['Labels']
    else:
        NotImplementedError
            
    return domain_names, data_insts, data_labels