import torch
from torch import nn
from torch import optim
#from utils import device
from src.destination_prediction.utils import device, haversine_np
import numpy as np

def train(model, train_set, mean_std_data, valid_set=None, n_epochs=20, start_lr=1e-4, trip_frac=1.,lr_decay=True):
    
    train_loss_tracker = []
    eval_loss_tracker = []
    
    for epoch in range(n_epochs):
        
        if lr_decay:
            if epoch < 100:
                lr = start_lr
            else:
                # learning rate decays linearly
                lr = (n_epochs - epoch)/n_epochs * start_lr
        else:
            lr = start_lr
#         print("learning rate : %.6f" % lr)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
                
        total_distance_error = 0
        n_pts = 0
        
        for i, (inputs, targets) in enumerate(train_set):
            
            optimizer.zero_grad()
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # truncate trip to keep only beginning
            inputs = inputs[:int(trip_frac*inputs.shape[0])]
            
            # standardize inputs and targets
            inputs[:,:,0] = (inputs[:,:,0] - mean_std_data["mean_lat"]) / mean_std_data["std_lat"]
            inputs[:,:,1] = (inputs[:,:,1] - mean_std_data["mean_long"]) / mean_std_data["std_long"]
            
            targets[:,0] = (targets[:,0] - mean_std_data["mean_lat"]) / mean_std_data["std_lat"]
            targets[:,1] = (targets[:,1] - mean_std_data["mean_long"]) / mean_std_data["std_long"]
            
            out_lat, out_long = model(inputs)

            loss = criterion(out_lat.squeeze(), targets[:,0])
            loss += criterion(out_lat.squeeze(), targets[:,1])
            
            loss.backward()
            optimizer.step()

            # compute mean distance error (km)
            n_pts += inputs.shape[1]
            
            out_lat = out_lat.squeeze().data.to('cpu').numpy()
            out_long = out_long.squeeze().data.to('cpu').numpy()
            tgt_lat = targets[:,0].squeeze().to('cpu').numpy()
            tgt_long = targets[:,1].squeeze().to('cpu').numpy()
            
            
#             print(out_lat, out_long, tgt_lat, tgt_long)
            
            # un-standardize data
            out_lat = out_lat * mean_std_data["std_lat"] + mean_std_data["mean_lat"]
            out_long = out_long * mean_std_data["std_long"] + mean_std_data["mean_long"]
            tgt_lat = tgt_lat * mean_std_data["std_lat"] + mean_std_data["mean_lat"]
            tgt_long = tgt_long * mean_std_data["std_long"] + mean_std_data["mean_long"]
            
#             print(out_lat, out_long, tgt_lat, tgt_long)

            total_distance_error += np.sum(haversine_np(out_long, out_lat, tgt_long, tgt_lat))
        
        mean_distance_error = total_distance_error / n_pts
        
        mean_valid_distance_error = evaluate(model, valid_set, mean_std_data, trip_frac=trip_frac)
            
        print("Epoch %d ----- mean distance error : %.3f ----- mean valid distance error : %.3f" % (epoch, mean_distance_error, mean_valid_distance_error))            

def train_clf(model, train_set, mean_std_data, valid_set=None, n_epochs=20, start_lr=1e-4, trip_frac=1.,lr_decay=True):
    
    import osmnx as ox
    
    train_loss_tracker = []
    eval_loss_tracker = []
    
    for epoch in range(n_epochs):
        
        if lr_decay:
            if epoch < 100:
                lr = start_lr
            else:
                # learning rate decays linearly
                lr = (n_epochs - epoch)/n_epochs * start_lr
        else:
            lr = start_lr
#         print("learning rate : %.6f" % lr)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
                
        total_distance_error = 0
        n_pts = 0
        
        for i, (inputs, targets) in enumerate(train_set):
            
            optimizer.zero_grad()
            
            inputs = inputs.to(device)
            
            # find nearest node for classification
            nearest_nodes = torch.LongTensor([ox.get_nearest_node(model.graph, (targets[i,0].item(), targets[i, 1].item())) for i in range(len(targets))]).to(device)
            
            # truncate trip to keep only beginning
            inputs = inputs[:int(trip_frac*inputs.shape[0])]
            
            # standardize inputs and targets
            inputs[:,:,0] = (inputs[:,:,0] - mean_std_data["mean_lat"]) / mean_std_data["std_lat"]
            inputs[:,:,1] = (inputs[:,:,1] - mean_std_data["mean_long"]) / mean_std_data["std_long"]
            
            output = model(inputs)
            
            loss = criterion(output, nearest_nodes)
            
            loss.backward()
            optimizer.step()

            # compute mean distance error (km)
            n_pts += inputs.shape[1]
            
            predicted_nodes = torch.argmax(output, dim=1).to('cpu').numpy()

            out_long = np.zeros(len(targets))
            out_lat = np.zeros(len(targets))
            for i in range(len(predicted_nodes)):
                out_long[i] = model.graph.node[predicted_nodes[i]]['x']
                out_lat[i] = model.graph.node[predicted_nodes[i]]['y']

            tgt_lat = targets[:,0].squeeze().to('cpu').numpy()
            tgt_long = targets[:,1].squeeze().to('cpu').numpy()

            total_distance_error += np.sum(haversine_np(out_long, out_lat, tgt_long, tgt_lat))
        
        mean_distance_error = total_distance_error / n_pts
        
        mean_valid_distance_error = evaluate_clf(model, valid_set, mean_std_data, trip_frac=trip_frac)
            
        print("Epoch %d ----- mean distance error : %.3f ----- mean valid distance error : %.3f" % (epoch, mean_distance_error, mean_valid_distance_error))            

        
def evaluate(model, valid_set, mean_std_data, trip_frac=1.):
        
    total_distance_error = 0
    n_pts = 0    
    
    for i, (inputs, targets) in enumerate(valid_set):
        
        inputs = inputs.to(device)
        targets = targets.to(device)

        # truncate trip to keep only beginning
        inputs = inputs[:int(trip_frac*inputs.shape[0])]

        # standardize inputs and targets
        inputs[:,:,0] = (inputs[:,:,0] - mean_std_data["mean_lat"]) / mean_std_data["std_lat"]
        inputs[:,:,1] = (inputs[:,:,1] - mean_std_data["mean_long"]) / mean_std_data["std_long"]

        targets[:,0] = (targets[:,0] - mean_std_data["mean_lat"]) / mean_std_data["std_lat"]
        targets[:,1] = (targets[:,1] - mean_std_data["mean_long"]) / mean_std_data["std_long"]
        
        out_lat, out_long = model(inputs)

        # compute mean distance error (km)
        n_pts += inputs.shape[1]

        out_lat = out_lat.squeeze().data.to('cpu').numpy()
        out_long = out_long.squeeze().data.to('cpu').numpy()
        tgt_lat = targets[:,0].squeeze().to('cpu').numpy()
        tgt_long = targets[:,1].squeeze().to('cpu').numpy()

        # un-standardize data
        out_lat = out_lat * mean_std_data["std_lat"] + mean_std_data["mean_lat"]
        out_long = out_long * mean_std_data["std_long"] + mean_std_data["mean_long"]
        tgt_lat = tgt_lat * mean_std_data["std_lat"] + mean_std_data["mean_lat"]
        tgt_long = tgt_long * mean_std_data["std_long"] + mean_std_data["mean_long"]
        
        total_distance_error += np.sum(haversine_np(out_long, out_lat, tgt_long, tgt_lat))

    mean_distance_error = total_distance_error / n_pts   
    return mean_distance_error

def evaluate_clf(model, valid_set, mean_std_data, trip_frac=1.):

    import osmnx as ox    
    
    total_distance_error = 0
    n_pts = 0    
    
    for i, (inputs, targets) in enumerate(valid_set):
        
        inputs = inputs.to(device)

        # find nearest node for classification
        nearest_nodes = torch.LongTensor([ox.get_nearest_node(model.graph, (targets[i,0].item(), targets[i, 1].item())) for i in range(len(targets))]).to(device)

        # truncate trip to keep only beginning
        inputs = inputs[:int(trip_frac*inputs.shape[0])]

        # standardize inputs and targets
        inputs[:,:,0] = (inputs[:,:,0] - mean_std_data["mean_lat"]) / mean_std_data["std_lat"]
        inputs[:,:,1] = (inputs[:,:,1] - mean_std_data["mean_long"]) / mean_std_data["std_long"]

        output = model(inputs)

        # compute mean distance error (km)
        n_pts += inputs.shape[1]
        
        predicted_nodes = torch.argmax(output, dim=1).to('cpu').numpy()

        out_long = np.zeros(len(targets))
        out_lat = np.zeros(len(targets))
        for i in range(len(predicted_nodes)):
            out_long[i] = model.graph.node[predicted_nodes[i]]['x']
            out_lat[i] = model.graph.node[predicted_nodes[i]]['y']

        tgt_lat = targets[:,0].squeeze().to('cpu').numpy()
        tgt_long = targets[:,1].squeeze().to('cpu').numpy()

        total_distance_error += np.sum(haversine_np(out_long, out_lat, tgt_long, tgt_lat))

    mean_distance_error = total_distance_error / n_pts   
    return mean_distance_error


if __name__ == "__main__":
    
    from model import DestinationLSTM
    from utils import device
    
    seq_len = 2000
    batch_size = 64
    input_size=2
    
    inputs = torch.ones((seq_len, batch_size, input_size)).to(device)
    targets = torch.ones((batch_size, input_size))
    
    model = DestinationLSTM().to(device)
    
    train(model, train_set=[(inputs, targets)], n_epochs=100)
    
    evaluate(model, valid_set=[(inputs, targets)])    