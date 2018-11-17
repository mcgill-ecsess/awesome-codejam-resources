import os
import sys
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def convert_dset_to_json(dataset):
    
    dataset_json = []
    for elem in dataset:
        dataset_json.append({})
        dataset_json[-1]["input"] = elem[0]
        dataset_json[-1]["target"] = elem[1]
    
    return dataset_json

def download_data(download_url="https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip",
                  folder_name="geolife_trajectories"):
    """
    Parameters
    ----------
    download_url : string
        url of Geolife dataset
    folder_name : string
        name of folder where the data will be stored
    """
    
    import requests
    import zipfile
    import io
    
    # download data if does not exist yet
    if not os.path.exists(os.path.join(os.getcwd(), folder_name)):
        print("Downloading the dataset...")
        r = requests.get(download_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
        os.rename("Geolife Trajectories 1.3", folder_name)   
        print("Done!")
    else:
        print("Data has already been downloaded!")

def show_trip_length_distribution(folder_name="geolife_trajectories", 
                                  n_bins=100, lower=50, upper=2000):
    """
    Parameters
    ----------
    folder_name : string
        name of folder where the data is stored
    lower : int
        shortest trip length shown by histogram
    upper : int
        longest trip length shown by histogram
    """
    print("Creating a histogram showing the trip length distribution...")
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # create list of trip lengths
    trip_lengths = []
    # loop over users
    for user in os.listdir(os.path.join(os.getcwd(), folder_name, "Data")):
        user_path = os.path.join(os.getcwd(), folder_name, "Data", user)
        # loop over trips of each user
        for trip in os.listdir(os.path.join(user_path, "Trajectory")):
            trip_path = os.path.join(user_path, "Trajectory", trip)
            data = pd.read_csv(trip_path, header=None, skiprows=6)
            trip_lengths.append(len(data))

    # create histogram to show trip length distribution
    plt.hist(trip_lengths, bins=n_bins, range=(lower, upper))
    plt.show()
    
def show_position_distribution(folder_name="geolife_trajectories", n_bins=100, 
                               long_lower=116.25, long_upper=116.50,
                               lat_lower=39.85, lat_upper=40.10):
    """
    Parameters
    ----------
    folder_name : string
        name of folder where the data is stored
    long_lower : int
        minimum longitude shown by histogram
    long_upper : int
        maximum longitude shown by histogram
    lat_lower : int
        minimum latitude shown by histogram
    lat_upper : int
        maximum latitude shown by histogram
    """
    print("Creating histograms showing the position distribution...")
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # create list of longitudes and latitudes
    longs = []
    lats = []
    # loop over users
    for user in os.listdir(os.path.join(os.getcwd(), folder_name, "Data")):
        user_path = os.path.join(os.getcwd(), folder_name, "Data", user)
        # loop over trips of each user
        for trip in os.listdir(os.path.join(user_path, "Trajectory")):
            trip_path = os.path.join(user_path, "Trajectory", trip)
            data = pd.read_csv(trip_path, header=None, skiprows=6)
            
            lats += [elem for elem in data.iloc[:,0].values]
            longs += [elem for elem in data.iloc[:,1].values]

    plt.hist(lats, range=(lat_lower, lat_upper), bins=n_bins)
    plt.title("Latitudes distribution")
    plt.xlabel("Latitude")
    plt.ylabel("Number of occurences")
    plt.show()
    plt.clf()

    plt.hist(longs, range=(long_lower, long_upper), bins=n_bins)
    plt.title("Longitudes distribution")
    plt.xlabel("Longitude")
    plt.ylabel("Number of occurences")
    plt.show()
    
def create_datasets(folder_name="geolife_trajectories", 
                    trip_length_lower=50, trip_length_upper=2000,
                    long_lower=116.25, long_upper=116.5,
                    lat_lower=39.85, lat_upper=40.1,
                    train_frac=0.8, valid_frac=0.1, standardize=True):
    """
    Parameters
    ----------
    folder_name : string
        name of folder where the data is stored
    lower : int
        Minimum length of kept trips
    upper : int
        Maximum length of kept trips
    train_frac : float
        Proportion of whole dataset dedicated to training set
    valid_frac : float
        Proportion of whole dataset dedicated to validation set
    """
    print("Splitting the dataset into train, valid, dev ...")
    import pandas as pd
    import numpy as np
    import json
    
    assert (train_frac + valid_frac) <= 1.

    # store each trip with proper length here
    combined_dset = []
    
    # used to compute stddev and mean without keeping everything point
    n_pts = 0
    sum_longs = 0
    sum_lats = 0
    sq_sum_longs = 0
    sq_sum_lats = 0 

    # loop over users
    for user in os.listdir(os.path.join(os.getcwd(), folder_name, "Data")):
        user_path = os.path.join(os.getcwd(), folder_name, "Data", user)
        out_of_bounds=False
        # loop over trips of each user
        for trip in os.listdir(os.path.join(user_path, "Trajectory")):
            trip_path = os.path.join(user_path, "Trajectory", trip)
            data = pd.read_csv(trip_path, header=None, skiprows=6)
            
            # filter trips by length
            if len(data) < trip_length_lower or len(data) > trip_length_upper:
                continue

            latitudes = data.iloc[:,0].values
            longitudes = data.iloc[:,1].values
            
            # filter trips by position
            for i in range(len(data)):
                if latitudes[i] < lat_lower or latitudes[i] > lat_upper or longitudes[i] < long_lower or longitudes[i] > long_upper:
                    out_of_bounds=True
                    continue
            if out_of_bounds:
                continue
            
            n_pts += len(longitudes)
            sum_longs += np.sum(longitudes)
            sum_lats += np.sum(latitudes)
            sq_sum_longs += np.sum(longitudes * longitudes)
            sq_sum_lats += np.sum(latitudes * latitudes)

            # store data so that full trip is trajectory, last point is target for supervised learning
            trajectory = [[latitudes[i], longitudes[i]] for i in range(len(data))]
            target = [latitudes[-1], longitudes[-1]]

            combined_dset.append((trajectory, target))
                
    if standardize:
        # Instead of standardizing right away, we standardize on the fly. 
        # this makes it easier to plot the trajectories on the maps
        mean_long = sum_longs / n_pts
        mean_lat = sum_lats / n_pts
        std_long = np.sqrt(sq_sum_longs / n_pts - mean_long * mean_long)
        std_lat = np.sqrt(sq_sum_lats / n_pts - mean_lat * mean_lat)
    
    # shuffle dataset
    np.random.shuffle(combined_dset)
    
    # split into train, valid, test set
    train_cutoff = int(len(combined_dset) * train_frac)
    valid_cutoff = int(len(combined_dset) * valid_frac) + train_cutoff
    train_dset = combined_dset[:train_cutoff]
    valid_dset = combined_dset[train_cutoff:valid_cutoff]
    test_dset = combined_dset[valid_cutoff:]
    
    print("Saving the datasets created...")
    # saving training set
    with open(os.path.join(os.getcwd(), folder_name, 'train.json'), 'w') as json_file:
        json.dump(convert_dset_to_json(train_dset), json_file)

    # saving valid set
    with open(os.path.join(os.getcwd(), folder_name, 'valid.json'), 'w') as json_file:
        json.dump(convert_dset_to_json(valid_dset), json_file)

    # saving valid set
    with open(os.path.join(os.getcwd(), folder_name, 'test.json'), 'w') as json_file:
        json.dump(convert_dset_to_json(test_dset), json_file)
        
    # saving mean and std info. Really important when comes the time to predict the output
    with open(os.path.join(os.getcwd(), folder_name, 'mean_std.json'), 'w') as json_file:
        json.dump({"mean_long": mean_long, "mean_lat": mean_lat,
                   "std_long": std_long, "std_lat": std_lat}, json_file)
    
def pad_batch(batch):
    # padding each trip with [0,0]
    
    max_length = len(max(batch, key=len))
    padded = [trip + [[0, 0]] * (max_length - len(trip)) for trip in batch]
    
    return padded

def create_batched_dset(dset, batch_size=64, sort_by_length=True, standardize=True, shuffle=False):
    
    if sort_by_length:
        dset.sort(key=lambda trip : len(trip["input"]))
        
    # extract inputs and targets separately (easier to construct batched dset after)
    # converting to torch tensor at the same time
    inputs = [trip["input"] for trip in dset]
    targets = [trip["target"] for trip in dset]

    batched_dset = []
    for i in range(len(dset) // batch_size):
        # grouping into batches
        
        if i != ((len(dset) // batch_size) - 1):
            input_batch = inputs[batch_size*i : batch_size*(i+1)]
            target_batch = targets[batch_size*i : batch_size*(i+1)]
        else:
            input_batch = inputs[batch_size*i:]
            target_batch = targets[batch_size*i:]
            
        # padding input so that all items are the same length as the longest one from that batch
        input_batch = pad_batch(input_batch)
        
        # converting to torch tensor and adding to batched dataset
        batched_dset.append((torch.FloatTensor(input_batch).permute(1, 0, 2), 
                             torch.FloatTensor(target_batch)))
    if shuffle:
        import numpy as np
        np.random.shuffle(batched_dset)
        
    return batched_dset

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    
    
    Taken from https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    """
    import numpy as np
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km
    
if __name__ == "__main__":
    
    #download_data()
    
    #show_trip_length_distribution()
    
    create_datasets()