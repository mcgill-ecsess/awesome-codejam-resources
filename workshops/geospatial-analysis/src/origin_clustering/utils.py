import os
import sys

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

def create_datasets(folder_name="geolife_trajectories",
                    long_lower=116.25, long_upper=116.5,
                    lat_lower=39.85, lat_upper=40.1):
    """
    Parameters
    ----------
    folder_name : string
        name of folder where the data is stored
    long_lower : int
        minimum longitude kept
    long_upper : int
        maximum longitude kept
    lat_lower : int
        minimum latitude kept
    lat_upper : int
        maximum latitude kept
    """

    import pandas as pd
    import numpy as np
    import json
    
    # store each trip with proper length here
    dset = []

    # loop over users
    print("Retrieving origins and destinations from all trips ...")
    for user in os.listdir(os.path.join(os.getcwd(), folder_name, "Data")):
        user_path = os.path.join(os.getcwd(), folder_name, "Data", user)
        out_of_bounds=False
        # loop over trips of each user
        for trip in os.listdir(os.path.join(user_path, "Trajectory")):
            trip_path = os.path.join(user_path, "Trajectory", trip)
            data = pd.read_csv(trip_path, header=None, skiprows=6)

            # we only keep beginning and end points
            for point in [data.iloc[0,:2].values, data.iloc[-1,:2].values]:
            #for point in [data.iloc[0,:2].values]:
                if point[0] < lat_lower or point[0] > lat_upper or point[1] < long_lower or point[1] > long_upper:
                    # out of bounds
                    continue
                else:
                    dset.append(point)

    print("Saving the dataset created...")
    np.savetxt(os.path.join(os.getcwd(), folder_name, "origins.csv"), dset)
