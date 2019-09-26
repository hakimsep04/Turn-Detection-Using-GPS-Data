'''
@author:    1. Abdul Hakim Shanavas
            2. Maha Krishnan Krishnan
@date:      04-18-2019
'''

# Importing necessary packages
import simplekml
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import DistanceMetric
import pandas as pd
import glob
from matplotlib import pyplot as plt
import numpy as np
from math import radians, cos, sin, asin, sqrt


def haversine(coordinate_1, coordinate_2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lat1, lon1, lat2, lon2 = coordinate_1[0], coordinate_1[1], coordinate_2[0], coordinate_2[1]

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6373  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def readFile(file_name):
    '''
    Reading the three generated text files which contains the coordinates of the stops, left_turns and right_turns
    in to the pandas dataframe
    :param file_name: Name of the particular text including path
    :return: 2D Numpy array
    '''

    # Reading the text files into the pandas dataframe
    dataFrame = pd.read_csv(file_name, sep=',', header=None)

    # Removing duplicate entries
    dataFrame.drop_duplicates(inplace=True)
    return dataFrame.values


def kmlPoint(kml, sign, coordinates):
    '''
    After agglomeration process of stops, left_turn and right_turn plotting the coordinates into the KML form
    :param kml: KML object to plot coordinates
    :param sign: Designated sign either left or right or stop
    :param coordinates: Latitude and Longitude coordinate
    :return: None
    '''

    # if the coordinate is for left_turn then mark it as 'Left' into KML
    if 'left' in sign:
        point = kml.newpoint(name='Left', coords=[(coordinates[1], coordinates[0])])
        point.style.iconstyle.color = simplekml.Color.red

    # if the coordinate is for right_turn then mark it as 'Right' into KML
    elif 'right' in sign:
        point = kml.newpoint(name='Right', coords=[(coordinates[1], coordinates[0])])
        point.style.iconstyle.color = simplekml.Color.green

    # if the coordinate is for stops then mark it as 'Stop' into KML
    elif 'stops' in sign:
        point = kml.newpoint(name='Stop', coords=[(coordinates[1], coordinates[0])])
        point.style.iconstyle.color = simplekml.Color.yellow


def aglo(processed_files, n_clusters=80):
    '''
    This function will agglomerate the group of closed coordinates into single coordinate, plot into KML
    and mark it as Left or Right or Stop
    :param processed_files: Containing full path of the three generated text files
    :param n_clusters: By default set to 80
    :return: None
    '''


    # Calling the readFile function to parse the text files into the dataframes and stored into the dictionary
    # key as full path name of the text file
    dataframes = {}
    for file in processed_files:
        dataframes[file] = readFile(file)

    kml1 = simplekml.Kml(open=1)
    for df in dataframes:
        for row in dataframes[df]:
            kmlPoint(kml1, df, row)
    kml1.save('Output_before_aglo.kml')

    # Performing agglomeration for the three dataframes and store the cluster labels for each of the three
    # Agglomerative process into the dictionary
    model_labels = {}
    for df in dataframes:
        dist = DistanceMetric.get_metric('haversine')
        dist_matrix = dist.pairwise(dataframes[df])
        hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='single')
        hc.fit(dist_matrix)
        model_labels[df] = hc.labels_

    # Finding the centroid values for each clusters of the three agglomerative clusters label objects and
    # store that into the dictionary
    centroids = {}

    # For each agglomerative cluster label objects
    for model in model_labels:
        centroid = []
        # Run it through each clusters
        for cluster in range(n_clusters):
            coordinates = []
            # Take all the coordinates for each clusters and find the mean centroid coordinate and store it
            latitude = dataframes[model][model_labels[model] == cluster, 0]
            longitude = dataframes[model][model_labels[model] == cluster, 1]
            coordinates.append(np.sum(latitude) / len(latitude))
            coordinates.append(np.sum(longitude) / len(longitude))
            centroid.append(coordinates)

        centroids[model] = centroid

    # Getting centroid values list for each of the tracks and store into the separate list
    stops = []
    left = []
    right = []
    for model in centroids:
        if 'stops' in model:
            stops = centroids[model]
        if 'left' in model:
            left = centroids[model]
        if 'right' in model:
            right = centroids[model]

    # If the stops and left are within the distance of 100 meters, then remove the particular stop coordinates
    # from the list. Used Haversine distance to find the distance between two coordinates
    for idx, stop_coordinate in enumerate(stops):
        for left_coordinate in left:
            distance = haversine(stop_coordinate, left_coordinate)
            if distance < 100:
                del stops[idx]
                break

    # If the stops and right are within the distance of 100 meters, then remove the particular stop coordinates
    # from the list. Used Haversine distance to find the distance between two coordinates
    for idx, stop_coordinate in enumerate(stops):
        for right_coordinate in right:
            distance = haversine(stop_coordinate, right_coordinate)
            if distance < 100:
                del stops[idx]
                break

    # Updating the new stop list into the centroids[stops] dictionary value
    for model in centroids:
        if 'stops' in model:
            centroids[model] = stops

    # Creating the object for KML
    kml = simplekml.Kml(open=1)

    # For each GPS tracks, plot the coordinates into the KML with the designated labels and save the
    # KML as Output.kml
    for centroid in centroids:
        for coordinates in centroids[centroid]:
            kmlPoint(kml, centroid, coordinates)
    kml.save('Output.kml')


if __name__ == '__main__':
    path = 'Results'
    processed_files = [file for file in glob.glob(path + "**/*.txt", recursive=True)]
    aglo(processed_files)
