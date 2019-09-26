"""
@author:    1. Abdul Hakim Shanavas
            2. Maha Krishnan Krishnan
@date:      04-10-2019
"""

# Importing necessary packages
import simplekml
import sys
import math
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from datetime import datetime
import glob
import os


def dmsTodd(deg, direction):
    """
    Converting from Degrees,Minutes,Seconds to Decimal degrees
    :param deg:
    :param direction:
    :return: decimal degrees
    """

    # Split the degrees variable by '.'
    fields = deg.split('.')

    # Identify the location of the stop slicing to split the given degree into degrees and minutes
    if len(fields[0]) == 5:
        stop = 3
    else:
        stop = 2

    # Get the degrees and minutes from the identified stop slicing
    degrees = float(deg[:stop])
    minutes = float(deg[stop:])
    seconds = 0.016667

    # Based on the direction of the degree, identify its respective decimal value
    decimal_degrees = degrees + ((minutes / seconds) / 3600)
    if direction == 'S' or direction == 'W':
        decimal_degrees *= -1

    return decimal_degrees


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # lat1, lon1, lat2, lon2 = coordinate_1[0], coordinate_1[1], coordinate_2[0], coordinate_2[1]
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6373  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def getLocalTime(UTC):
    """
    converts UTC time to local time
    time_to_convert: time to be converted
    return: converted time
    """

    # Casting the ginven UTC into float type
    UTC = float(UTC)

    # Converting that into hours, minutes and seconds
    hours = int(UTC / 10000)
    mins = int((UTC % 10000) / 100)
    sec = int((UTC) % 100)

    # Storing that into the list and return it
    localTime = [hours, mins, sec]

    return localTime


def kmlPoint(kml, sign, row):
    '''
    Plotting the coordinates into the KML form for stops, left_turn and right_turn
    :param kml: KML object to plot coordinates
    :param sign: Designated sign either left or right or stop
    :param row: Containing Latitude and Longitude coordinates
    :return: None
    '''

    # if the coordinate is for left_turn then mark it as 'Left' into KML
    if sign is 'Left':
        point = kml.newpoint(name='Left', coords=[(row['Lon'], row['Lat'])])
        point.style.iconstyle.color = simplekml.Color.red

    # if the coordinate is for right_turn then mark it as 'Right' into KML
    elif sign is 'Right':
        point = kml.newpoint(name='Right', coords=[(row['Lon'], row['Lat'])])
        point.style.iconstyle.color = simplekml.Color.green

    # if the coordinate is for stops then mark it as 'Stop' into KML
    elif sign is 'Stop':
        point = kml.newpoint(name='Stop', coords=[(row['Lon'], row['Lat'])])
        point.style.iconstyle.color = simplekml.Color.yellow


def getTurns_1(gps_dataFrame, kml):
    '''
    Identify which of the coordinates are turned for the left and right turn from each data frame
    and store that into the kml to visualize for each dataframe in google earth and consoildate those
    coordinates with rest of the left and rightfile for the end process
    :param gps_dataFrame: Pandas dataframe
    :param kml: simple kml object
    :return: None
    :param gps_dataFrame:
    :param kml:
    :return:
    '''
    index = 0

    # Opening the left and right turn files
    with open('Results/left_turns.txt', 'a') as left_turns, open('Results/right_turns.txt', 'a') as right_turns:

        # Providing the sliding window of 10 to compute the left and right turn
        # Iterating through all the datapoints in the data frame and compute the angle between the first coordinate
        # and last coordinate of the sliding window
        while index < len(gps_dataFrame.index) - 10:

            # Based on the sliding window find the angle between two coordinates
            angle_1 = gps_dataFrame.iloc[index]['Angle']
            angle_2 = gps_dataFrame.iloc[index + 10]['Angle']
            angle = angle_1 - angle_2

            row = gps_dataFrame.iloc[index + 10]

            # if the angle is between the range of -50 to -90 or 260 to 300 assume that the vehicle is
            # turning toward right
            if (angle < -50 and angle > -90) or (angle > 260 and angle < 300):
                kmlPoint(kml, 'Right', gps_dataFrame.iloc[index + 10])
                right_turns.write(str(row['Lat']) + ',' + str(row['Lon']) + '\n')

            # if the angle is between the range of 55 to 100 assume that the vehicle is
            # turning toward left
            if (angle > 55 and angle < 100):
                kmlPoint(kml, 'Left', gps_dataFrame.iloc[index + 10])
                left_turns.write(str(row['Lat']) + ',' + str(row['Lon']) + '\n')

            index += 10


def getStopsList(gps_dataFrame, kml):
    '''
    Identify which of the coordinates are stopped for the stop sign from each data frame
    and store that into the kml to visualize for each dataframe in google earth and consoildate those
    coordinates with rest of the stopfile for the end process
    :param gps_dataFrame: Pandas dataframe
    :param kml: simple kml object
    :return: None
    '''

    # Opening the stop file containing all the consolidated stop coordinates
    with open("Results/stops.txt", "a") as stopFile:

        # iterating the dataframes
        for index, row in gps_dataFrame.iterrows():

            # if speed in miles is 0.0 then add into the kml and stop list
            if row['Speed'] == 0.0:
                kmlPoint(kml, 'Stop', row)
                stopFile.write(str(row['Lat']) + ',' + str(row['Lon']) + '\n')

            # if speed is less than 10 miles then check the distance between the current and previous coordinates,
            # if its less than 0.005 then assume that the vehicle is stopped for the stop sign and add it into the
            # stop list and kml
            elif row['Speed'] < 10.0:
                if index > 0:
                    lat1 = row['Lat']
                    lon1 = row['Lon']
                    lat2 = gps_dataFrame.iloc[index - 1]['Lat']
                    lon2 = gps_dataFrame.iloc[index - 1]['Lon']
                    distance = haversine(lat1, lon1, lat2, lon2)
                    if distance > 0.005:
                        kmlPoint(kml, 'Stop', row)
                        stopFile.write(str(row['Lat']) + ',' + str(row['Lon']) + '\n')
                else:
                    # If accessing the first value, then add it into the kml and stoplist
                    kmlPoint(kml, 'Stop', row)
                    stopFile.write(str(row['Lat']) + ',' + str(row['Lon']) + '\n')


def preprocess(inputFile):
    '''
    Preprocessing the GPRMC data from the .txt file and store that into the dataframe
    :param inputFile: full file path name
    :return: pandas dataframe containing the preprocessed GPRMC data
    '''

    # Reading each line from the particular file and store into the list
    file = open(inputFile, 'r')
    lines = file.readlines()
    file.close()

    # Headers for the dataframe
    columns = ['Time', 'Lat', 'Lon', 'Speed', 'Angle']
    gps_dataFrame = pd.DataFrame(columns=columns)

    # Conversion rate from Nautical miles unit into standard mile
    speed_conversion_constant = 1.15078
    index = 0
    previous_second = 0

    # for each line in the .txt file
    for line in lines:

        # Process only the GPRMC data
        if '$GPRMC' in line and line.count('$') == 1:

            # Split the line by ','
            fields = line.split(',')
            row = []

            # if the particular field values in the list are not empty then append those values into the list
            if fields[1] != '' and fields[3] != '' and fields[4] != '' and \
                    fields[5] != '' and fields[6] != '' and fields[7] != '' and fields[8] != '':

                # Converting UTC to local time
                row.append(getLocalTime(fields[1]))
                row.append(dmsTodd(fields[3], fields[4]))
                row.append(dmsTodd(fields[5], fields[6]))
                row.append(float(fields[7]) * speed_conversion_constant)
                row.append(float(fields[8]))

                if index > 0:
                    # if the time difference between the current and previous line based onthe seconds unit
                    # is zero then dont add the particular detail into the dataframe
                    time_difference = row[0][2] - previous_second
                    if time_difference != 0:
                        gps_dataFrame.loc[index] = row

                # if its the first value into the dataframe then add
                else:
                    gps_dataFrame.loc[index] = row

                # Update the previous_second with the current time second value
                previous_second = row[0][2]
                index += 1

    # dropping the duplicates of combined latitude and longitude coordinates and reset the index into the dataframe
    gps_dataFrame.drop_duplicates(subset=['Lat', 'Lon'], inplace=True)
    gps_dataFrame = gps_dataFrame.reset_index()
    print(gps_dataFrame)

    return gps_dataFrame


def gps(inputFiles):
    '''
    This function will process the GPS RMC data from .txt files and parse into KML containing the line string
    of coordinates and speed and store in as 'name'.kml in the kml_files directory
    :param inputFiles: Containing the list of all the .txt files in the specified path
    :return: None
    '''

    # Creating simple kml object
    kml = simplekml.Kml(open=1)

    # Path of the directory to store the generated KML's for each .txt file
    save_path = 'kml_files'

    # For each .txt file
    for file in inputFiles:

        # Preprocess only the RMC data into the pandas dataframe
        gps_dataFrame = preprocess(file)

        # Get stops and turns list from the data frame
        getStopsList(gps_dataFrame, kml)
        getTurns_1(gps_dataFrame, kml)

        # Store the gps Latitude, Longitide and Speed information into the KML linestring object
        line_string = kml.newlinestring(name='A LineString', description='Speed in Knots, instead of altitude.')
        for index, row in gps_dataFrame.iterrows():
            line_string.coords.addcoordinates([(row['Lon'], row['Lat'], row['Speed'])])
        line_string.extrude = 1

        # Specifying the line string as yellow, width as 12
        line_string.altitudemode = simplekml.AltitudeMode.absolute
        line_string.style.linestyle.width = 12
        line_string.style.linestyle.color = simplekml.Color.yellow

        # Getting the file name from the .txt and generate the same with .kml extension
        kml_file = file.split('.')
        kml_file = kml_file[0].split('/')
        kml_file_name = kml_file[1] + '.kml'
        completeName = os.path.join(save_path, kml_file_name)

        # store the simple kml path into the particular path
        kml.save(completeName)

    print('Done!')


if __name__ == '__main__':
    # if len(sys.argv) < 3:
    #     print('Usage: python3 CS720_GPS_Project_MK_Hakim_kml_file_generator <inputFileName> <kmlFileName>')
    #     sys.exit(1)
    # # path = sys.argv[1]
    # # kmlFileName = sys.argv[2]

    path = 'input_files'

    # Pick all the .txt file containing in the specified path and store that into the list
    inputFiles = [file for file in glob.glob(path + "**/*.txt", recursive=True)]

    # Call this function to process the GPS data
    gps(inputFiles)
