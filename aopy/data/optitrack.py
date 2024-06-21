import csv
import os
import warnings
from pandas import read_csv

def load_optitrack_metadata(data_dir, filename, metadata_row=0):
    '''
    This function loads optitrack metadata from .csv file that has 1 rigid body
    exported with the following settings:

        | **Markers:** Off
        | **Unlabeled markers:** Off
        | **Quality Statistics:** Off
        | **Rigid Bodies:** On
        | **Rigid Body Markers:** Off
        | **Bones:** Off
        | **Bone Markers:** Off
        | **Header Information:** On
        | **Optitrack format Version(s):** 1.23

    Required packages: csv, pandas

    Args:
        data_dir (string): Directory to load data from
        filename (string): File name to load within the data directory

    Returns:
        dict: Dictionary of metadata for for an optitrack datafile
    '''

    # Constants for indexing into mocap data
    mocap_data_column_idx = 2 # Column index where data begins
    rigid_body_name_idx_csvrow = 3 #csv row that contains the rigid body name
    column_type_idx_csvrow = 5 #csv row that contains the column type 
    column_names_idx_csvrow = 6  #csv row that contains the column names

    # Initialize empty dict to add to
    mocap_metadata = {}

    filepath = os.path.join(data_dir, filename)
    # Create dict with .csv meta data
    with open(filepath) as csvfile:
        mocap_reader = csv.reader(csvfile, delimiter= ',')  # csv.reader object for optitrack data loading

        for idx_csvrow, row in enumerate(mocap_reader):
            # Create a dictionary of metadata for the first line
            if idx_csvrow == metadata_row:
                # For every pair of values in the list add to dict
                for idx_dictval in range(0, len(row), 2):
                    mocap_metadata[row[idx_dictval]]= row[idx_dictval + 1]

            # Get Rigid body name for each column
            elif idx_csvrow == rigid_body_name_idx_csvrow:
                mocap_metadata['Rigid Body Name'] = row[mocap_data_column_idx:][0]

            # Load whether column is rotation/position
            elif idx_csvrow == column_type_idx_csvrow:
                mocap_metadata['Data Column Motion Types'] = row[mocap_data_column_idx:]

            # Get column names for data
            elif idx_csvrow == column_names_idx_csvrow:
                mocap_metadata['Data Column Motion Names'] = row[mocap_data_column_idx:]  

    if 'Export Frame Rate' in mocap_metadata:
        mocap_metadata['samplerate'] = float(mocap_metadata['Export Frame Rate'])
    return mocap_metadata

def load_optitrack_data(data_dir, filename):
    '''
    This function loads a series of x, y, z positional data from the optitrack
    .csv file that has 1 rigid body exported with the following settings:

        | **Markers:** Off
        | **Unlabeled markers:** Off
        | **Quality Statistics:** Off
        | **Rigid Bodies:** On
        | **Rigid Body Markers:** Off
        | **Bones:** Off
        | **Bone Markers:** Off
        | **Header Information:** On
        | **Optitrack format Version(s):** 1.23

    Required packages: pandas, numpy

    Args:
        data_dir (string): Directory to load data from
        filename (string): File name to load within the data directory

    Returns:
        tuple: Tuple containing:
            | **mocap_data_pos (nt, 3):** Positional mocap data
            | **mocap_data_rot (nt, 4):** Rotational mocap data
    '''

    # Load the metadata to check the columns are going to line up
    mocap_metadata = load_optitrack_metadata(data_dir, filename)
    if not mocap_metadata:
        raise Exception('No metadata found for optitrack file')
    if mocap_metadata['Rotation Type'] != 'Quaternion':
        warnings.warn('Rotation type must be Quaternion')
    if mocap_metadata['Format Version'] not in ['1.23', '1.24']:
        warnings.warn(f"Export version {mocap_metadata['Format Version']} not supported")

    # Load the data columns
    column_names_idx_csvrow = 5 # Header row index
    mocap_data_rot_column_idx = range(2,6) # Column index for rotation data
    mocap_data_pos_column_idx = range(6,9) # Column indices for position data
    filepath = os.path.join(data_dir, filename)
    # Load .csv file as a pandas data frame, convert to a numpy array, and remove
    # the 'Frame' and 'Time (Seconds)' columns.
    mocap_data_rot = read_csv(filepath, header=column_names_idx_csvrow).to_numpy()[:,mocap_data_rot_column_idx]
    mocap_data_pos = read_csv(filepath, header=column_names_idx_csvrow).to_numpy()[:,mocap_data_pos_column_idx]

    return mocap_data_pos, mocap_data_rot

def load_optitrack_time(data_dir, filename):
    '''
    This function loads timestamps from the optitrack .csv file 

    Required packages: pandas, numpy

    Args:
        data_dir (string): Directory to load data from
        filename (string): File name to load within the data directory

    Returns:
        (nt): Array of timestamps for each captured frame
    '''

    column_names_idx_csvrow = 5 # Header row index
    timestamp_column_idx = 1 # Column index for time data
    filepath = os.path.join(data_dir, filename)
    # Load .csv file as a pandas data frame, convert to a numpy array, and only
    # return the 'Time (Seconds)' column
    timestamps = read_csv(filepath, header=column_names_idx_csvrow).to_numpy()[:,timestamp_column_idx]
    return timestamps
