import numpy as np
import re
import aopy
from aopy.preproc.bmi3d import segment

TRIAL_START = 2
REWARD = 48

def build_file_df(df, file_names_sorted, data_dir, traj = True):
    '''
    Builds a dataframe from given files with metadata (runtime, reward trial length,
    trajectory amplitude). Reports problematic files.
    
    Args:
        df (pd dataframe): dataframe with a row for each file to process
        file_names_sorted (list): list of file names, sorted by TE number
        data_dir (str): directory where files are found
        traj (bool): whether to include trajectory amplitude as a column
    
    Returns:
        pd dataframe: dataframe with processed metadata information for each given
        file, where each row is one file
    '''

    time2 = 5
    problem_flag = False
    
    print('Building dataset...')
    
    for i, f in enumerate(file_names_sorted):
        files = dict(hdf=f)
        data, metadata = aopy.preproc.bmi3d._parse_bmi3d_v1(data_dir, files)
        bmi3d_task = data['bmi3d_task']
        
        pattern = r'"runtime":\s*([0-9.]+)'

        try:
            match = re.search(pattern, metadata['report'])
            runtime = round(float(match.group(1)))

        except:
            if not problem_flag:
                print('Problematic files: \n')
                problem_flag = True
            print(f)
            continue
        
        df.loc[i, 'Runtime'] = runtime
        
        if traj:
            try:
                df.loc[i, 'Trajectory Amplitude'] = np.ceil(max(abs(min(bmi3d_task['current_target'][:, 2])), 
                                                                abs(max(bmi3d_task['current_target'][:, 2]))))
            except:
                if not problem_flag:
                    print('Problematic files: \n')
                    problem_flag = True
                print(f)
                continue
            
        # trial length
        rewarded = segment([TRIAL_START], [REWARD], data)[1]
        time = []
        for j in (rewarded):
            time.append(j[-1] - j[0])
            time2 = round((np.median(time))/metadata['fps'])
    
        df.loc[i, 'Reward trial length'] = time2
    return df

