'''
Interface between database methods/models and data analysis code
'''
import json
import os
import datetime
from collections import defaultdict
try:
    from db import dbfunctions as bmi3d
except:
    pass
import warnings

'''
Lookup
'''
DB_TYPE = 'bmi3d'
BMI3D_DBNAME = 'default'

def lookup_sessions(subject=None, date=None, task_name=None, task_desc=None, session=None, project=None, 
                    experimenter=None, has_features=None,
                    exclude_ids=[], filter_fn=lambda x:True, **kwargs):
    '''
    Returns list of entries for all sessions on the given date
    '''
    if DB_TYPE == 'bmi3d':
        if subject:
            kwargs['subj'] = subject
        if date:
            kwargs['date'] = date
        if task_name:
            kwargs['task'] = task_name
        if task_desc:
            kwargs['entry_name'] = task_desc
        if session:
            kwargs['session'] = session
        if project:
            kwargs['project'] = project
        if experimenter:
            kwargs['experimenter__name'] = experimenter
        entries = bmi3d.get_task_entries(dbname=BMI3D_DBNAME, **kwargs)
        if has_features:
            filter_fn = filter_fn and (lambda x: all([has_feature(x, f) for f in has_features]))
        if len(entries) == 0:
            warnings.warn("No entries found")
            return []
        return [e for e in entries if filter_fn(e) and e.id not in exclude_ids]
    else:
        warnings.warn("Unsupported db type!")
    
def lookup_mc_sessions(subject, date, mc_task_name='manual control', **kwargs):
    '''
    Returns list of entries for all bmi control sessions on the given date
    '''
    return lookup_sessions(subject, date, task_name=mc_task_name, **kwargs)
    
def lookup_bmi_sessions(subject, date, session_name='training', bmi_task_name='bmi control', **kwargs):
    '''
    Returns list of entries for all bmi control sessions on the given date
    '''
    return lookup_sessions(subject, date, session_name=session_name, task_name=bmi_task_name, **kwargs)

'''
Paramters
'''
def get_subject(entry):
    '''
    Returns name of subject for session.
    Takes TaskEntry object.
    '''
    return bmi3d.get_subject(entry, dbname=BMI3D_DBNAME)

def get_id(entry):
    '''
    Returns id for session.
    Takes TaskEntry object.
    '''
    return entry.id

def get_time(entry):
    '''
    Returns date for session.
    Takes TaskEntry object.
    '''
    return bmi3d.get_date(entry, dbname=BMI3D_DBNAME)

def get_date(entry):
    '''
    Returns date for session.
    Takes TaskEntry object.
    '''
    return bmi3d.get_date(entry, dbname=BMI3D_DBNAME).date()

def get_features(entry):
    return [f.name for f in entry.feats.all()]

def has_feature(entry, featname):
    features = get_features(entry)
    return featname in features

def get_params(entry):
    '''
    Returns a dict of all task params for session.
    Takes TaskEntry object.
    '''
    return bmi3d.get_params(entry)

def get_param(entry, paramname):
    '''
    Returns parameter value.
    Takes TaskEntry object.
    '''
    params = bmi3d.get_params(entry)
    if paramname not in params:
        return None
    return params[paramname]

def get_task_name(entry):
    '''
    Returns name of task used for session.
    Takes TaskEntry object.
    '''
    return bmi3d.get_task_name(entry, dbname=BMI3D_DBNAME)

def get_notes(entry):
    '''
    Returns notes for session.
    Takes TaskEntry object.
    '''
    return bmi3d.get_notes(entry)

def get_length(entry):
    '''
    Returns length of session in seconds.
    Takes TaskEntry object.
    '''
    return bmi3d.get_length(entry)
    
def get_raw_files(entry, system_subfolders=None):
    '''
    Gets the raw data files associated with each task entry 
    
    Args:
        entry (TaskEntry): recording to find raw files for 
        system_subfolders (dict, optional): dictionary of system subfolders where the 
           data for that system is located. If None, defaults to the system name
    
    Returns: 
        files : list of (system, filepath) for each datafile associated with this task entry 
    '''
    return bmi3d.get_rawfiles_for_taskentry(entry, system_subfolders=system_subfolders)

'''
Wrappers
'''
def get_entry_details(entry):
    '''
    Returns subject, id, and date for the given entry
    '''
    subject = get_subject(entry)
    te = get_subject(entry)
    date = get_date(entry)
    return subject, te, date

def get_entries_details(entries):
    return zip(*[get_entry_details(e) for e in entries])

def get_preprocessed_sources(entry):
    '''
    Returns a list of datasource names that should be preprocessed for the given entry
    '''
    params = get_params(entry)
    sources = ['exp']
    if 'record_headstage' in params and params['record_headstage']:
        sources.append('broadband')
        sources.append('lfp')
    sources.append('eye')
    
    return sources

def group_entries(entries, grouping_fn=lambda te: get_date(te)):
    '''
    Automatically group together a flat list of database IDs

    Parameters
    ----------
    ids: iterable
        iterable of ints representing the ID numbers of TaskEntry objects to group
    grouping_fn: callable, optional (default=sort by date); call signature: grouping_fn(task_entry)
        Takes a dbfn.TaskEntry as its only argument and returns a hashable and sortable object
        by which to group the ids
    '''
    keyed_ids = defaultdict(list)
    for te in entries:
        key = grouping_fn(te)
        keyed_ids[key].append(id)

    keys = list(keyed_ids.keys())
    keys.sort()

    grouped_ids = []
    for date in keys:
        grouped_ids.append(tuple(keyed_ids[date]))
    return grouped_ids


'''
Filtering
'''
