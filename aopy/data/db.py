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
def get_task_entries(subj=None, date=None, task=None, dbname='default', **kwargs):
    '''
    Get all the task entries for a particular date

    Args:
        subj: string, optional, default=None
            Specify the beginning of the name of the subject or the full name. If not specified, blocks from all subjects are returned
        date: multiple types, optional, default=today
            Query date for blocks. The date can be specified as a datetime.date object, a tuple of (start, end) datetime.date objects,
            or a 3-tuple of ints (year, month, day).
        task: string
            name of task to filter
        kwargs: dict, optional
            Additional keyword arguments

    Returns:
        database entries
    '''
    return bmi3d.get_task_entries(subj=subj, date=date, task=task, dbname=dbname, **kwargs)

def get_sessions(subject, date, task_name, session_name=None, exclude_ids=[], filter_fn=lambda x:True, dbname='default'):
    '''
    Returns list of entries for all sessions on the given date
    '''
    if session_name:
        entries = get_task_entries(subj=subject, task=task_name, date=date, session=session_name, dbname=dbname)
    else:
        entries = get_task_entries(subj=subject, task=task_name, date=date, dbname=dbname)
    if len(entries) == 0:
        warnings.warn("No entries found")
        return []
    return [e for e in entries if filter_fn(e) and e.id not in exclude_ids]
    
def get_mc_sessions(subject, date, mc_task_name='manual control', dbname='default', **kwargs):
    '''
    Returns list of entries for all bmi control sessions on the given date
    '''
    return get_sessions(subject, date, task_name=mc_task_name, dbname=dbname, **kwargs)
    
def get_bmi_sessions(subject, date, session_name='training', bmi_task_name='bmi control', dbname='default', **kwargs):
    '''
    Returns list of entries for all bmi control sessions on the given date
    '''
    return get_sessions(subject, date, session_name=session_name, task_name=bmi_task_name, dbname=dbname, **kwargs)


'''
Paramters
'''
def get_subject(entry, dbname='default'):
    '''
    Returns name of subject for session.
    Takes TaskEntry object.
    '''
    return bmi3d.get_subject(entry, dbname=dbname)

def get_id(entry, dbname='default'):
    '''
    Returns id for session.
    Takes TaskEntry object.
    '''
    return bmi3d.get_id(entry, dbname=dbname) # this doesn't exist yet

def get_time(entry, dbname='default'):
    '''
    Returns date for session.
    Takes TaskEntry object.
    '''
    return bmi3d.get_date(entry, dbname=dbname)

def get_date(entry, dbname='default'):
    '''
    Returns date for session.
    Takes TaskEntry object.
    '''
    return bmi3d.get_date(entry, dbname=dbname).date()

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
    return bmi3d.get_params(entry)[paramname]

def get_task_name(entry, dbname='default'):
    '''
    Returns name of task used for session.
    Takes TaskEntry object.
    '''
    return bmi3d.get_task_name(entry, dbname=dbname)

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

def group_entries(entries, grouping_fn=lambda te: te.calendar_date):
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
