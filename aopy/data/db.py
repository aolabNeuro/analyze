'''
Interface between database methods/models and data analysis code
'''
import json
import os
import pickle
import sys
from collections import defaultdict
import warnings
import traceback

import pandas as pd
import numpy as np
try:
    from db import dbfunctions as bmi3d
    from db.tracker import dbq
except:
    warnings.warn("Database not configured")
    traceback.print_exc()

from .. import data as aodata
from .. import preproc

'''
Lookup
'''

# Some global variable shenanigans.
this = sys.modules[__name__]
this.DB_TYPE = 'bmi3d' # can be modified with `db.DB_TYPE` after `from aopy.data import db``
this.BMI3D_DBNAME = 'default' # can be modified with `db.BMI3D_DBNAME` after `from aopy.data import db`

def lookup_sessions(id=None, subject=None, date=None, task_name=None, task_desc=None, session=None, project=None, 
                    experimenter=None, exclude_ids=[], filter_fn=lambda x:True, **kwargs):
    '''
    Returns list of entries for all sessions on the given date

    Args:
        id (int or list, optional): Lookup sessions with the given ids, if provided.
        subject (str, optional): Lookup sessions with the given subject, if provided.
        date (multiple, optional): Lookup sessions from the given date, if provided. Accepts multiple formats:
            | datetime.date object
            | (start, end) tuple of datetime.date objects
            | (start, end) tuple of strings in the format 'YYYY-MM-DD'
            | (year, month, day) tuple of integers
        task_name (str, optional): Lookup sessions with the given task name, if provided. Examples include
            `manual control`, `tracking`, `nothing`, etc.
        task_desc (str, optional): Lookup sessions with the given task description, if provided. Examples include
            `flash`, `simple center out`, `resting state`, etc.
        session (str, optional): Lookup sessions with the given session name, if provided.
        project (str, optional): Lookup sessions with the given project name, if provided.
        experimenter (str, optional): Lookup sessions with the given experimenter, if provided.
        exclude_ids (list, optional): Exclude sessions with matching task entry ids, if provided.
        filter_fn (function, optional): Additional filtering, of signature `fn(session)->bool. 
            Defaults to `lambda x:True`.
        kwargs (dict, optional): optional keyword arguments to pass to database lookup function.

    Returns:
        list: list of TaskEntry sessions matching the query
    '''
    if this.DB_TYPE == 'bmi3d':
        if id is not None and isinstance(id, list):
            kwargs['pk__in'] = id
        elif id is not None:
            kwargs['pk'] = id
        if subject is not None:
            kwargs['subj'] = subject
        if date is not None:
            kwargs['date'] = date
        if task_name is not None:
            kwargs['task'] = task_name
        if task_desc is not None:
            kwargs['entry_name'] = task_desc
        if session is not None:
            kwargs['session'] = session
        if project is not None:
            kwargs['project'] = project
        if experimenter is not None:
            kwargs['experimenter__name'] = experimenter
        
        # Initial lookup, wrapped into BMI3DTaskEntry objects
        dbname = kwargs.pop('dbname', this.BMI3D_DBNAME)
        entries = bmi3d.get_task_entries(dbname=dbname, **kwargs)
        entries = [BMI3DTaskEntry(e, dbname=dbname) for e in entries]

        # Additional filtering
        if len(entries) == 0:
            warnings.warn("No entries found")
            return []
        return [e for e in entries if filter_fn(e) and e.id not in exclude_ids]
    else:
        warnings.warn("Unsupported db type!")
        return []
    
def lookup_flash_sessions(mc_task_name='manual control', **kwargs):
    '''
    Returns list of entries for all flash sessions on the given date.
    See :func:`~aopy.data.db.lookup_sessions` for details.
    '''
    return lookup_sessions(task_name=mc_task_name, entry_name__contains='flash', **kwargs)

def lookup_mc_sessions(mc_task_name='manual control', **kwargs):
    '''
    Returns list of entries for all manual control sessions on the given date
    See :func:`~aopy.data.db.lookup_sessions` for details.
    '''
    sessions = lookup_sessions(task_name=mc_task_name, **kwargs)
    return [s for s in sessions if s.task_desc != 'flash']

def lookup_tracking_sessions(tracking_task_name='tracking', **kwargs):
    '''
    Returns list of entries for all tracking sessions on the given date
    See :func:`~aopy.data.db.lookup_sessions` for details.
    '''
    return lookup_sessions(task_name=tracking_task_name, **kwargs)

def lookup_bmi_sessions(bmi_task_name='bmi control', **kwargs):
    '''
    Returns list of entries for all bmi control sessions on the given date
    See :func:`~aopy.data.db.lookup_sessions` for details.
    '''
    return lookup_sessions(task_name=bmi_task_name, **kwargs)

def lookup_decoder_parent(task_name='nothing', task_desc='decoder parent', **kwargs):
    '''
    Lookup by project and session
    '''
    return lookup_sessions(task_name=task_name, task_desc=task_desc, **kwargs)

def lookup_decoders(id=None, parent_id=None, **kwargs):
    '''
    Returns list of decoders with the given filter parameters

    Args:
        id (int or list, optional): Lookup decoders with the given ids, if provided.
        parent_id (int, optional): Lookup decoders with the given parent ids, if provided.
        kwargs (dict, optional): optional keyword arguments to pass to database lookup function.

    Returns:
        list: list of Decoder records matching the query
    '''
    if id is not None and isinstance(id, list):
        kwargs['id__in'] = id
    elif id is not None:
        kwargs['id'] = id
    if parent_id is not None:
        kwargs['entry_id'] = parent_id

    dbname = kwargs.pop('dbname', this.BMI3D_DBNAME)
    records = bmi3d.models.Decoder.objects.using(dbname).filter(**kwargs)
    return [BMI3DDecoder(r) for r in records]

'''
Filter functions
'''
def filter_has_features(features):
    '''
    Filter function to select sessions only if they had the given features enabled

    Args:
        features (list or str): a list of feature names, or a single feature

    Returns:
        function: a filter function to pass to `lookup_sessions`
    '''
    if not isinstance(features, list):
        filter_fn = lambda x: x.has_feature(features)
    else:
        filter_fn = lambda x: all([x.has_feature(f) for f in features])
    return filter_fn

def filter_has_neural_data(datasource):
    '''
    Filter function to select sessions only if they contain neural data recordings

    Args:
        datasource (str): 'ecog' or 'neuropixel'

    Returns:
        function: a filter function to pass to `lookup_sessions`

    '''
    if datasource == 'ecog':
        return lambda x: x.get_task_param('record_headstage')
    elif datasource == 'neuropixel':
        return lambda x: x.has_feature('neuropixel')
    else:
        warnings.warn('No datasource matching description')
        return lambda x: False

'''
Database objects
'''
class BMI3DTaskEntry():
    '''
    Wrapper class for bmi3d database entry classes. Written like this 
    so that other database types can implement their own classes with
    the same methods without needing to modfiy their database model.
    '''

    def __init__(self, task_entry, dbname='default'):
        '''
        Constructor

        Args:
            task_entry (TaskEntry): a database task entry from BMI3D
            dbname (str, optional): name of the database to connect to. Defaults to 'default'.
        '''
        self.dbname = dbname
        self.record = task_entry
        self.id = self.record.id
        self.datetime = self.record.date
        self.date = self.record.date.date()
        self.subject = self.record.subject.name
        self.session = self.record.session
        self.project = self.record.project

    def __repr__(self):
        '''
        Call __repr__ on the database record

        Returns:
            str: repr
        '''
        return repr(self.record)
    
    def __str__(self):
        '''
        Call __str__ on the database record

        Returns:
            str: str
        '''
        return str(self.record)
    
    @property
    def experimenter(self):
        '''
        Experimenter

        Returns:
            str: name of the experimenter
        '''
        if self.record.experimenter is None:
            return ''
        return self.record.experimenter.name
    
    @property
    def notes(self):
        '''
        Notes

        Returns:
            str: notes
        '''
        return self.record.notes

    @property
    def features(self):
        '''
        List of features that were enabled during recording

        Returns:
            list: enabled features
        '''
        return [f.name for f in self.record.feats.all()]

    @property
    def task_params(self):
        '''
        All task parameters

        Returns:
            dict: task params
        '''
        return self.record.task_params

    @property
    def sequence_name(self):
        '''
        Sequence name, e.g. `centerout_2D`

        Returns:
            str: sequence name
        '''
        return self.record.sequence.generator.name

    @property
    def sequence_params(self):
        '''
        All sequence parameters, e.g. `ntargets` or `target_radius`

        Returns:
            dict: sequence params
        '''
        return self.record.sequence_params

    @property
    def task_name(self):
        '''
        Task name, e.g. `manual control`

        Returns:
            str: task name
        '''
        return bmi3d.get_task_name(self.record, dbname=self.dbname)

    @property
    def task_desc(self):
        '''
        Task description, e.g. `flash`

        Returns:
            str: task description
        '''
        if self.record.entry_name is None:
            return ''
        return self.record.entry_name

    @property
    def duration(self):
        '''
        Duration of recording in seconds

        Returns:
            float: duration
        '''
        try:
            report = json.loads(self.record.report)
            return report['runtime']
        except:
            return 0.0
        
    @property
    def n_trials(self):
        '''
        Number of total trials presented

        Returns:
            int: number of total trials
        '''
        try:
            report = json.loads(self.record.report)
        except: 
            return 0
        total = report['n_trials']
        return total
    
    @property
    def n_rewards(self):
        '''
        Number of rewarded trials

        Returns:
            int: number of rewarded trials
        '''
        try:
            report = json.loads(self.record.report)
        except: 
            return 0
        rewards = report['n_success_trials']
        return rewards        

    def get_decoder(self, decoder_dir=None):
        '''
        Fetch the decoder object from the database, if there is one.

        Returns:
            Decoder: decoder object (type depends on which decoder is being loaded)
        '''
        if decoder_dir is not None:
            filename = bmi3d.get_decoder_name(self.record, dbname=self.dbname)
            filename = os.path.join(decoder_dir, filename)
        else:
            decoder_basename = bmi3d.get_decoder_name(self.record, dbname=self.dbname)
            sys_path = bmi3d.models.System.objects.using(self.dbname).get(name='bmi').path
            filename =  os.path.join(sys_path, decoder_basename)
        dec = pickle.load(open(filename, 'rb'))
        dec.db_entry = self.get_decoder_record()
        return dec
    
    def get_decoder_record(self):
        '''
        Fetch the database models.Decoder record for this recording, if there is one.

        Returns:
            models.Decoder: decoder record
        '''
        return bmi3d.get_decoder_entry(self.record, dbname=self.dbname)

    def has_feature(self, featname):
        '''
        Check whether a feature was included in this recording

        Args:
            featname (str): name of the feature to check

        Returns:
            bool: whether or not the feature was enabled
        '''
        return featname in self.features

    def get_task_param(self, paramname):
        '''
        Get a specific task parameter

        Args:
            paramname (str): name of the parameter to get

        Returns:
            object: parameter value
        '''
        params = self.task_params
        if paramname not in params:
            return None
        return params[paramname]
    
    def get_sequence_param(self, paramname):
        '''
        Get a specific sequence parameter

        Args:
            paramname (str): name of the parameter to get

        Returns:
            object: parameter value
        '''
        params = self.sequence_params
        if paramname not in params:
            return None
        return params[paramname]

    def get_raw_files(self, system_subfolders=None):
        '''
        Gets the raw data files associated with this task entry 
        
        Args:
            system_subfolders (dict, optional): dictionary of system subfolders where the 
                data for that system is located. If None, defaults to the system name
        
        Returns: 
            files : list of (system, filepath) for each datafile associated with this task entry 
        '''
        return bmi3d.get_rawfiles_for_taskentry(self.record, system_subfolders=system_subfolders)

    def get_preprocessed_sources(self):
        '''
        Returns a list of datasource names that should be preprocessed for this task entry. Always
        includes experiment data (`exp`) and eye data (`eye`), and additionally includes broadband,
        lfp, and spike data if there are associated datafiles with appropriate neural data.

        Returns:
            list: preprocessed sources for this task entry
        '''
        sources = ['exp', 'eye']
        if 'quatt_bmi' in self.features:
            sources.append('emg')
        params = self.task_params
        if 'record_headstage' in params and params['record_headstage']:
            sources.append('broadband')
            sources.append('lfp')
        if 'neuropixels' in self.features:
            sources.append('spike')
            sources.append('lfp')
        
        return sources
    
    def preprocess(self, data_dir, preproc_dir, overwrite=False, exclude_sources=[], system_subfolders=None, **kwargs):
        '''
        Preprocess the data associated with this task entry

        Args:
            data_dir (str): directory where the raw data is stored
            preproc_dir (str): directory where the preprocessed data will be written
            overwrite (bool, optional): whether or not to overwrite existing preprocessed data. Defaults to False.
            exclude_sources (list, optional): list of sources to exclude from preprocessing. Defaults to [].
            system_subfolders (dict, optional): dictionary of system subfolders where the
                data for that system is located. If None, defaults to the system name
            kwargs (dict, optional): additional keyword arguments to pass to the preprocessing function

        Returns:
            str: error message if there was an error during preprocessing
        '''

        # Get the raw data files
        files = self.get_raw_files(system_subfolders=system_subfolders)
        if len(files) == 0:
            print("No files for entry!")
            return
        if 'hdf' in files.keys():
            self.record.make_hdf_self_contained(data_dir=data_dir, dbname=self.dbname) # update the hdf metadata from the database
        
        # Choose which sources to preprocess
        sources = self.get_preprocessed_sources()
        for src in exclude_sources:
            sources.remove(src)

        # Prepare the preproc directory
        preproc_dir = os.path.join(preproc_dir, self.subject)
        if not os.path.exists(preproc_dir):
            os.mkdir(preproc_dir)
        
        # Preprocess the data, keeping track of any errors and returning them
        error = None
        try:
            preproc.proc_single(data_dir, files, preproc_dir, self.subject, self.id, self.date, 
                                sources, overwrite=overwrite, **kwargs)
        except Exception as exc:
            traceback.print_exc()
            error = traceback.format_exc()
        return error
    
    def get_db_object(self):
        '''
        Get the raw database object representing this task entry

        Returns:
            models.TaskEntry: bmi3d task entry object
        '''
        return self.record

class BMI3DDecoder():
    '''
    Wrapper for BMI3D Decoder objects. Written like this 
    so that other database types can implement their own decoder classes with
    the same methods without needing to modfiy their database model.
    '''
    
    def __init__(self, decoder, dbname='default'):
        '''
        Constructor

        Args:
            task_entry (TaskEntry): a database task entry from BMI3D
            dbname (str, optional): name of the database to connect to. Defaults to 'default'.
        '''
        self.dbname = dbname
        self.record = decoder
        self.id = self.record.id
        if decoder.entry_id is not None:
            self.parent = lookup_sessions(id=decoder.entry_id)[0]
        else:
            self.parent = None
        self.project = self.parent.project
        self.session = self.parent.session
        self.name = self.record.name
        
    def __repr__(self):
        '''
        Call __repr__ on the database record

        Returns:
            str: repr
        '''
        if self.parent is None:
            return f"Decoder[{self.id}:{self.name}]"
        else:
            return f"Decoder[{self.id}:{self.name}] trained from {repr(self.parent)}"
    
    def __str__(self):
        '''
        Call __str__ on the database record

        Returns:
            str: str
        '''
        return str(self.record)
        
    @property
    def decoder(self):
        '''
        The decoder object

        Returns:
            object: decoder object
        '''
        if hasattr(self, 'dec'):
            return self.dec
        else:
            self.dec = self.get()
            return self.dec
        
    @property
    def channels(self):
        '''
        The decoder channels

        Returns:
            list: channels used by the decoder
        '''
        return self.decoder.channels

    @property
    def filt(self):
        '''
        The decoder filter

        Returns:
            object: decoder filter
        '''
        return self.decoder.filt

    def get(self, decoder_dir=None):
        '''
        Fetch the decoder object from the database, if there is one.

        Returns:
            Decoder: decoder object (type depends on which decoder is being loaded)
        '''
        filename = self.record.path
        if decoder_dir is not None:
            filepath = os.path.join(decoder_dir, filename)
        else:
            sys_path = bmi3d.models.System.objects.using(self.dbname).get(name='bmi').path
            filepath =  os.path.join(sys_path, filename)
        dec = pickle.load(open(filepath, 'rb'))
        self.dec = dec
        return dec

'''
Create
'''
def create_decoder_parent(project, session, task_name='nothing', task_desc='decoder parent', **kwargs):
    '''
    Create a new decoder parent entry (a TaskEntry) in the database. These are used to keep track of
    decoders that weren't trained on a specific session.

    Args:
        project (str): project name
        session (str): session name
        task_name (str, optional): task name. Defaults to 'nothing'.
        task_desc (str, optional): task description. Defaults to 'decoder parent'.
        kwargs (dict, optional): optional keyword arguments, including `dbname` to specify the database

    Returns:
        TaskEntry: the new decoder parent entry
    '''
    dbname = kwargs.pop('dbname', this.BMI3D_DBNAME)

    subj = bmi3d.models.Subject.objects.using(dbname).get(name='test')
    task = bmi3d.models.Task.objects.using(dbname).get(name=task_name)
    
    te = bmi3d.models.TaskEntry(subject_id=subj.id, task_id=task.id)
    te.entry_name = task_desc
    te.project = project
    te.session = session
    te.save(using=dbname)
    
    return te

def save_decoder(decoder_parent, decoder, suffix, **kwargs):
    '''
    Save a new decoder to the database, associated with the given parent TaskEntry. If the decoder
    was trained on a specific session, use that as the parent. If not, use 
    :func:`~aopy.data.db.lookup_decoder_parent` or :func:`~aopy.data.db.create_decoder_parent` to 
    look up or create a new parent entry, respectively.

    Args:
        decoder_parent (TaskEntry): the parent decoder entry
        decoder (object): the decoder object to save
        suffix (str): suffix to append to the decoder name
        kwargs (dict, optional): optional keyword arguments, including `dbname` to specify the database

    Note:
        This only works if you have the `bmi` system path locally. See the BMI3D setup page
        to find this path and make it available on your system.
    ''' 
    te_id = decoder_parent.id
    new_decoder_fname = decoder.save()
    new_decoder_name = f"{decoder_parent.project}_{decoder_parent.session}_{suffix}" 
        
    print("Saving new decoder:", new_decoder_name)
    dbname = kwargs.pop('dbname', this.BMI3D_DBNAME)
    dbq.save_bmi(new_decoder_name, te_id, new_decoder_fname, dbname=dbname)

'''
Wrappers
'''
def list_entry_details(sessions):
    '''
    Returns (subject, te_id, date) for each given session.

    Args:
        sessions (list of TaskEntry): list of sessions

    Returns:
        tuple: tuple containing
            | **subject (list):** list of subject names
            | **te_id (list):** list of task entry ids
            | **date (list):** list of dates
    '''
    return zip(*[(te.subject, te.id, te.date) for te in sessions])

def group_entries(sessions, grouping_fn=lambda te: te.date):
    '''
    Automatically group together a flat list of database IDs

    Args:
        sessions (list of task entries): TaskEntry objects to group
        grouping_fn (callable, optional): grouping_fn(task_entry) takes a TaskEntry as 
            its only argument and returns a hashable and sortable object by which to group the ids

    Returns:
        list: list of tuples, each tuple containing a group of sessions
    '''
    keyed_ids = defaultdict(list)
    for te in sessions:
        key = grouping_fn(te)
        keyed_ids[key].append(te)

    keys = list(keyed_ids.keys())
    keys.sort()

    grouped_ids = []
    for date in keys:
        grouped_ids.append(tuple(keyed_ids[date]))
    return grouped_ids

def summarize_entries(entries, sum_trials=False):
    '''
    Generates a dataframe summarizing the subject, date, task, number of trials, 
    and duration in minutes of each entry in the input list. Optionally sum the
    number of trials and duration for unique tasks across days for each subject
    
    Args:
        entries (list): list of bmi3d task entries
        sum_trials (bool, optional): sum the number of trials and duration across
            unique tasks for each day for each subject

    Returns:
        pd.DataFrame: dataframe of entry summaries

    Examples:

        .. code-block:: python

            date_obj = date.fromisoformat('2023-02-06')
            entries = db.lookup_sessions(date=date_obj)
            df = db.summarize_entries(entries)
            display(df)

        .. image:: _images/db_summarize_sessions.png

        .. code-block:: python

            df_unique = db.summarize_entries(entries, sum_trials=True)
            display(df_unique)

        .. image:: _images/db_summarize_sessions_sum.png

    '''

    # Generate a summary dataframe
    desc = {
        'subject': [e.subject for e in entries],
        'te_id': [e.id for e in entries],
        'date': [e.date for e in entries],
        'time': [e.datetime.time().replace(microsecond=0) for e in entries],
        'task_name': [e.task_name for e in entries],
        'task_desc': [e.task_desc for e in entries],
        'n_rewards': [e.n_rewards for e in entries],
        'n_trials': [e.n_trials for e in entries],
        'duration_minutes': [np.round(e.duration/60, 1) for e in entries],
    }
    all_sessions = pd.DataFrame(desc)
    if sum_trials is False:
        return all_sessions

    # Optionally sum the dataframe across unique tasks on each day for each subject
    unique_sessions = all_sessions.drop('te_id', axis=1).groupby(
        ['subject', 'date', 'task_name', 'task_desc']).sum(numeric_only=True)
    return unique_sessions

def encode_onehot_sequence_name(sessions, sequence_types):
    '''
    Generates a dataframe summarizing the id, subject, date and by onehot 
    encoding the sequences of interest of each entry in the input session list.

    Args:
        sessions (list): list of bmi3d task entries
        sequence_types (list): Array of sequence_name strings. Can only be a list of strings

    Returns:
        pd.Dataframe: Dataframe of entry summaries containing sequence name occurance
            
    Examples:

        .. code-block:: python
            
            sessions = db.lookup_mc_sessions()
            sequence_types = ['rand_target_chain_2D', 'centerout_2D', 'out_2D', 
                            'rand_target_chain_3D', 'corners_2D', 'centerout_2D_different_center', 
                            'sequence_2D', 'centerout_2D_select', 'single_laser_pulse']
                            
            df = db.encode_onehot_sequence_name(entries, sequence_types)
            display(df)

        .. image:: _images/db_encode_onehot_sequence_name.png  
    '''
    
    # sets row and col count
    row_count = len(sessions)
    col_count = ['id','subject','date'] + sequence_types

    # creates correct size matrix with all 0s as inputs
    df_matrix = [[0 for _ in range(len(col_count))] for _ in range(row_count)]

    for row_id, entry in enumerate(sessions):
        df_matrix[row_id][0] = entry.id
        df_matrix[row_id][1] = entry.subject
        df_matrix[row_id][2] = entry.date
        try:
            for col_id, sequence in enumerate(sequence_types):
                if entry.sequence_name == sequence:
                    df_matrix[row_id][col_id + 3] = 1
        except:
            pass

    df = pd.DataFrame(df_matrix, columns = col_count)
    return df

def add_metadata_columns(df, sessions, column_names, apply_fns):
    '''
    Adds metadata columns (in-place) to a dataframe keyed on session id (e.g. from 
    :func:`~aopy.data.tabulate_behavior_data`). Specify the same number of column names
    as functions. Each function should take a single session as input and return a
    single value of any type. The return value will be appended to the dataframe in all
    rows where the task entry id (te_id) matches the input session. 

    Args:
        df (pd.DataFrame): dataframe of session summaries
        sessions (list): list of bmi3d task entry objects
        column_names (list of str): list of column names to append to the dataframe
        apply_fns (list of functions): functions to apply to each session to generate metadata columns

    Examples:

        Addding a metadata column to a dataframe of session summaries

        .. code-block:: python

            date_obj = date.fromisoformat('2023-02-06')
            entries = db.lookup_sessions(date=date_obj)
            df = db.summarize_entries(entries)
            db.append_metadata_columns(df, entries, 'hs_data', lambda x: x.get_task_param('record_headstage'))
            display(df)

        Adding session and experimenter info after tabulating behavior data
        
        .. code-block:: python

            date_obj = date.fromisoformat('2023-02-06')
            entries = db.lookup_sessions(date=date_obj)
            df = aopy.data.tabulate_behavior_data(entries)
            db.append_metadata_columns(df, entries, ['session', 'experimenter'], 
                                                    [lambda x: x.session, lambda x: x.experimenter])
            display(df)

        More information about `entries` can be found in :class:`~aopy.data.db.BMI3DTaskEntry`
    '''
    try:
        len(column_names) == len(apply_fns)
    except TypeError:
        column_names = [column_names]
        apply_fns = [apply_fns]
    if len(column_names) != len(apply_fns):
        raise ValueError("column_names and apply_fns must be the same length")
    
    for col, fn in zip(column_names, apply_fns):
        for entry in (sessions):
            df.loc[df['te_id'] == entry.id, col] = fn(entry)
