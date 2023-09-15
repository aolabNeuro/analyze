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
try:
    from db import dbfunctions as bmi3d
except:
    warnings.warn("Database not configured")
    traceback.print_exc()

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
        entries = bmi3d.get_task_entries(dbname=this.BMI3D_DBNAME, **kwargs)
        entries = [BMI3DTaskEntry(e) for e in entries]

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
    filter_fn = kwargs.pop('filter_fn', lambda x:True) and (lambda te: 'flash' in te.task_desc)
    return lookup_sessions(task_name=mc_task_name, filter_fn=filter_fn, **kwargs)

def lookup_mc_sessions(mc_task_name='manual control', **kwargs):
    '''
    Returns list of entries for all manual control sessions on the given date
    See :func:`~aopy.data.db.lookup_sessions` for details.
    '''
    filter_fn = kwargs.pop('filter_fn', lambda x:True) and (lambda te: 'flash' not in te.task_desc)
    return lookup_sessions(task_name=mc_task_name, filter_fn=filter_fn, **kwargs)

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

    def __init__(self, task_entry, dbname='default', **kwargs):
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
        Number of rewarded trials

        Returns:
            int: number of rewarded trials
        '''
        try:
            report = json.loads(self.record.report)
        except: 
            return 0
        rewards = report['n_success_trials']
        total = report['n_trials']
        if rewards == 0:
            return total # in the case of laser or flash trials, no rewards
        
        return rewards

    def get_decoder(self, decoder_dir=None):
        '''
        Fetch the decoder object from the database, if there is one.

        Returns:
            Decoder: decoder object (type depends on which decoder is being loaded)
        '''
        if decoder_dir is not None:
            filename = bmi3d.get_decoder_name(self.record)
            filename = os.path.join(decoder_dir, filename)
        else:
            filename = bmi3d.get_decoder_name_full(self.record)
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
        params = self.task_params
        sources = ['exp']
        if 'record_headstage' in params and params['record_headstage']:
            sources.append('broadband')
            sources.append('lfp')
        if 'neuropixels' in self.features:
            sources.append('spike')
            sources.append('lfp')
        sources.append('eye')
        
        return sources
    
    def get_db_object(self):
        '''
        Get the raw database object representing this task entry

        Returns:
            models.TaskEntry: bmi3d task entry object
        '''
        return self.record

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
