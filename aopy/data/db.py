'''
Interface between database methods/models and data analysis code
'''
import json
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

def lookup_sessions(subject=None, date=None, task_name=None, task_desc=None, session=None, project=None, 
                    experimenter=None, has_features=None,
                    exclude_ids=[], filter_fn=lambda x:True, **kwargs):
    '''
    Returns list of entries for all sessions on the given date
    '''
    if this.DB_TYPE == 'bmi3d':
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
        entries = [BMI3DTaskEntry(e) for e in bmi3d.get_task_entries(dbname=this.BMI3D_DBNAME, **kwargs)]
        if has_features and not isinstance(has_features, list):
            filter_fn = filter_fn and (lambda x: x.has_feature(has_features))
        elif has_features:
            filter_fn = filter_fn and (lambda x: all([x.has_feature(f) for f in has_features]))
        if len(entries) == 0:
            warnings.warn("No entries found")
            return []
        return [e for e in entries if filter_fn(e) and e.id not in exclude_ids]
    else:
        warnings.warn("Unsupported db type!")
        return []
    
def lookup_flash_sessions(subject, date, mc_task_name='manual control', **kwargs):
    '''
    Returns list of entries for all flash sessions on the given date
    '''
    filter_fn = kwargs.pop('filter_fn', lambda x:True) and (lambda te: 'flash' in te.task_desc)
    return lookup_sessions(subject, date, task_name=mc_task_name, filter_fn=filter_fn, **kwargs)

def lookup_mc_sessions(subject, date, mc_task_name='manual control', **kwargs):
    '''
    Returns list of entries for all manual control sessions on the given date
    '''
    filter_fn = kwargs.pop('filter_fn', lambda x:True) and (lambda te: 'flash' not in te.task_desc)
    return lookup_sessions(subject, date, task_name=mc_task_name, filter_fn=filter_fn, **kwargs)

def lookup_tracking_sessions(subject, date, tracking_task_name='tracking', **kwargs):
    '''
    Returns list of entries for all tracking sessions on the given date
    '''
    return lookup_sessions(subject, date, task_name=tracking_task_name, **kwargs)

def lookup_bmi_sessions(subject, date, bmi_task_name='bmi control', **kwargs):
    '''
    Returns list of entries for all bmi control sessions on the given date
    '''
    return lookup_sessions(subject, date, task_name=bmi_task_name, **kwargs)

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

    def get_decoder(self):
        '''
        Fetch the decoder object from the database, if there is one.

        Returns:
            Decoder: decoder object (type depends on which decoder is being loaded)
        '''
        return bmi3d.get_decoder(self.record, dbname=self.dbname)
    
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
    return zip(*[(te.subject, str(te.id), str(te.date)) for te in sessions])

def group_entries(sessions, grouping_fn=lambda te: te.date):
    '''
    Automatically group together a flat list of database IDs

    Args:
        sessions (list of task entries): TaskEntry objects to group
        grouping_fn (callable, optional): grouping_fn(task_entry) takes a TaskEntry as 
            its only argument and returns a hashable and sortable object by which to group the ids
    '''
    keyed_ids = defaultdict(list)
    for te in sessions:
        key = grouping_fn(te)
        keyed_ids[key].append(id)

    keys = list(keyed_ids.keys())
    keys.sort()

    grouped_ids = []
    for date in keys:
        grouped_ids.append(tuple(keyed_ids[date]))
    return grouped_ids
