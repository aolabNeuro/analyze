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
        entries = [TaskEntry(e) for e in bmi3d.get_task_entries(dbname=BMI3D_DBNAME, **kwargs)]
        if has_features:
            filter_fn = filter_fn and (lambda x: all([x.has_feature(f) for f in has_features]))
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
class BMI3DTaskEntry():
    '''
    Wrapper class for bmi3d database entry classes. Written like this 
    so that other database types can implement their own classes with
    the same methods without needing to modfiy their database model.
    '''

    def __init__(self, task_entry=None, dbname='default', **kwargs):
        '''
        '''
        self.dbname = dbname
        self.record = task_entry

    @property
    def id(self):
        return self.record.id
    
    @property
    def params(self):
        return self.record.task_params
    
    @property
    def time(self):
        return self.record.date
    
    @property
    def date(self):
        self.record.date.date()

    @property
    def notes(self):
        return self.record.notes
        
    @property
    def subject(self):
        return self.record.subject.name
    
    @property
    def decoder_record(self):
        # Load decoder record
        if 'decoder' in self.params:
            self.decoder_record = models.Decoder.objects.using(self.record._state.db).get(pk=self.params['decoder'])
        elif 'bmi' in self.params:
            self.decoder_record = models.Decoder.objects.using(self.record._state.db).get(pk=self.params['bmi'])
        else: # Try direct lookup
            try:
                self.decoder_record = models.Decoder.objects.using(self.record._state.db).get(entry_id=self.id)
            except:
                self.decoder_record = None

        # Load the event log (report)
        try:
            self.report = json.loads(self.record.report)
        except:
            self.report = ''

    @property
    def features(self):
        return [f.name for f in self.record.feats.all()]

    def has_feature(self, featname):
        return featname in self.features

    @property
    def params(self):
        '''
        Returns a dict of all task params for session.
        Takes TaskEntry object.
        '''
        return self.record.task_params

    @property
    def sequence_params(self):
        return self.record.sequence_params

    def get_param(self, paramname):
        '''
        Returns parameter value.
        Takes TaskEntry object.
        '''
        params = self.params
        if paramname not in params:
            return None
        return params[paramname]
    
    def get_sequence_param(self, paramname):
        '''
        Returns parameter value.
        Takes TaskEntry object.
        '''
        params = self.sequence_params
        if paramname not in params:
            return None
        return params[paramname]

    @property
    def task_name(self):
        '''
        Returns name of task used for session.
        Takes TaskEntry object.
        '''
        return bmi3d.get_task_name(self.record, dbname=self.dbname)

    @property
    def notes(self):
        '''
        Returns notes for session.
        Takes TaskEntry object.
        '''
        return self.record.notes

    @property
    def length(self):
        '''
        Returns length of session in seconds.
        Takes TaskEntry object.
        '''
        return bmi3d.get_length(self.record)
        
    @property
    def raw_files(self, system_subfolders=None):
        '''
        Gets the raw data files associated with each task entry 
        
        Args:
            entry (TaskEntry): recording to find raw files for 
            system_subfolders (dict, optional): dictionary of system subfolders where the 
            data for that system is located. If None, defaults to the system name
        
        Returns: 
            files : list of (system, filepath) for each datafile associated with this task entry 
        '''
        return bmi3d.get_rawfiles_for_taskentry(self.record, system_subfolders=system_subfolders)

    @property
    def preprocessed_sources(self):
        '''
        Returns a list of datasource names that should be preprocessed for the given entry
        '''
        params = self.params
        sources = ['exp']
        if 'record_headstage' in params and params['record_headstage']:
            sources.append('broadband')
            sources.append('lfp')
        if 'neuropixels' in self.features:
            sources.append('spike')
            sources.append('lfp')
        sources.append('eye')
        
        return sources

'''
Wrappers
'''
def list_entry_details(entries):
    return zip(*[(te.subject, te.id, te.date) for te in entries])

def group_entries(entries, grouping_fn=lambda te: te.date):
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
