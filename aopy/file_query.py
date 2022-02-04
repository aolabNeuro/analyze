import os
import re
import datetime

class FileQuery:
    """
    A helper class to query for different file paths and sessions that exist for the animals
    """

    def __init__(self, base_dir, animal):
        """
        :param base_dir: (str) base directory
        :param animal: (str) name of animal, beignet or affi,
        """
        self.base_dir = base_dir
        self.animal = animal
        self.raw_dir = os.path.join(base_dir, "raw")
        self.preprocessed_dir = os.path.join(base_dir, "preprocessed", animal)

    def get_raw_dir(self):
        return self.raw_dir

    def get_preprocess_dir(self):
        return self.preprocessed_dir

    def get_n_most_recent_preprocessed_sessions(self, n):
        """
        :param n: (int) to query for
        :return: (list[int]) of recent sessions of length n, in reverse order
        """
        file_names = [f for f in os.listdir(self.preprocessed_dir) if f.startswith("preprocessed_")]
        sessions = [re.findall(r'\d+', f)[0] for f in file_names]
        sessions.sort(reverse=True)
        return sessions[:n]

    def get_raw_sessions_from_date_range(self, start, end):
        """
        :param start: (DateTime, None) start date to query from, inclusive
        :param end: (DateTime, None) end date to query to, inclusive
        :return: (list[int]) sessions to
        """
        if start is None:
            # TODO: handle start is None situation more gracefully
            # currently just set to 2020 since oldest data is in 2021 it seems.
            start = datetime.datetime(2020, 12, 31)
        if end is None:
            end = datetime.now()
        days_between = (end - start).days + 1  # inclusive
        date_list = [(start + datetime.timedelta(days=idx)) for idx in range(days_between)]
        date_str_list = [x.strftime("%Y%m%d") for x in date_list]
        sessions = []
        for date_str in date_str_list:
            # look in hdf directory
            file_names = [f for f in os.listdir(f"{self.raw_dir}/hdf") if date_str in f]
            sessions_for_date = [int(re.findall(r'te\d+', f)[0]) for f in file_names]
            sessions = sessions + sessions_for_date
        return sessions

    def get_raw_filenames_for_session(self, session_id):
        """
        Finds hdf and ecube paths for session
        NOTE: only if there are multiple hdf/ecode files for session, only returns the first.
        :param (int) session_id:
        :return: (dict[str, str]) of relative hdf and ecube file paths for this session
        """
        els_hdf = os.listdir(f"{self.raw_dir}/hdf")
        hdf_name = [el for el in els_hdf if session_id in el][0]
        els_ecube = os.listdir(f"{self.raw_dir}/ecube")
        ecube_name = [el for el in els_ecube if session_id in el][0]
        return {
            "hdf": f"hdf/{hdf_name}",
            "ecube": f"ecube/{ecube_name}",
        }

    def get_preprocessed_filename(self, session_id):
        return f"preprocessed_te{session_id}.hdf"




