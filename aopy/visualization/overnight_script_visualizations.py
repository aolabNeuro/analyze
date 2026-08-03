import numpy as np
from matplotlib import pyplot as plt
import aopy
from aopy.data import base, db
from aopy.data.bmi3d import get_target_locations

def visualize_co_behavior_data(preproc_dir, te_ids):
    """
    Plot eye and cursor/hand trajectories for center-out behavioral sessions, split by trial outcome.

    For each task entry (TE) ID, this loads preprocessed experiment data and tabulated
    behavioral data, then splits trials into three mutually exclusive outcome categories:

        - 'completed_trials': trials where the delay was completed, the reach was
          completed, and the trial was rewarded (segmented from go_cue_time).
        - 'failed_delay': trials where the delay was NOT completed
          (segmented from center_target_on_time).
        - 'failed_reach': trials where the delay was completed but the reach was NOT
          completed (segmented from go_cue_time).

    Trials are assigned to categories in the order listed above, and once a trial has
    been claimed by an earlier category it is excluded from later ones (via the
    `no_repeats_bool` mask), so each trial appears in at most one category/column.

    For each category, trial segments run from the category's start event to either
    `penalty_start_time` (if present) or `trial_end_time` (if `penalty_start_time` is
    NaN). Eye and cursor kinematic data are tabulated over these segments and plotted
    as trajectories overlaid on the session's unique target locations.

    Produces one figure per TE ID with a 2x3 grid of subplots:
        - Row 0: eye trajectories for each category (completed / failed_delay / failed_reach)
        - Row 1: cursor/hand trajectories for the same categories

    Parameters
    ----------
    preproc_dir : str
        Path to the directory containing preprocessed experiment data.
    te_ids : Sequence[int]
        Iterable of task entry IDs to look up, process, and plot. One figure is
        generated per ID.

    Returns
    -------
    None
        Displays/creates one matplotlib figure per TE ID; does not return a value.

    Notes
    -----
    - Requires `db`, `aopy`, `plt`, `np`, and `get_target_locations` to be available
      in the calling scope/module.
    - If a category has no matching trials, a message is printed and that category
      is skipped (its subplot axes remain unpopulated), but it still consumes a
      column in the 2x3 grid.
    - Hand/optitrack segment tabulation is currently disabled (`if False:`), so the
      'user_world' datatype is not tabulated even when optitrack features are present.
    - Assumes exactly one matching session entry is returned by `db.lookup_sessions(id=id)`.
    """
    bounds = [-10, 10, -10, 10]

    categories = {'completed_trials': ('reward', True, 'go_cue_time'), 
                  'failed_delay': ('delay_completed', False, 'center_target_on_time'),
                  'failed_reach': ('reach_completed', False, 'go_cue_time')}
    
    for id in te_ids:
        entry = db.lookup_sessions(id=id)[0]
        df = db.summarize_entries([entry])

        fig, ax = plt.subplots(2,3, figsize=(15, 10))
        fig.suptitle(f'TE ID: {id} - {entry.task_desc}')

        b = aopy.data.base.load_preproc_exp_data(preproc_dir, df['subject'][0], df['te_id'][0], df['date'][0])

        bh_df = aopy.data.tabulate_behavior_data_center_out(preproc_dir, df['subject'], df['te_id'], df['date'])

        no_repeats_bool = np.ones((bh_df.shape[0],), dtype=bool)

        for i,key in enumerate(categories.keys()):
            print(f'Generating plots for {key}...')
            inclusion_boolean, inclusion_value, start_event = categories[key]

            tmp_bool = bh_df[inclusion_boolean] == inclusion_value

            filtered_df = bh_df[tmp_bool & no_repeats_bool]

            start_times = filtered_df[start_event].values
            end_times = filtered_df['penalty_start_time'].values
            end_times[np.isnan(end_times)] = filtered_df['trial_end_time'].values[np.isnan(end_times)]

            if filtered_df.shape[0] == 0:
                print(f'No trials found for {key}. Skipping...')
                continue
            no_repeats_bool = no_repeats_bool & ~tmp_bool

            cursor_segments_reward = aopy.data.tabulate_kinematic_data(
                preproc_dir, filtered_df['subject'], filtered_df['te_id'], filtered_df['date'],
                start_times, end_times,
                datatype='cursor'
            )

            eye_segments_reward = aopy.data.tabulate_kinematic_data(
                preproc_dir, filtered_df['subject'], filtered_df['te_id'], filtered_df['date'],
                start_times, end_times,
                datatype='eye'
            )
            if False:#'optitrack' in b[1]['features'].astype('str'):
                hand_segments_reward = aopy.data.tabulate_kinematic_data(
                                preproc_dir, filtered_df['subject'], filtered_df['te_id'], filtered_df['date'],
                                start_times, end_times,
                                datatype='user_world'
                            )
            
            subject, te_id, date = aopy.data.db.list_entry_details([entry])
            targs = range(0,9)

            eye_ax = ax[0, i]
            eye_ax.set_title(f'{key} - Eye')
            hand_ax = ax[1, i]
            hand_ax.set_title(f'{key} - Hand')

            unique_targets = get_target_locations(preproc_dir, subject[0], te_id[0], date[0], targs)
            aopy.visualization.plot_targets(unique_targets, 2, bounds, ax=eye_ax)
            aopy.visualization.plot_trajectories(eye_segments_reward, bounds, ax=eye_ax)

            
            aopy.visualization.plot_targets(unique_targets, 2, bounds, ax=hand_ax)
            aopy.visualization.plot_trajectories(cursor_segments_reward, bounds, ax=hand_ax)
        
visualize_co_behavior_data(preproc_dir, te_ids)