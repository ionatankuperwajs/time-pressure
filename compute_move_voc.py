############################
# IMPORT PACKAGES TO USE

import os
import chess
import chess.engine
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import time
from convert_stockfish_scores import *

########################################
# SET ARRAY JOB IDX IF CLUSTER ARRAY JOB

# on_cluster = True
# is_array_job = True
#
# if is_array_job:
#     job_idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
# else:
#     job_idx = 1

###################################
# LINK TO DATA AND ENGINE FOLDERS


# DOWNLOAD CHESS DATA FILE FROM http://csslab.cs.toronto.edu/datasets/#maia_kdd
# e.g. http://csslab.cs.toronto.edu/data/chess/monthly/lichess_db_standard_rated_2019-01.csv.bz2 and unzip

# DOWNLOAD STOCKFISH FROM https://stockfishchess.org/download/

# chess_csv_file = 'lichess_db_standard_rated_2019-01.csv'
#
# if on_cluster:
#
#     working_folder = '/home/erussek/projects/Chess_Project';
#     stockfish_file = "stockfish_14_linux_x64_avx2/stockfish_14_x64_avx2"
#
#     engine_folder = '/home/erussek/projects/utils'
#     stockfish_path = os.path.join(engine_folder, stockfish_file)
#     chess_data_folder = "/scratch/gpfs/erussek/Chess_project/Lichess_2019_data"
#     to_save_folder = "/scratch/gpfs/erussek/Chess_project/analysis_results_Jan_2019_full_3_sf14" # this is fine still
#
# else:
#
#     working_folder = '/Users/evanrussek/Dropbox/Griffiths_Lab_Stuff/Code/Chess_Project';
#     stockfish_path = "/Users/evanrussek/stockfish/14/bin/stockfish"
#     chess_data_folder = '/Users/evanrussek/Dropbox/Griffiths_Lab_Stuff/Chess_Data/Lichess_2019_data'
#     to_save_folder = '/Users/evanrussek/Dropbox/Griffiths_Lab_Stuff/Chess_Data/analysis_results_Jan_2019_full_3_sf14';
#
# os.chdir(working_folder)

##########################################################################
# TO GO THROUGH FILE, HOW MANY TOTAL JOBS (RUNS) AND HOW MANY MOVES PER RUN
moves_per_run = 50000
n_total_runs = 5000

###########################################
########### FUNCTIONS TO RUN ANALYSIS
###############################

def get_move_at_each_depth(board, engine, limit_dict, white_active, clear_hash = True):

    """
    Generate move selected by stockfish at each depth up to depth limit
    Args:
        board: board rep from python chess
        engine: stockfish engine from python chess
        limit_dict: dictionary containing type of limit (e.g. depth) and it's value (e.g. 15)
    Returns:
        dataframe containing 1 row for each search depth w/ move, depth, nodes, time
    """

    # set to not use tables
    engine.configure({"SyzygyProbeLimit": 0})


    # first clear the hash
    if clear_hash:
        engine.configure({"Clear Hash": None})

    max_list_size = 20

    # what to store for each computational iteration - would pre-allocating make this faster?
    depth_list = []
    node_list = []
    move_list = []
    time_list = []

    # note that we compute all moves in a single pass - this speeds it up from repeated calls for each depth
    n_info = 0
    last_depth = 0
    with engine.analysis(board) as analysis:
        for info in analysis:

            n_info += 1
            if info.get("pv") != None:

                this_depth = info.get("depth")

                # Handle researches - just skip move if it tries to repeat.
                if this_depth != last_depth:
                    depth_list.append(info.get("depth"))
                    node_list.append(info.get("nodes"))
                    move_list.append(info.get("pv")[0].uci())
                    time_list.append(info.get("time"))

                last_depth = this_depth

                # Stop condition.
                if (info.get(limit_dict['type'],0) >= limit_dict['val']) | (n_info > max_list_size):
                    break

    board_score  = info.get("score");
    board_score_dict = process_score(board_score, white_active)

    return pd.DataFrame({'move': move_list, 'depth': depth_list, 'nodes': node_list, 'time': time_list}), board_score_dict

def evaluate_moves_at_each_depth(move_at_depth_df, board, engine, eval_depths_arr, white_active, clear_hash = True):

    """
    Evaluate each move in move_at_depth_df (move selected at each depth) at each depth in eval_depth_arr
    Args:
        move_at_depth_df: Output from get_move_at_each_depth function. Dataframe with one row per depth w/ SF move
        board: python chess board object with current position
        engine: stockfish engine from python chess
        eval_depths_arr: array of depths at which to evaluate each move
        white_active: boolean with whether white is the active player
        clear_hash: should SFs cash be cleared?
    """

    # check that eval_depths is a numpy array
    assert(type(eval_depths_arr) == np.ndarray)

    # for each unique move, evaluate it at each depth and store in a dict
    unique_move_scores_at_depths = {}

    n_unique_moves = len(move_at_depth_df['move'].unique())

    # for each unique move, generate a score at each eval_depth, store in dict
    for move in move_at_depth_df['move'].unique():


        # clear the engine's hash
        if clear_hash:
            engine.configure({"Clear Hash": None})

        # create a new board and push the move
        new_board = board.copy()
        new_board.push_uci(move)

        # check if the new board is stalemate
        if new_board.is_stalemate():
            info = engine.analyse(new_board, chess.engine.Limit(time=0.001))
            score = info["score"]
            move_eval_score_list = len(eval_depths_arr)*[score]
        else:
            # push score to a list for each evaluation depth
            move_eval_score_list = []

            last_eval_depth = 0
            with engine.analysis(new_board) as analysis:
                for info in analysis:

                    this_eval_depth = info.get("depth")
                    this_score = info.get("score")

                    # if it returns a mate, then the remainder will also return a mate
                    if (this_score != None):
                        if this_score.is_mate():
                            # append the mate to the rest of the list repeatedly...
                            n_depths_remaining = len(eval_depths_arr) - len(move_eval_score_list)
                            move_eval_score_list = [*move_eval_score_list, *[this_score]*n_depths_remaining]
                            break

                    # if this is a depth that we want, then record it
                    if (this_eval_depth in eval_depths_arr) & (info.get("score") != None):

                        # look for repeats of depth - this means a research and we should replace the previous answer w/ the new one
                        if this_eval_depth == last_eval_depth:
                            move_eval_score_list = move_eval_score_list[:-1]

                        move_eval_score_list.append(this_score)

                        # reset last_eval_depth
                        last_eval_depth = this_eval_depth

                        if (this_eval_depth >= eval_depths_arr[-1]):
                            break


        assert(len(eval_depths_arr) == len(move_eval_score_list))

        # list of scores at each depth for each unique move
        unique_move_scores_at_depths[move] = move_eval_score_list

    # for each depth and eval depth get the eval score for move selected (arrange so we can stack the original dataframe n_eval_depth times)
    n_eval_depths = len(eval_depths_arr)
    scores_long = [unique_move_scores_at_depths[move][ed_idx] for ed_idx in range(n_eval_depths) for move in move_at_depth_df['move']]

    # stack the move df for each eval depth and add the score and depth to which it corresponds
    move_df_stacked = pd.concat([move_at_depth_df]*n_eval_depths, ignore_index = True)
    move_df_stacked_score = [unique_move_scores_at_depths[move][ed_idx] for ed_idx in range(n_eval_depths) for move in move_at_depth_df['move']]

    # process the score
    move_df_score_dict_list = [process_score(s, white_active) for s in move_df_stacked_score]
    move_df_stacked['score_val'] = [s['val'] for s in move_df_score_dict_list]
    move_df_stacked['score_type'] = [s['type'] for s in move_df_score_dict_list]
    move_df_stacked['score_wp'] = [s['wp'] for s in move_df_score_dict_list]

    n_moves = move_at_depth_df.shape[0] # number of original depths
    move_df_stacked['eval_depth'] = np.repeat(eval_depths_arr, n_moves)

    return move_df_stacked

def comp_moves_at_depth_and_eval(move_data, engine, limit_dict, eval_depths_arr):

    """
    Computes SF selected move at each depth from 1 to limit and evaluates those moves at some depths (eval_depths_arr)
    Args:
        move_data: 1 row of dataframe containing starting move data -- contains the fen, white_active,, move_ply, etc...
        engine: SF engine represented with python chess
        limit_dict: dictionary
    """

    pos_fen = move_data['board']
    board = chess.Board(pos_fen) # make board for
    white_active = move_data['white_active']

    # process the move to get mult evals at each depth
    move_at_depth_df, board_score_dict = get_move_at_each_depth(board, engine, limit_dict, white_active)
    move_evals_df = evaluate_moves_at_each_depth(move_at_depth_df, board, engine, eval_depths_arr, white_active)

    # convert to active from white
    move_evals_df['score_wp_active'] = move_evals_df['score_wp'] if white_active else -1*move_evals_df['score_wp']
    move_evals_df['fen'] = pos_fen
    move_evals_df['white_active'] = white_active
    move_evals_df['move_ply'] = move_data['move_ply']
    move_evals_df['game_id'] = move_data['game_id']

    # then also compute the slope...

    return move_evals_df, board_score_dict


def compute_voc(move_df):

    """ Compute voc for move

    Args:
        move_df: dataframe returned from evaluate_depths_at_position
        contains value estimate for each depth
    Returns:
        VOC for move - value of move which maximizes eval depth minus value of move at depth 1

    """

    voc = np.amax(move_df['score_wp_active'].values) - move_df['score_wp_active'].values[0] # this is

    return voc


def add_rt(game_df):

    """
    Computes response time for each move in input dataframe and returns dataframe with this as a new row.
    Note that for inital run this was mistaken and didn't include games '+10' and '+20', so this is corrected for in later scripts
    """

    game_df = game_df.reset_index(drop = True)
    game_time_setting = game_df.time_control[0]
    time_back = 0

    if '+1' in game_time_setting:
        time_back = 1
    elif '+2' in game_time_setting:
        time_back = 2
    elif '+3' in game_time_setting:
        time_back = 3
    elif '+5' in game_time_setting:
        time_back = 5
    elif '+10' in game_time_setting:
        time_back = 10
    elif '+20' in game_time_setting:
        time_back = 20

    white_move_rts = time_back + game_df.loc[game_df.white_active == True, 'clock'].diff()*-1
    black_move_rts = time_back + game_df.loc[game_df.white_active == False, 'clock'].diff()*-1
    game_df.loc[game_df.white_active == True, 'rt'] = white_move_rts
    game_df.loc[game_df.white_active == False, 'rt'] = black_move_rts

    return game_df

# remove first move of each game from each player b/c no rt for that
def filter_moves(game_data):
    filt_game_data = game_data.loc[game_data.move_ply > 2].reset_index(drop = True)
    return filt_game_data

########### END FUNCTIONS


# RUN ANALYSIS
if __name__ == "__main__":

    # LOAD DATA FOR RUN
    run_by_idxs = np.reshape(np.arange(moves_per_run*n_total_runs), (n_total_runs, moves_per_run))
    run_idxs = run_by_idxs[job_idx,:]
    chess_data_fullfile = os.path.join(chess_data_folder,chess_csv_file)
    data_first = pd.read_csv(chess_data_fullfile, nrows=2)
    cols = data_first.columns
    data_raw = pd.read_csv(chess_data_fullfile, skiprows = run_idxs[0]+1, nrows=moves_per_run, names = cols)

    # MAKE FOLDERS TO SAVE DATA IF THEY DON'T EXIST
    if not os.path.exists(to_save_folder):
      os.makedirs(to_save_folder)

    move_level_folder = os.path.join(to_save_folder, 'move_level')
    depth_level_folder = os.path.join(to_save_folder, 'depth_level')
    move_process_times_folder = os.path.join(to_save_folder, 'move_process_times_eval_15')

    if not os.path.exists(move_level_folder):
      os.makedirs(move_level_folder)

    if not os.path.exists(depth_level_folder):
      os.makedirs(depth_level_folder)

    if not os.path.exists(move_process_times_folder):
      os.makedirs(move_process_times_folder)


    # ADD RT AND FILTER FIRST MOVES
    data_rt = data_raw.groupby('game_id').apply(add_rt).reset_index(drop = True)
    data_filt = data_rt.groupby('game_id').apply(filter_moves).reset_index(drop=True)

    # SELECT TIME-CONTROLS OF INTEREST AND FILTER THESE
    game_time_types = ['60+0', '120+1', '180+0',
                       '180+2', '300+0', '300+3',
                       '600+0', '600+5', '900+10',
                       '1800+0', '1800+20']

    data_filt = data_filt.loc[data_filt.time_control.isin(game_time_types)].reset_index(drop = True)

    # SET UP THE ENGINE
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    # LIMIT DICT SETS WHICH DEPTHS INCLUDED IN CONSIDERATION SET (up to what depth
    limit_dict = {'type': 'depth', 'val': 16} # do it to 16

    # WHAT DEPTHS WILL WE EVALUATE MOVES (HIGH COMPUTATION DEPTH)
    eval_depths_arr = np.array([15]) # assert that this is a numpy array

    # MAKE FOLDER FOR EACH DEPTH IN EVAL_DEPTH FOR SAVING
    for ed in eval_depths_arr:
        folder_name = 'eval_depth_'+str(ed)
        folder_path = os.path.join(depth_level_folder, folder_name)
        if not os.path.exists(folder_path):
          os.makedirs(folder_path)

    # STORE HOW LONG EACH MOVE TOOK (TO FINETUNE)
    move_process_time_list = []

    # STORE RESULTS IN LISTS AND DICT
    move_info_dfs = []
    move_eval_dict = {}
    for ed in eval_depths_arr:
        move_eval_dict['eval_depth_'+str(ed)] = []

    # LOOP THROUGH MOVES
    n_moves_remaining = data_filt.shape[0]

    initial_start = time.time()

    for move_idx in range(n_moves_remaining):

        start = time.time()

        if (move_idx%10 == 0):
            print(move_idx, end = ' ')

        if (move_idx%1000 == 0):
            curr_time = (time.time()-initial_start)/3600;
            print(str(np.round(curr_time,decimals = 2))+' hours so far')

        # GET INFO FOR THIS MOVE AND COMPUTE VOC
        move_df = data_filt.loc[move_idx]

        depths_sel_and_eval_df, board_score_dict = comp_moves_at_depth_and_eval(move_df, engine, limit_dict, eval_depths_arr)
        voc = compute_voc(depths_sel_and_eval_df)

        # STORE AS NEW COLUMN
        move_res = move_df
        move_res['voc'] = voc

        # ADD SF evaluation of current position.
        move_res['board_score_type'] = board_score_dict['type']
        move_res['board_score_val'] = board_score_dict['val']

        # APPEND TO LIST OF RESULTS FOR EACH MOVE
        move_info_dfs.append(pd.DataFrame(move_res).transpose())

        # APPEND MOVE SELECTED AT EACH DEPTH TO SPECIFIC LIST FOR THAT DEPTH
        for ed in eval_depths_arr:
            move_eval_dict['eval_depth_'+str(ed)].append(depths_sel_and_eval_df.loc[depths_sel_and_eval_df.eval_depth == ed])

        move_process_time = time.time()-start
        move_process_time_list.append(move_process_time)

    # SAVE RESULTS

    start_idx = run_idxs[0];
    end_idx = run_idxs[-1];
    file_name = 'job_'+str(job_idx)+'_start_'+str(start_idx)+'_end_'+str(end_idx)+'.csv'

    print('saving move level df')
    move_level_df = pd.concat(move_info_dfs, ignore_index = True)
    move_level_df.to_csv(os.path.join(move_level_folder, file_name))

    print('saving depth level dfs')
    for key in move_eval_dict:
        this_depth_level_df = pd.concat(move_eval_dict[key], ignore_index = True)
        this_depth_level_df.to_csv(os.path.join(depth_level_folder, key, file_name))

    print('saving time info')

    file_name = 'job_'+str(job_idx)+'_start_'+str(start_idx)+'_end_'+str(end_idx)+'.npy'
    with open(os.path.join(move_process_times_folder, file_name), 'wb') as f:
        np.save(f, np.array(move_process_time_list))

    # CLOSE CHESS ENGINE
    engine.quit()
    print('script is done!')
