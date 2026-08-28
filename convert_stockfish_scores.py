import numpy as np

# for cp to wp
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cp_to_wp(score_val_cp_white):
    """
    Converts centipawn score val (signed with white pos/black neg) to win prob for white

    Args:
        score_val_cp_white: centipawn value (signed with white positive black negative) for position
    Returns:
        win probability: measure between 0 and 1 giving probability of white win
        Note: taken for logistic regression model applied to predict win given centipawn games
    """

    # these values were discovered by logistic regression mapping SF cp evals to win-rates
    this_const = -0.045821224987387915;
    this_board_score_val = 0.002226678310106879

    return sigmoid(this_const + this_board_score_val*score_val_cp_white)

def mate_to_wp(score_val_mate):
    """
    Converts plys from mate to win prob (for the player who is within reach of mate)
    """
    # these values were discovered by logistic regression mapping SF mate evals to win-rates
    return sigmoid(3.6986223572286208 + -0.05930670977061789*np.abs(score_val_mate))


def process_score(this_score, white_active):
    """
    this score is what is returned from stockfish
    """

    white_score = this_score.white()
    is_mate = white_score.is_mate()

    if is_mate:
        score_type = 'mate'
        score_val = white_score.mate() # this is framed as white'
        if score_val > 0:
            wp = mate_to_wp(score_val) # this is framed as whichever side is close to mate
        elif score_val < 0:
            wp = 1 - mate_to_wp(score_val)
        elif score_val == 0:
            wp = 1 if white_active else 0
            # 1 or -1 depending on whose move this is.... so process score needs to take in who is active
    else: # centipawn...
        score_type = 'cp'
        score_val = white_score.score()
        wp = cp_to_wp(score_val)

    score_dict = {'type': score_type, 'val': score_val, 'wp': wp}
    return score_dict
