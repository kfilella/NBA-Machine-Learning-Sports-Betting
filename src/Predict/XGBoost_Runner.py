import copy

import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
import morochobot


# from src.Utils.Dictionaries import team_index_current
# from src.Utils.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games
init()
xgb_ml = xgb.Booster()
xgb_ml.load_model('Models/XGBoost_Models/XGBoost_68.6%_ML-2.json')
xgb_uo = xgb.Booster()
xgb_uo.load_model('Models/XGBoost_Models/XGBoost_54.8%_UO-8.json')


def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds):
    ml_predictions_array = []
    odd_str = "\n---------------XGBoost Model Predictions---------------\n"

    for row in data:
        ml_predictions_array.append(xgb_ml.predict(xgb.DMatrix(np.array([row]))))

    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    data = frame_uo.values
    data = data.astype(float)

    ou_predictions_array = []

    for row in data:
        ou_predictions_array.append(xgb_uo.predict(xgb.DMatrix(np.array([row]))))

    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        winner = int(np.argmax(ml_predictions_array[count]))
        under_over = int(np.argmax(ou_predictions_array[count]))
        winner_confidence = ml_predictions_array[count]
        un_confidence = ou_predictions_array[count]
        if winner == 1:
            winner_confidence = round(winner_confidence[0][1] * 100, 1)
            if under_over == 0:
                un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
                odd_str = odd_str + home_team + f" ({winner_confidence}%)" + ' vs ' + away_team + ': ' + 'UNDER ' + str(todays_games_uo[count]) + f" ({un_confidence}%)\n"
            else:
                un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                odd_str = odd_str + home_team + f" ({winner_confidence}%)" + ' vs ' + away_team + ': ' + 'OVER ' + str(todays_games_uo[count]) + f" ({un_confidence}%)\n"
        else:
            winner_confidence = round(winner_confidence[0][0] * 100, 1)
            if under_over == 0:
                un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
                odd_str = odd_str + home_team + ' vs ' + away_team + f" ({winner_confidence}%)" + ': ' +'UNDER ' + str(todays_games_uo[count]) + f" ({un_confidence}%)\n"
            else:
                un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                odd_str = odd_str + home_team + ' vs ' + away_team + f" ({winner_confidence}%)" + ': ' +'OVER ' + str(todays_games_uo[count]) + f" ({un_confidence}%)\n"
        count += 1
    odd_str = odd_str + "\n--------------------Expected Value---------------------\n"
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        ev_home = ev_away = 0
        if home_team_odds[count] and away_team_odds[count]:
            ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
            ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
        if ev_home > 0:
            odd_str = odd_str + home_team + ' EV: ' + str(ev_home) + '\n'
        else:
            odd_str = odd_str + home_team + ' EV: ' + str(ev_home) + '\n'

        if ev_away > 0:
            odd_str = odd_str + away_team + ' EV: ' + str(ev_away) + '\n'
        else:
            odd_str = odd_str + away_team + ' EV: ' + str(ev_away) + '\n'
        count += 1
    odd_str = odd_str + "-------------------------------------------------------\n"
    morochobot.enviar_mensaje_colores(odd_str)
    deinit()
