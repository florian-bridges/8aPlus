import os
import requests
import json
import numpy as np
import pandas as pd
from utils.credentials import username, password

BOARD_WIDTH = 11
BOARD_HEIGHT = 18
NUM_HOLD_TYPES = 3

DATA_PATH = "data"


def request_data(username, password):
    url = "https://restapimoonboard.ems-x.com"

    payload = f"username={username}&password={password}&grant_type=password&client_id=com.moonclimbing.mb"
    headers = {
        "Authorization": "BEARER",
        "User-Agent": "MoonBoard/1.0",
        "Accept-Encoding": "gzip, gzip",
        "Host": "restapimoonboard.ems-x.com",
    }

    response = requests.request("POST", url + "/token", headers=headers, data=payload)
    token = "Bearer " + response.json()["access_token"]

    headers["Authorization"] = token

    response = requests.request(
        "GET", url + "/v1/_moonapi/problems/v3/17/1/0", headers=headers, data={}
    )
    data = response.json()["data"]

    return data


def hold_to_hold_type(hold):

    if hold["isStart"]:
        return 0
    if hold["isEnd"]:
        return 1
    else:
        return 2


def convert_moves_to_numpy(moves):
    moves_np = np.zeros((NUM_HOLD_TYPES, BOARD_HEIGHT, BOARD_WIDTH))

    for hold in moves:
        x = int(ord(hold["description"][0]) - 65)
        y = int(hold["description"][1:])
        hold_type = hold_to_hold_type(hold)

        moves_np[hold_type][BOARD_HEIGHT - y][x] = 1

    return moves_np


def save_data(data):

    for boulder in data:

        id = str(boulder["apiId"])
        moves_np = convert_moves_to_numpy(boulder["moves"])

        # save numpy
        np.savez(os.path.join(DATA_PATH, id + ".npz"), moves_np)

        # save json
        with open(os.path.join(DATA_PATH, id + ".json"), "w") as file:
            json.dump(boulder, file, indent=4)


if __name__ == "__main__":
    data = request_data(username, password)
    save_data(data)
