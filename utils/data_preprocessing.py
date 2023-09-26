
import requests
import numpy as np
import pandas as pd
from utils.credentials import username, password

def request_data(username, password):
    url = "https://restapimoonboard.ems-x.com"

    payload=f'username={username}&password={password}&grant_type=password&client_id=com.moonclimbing.mb'
    headers = {
    'Authorization': 'BEARER',
    'User-Agent': 'MoonBoard/1.0',
    'Accept-Encoding': 'gzip, gzip',
    #'Content-Type': 'application/x-www-form-urlencoded',
    'Host': 'restapimoonboard.ems-x.com',
    #'Content-Length': '93',
    }

    response = requests.request("POST", url + "/token", headers=headers, data=payload)
    token = "Bearer " + response.json()["access_token"]


    headers["Authorization"] = token
    #del headers["Content-Type"]
    #del headers["Content-Length"]



    response = requests.request("GET", url + "/v1/_moonapi/problems/v3/17/1/0", headers=headers, data={})
    data = response.json()["data"]

    return data



"""boulders = np.zeros((len(data),3,11,18))
for boulder in data:
    i = data.index(boulder)
    for hold in boulder["moves"]:
        x = int(ord(hold["description"][0]) - 65)
        y = int(hold["description"][1:]) - 1
        if hold["isStart"]:
           boulders[i][0][x][y] = 1 
        if hold["isEnd"]:
            boulders[i][1][x][y] = 1
        else:
            boulders[i][2][x][y] = 1
    


df = pd.DataFrame.from_records(data)"""

if __name__ == "__main__":
    data = request_data(username, password)
    print(data)