# import threading
import time
import requests
from datetime import datetime
# from app import price_dict

global category_dict
global price_dict

category_dict = {'0': 'Large', '1':'Medium', '2':'Small'}
price_dict = {'0': 'Rs.230/ltr', '1':'Rs.180/ltr', '2':'Rs.130/ltr'}

end_api = False

URL = "https://63e944f34f3c6aa6e7cae9e9.mockapi.io/rate/petrol"
ratelist=[111,222,333]
timeinforce =[2000,1,1,1,1,1]
activetime = datetime.now().strftime("%d-%m-%y %H-%M-%S") 


def API_call():
    try:
        answer = requests.get(URL, timeout=3)
    except:
        pass
    return answer

def run_api():
    response = API_call()

    # request was successful
    if response.status_code == 200:
        data = response.json()
        ratelist[0:3]=data[0:3]
        tif=data[3]
        print(tif)
        return tif

    else:
        print(f"Request failed with status code {response.status_code}")

def check_timeinforce(tif):
    current_datetime = datetime.now()
    
    print("time in force  ----")
    print(tif)
    target_datetime = datetime(tif[0],tif[1],tif[2],tif[3],tif[4],tif[5])
    
    print(current_datetime)
    print(target_datetime)
    # Check if the target datetime has passed
    if current_datetime >= target_datetime:
        print(price_dict)
        price_dict["0"]=ratelist[2]
        price_dict["1"]=ratelist[1]
        price_dict["2"]=ratelist[0]
        print(price_dict)
        print("  --  --")

def thread_function():
    while end_api==False:
        # Call the function every minute
        tif = run_api()
        check_timeinforce(tif)
        time.sleep(5)

