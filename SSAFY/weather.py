import requests
from bs4 import BeautifulSoup

url = "https://api.openweathermap.org/data/2.5/weather?q=Seoul&appid=" # 각자의 주소를 입력

data = requests.get(url).json()

weather = data['weather'][0]['main']
temp = data['main']['temp'] - 273.15
temp_min = data['main']['temp_min'] - 273.15
temp_max = data['main']['temp_max'] - 273.15

print(f'서울의 현재 날씨는 {weather}이고, {temp:0.1f}도 입니다.')
print(f'최저/최고 온도는 각각 {temp_min}, {temp_max}도 입니다.')
