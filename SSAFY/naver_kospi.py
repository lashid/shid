import requests
from bs4 import BeautifulSoup

url = "https://finance.naver.com/sise/"
response = requests.get(url).text

data = BeautifulSoup(response, 'html.parser')
selected = data.select_one('#KOSPI_now').text
print(f'오늘의 코스피 지수는 {selected}입니다.')