import os

import requests
from dotenv import load_dotenv

load_dotenv("C:\\code\\swe-bench\\.env")
os.getenv("GITHUB_TOKEN")

url = "https://api.github.com/repos/numpy/numpy/languages"
headers = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"Bearer {os.getenv("GITHUB_TOKEN")}",
}
response = requests.get(url, headers=headers)

x = response.json()

pass
