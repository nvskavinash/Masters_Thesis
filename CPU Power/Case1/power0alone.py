import psutil
import sys
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException as wde

chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
try:
	driver = webdriver.Chrome('/home/erc/Desktop/nvsk/papers/chromedriver')  # Optional argument, if not specified will search path.
except wde as e:
	print("\nChrome crashed on launch:")
	print(e)
	print("Trying again in 10 seconds..")
	time.sleep(1)
	driver = webdriver.Chrome('/home/erc/Desktop/nvsk/papers/chromedriver')
	print("Success!\n")
except Exception as e:
	raise Exception(e)
driver.get("https://webchaintrail.blogspot.com/")
time.sleep(1)  # Let the user actually see something!

pid=os.getpid()
p = psutil.Process(pid=pid)
timeout = time.time() + 60
while True:
	if(time.time() > timeout):
		driver.quit()
		break
	print(psutil.cpu_percent(interval=0.1))

print("\n")
