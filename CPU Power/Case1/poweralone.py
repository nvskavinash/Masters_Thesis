'''
@author: nvsk.avinash
'''


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
#p = psutil.Process()
#i=1 #has to be changed to argument
i=sys.argv[1]
orig_stdout=sys.stdout
sys.stdout = open('/home/erc/Desktop/nvsk/papers/javascript/power/result'+str(i)+'.txt', 'w+')
#s = ""
#x = 10
psutil.cpu_percent(interval=0.1)
timeout = time.time() + 60
while True:
	#now = time.time()
	#future = now+10
	#if time.time()>future:
	#	break
	#print(psutil.cpu_percent(interval=0.1))
	#s=s+str(psutil.cpu_percent(interval=0.1))+"\n"
	if(time.time() > timeout):
		driver.quit()
		break
	#psutil.cpu_percent(interval=None)
	print(psutil.cpu_percent(interval=0.1))
	#x=x-1
	#time.sleep(10)
#print(s)
sys.stdout.close()	
sys.stdout=orig_stdout 
