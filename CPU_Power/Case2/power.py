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
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
try:
	driver = webdriver.Chrome('/home/erc/Desktop/nvsk2/papers/chromedriver')  # Optional argument, if not specified will search path.
except wde as e:
	print("\nChrome crashed on launch:")
	print(e)
	print("Trying again in 10 seconds..")
	time.sleep(1)
	driver = webdriver.Chrome('/home/erc/Desktop/nvsk2/papers/chromedriver')
	print("Success!\n")
except Exception as e:
	raise Exception(e)
#a=[]
driver.get("https://webchaintrail.blogspot.com/")
#page = driver.find_element_by_tag_name('body')
#page.send_keys(Keys.CONTROL + 't') 
driver.execute_script("window.open('about:blank', 'tab2');")
driver.switch_to.window("tab2")
#browser.get('http://bing.com')
driver.get("https://www.youtube.com/watch?v=bBC-nXj3Ng4")
button = WebDriverWait(driver,10).until(EC.element_to_be_clickable((By.XPATH,"//button[@class='ytp-play-button ytp-button']")))
    # a = button.get_attribute("aria-label")
if button.get_attribute("aria-label")=='Play (k)':
	print('start to play')
	button.click()
else:
	print('video is already playing')
time.sleep(1)  # Let the user actually see something!

pid=os.getpid()
p = psutil.Process(pid=pid)
#p = psutil.Process()
#i=1 #has to be changed to argument
i=sys.argv[1]
orig_stdout=sys.stdout
sys.stdout = open('/home/erc/Desktop/nvsk2/papers/javascript/power/result'+str(i)+'.txt', 'w+')
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
