import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException as wde
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import sys

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
driver.get("https://webchaintrail.blogspot.com/")
driver.execute_script("window.open('about:blank', 'tab2');")
driver.switch_to.window("tab2")
driver.get("https://www.youtube.com/watch?v=bBC-nXj3Ng4")
button = WebDriverWait(driver,10).until(EC.element_to_be_clickable((By.XPATH,"//button[@class='ytp-play-button ytp-button']")))
    # a = button.get_attribute("aria-label")
if button.get_attribute("aria-label")=='Play (k)':
	#print('start to play')
	button.click()
#else:
	#print('video is already playing')
#time.sleep(1)  # Let the user actually see something!
a=sys.argv[1]
for i in range(150):
	os.system('perf stat --append -o output'+str(a)+'.txt -a -e L1-dcache-loads,L1-dcache-load-misses sleep 0.04')

driver.quit()
#os.system('perf stat -o output.txt -a -e L1-dcache-loads,L1-dcache-load-misses sleep 1')

#import subprocess

#with open('output.txt', 'a') as outfile:
#	subprocess.call("perf stat -o output.txt -a -e L1-dcache-loads,L1-dcache-load-misses sleep 0.01", stdout=outfile)

#import os
#import sys
#orig_stdout=sys.stdout
#sys.stdout = open('/home/erc/Desktop/nvsk/papers/javascript/power/output.txt', 'w+')
#with open('output.txt','a') as outfile:
#s=""
#for i in range(3):
	#outfile.write(os.popen('perf stat -a -e L1-dcache-loads,L1-dcache-load-misses sleep 0.01').read()+"\n")
	#print(os.popen('perf stat -a -e L1-dcache-loads,L1-dcache-load-misses sleep 0.01').read()+"\n")
	#s=s+os.popen('perf stat -a -e L1-dcache-loads,L1-dcache-load-misses sleep 0.01').read()+"\n"
	#s=s+str(os.popen('perf stat -a -e L1-dcache-loads,L1-dcache-load-misses sleep 0.01').read())+"\n"
#print(s)
#sys.stdout.close()	
#sys.stdout=orig_stdout

#import subprocess
#s=""
#s = s+str(subprocess.check_output('perf stat -a -e L1-dcache-loads,L1-dcache-load-misses sleep 0.01', shell=True))+"\n"
#print(s)

#from subprocess import PIPE, Popen
#import sys

#def cmdline(command):
#    process = Popen(
#        args=command,
#        stdout=PIPE,
#        shell=True	
#    )
#    return process.communicate()[0]

#orig_stdout=sys.stdout
#sys.stdout = open('/home/erc/Desktop/nvsk/papers/javascript/power/output.txt', 'w+')
#print(cmdline('perf stat -a -e L1-dcache-loads,L1-dcache-load-misses sleep 0.01'))
#sys.stdout.close()	
#sys.stdout=orig_stdout
