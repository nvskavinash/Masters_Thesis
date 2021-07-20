'''

@author: nvsk.avinash

'''

import os
import sys
import re

a1=sys.argv[1]
#a2=sys.argv[2]
f=open('/home/erc/Desktop/nvsk/papers/javascript/cache/cpu0/output'+a1+'.txt','r')
count1=1
count2=1
c=""
d=""
temp=""
toggle=False
orig_stdout=sys.stdout
sys.stdout = open('/home/erc/Desktop/nvsk/papers/javascript/cache/stage2test1/cpufinal/output_stage2.txt', 'a')
for x in f:
	#count=count+1
	#if(count<=100):
	if("L1-dcache-loads" in x):
		if(toggle==False):
			#getting the value of cache hits 
			c=x.split(None,1)[0]
			#cleanng the data
			c=re.sub('[^A-Za-z0-9]+','',str(c))
			#setting a parameter called toggle so that my file doesn't start with a comma
			toggle=True
			count1=count1+1
		else:
			temp=x.split(None,1)[0]
			#collecting the initial 100 readings temporarly
			if(count1<=100):
				c=c+','+re.sub('[^A-Za-z0-9]+','',str(temp))
				count1=count1+1
			temp=""	
	#toggle=False	
	elif("L1-dcache-load-misses" in x):
		if(toggle==False):
			d=x.split(None,1)[0]
			d=re.sub('[^A-Za-z0-9]+','',str(d))
			toggle=True
			count2=count2+1
		else:
			temp=x.split(None,1)[0]
			if(count2<=100):	
				d=d+','+re.sub('[^A-Za-z0-9]+','',str(temp))
				count2=count2+1
			temp=""
c=c+d+',0'
print(c)	
sys.stdout.close()	
sys.stdout=orig_stdout
