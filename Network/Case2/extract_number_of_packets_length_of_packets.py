'''

@author: nvsk.avinash

'''

import sys
from pathlib import Path
import csv
import os
import dpkt
from scapy.all import *

a1=sys.argv[1]
a2=sys.argv[2]
path = '/home/erc/Desktop/nvsk2/papers/javascript/pcap/website_pcapFiles_100/test'+str(a1)+'/test_'+str(a2)+'_Chrome.pcap'
scapy_cap = rdpcap('/home/erc/Desktop/nvsk2/papers/javascript/pcap/website_pcapFiles_100/test'+str(a1)+'/test_'+str(a2)+'_Chrome.pcap')
#setting the first value from the pcap file as t0, as .time function is producing absolute time instead of excat time
t0=scapy_cap[0].time
#creating a dictionary named items to store the (difference of time as key: (1,length of that particular packet as value))
items={}
#creating a blank 2d array with 600 rows and 2 columns with all filled with 0 initially 
a=[[0 for i in range(2)]for j in range(600)]
c=0
#appending into hashmap
for i in range(len(scapy_cap)):
	c=scapy_cap[i].time-t0
	items[c]=(1,len(scapy_cap[i]))

#for each key value I'll check in which region it falls 0-100/100-200/..... and then add the values to corresponding locations in the 2d array
for key in items:
	intt=int(key/100)
	a[intt][0]=items[key][0]+a[intt][0]
	a[intt][1]=items[key][1]+a[intt][1]

#writing the output to a file 
orig_stdout=sys.stdout
sys.stdout = open('/home/erc/Desktop/nvsk2/papers/javascript/pcap/final/cpu'+str(a1)+'/result'+str(a2)+'.txt', 'w+')
for i in range(len(a)):
	print(str(a[i][0])+","+str(a[i][1]))
sys.stdout.close()	
sys.stdout=orig_stdout 

