'''

@author: nvsk.avinash

'''

import sys
a1=sys.argv[1]

orig_stdout=sys.stdout

sys.stdout = open('/home/nvsk/Downloads/case2/combined/output_cachealone_case2.csv', 'a')

f=open('/home/nvsk/Downloads/case2/final/cpu100k/result'+str(a1)+'.txt', 'r').read().splitlines()

count=0
c=''
d=''
while(count<100):
	if(count==0):
		c=f[count].split(',')[0]
		d=f[count].split(',')[1]
	else:
		c=c+','+f[count].split(',')[0]
		d=d+','+f[count].split(',')[1]
	count+=1
print(c+','+d+',5')

sys.stdout.close()

sys.stdout=orig_stdout
