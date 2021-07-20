'''
@author: nvsk.avinash
'''


import webbrowser
import time
import os
import pyshark

def web_crawler(category, address, ip, timeout, idx):
	new=2
	browserExe="chrome"
	url = address
	#url = 'https://webchaintrail.blogspot.com/'
	#url = 'https://www.google.com'

	webbrowser.open(url, new=new)
	dst = '/home/erc/Desktop/nvsk/papers/javascript/pcap/website_pcapFiles_100/'+ category
	if not os.path.isdir(dst):
	    os.makedirs(dst)
	#filter = 'ip host ' + str('192.168.200.39') + ' || tcp'
	filter = 'ip host ' + ip + ' || tcp'
	capture = pyshark.LiveCapture(interface='enp0s31f6', bpf_filter= filter,
		                          output_file=dst + '/' + 'test' + '_' + str(idx) +'_Chrome.pcap')

	capture.sniff(timeout = float(timeout))
	#print(category + '_' + str(idx) +'_Chrome.pcap finished!!')
	capture.close()
	#time.sleep(10)
	#webbrowser.find_element_by_tag_name('body').send_keys(Keys.CONTROL + 'w')
	#webbrowser.close()
	#filter = 'ip host ' + ip + ' || tcp'
	os.system("pkill "+browserExe)

if __name__ == '__main__':
    #main(sys.argv[1:]
    for i in range(100):
        web_crawler('test11', 'https://webchaintrail.blogspot.com/', '192.168.200.39', 60, i)
    #web_crawler('test_google', 'https://google.com/', '192.168.200.39', 30, 29)
    #main()
