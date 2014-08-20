import urllib
import time
def download_files(file_name, url_name, dir):
	success = 0
	try:
		dl = urllib.urlopen(url_name).read()
		success = 1

	except:
		time.sleep(1)
		dl = urllib.urlopen(url_name).read()
		success = 1
	finally:
		pass

	if success == 1:
		f = open(dir + file_name,'wb')
		f.write(dl)
		f.close()
		print 'Success: ' + file_name
	else:
		print 'Failed: ' + file_name	

def generate_url_list(base_url, year):
	url_list = []
	file_list = []
	year_str = str(year)
	for month in range(1,13):
		month_str = str(month)
		if month<10:
			month_str = '0'+ month_str

		for day in range(1,32):
			day_str = str(day)
			if day<10:
				day_str = '0'+ day_str

			for hour in range(0,24):
				hour_str= str(hour)
				if hour<10:
					hour_str = '0'+ hour_str 				

				for minute_str in ['0000', '1500', '3000', '4500']:
					file_str = year_str + month_str + day_str + '_' + hour_str + minute_str + '_M_512.jpg'
					file_list.append(file_str)
					url_list.append(base_url + year_str + '/' + month_str + '/' + day_str + '/' +  file_str)
	return url_list, file_list

if __name__ == "__main__":
	url_list, file_list = generate_url_list('http://jsoc.stanford.edu/data/hmi/images/', 2011)
	# count = 0
	# time_total = 0
	total_files = len(url_list)
	for url_name, file_name in zip(url_list[0:3], file_list[0:3]):
		# t0 = time.time()
		
		download_files(file_name, url_name, './2011/')
		# t1 = time.time()
		# total = (t1-t0)
		# time_total += total
		# total = str(total)
		# count +=1
		# print 'Took ' + total + 's.'+ ' of ' + str(total_files) + ' done. ' + str((time_total/count)*(total_files-count)) + 's remaining...'
		

		

		

