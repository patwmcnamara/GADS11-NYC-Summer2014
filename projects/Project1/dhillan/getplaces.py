import urllib2
import json
import csv
import math
import time

# Set the Places API key for your application
AUTH_KEY = 'XXXXXXXXXXXXXXXXXXXXXXX'

# Search bounds for Manhattan and Brooklyn
LAT_MAX = 40.885457
LAT_MIN = 40.568874
LON_MAX = -73.854017
LON_MIN = -74.036351

# Lazy estimate of degrees per meter 
# TODO Use Haversine formula
LAT_DEG_P_M = (40.727740 - 40.726840)/100
LON_DEG_P_M = (-73.978720 - -73.979907 )/100

# Define the radius (in meters) for the search
# We are making the assumption that there are no more than
# 20 establishments within a 100m radius
RADIUS = 100
SPACING = 2*RADIUS/math.sqrt(2)

# Steps in degrees
LAT_STEP = SPACING*LAT_DEG_P_M
LON_STEP = SPACING*LON_DEG_P_M

# Estimate number of calls required
LAT_CALLS = int(math.floor((LAT_MAX - LAT_MIN + LAT_STEP)/(LAT_STEP)))
LON_CALLS = int(math.floor((LON_MAX - LON_MIN + LON_STEP)/(LON_STEP)))
print 'Total LAT calls: ', (LAT_CALLS)
print 'Total LON calls: ', (LON_CALLS)
print 'Total calls: ', (LAT_CALLS*LON_CALLS)

# Build a list of lons and lats to call
location_list = []
for i in range(LAT_CALLS):
	cur_lat = LAT_MIN + i*LAT_STEP
	for j in range(LON_CALLS):
		cur_lon = LON_MIN + j*LON_STEP
		cur_str = '{:.6f},{:.6f}' .format(cur_lat,cur_lon)
		location_list.append(cur_str)

# Set an upper bound on the number of calls (so we don't get banned?)
CALL_COUNT_MAX = 50000

#Key word search, for multiple 'coffee+cafe'
KEYWORD = 'coffee'

# Make a dictionary of our places
place_ref_dict = {}

#Keep track of time and counts
call_count = 0
total_time = 0

with open('data.csv','wb') as f1:
	# Write out our header for CSV
	writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
	writer.writerow(['rating','name','reference','price_level','lat','lon','opening_hours','vicinity','photos','id','types','icon'])
	
	# For every lon/lat location
	for LOCATION in location_list:
		print 'Current location: ', LOCATION
		start_time = time.time()

		# Build URL
		url =  ('https://maps.googleapis.com/maps/api/place/search/json?keyword=%s&location=%s'
		         '&radius=%s&sensor=false&key=%s') % (KEYWORD, LOCATION, RADIUS, AUTH_KEY)

		# Handle any timeout errors with max 5 attempts
		attempt = 0
		max_attempts = 5
		while attempt < max_attempts:
			try:
				response = urllib2.urlopen(url)
				attempt = max_attempts
			except:
				print 'Error, pausing then attempting retry'
				time.sleep(5)
				attempt += 1

		# Get the response and use the JSON library to decode the JSON
		json_raw = response.read()
		json_data = json.loads(json_raw)

		# Iterate through the results and add them to our dictionary
		if json_data['status'] == 'OK':
			for place in json_data['results']:

				# Only add the establishment if it is unique
				if str(place['id']) not in place_ref_dict.keys():
					place_ref_dict[str(place['id'])] = ''
					place_list = []

					# Paranoid error checking for missing data - build result string
					if 'rating' in place.keys():
						place_list.append(str(place['rating']))
					else:
						place_list.append('')

					if 'name' in place.keys():
						place_list.append((place['name']).encode('utf8'))
					else:
						place_list.append('')

					if 'reference' in place.keys():
						place_list.append(str(place['reference']))
					else:
						place_list.append('')

					if 'price_level' in place.keys():
						place_list.append(str(place['price_level']))
					else:
						place_list.append('')

					if 'geometry' in place.keys():
						if 'location' in place['geometry'].keys():
							place_list.append(str(place['geometry']['location']['lat']))
							place_list.append(str(place['geometry']['location']['lng']))
						else:
							place_list.append('')
							place_list.append('')
					else:
						place_list.append('')
						place_list.append('')

					if 'opening_hours' in place.keys():
						place_list.append(str(place['opening_hours']))
					else:
						place_list.append('')

					if 'vicinity' in place.keys():
						place_list.append((place['vicinity']).encode('utf8'))
					else:
						place_list.append('')

					if 'photos' in place.keys():
						place_list.append(str(place['photos']))
					else:
						place_list.append('')

					if 'id' in place.keys():
						place_list.append(str(place['id']))
					else:
						place_list.append('')	

					if 'types' in place.keys():
						place_list.append(str(place['types']))
					else:
						place_list.append('')

					if 'icon' in place.keys():
						place_list.append(str(place['icon']))
					else:
						place_list.append('')

					# Write out this row of data
					writer.writerow(place_list)

		# Keep track of average time and let user know how long to go			
		cur_time = time.time() - start_time
		total_time += cur_time
		call_count += 1
		average_time = total_time/call_count

		print 'Call number: ', call_count, ' took (s): ', cur_time
		print 'Current number of unique items: ', len(place_ref_dict)
		print 'Time remaining (s): ', average_time*(LAT_CALLS*LON_CALLS - call_count)

		# Break out if we exceed our calls
		if call_count > CALL_COUNT_MAX:
			print 'Call count too high: ', call_count
			break