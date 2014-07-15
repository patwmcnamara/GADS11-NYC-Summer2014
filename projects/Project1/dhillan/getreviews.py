import urllib2
import json
import csv
import math
import time

# Set the Places API key for your application
AUTH_KEY = 'XXXXXXXXXXXXXXXXXXXXXXX'

# Initialise place list
place_list = []
# Open existing data to retrieve reference codes
with open('data.csv', 'rb') as r: 
	reader = csv.reader(r)
	# Build our list of places
	for x in reader:
		place_list.append(x)

# Initialise the number of ratings and the score to 0
(place_list[0]).append('rating_count')
(place_list[0]).append('rating_score')
count = 0
for item in place_list:
	# Skip header
	if count > 0:
		place_list[count].append(str(0.0))
		place_list[count].append(str(0.0))
	count += 1

# For every unique place, retrieve details on that place
count = 0
for ref in place_list:
	# Skip header
	if count >0:
		# Get unique reference code
		REFERENCE = ref[2]
		# Build URL
		url =  ('https://maps.googleapis.com/maps/api/place/details/json?reference=%s&sensor=false&key=%s') % (REFERENCE, AUTH_KEY)
	
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

		# Iterate through the results 
		if json_data['status'] == 'OK':
			place = json_data['result']
			# If reviews exist for a plce
			if 'reviews' in place.keys(): 

				# Work out the average score for that place
				# And add the number of reviews for that place		
				cur_score = 0.0
				score_count = 0.0
				for review in place['reviews']:
					cur_score += review['rating']
					score_count +=1.0
				cur_score = cur_score/score_count
				place_list[count][-2] = str(score_count)
				place_list[count][-1] = str(cur_score)

	count +=1
	print count

# Write out our new place_list with average review score and count added
with open('data_with_review_count.csv','wb') as f1:
	writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
	for x in place_list:
		writer.writerow(x)