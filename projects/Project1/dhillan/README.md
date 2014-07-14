#Coffee in New York with Google Places API

## Where to find good coffee in New York?

This project attempts to locate all the coffee outlets in Manhattan and Brooklyn, obtain their Google Places review scores, and find the areas and stores with the best coffee. A summary of the results can be found on my blog 'www.datadino.ghost.io' with results plotted to a geoJSON file by postcode, for example.

## Files:

* 5_hists_by_reviewer_number.png - 5 Histograms showing how the scores recorded by 1-5+ reviewers differ from the global distribution of Google Places Scores.

* all_score_hist.png - A histogram of the full distribution of official Google Places scores.

* coffee_data.ipynb - IPython Notebook to generate some plots from the data and to analyse results.

* data.csv - CSV file of all unique 'coffee' places in Manhattan and Brooklyn.

* data_with_review_count.csv - Appended CSV file with the average review from the individual user reviews and the number of reviews for each establishment.

* getplaces.py - Uses a Place search from the Google Places API to search the latitude and longitude bounds of Manhattan and Brooklyn for search term 'coffee'. Many API calls are required to obtain all the unique establishments. Results are written to data.csv.

* getreviews.py - Uses the results in data.csv to make a different API call to retrieve details about an establishment. This gives us up to 5 reviews, and allows us to add in the number of user reviews and the average score.

* slides.* - Presentation of results.

## Places API summary:

There are two separate components to the searches in this project: 1) Search for a keyword 'coffee' using a Place search on location. 2) Use the unique identifier from each establishment identified in 1) to extract further information via a Place detail search.

* Usage limits - 1000 calls an hour which may be increased to 100000 by 'verifying your identity' by providing your credit card to Google... The project as stands makes of the order of 25000 requests.

* Place search call - Will return up to the top 20 places within a user given search radius for a given latitude and longitude. To ensure I obtain _all_ places within the search area, I set the search radius to r = 100m, and we step the lat/long points by 2*r/sqrt(2). I capture the following returned fields : 'rating', 'name', 'reference', 'price_level', 'lat', 'lon', 'opening_hours', 'vicinity', 'photos', 'id', 'types', 'icon'. The main ones of interest are the ID, reference, rating, lat, and lon. The rating allows the list to be sorted for the best coffee and the data analysed via histograms. Using the lat/lon I am able to plot the results by postcode as on my blog.

* Place details call - Using the unique identifier from the Place search, we may now fetch more information about an establishment i.e. reviews, scores, etc. It was my original intention to get all the user reviews and/or the number of reviews via a details call. However, Google only returns _up_ _to_ 5 reviews. This is disappointing but also means that we can analyse if there is a bias in scores when an establishment has only a few reviews i.e. are the owners or friends adding in the first few favorable reviews?

* Other - Another way similar results might be achieved is to use a 'Radar search' from the Places API which will return up to 200 results for a given location and radius. There is a 5-times multiplier on the number of API calls for one of these requests and it returns fewer details.

## TODO:
* Asynchronous version to improve speed.
* Restrict calls to those that only fall within a geoJSON polygon definition to improve speed / decrease API calls.
* Cover all of New York City in the search.
* Find other methods to get the actual number of reviewers for a Place, rather than just up to 5.
* Find other methods to get the actual reviews from individuals for sentiment analysis or predictive capabilities.

## Note:

Breaking news - the 'ID' and 'references' that I have provided will soon be deprecated. From the Google Places API website: "The id and reference fields are deprecated as of June 24, 2014. They are replaced by the new place ID, a unique identifier that can be used to compare places and to retrieve information about a place. The Places API currently returns a place_id in all responses, and accepts a placeid in the Place Details and Place Delete requests. Soon after June 24, 2015, the API will stop returning the id and reference fields in responses. Some time later, the API will no longer accept the reference in requests. We recommend that you update your code to use the new place ID instead of id and reference as soon as possible."


