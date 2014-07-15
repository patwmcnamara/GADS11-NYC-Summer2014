# Project 1: Scraping, APIs, and Data Visualization
## Robert D. West

## Requirements

* Expected requirements: `pandas`, `urllib2`, `bs4`

* I created the Module `automated_craigslist_search` to complete this project.  This can be cloned an installed as follows:

		$ git clone git@github.com:robertdavidwest/api-automated_craigslist_search.git #assuming ssh install
		$ cd automated_craigslist_search
		$ python setup.py install

* I also used the python package `nltk` installed from pip. As well as installing this package. I also had to run the following command to get the data I needed:

		$ import nltk
		$ nltk.download()

	A separate window will appear. Download the `stopwords` corpus. This data is used to obtain a list of Natural Language regular vocabulary.

* `automated_craigslist_search` also uses the module `email` but this was not used in Project01.

## Write-Up

* `automated_craigslist_search`: Using the module I have created called `automated_craigslist_search` I am able to obtain data from Craigslist searches and store the results in a Dataframe. The module also allows you to create a html output of the results and send them out in an e-mail list.
	* I obtained the search results from Craigslist using `urllib2` and `bs4` to obtain and filter the data. I then stored the results in a `pandas.DataFrame`. It wasn't pretty but I think I should be able to use this going forward.
	
	* So far **the fields I have obtained** from the search results are: **Date, Location, Price, the Description, and the live link**.   

* Using the file `project_1_robert_west.py` I set out to find the most popular activities from the Craigslist activity partner section across different major us cities. i.e. **My question was: Can we find the most popular desired activities by scraping the Craigslist activity section across multiple cities?**
	* Using this module I downloaded datasets from the `activity` section of Craigslist across 7 major cities in the United States.
	 
	**The changes that need to be made to the data to obtain the answer to the question are the following**:
 
	* I then cleaned the data, setting the results to lower case, removing all numbers and punctuation and commonly used vocabulary including standard Natural Language words and some idiosyncratic words associated with the search: e.g. "need", "want", "looking" and locations e.g "nyc" etc
	* I then found the most popular search words for each city (one per search result) to gain an understanding of the most desired activities across the different cities

I haven't blogged about this yet. I will blog about the `automated_craigslist_search` module that I have created once I get it up and running on a server. I've just opened an AWS account and have an EC2 up and running, I just need to figure out how to use it....