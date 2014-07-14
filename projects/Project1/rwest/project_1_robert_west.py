###################################################
# Project 1: Scraping, APIs, and Data Visualization
# Robert D. West

import automated_craigslist_search.connect_to_craigslist as connect_to_craigslist
import pandas
import string
from nltk.corpus import stopwords

def is_number(s):
    '''Returns True is the string 's' contains a number, false otherwise.'''
    try:
        float(s)
        return True
    except ValueError:
        return False
        
def top_words(df):
    ''' Takes as an argument a pandas dataFrame containing craigslist search resuults from connect_to_craigslist.search_craigslist(). Returns a list of the most popular words in the search results (with a limited count of one entry per search result and excludes certain words such as the stopwirds provides by the module 'nltk' and various other keywords. Returns a pandas.DataFrame of results ordered by frequency
    
    :param df: a pandas DataFrame containing craigslist search resuults from connect_to_craigslist.search_craigslist()
    :type seach_key_words: pandas.DataFrame
    
    :param html: a pandas DataFrame containing a list of the most frequently seen words in search results
    :type seach_key_words: pandas.DataFrame
    
    '''
    # Remove duplicate listings
    all_words = df.Results.unique()
    
    # make all entries lower case
    all_words = [x.lower() for x in all_words]
    
    # remove punctuation
    exclude = set(string.punctuation)
    without_punc = []
    for s in all_words:
        without_punc.append(''.join(ch for ch in s if ch not in exclude))
    all_words = without_punc

    # remove numbers
    without_numbers = []
    for s in all_words: 
        without_numbers.append(''.join(ch for ch in s if not is_number(ch)))
    all_words = without_numbers
         
    # split the string on each row into single words
    split_words = [x.split() for x in all_words]

        
    # concatenate all words
    full_list = []
    for x in split_words:
        full_list = full_list + x
    
    # Remove standard words using NLTK library
    std_word_set=set(stopwords.words('english'))
    full_list = [x if x not in std_word_set else None for x in full_list]                   
    
    # remove non activity entries
    location_words = ['manhattan','midtown',"nyc","houston","san","antonio","pheonix","city","chicago","phoenix"]
    needy_words = ["looking","seeking","want","wanted","needed","need","help"]
    others = ['sick','action',"new",'opening',"starts","players","player","free","partner","team","group","july","get","th"]
    non_actitivity_words = location_words + needy_words + others
    full_list = [x if x not in non_actitivity_words else None for x in full_list]                                                                            
                                                                                                                                                                                  
    # find the list of unique words    
    unique_list = []
    for x in full_list:
        if x not in unique_list:
            unique_list.append(x)
    
    # count the occurence of each word, 1 per row of the dataset
    count_entries = []
    for y in unique_list :
        count = 0
        for x in split_words:
            if y in x:
                count = count + 1
        
        count_entries.append(count)
    
    # stores the unique entry results in a dataframe  
    unique_words_df = pandas.DataFrame({"Unique Words" : unique_list, "frequency" : count_entries})
    
    # drop null values  
    unique_words_df = unique_words_df[~unique_words_df['Unique Words'].isnull()]
 
    # sort dataframe
    unique_words_df = unique_words_df.sort('frequency',ascending=False)

    return unique_words_df


if __name__ == "__main__":

    # top 10 us cities by population
    cities = ['newyork','losangeles','chicago','houston','philadelphia','phoenix','sanantonio'] #,'sandiego','dallas','sanjose']
   
    ################################################ 
    # Activity Partners top 20 across mutiple cities
    all_data = pandas.DataFrame()
    for current_city in cities:
        df = connect_to_craigslist.search_craigslist('',category='activity partners', city=current_city)
        # top 20 from each city
        topwords_df = top_words(df)
        topwords_df['city'] = current_city 
        top20 = topwords_df[:20]
        top20['idx'] = range(20)
        all_data = all_data.append(top20)        
        
    # reshape data
    df_pivot = all_data.pivot(index='idx', columns='city')
    # export
    df_pivot.to_csv('activities2.csv')
    #df_pivot.to_csv('personal.csv')
    