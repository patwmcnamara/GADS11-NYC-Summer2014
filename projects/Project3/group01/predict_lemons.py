#!/usr/bin/env python
# encoding: utf-8
"""
<script_name>.py

Created by Benjamin Gross on <insert date here>.

INPUTS:
--------

RETURNS:
--------

TESTING:
--------


"""

import argparse
import pandas
import numpy
import itertools
import sys
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score

def sig_veh_buy(data):
    """
    Extract the MMR columns that are valuable when compared to the
    VehBCost (i.e. what someone paid at the auction)

    ARGS:

        data: :class:`pandas.DataFrame` of the lemon training data

    RETURNS:

        :class:`pandas.DataFrame` of the significant MMR pairings
        divided by the 'VehBCost' or Vehicle Buy Cost
    """
    
    cols = ['MMRAcquisitionAuctionAveragePrice','MMRAcquisitionAuctionCleanPrice',
            'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice',
            'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 
            'MMRCurrentRetailAveragePrice','MMRCurrentRetailCleanPrice']

    to_use = {}
    # go through and construct a rough cut based on .95 p-value
    for col in cols:
        xs = data['VehBCost'].div(data[col])
        is_inf = numpy.isinf(xs)
        xs[is_inf] = numpy.nan
        ols = pandas.ols(x = xs, y = data['IsBadBuy'])
        if ols.p_value['x'] < .05:
            to_use[col] = xs
    
    is_sig = 1e-3
    not_parsimonious = True
    while not_parsimonious:
    #now trim down to the most parsimonious model
        buy_df = pandas.DataFrame(to_use)
        ols = pandas.ols(x = buy_df, y = data['IsBadBuy'])
        if any(ols.p_value > is_sig):
            for val in ols.p_value[ols.p_value > is_sig].index:
                try:
                    to_use.pop(val)
                except:
                    print "Intercept not significant"
        else:
            not_parsimonious = False
    return buy_df

def sig_MMR(data):
    """
    Extract the MMR columns that are valuable based on the multivariate
    regression run by statsmodels

    ARGS:

        data: :class:`pandas.DataFrame` of the lemon training data

    RETURNS:

        :class:`pandas.DataFrame` of the significant MMR pairings
    """
    cols = ['MMRAcquisitionAuctionAveragePrice','MMRAcquisitionAuctionCleanPrice',
            'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice',
            'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 
            'MMRCurrentRetailAveragePrice','MMRCurrentRetailCleanPrice']

    to_use = {}
    pairs = itertools.combinations(range(len(cols)), 2)
    # go through and construct a rough cut based on .95 p-value
    for x, y in pairs:
        xs = data[cols[x]].div(data[cols[y]])
        is_inf = numpy.isinf(xs)
        xs[is_inf] = numpy.nan
        ols = pandas.ols(x = xs, y = data['IsBadBuy'])
        if ols.p_value['x'] < .05:
            to_use[str(x) + ',' + str(y)] = xs
    
    is_sig = 1e-3
    not_parsimonious = True
    while not_parsimonious:
    #now trim down to the most parsimonious model
        mmr_df = pandas.DataFrame(to_use)
        ols = pandas.ols(x = mmr_df, y = data['IsBadBuy'])
        if any(ols.p_value > is_sig):
            for val in ols.p_value[ols.p_value > is_sig].index:
                try:
                    to_use.pop(val)
                except:
                    print "Intercept not significant"
        else:
            not_parsimonious = False
    return mmr_df
       
def miles_per_year(data):
    """
    Calculate the number of miles per year for a given car instead of 
    simply using the odometer

    ARGS:

        data: :class:`pandas.DataFrame` of the lemon training data

    RETURNS:

        :class:`pandas.Series` of the miles per year of each car
    
    """
    mpy = data['VehOdo'].div(data['VehicleAge'])
    is_inf = numpy.isinf(mpy)
    mpy[is_inf] = numpy.nan
    mpy.name = 'miles_per_year'
    return mpy

def truncated_zipcode(data):
    """
    Because the zipcode is important, however, the **exact** zipcode
    is not, this takes the first three digits of the zipcode

    ARGS:

        data: :class:`pandas.DataFrame` of the lemon training data

    RETURNS:

        :class:`pandas.Series` of the first 3 digits of the zipcode
    
    """
    ret_series = (data['VNZIP1']/100.).apply(numpy.floor)
    ret_series.name = 'trunc_zip'
    return ret_series

def parse_data(file_path):
    """
    Run the `sig_MMR`, `truncated_zipcode`, and `miles_per_year` functions
    and aggregate them into a single :class:`pandas.DataFrame`

    ARGS:

        file_path: :class:`string` of the path to the `lemon_training.csv`
        file

    RETURNS:

        :class:`pandas.DataFrame` of the aggregated feature calcs
    
    """
    data = pandas.DataFrame.from_csv(file_path + 'lemon_training.csv',
                                     index_col = None)
    mpy = miles_per_year(data)
    zip_code = general_zip_code(data)
    mmr = sig_MMR(data)
    buy_df = sig_veh_buy(data)
    return pandas.concat( [mpy, zip_code, mmr, buy_df], axis = 1)

def bagging_predict_lemons(x_train, x_test, y_train, y_test, rands = None):
    """
    Predict the lemons using a Bagging Classifier and a random seed
    both for the number of features, as well as for the size of the
    sample to train the data on

    ARGS:

        - x_train: :class:`pandas.DataFrame` of the x_training data

        - y_train: :class:`pandas.Series` of the y_training data

        - x_test: :class:`pandas.DataFrame` of the x_testing data

        - y_test: :class:`pandas.Series` of the y_testing data

        - rands: a :class:`tuple` of the (rs, rf) to seed the sample
        and features of the BaggingClassifier.  If `None`, then
        rands are generated and provided in the return `Series`

    RETURNS:

        :class:`pandas.Series` of the f1-scores and random seeds
    """
    #create a dictionary for the return values
    ret_d = {'train-f1':[], 'test-f1':[], 'rs':[], 'rf':[]}

    #use the randoms provided if there are any, otherwise generate them
    if not rands:
        rs =  numpy.random.rand()
        rf = numpy.random.rand()
        while rf < 0.1:
            rf = numpy.random.rand()
    else:
        rs, rf = rands[0], rands[1]
    #place them into the dictionary
    ret_d['rs'], ret_d['rf'] = rs, rf
    #create and run the bagging classifier
    bc = BaggingClassifier(n_estimators = 100, max_samples = rs,
                           max_features = rf, n_jobs = -1)

    bc.fit(x_train, y_train)
    y_hat_train = bc.predict(x_train)
    ret_d['train-f1'] = f1_score(y_train, y_hat_train)
    y_hat_test = bc.predict(x_test)
    ret_d['test-f1'] = f1_score(y_test, y_hat_test)
    return pandas.Series(ret_d)
    
def create_in_out_samples(data, in_sample_size):
    """
    Construct in-sample and out-of sample data

    Args:
    ------
    - data: `pandas.DataFrame` of the data
    - in_sample_size: integer of the size of the in-sample data (the
      out of sample data will be the rest of the data)

    Returns:
    --------
    - isi: `pandas.Index` of the in-sample data
    - in_sample: `pandas.DataFrame` of the in-sample data
    - osi: `pandas.Index` of the out-of-sample data
    - out_sample: `pandas.DataFrame` of the out-of-sample data, i.e.
      the rest of the data not part of the in_sample)
    """
    #in-sample index and out-of-sample index
    isi = numpy.random.choice(data.index, in_sample_size)
    osi = data.index[~data.index.isin(isi)]

    #create in-sample and out-of-sample DataFrames
    in_sample = data.loc[ isi, :].copy()
    out_sample = data.loc[ osi, :].copy()

    ##Fill the in-sample data with the means if there are nan values
    if in_sample.isnull().any().any():
        fill_data = in_sample.mean().apply(numpy.floor)
        in_sample.fillna( fill_data, inplace = True)

    #Fill the out-of-sample with the means from the in-sample
    if out_sample.isnull().any().any():
        out_sample.fillna( fill_data, inplace = True)

    return isi, in_sample, osi, out_sample

def load_data_and_run_bagging_classifier(file_path):
    data = pandas.DataFrame.from_csv(file_path + 'lemon_training.csv',
                                     index_col = None)

    my_features = parse_data(file_path)
    agg_data = pandas.concat([my_features, data['IsBadBuy']], axis = 1)
    agg_data.dropna(inplace = True)
    x_cols = agg_data.columns[agg_data.columns != 'IsBadBuy']
    isi, in_sample, osi, out_sample = create_in_out_samples(
        agg_data[x_cols], int(agg_data.shape[0]/2.) )

    #run the bagging classifier 1000 times
    d = {}
    for i in numpy.arange(100):
        print "Now on " + str(i) + " out of " + str(100)
        d['sim_'+str(i)] = bagging_predict_lemons(in_sample, out_sample,
            agg_data['IsBadBuy'][isi], agg_data['IsBadBuy'][osi])
        
    return pandas.DataFrame(d).transpose()

if __name__ == '__main__':
	
    usage = sys.argv[0] + "usage instructions"
    description = "describe the function"
    parser = argparse.ArgumentParser(description = description, usage = usage)
    parser.add_argument('name_1', nargs = 1, type = str, help = 'describe input 1')
    parser.add_argument('name_2', nargs = '+', type = int, help = "describe input 2")

    args = parser.parse_args()
	
    script_function(input_1 = args.name_1[0], input_2 = args.name_2)
