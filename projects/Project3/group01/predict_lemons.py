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

def scipt_function(arg_1, arg_2):
	return None

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
    # go through and construct a rough cut first
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
    #now trim down the most parsimonious model
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
    """
    mpy = data['VehOdo'].div(data['VehicleAge'])
    is_inf = numpy.isinf(mpy)
    mpy[is_inf] = numpy.nan
    return mpy

def general_zip_code(data):
    """
    Because the zipcode is important, however, the **exact** zipcode
    is not, this takes the first three digits of the zipcode
    """
    return (data['VNZIP1']/100.).apply(numpy.floor)

def parse_data(file_path):
    data = pandas.DataFrame.from_csv(file_path + 'lemon_training.csv')
    mpy = miles_per_year(data)
    zip_code = general_zip_code(data)
    mmr = sig_MMR(data)
    return pandas.concat( [mpy, zip_code, mmr], axis = 1)

if __name__ == '__main__':
	
    usage = sys.argv[0] + "usage instructions"
    description = "describe the function"
    parser = argparse.ArgumentParser(description = description, usage = usage)
    parser.add_argument('name_1', nargs = 1, type = str, help = 'describe input 1')
    parser.add_argument('name_2', nargs = '+', type = int, help = "describe input 2")

    args = parser.parse_args()
	
    script_function(input_1 = args.name_1[0], input_2 = args.name_2)
