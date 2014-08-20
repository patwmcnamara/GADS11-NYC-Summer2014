import pandas 
import numpy
import statsmodels.api as sm
import pdb
import matplotlib.pyplot as plt

def rolling_ols(df,x_label,y_label,window_len):
    ''' Takes in a pandas.DataFrame, performs a 1 variable regression, regressing df.xlabel on df.ylabel (with an intercept).
    Performs a regression on a rolling basis over a window=window_len 
    
    :param df: a pandas DataFrame containing data for the regression
    :type df: pandas.DataFrame
    
    :param xlabel: the name of the column in df that represents the indepedent variable in the regression
    :type xlabel: string
     
    :param ylabel: the name of the column in df that represents the depedent variable in the regression   
    :type ylabel: string


    :param window_len: length of data to be used in the regression, this will roll over the whole dataset
    :type window_len: integer   
    
    :returns: the same pandas.DataFrame that was input, now including intercept and slope columns
    :rtype df: pandas.DataFrame
    
    '''
    # Pre-Conditions
    #assert( len(df) >= window_len, 'window_len must be less than or equal to len(df)')
    
    # the first n-1 values are NaN since we need at least n datapoints to run the regressions (where n = window_len), then our first prediction occurs at the nth step
    intercept = [numpy.NAN]*(window_len-1)
    slope = [numpy.NAN]*(window_len-1)
    # loop through dataset performing regression on rolloing n rows
    # add the intercept and slope from the regression to the DataFrame, then output the DataFrame
    for i in xrange(len(df)-window_len+1):  
        a = i 
        b = i + window_len
        
        rollin_data = df[a:b]
        X = rollin_data[x_label].values
        
        ############################################################################################################
        ### This is a stupid hack because sm.add_constant() wasn't appending ones when X was all zeros, bleurgh! ###
        if sum(X) == 0:
            tmp = X[0]
            X[0] = 5
            X = sm.add_constant(X,prepend=False)
            X[0,0] = tmp
            
        else :
            X = sm.add_constant(X)   
        ############################################################################################################
        
        #X = sm.add_constant(X) # in the hack above
        Y = rollin_data[y_label].values
        model = sm.OLS(Y,X)
        results = model.fit()
        
        # use previous run if coeffs are negative
        if results.params[1] <0 or results.params[0] <=0  :
            assert(len(intercept)<=(window_len-1),'1st coeffs are negative')
            intercept.append(intercept[-1])
            slope.append(slope[-1])
        else :
            intercept.append(results.params[0])
            slope.append(results.params[1])
    
    df['intercept'] = intercept   
    df['slope'] = slope
    return df
                     

def run_model(window_len):
     
    df = pandas.io.parsers.read_csv('Organic Sales.csv',index_col=0,parse_dates=True)
     
    #window_len = 10
    
    # run rolling regression
    df = rolling_ols(df,'Paid Sales','Organic Sales',window_len)
        
    # estimate weights or 'true organic' and 'paid organic'    
    df['w_organic'] = numpy.divide(df['intercept'], df['intercept'] + df['slope']*df['Paid Sales'])
    df['w_paid'] = 1 - df['w_organic']
    # estimate True Organic sales and True Paid sales (round to nearest integer)
    df['True Organic'] = (df['w_organic']*df['Organic Sales']).apply(round)
    df['Paid Organic'] = (df['w_paid']*df['Organic Sales']).apply(round)   
    df['True Paid'] = df['Paid Sales'] + df['Paid Organic']
    #df['cheap true paid'] = (df['Paid Sales'] + df['Organic Sales'] - df['intercept']).apply(round)
    
    # remove days when there was no spend
    #df = df[df.Spent!=0]
 
    # remove the nan rows
    df = df.dropna()
       
    # Rolling correlation
    #df['corr_Spend_vs_Paid_Sales']  = pandas.rolling_corr(df['Spent'],df['Paid Sales'],window_len)
    #df['corr_Spend_vs_Organic_Sales']  = pandas.rolling_corr(df['Spent'],df['Organic Sales'],window_len)
    #df['corr_Spend_vs_True Paid']  = pandas.rolling_corr(df['Spent'],df['True Paid'],window_len)
    #df['corr_Spend_vs_Paid Organic']  = pandas.rolling_corr(df['Spent'],df['Paid Organic'],window_len)
    #df['corr_Spend_vs_True Organic']  = pandas.rolling_corr(df['Spent'],df['True Organic'],window_len)
    print len(df)         
    
    # Correlation   
    # only on remaining 111 datapoints so can have a range of 5 to 15 for the window_len and compare results on the same data
    df = df[-111:]
    paid_r = numpy.corrcoef(df['Spent'],df['Paid Sales'])[0,1]
    organic_r = numpy.corrcoef(df['Spent'],df['Organic Sales'])[0,1]
    true_paid_r = numpy.corrcoef(df['Spent'],df['True Paid'])[0,1]
    paid_organic_r = numpy.corrcoef(df['Spent'],df['Paid Organic'])[0,1]
    true_organic_r = numpy.corrcoef(df['Spent'],df['True Organic'])[0,1]
    '''
    print '____________________________________________________________________'
    print '____________________________________________________________________'
    print 'window_len = ', window_len
    print '____________________________________________________________________'
    print 'Recorded Paid correlation with Spend    = %.2f%%' % (paid_r*100)
    print 'Recorded Organic correlation with Spend = %.2f%%' %  (organic_r*100)
    print ''    
    print 'Paid Untracked correlation with Spend     = %.2f%%' % (paid_organic_r*100)
    print 'True Organic correlation with Spend     = %.2f%%' % (true_organic_r*100)
    print 'True Paid correlation with Spend        = %.2f%%' % (true_paid_r*100)
    print '____________________________________________________________________'
    '''
    return organic_r, paid_organic_r, true_organic_r
    
    # Rolling correlation over time
    #df[['corr_Spend_vs_Paid_Sales','corr_Spend_vs_Organic_Sales','corr_Spend_vs_Paid Organic','corr_Spend_vs_True Organic']].plot()
    '''
    ###################
    ##### PLOTS #######
    ###################
    
    # Bars - Organic Sales  
    fig, ax1 = plt.subplots()
    p1 = ax1.bar(df.index, df['Organic Sales']   , color='#D1B9D4')  
    ax1.set_ylabel('Recorded Organic Sales')
    plt.title('Recorded Organic Sales')
    plt.setp(ax1.get_xticklabels(), rotation=30, fontsize=10)
    plt.savefig('plt1..jpg')
    plt.show()
    plt.close()
    
    # Stacked bars - Organic Sales  
    fig, ax1 = plt.subplots()
    p1 = ax1.bar(df.index, df['Paid Organic']   , color='#D1B9D4')
    p2 = ax1.bar(df.index, df['True Organic'], color='#D1D171') 
    ax1.set_ylabel('Untracked Sales')
    plt.title('Untracked Sales: Paid VS True Organic')
    ax1.legend( (p1[0], p2[0]), ('Untracked Paid Sales', 'True Organic Sales') )   
    plt.setp(ax1.get_xticklabels(), rotation=30, fontsize=10)
    plt.savefig('plt2..jpg')
    plt.show()
    plt.close()
  
    # Stacked bars - Organic Sales + Spend overtime on 2 axis  
    fig, ax1 = plt.subplots()
    p1 = ax1.bar(df.index, df['Paid Organic']   , color='#D1B9D4')
    p2 = ax1.bar(df.index, df['True Organic'], color='#D1D171') 
    ax1.set_ylabel('Recorded Organic Sales')
    plt.setp(ax1.get_xticklabels(), rotation=30, fontsize=10)
    plt.title('Untracked Sales: Paid VS True Organic & Spend')
    ax2 = ax1.twinx()
    p3 = ax2.plot(df.index, df['Spent'], color='#84DEBD' ,linewidth=3)
    ax2.set_ylabel('Spend')
    ax2.legend( (p1[0], p2[0],p3[0]), ('Untracked Paid Sales', 'True Organic Sales','Spend'),prop={'size':10} )   
    plt.savefig('plt3..jpg')
    plt.show()   
    plt.close()

    # subplots of bars - Organic Sales + Spend overtime on 2 axis 
    f, axarr = plt.subplots(2) 
    p1 = axarr[0].bar(df.index, df['Paid Organic']   , color='#D1B9D4')
    axarr[0].set_title('Untracked Sales: Paid VS True & Spend')
    axarr[0].set_ylabel('Estimated Paid Organic Sales')
    plt.setp(axarr[0].get_xticklabels(), rotation=30, fontsize=6)
    ax2 =axarr[0].twinx()
    p3 = ax2.plot(df.index, df['Spent'], color='#84DEBD' ,linewidth=3)
    ax2.set_ylabel('Spend')
    xlegend = r'Untracked Paid Sales, $\rho$ = ' + str(round(100*paid_organic_r,2)) + '%'
    next_legend = r'(Untracked Sales $\rho$ = ' + str(round(100*organic_r,2)) + '%)'
    ax2.legend( (p1[0],p3[0],p3[0]), (xlegend, 'Spend',next_legend),prop={'size':8})   

    p1 = axarr[1].bar(df.index, df['True Organic'], color='#D1D171')        
    axarr[1].set_ylabel('True Organic Sales')
    ax2 =axarr[1].twinx()
    plt.setp(ax2.get_xticklabels(), rotation=30, fontsize=6)
    p3 = ax2.plot(df.index, df['Spent'], color='#84DEBD' ,linewidth=3)
    ax2.set_ylabel('Spend')
    plt.setp(axarr[1].get_xticklabels(), rotation=30, fontsize=6)
    xlegend = r'True Organic Sales, $\rho$ = ' + str(round(100*true_organic_r,2)) + '%'
    ax2.legend( (p1[0],p3[0]), (xlegend,'Spend'),prop={'size':8} )   
    plt.savefig('plt4..jpg')
    plt.show()
    plt.close()
    '''
    
if __name__ == "__main__":                      
    
    organic_rs= []
    paid_organic_rs = []
    true_organic_rs = []
    
    for i in xrange(11):
        organic_r, paid_organic_r, true_organic_r = run_model(i+5)
        organic_rs.append(round(organic_r*100,2))
        paid_organic_rs.append(round(paid_organic_r*100,2))
        true_organic_rs.append(round(true_organic_r*100,2))
    
    df = pandas.DataFrame({'organic_r' : organic_rs, 'paid_organic_r' : paid_organic_rs, 'true_organic_r': true_organic_rs})
    print df