# Investigating the relationship between solar magnetic activity and coronal mass ejections

## Summary

A coronal mass ejection (CME) is an energetic event in which the sun ejects a large body of plasma along with a frozen in magnetic field configuration into the solar wind. A so called 'halo' event occurs when this ejection appears primarily along the earth-sun field of view. It is these events in particular which pose risk to earth because of the strong association with space weather and geomagnetic storms arising from the CME's magnetic interaction with the Earth's magnetic field. This poses problems for astronauts, satellites, high frequency communications, and may induce currents in any long conductors near the earth's poles (pipelines, powerlines etc.) It is clearly of interest to understand and predict these events.

Because of the strong association of CMEs and the solar magnetic field configuration, a natural place to start looking at predictors is the sun spot cycle (sun spots occur because the twisted magnetic configurations inhibit heat convection and hence the plasma cools appearing dark, especially in hydrogen-alpha line images of the sun). Excellent data on sun spot numbers exists as far back as the 1800s, and more recently the latitude and longitude of sun spot areas on the sun is documented daily. Other indicators of solar activity include the solar irradiance or radio flux (akin to the power output of the sun) which can be observed at the Lyman Alpha line or 10.7cm Hydrogen line, respectively.  Even better are the vector magnetograms that directly image the magnetic field configuration on the sun's surface for which we have images produced every 15 minutes since 2010. Furthermore, to train our classifiers we have the LASCO catalogue of CMEs which has documented every (observerable/detectable) CME since 1997.

## Project files
### Analysis
`cmes.ipynb` - looks at the association of more qualitative sun activity measures on predicting CMEs on time scales of days to a month.

`magnetograms.ipynb` - uses vector magnetogram data from 2010-2013 to analyse solar activity and attempt to predict CMEs on time scales of 15-30 mins.

### Feature data
`issn.md` - international sunspot number record.

`daily_area.txt` - daily area of the sun covered in sunspots.

`lyman_alpha.txt` - daily solar irradiance.

`f10.7.txt` - daily radio flux.

`g*.txt` - Greenwich sunspot grouping data (tracks groups of sunspots).

`/data/all_256` - (Not included here) 3.4GB of vector magnetogram data from 2010-2013.

### Target data
`cme_catalogue.md` - List of times and other attributes for all CMEs since 1997. Objective is to predict the number of CMEs in a given time interval.

![alt text](https://github.com/dinob0t/ga_project_final/blob/master/monthly_mean_cmes.png)<br>
(Fig 1: Mean CMEs per month grouped by year)

## Part 1 - Proxies for solar activity - analysis and prediction

### Known features that correlate with solar activity
- The number of sun spots

![alt text](https://github.com/dinob0t/ga_project_final/blob/master/monthly_mean_ssn_clip.png)<br>
(Fig 2: Mean CMEs per month grouped by year)

- The area of the solar disc covered by sun spots

![alt text](https://github.com/dinob0t/ga_project_final/blob/master/monthly_mean_ssa.png)<br>
(Fig 3: Mean area of solar disc covered per month grouped by year)

- The solar irradiance 

![alt text](https://github.com/dinob0t/ga_project_final/blob/master/monthly_mean_lyman.png)<br>
(Fig 4: Mean solar irradiance grouped by year)

- The solar radio flux

![alt text](https://github.com/dinob0t/ga_project_final/blob/master/monthly_mean_10.7.png)<br>
(Fig 5: Mean F10.7 radio emission by year)

Figures 2-5 show that all these measures are highly correlated with each other and are likely to be related to the same variation that we are trying to model. We require additional independent features.

### Tracking sun spot groups

The Greenwich sunspot grouping data tracks information about any number of sunspots that exist on the solar disc at one time - sometimes there are no sunspots and sometimes there are many. In order to incorporate the sunspot information as additional features we can choose a number (N) of sunspots to track, chosen by sorting on a particular aspect of the sunspot group (i.e. size, position, etc). In the absence of any sunspots, the columns representing the parameters of the sunspot data are set to zero.

### Lagging previous CME information

The information required to predict the number of CMEs on a particular day is likely to be related to CME activity on the day before. Therefore a number of CME parameters such as size, velocity, onset hour, etc., are lagged by one day and added as additional features. In the case that there are multiple CMEs the day before then the average values of the features is used.

### Attempted target

1) Predict the number of CMEs tomorrow based on the information for today. Fig 6 shows the relationship between sunspots and CMEs on a daily basis. They do not appear to be strongly correlated. 

![alt text](https://github.com/dinob0t/ga_project_final/blob/master/correlation_cme_ssn_daily.png)<br>
(Fig 6: Showing correlation between sunspot number and CME number per day)

2) Predict the number of CMEs next month based on the information for this month (1 month is approximately 1 solar cycle). Fig 7 shows the relationship between sunspots and CMEs on a monthly basis. There is a better correlation over the course of a month compared to per day.

![alt text](https://github.com/dinob0t/ga_project_final/blob/master/correlation_cme_ssn_monthly.png)<br>
(Fig 7: Showing correlation between sunspot number and CME number per month))

### Classifiers

1) Random Forest for attempted target 1 (above), as the number of categories representing the number of CMEs on any given day is limited to around 10.
2) Linear Model Regression for attempted target 2 (above), as there are a large number of CMEs occurring over the course of a month.

## Results
1) Random Forest - predicting for each day.
```
R^2 train: 0.340327415264
R^2 test: 0.32688172043
Classification Report
             precision    recall  f1-score   support

        0.0       0.49      0.75      0.59      1079
        1.0       0.27      0.23      0.25       959
        2.0       0.26      0.31      0.28       762
        3.0       0.22      0.35      0.27       583
        4.0       0.00      0.00      0.00       418
        5.0       0.00      0.00      0.00       246
        6.0       0.00      0.00      0.00       152
        7.0       0.00      0.00      0.00        68
        8.0       0.00      0.00      0.00        32
        9.0       0.00      0.00      0.00        24
       10.0       0.00      0.00      0.00         6
       11.0       0.00      0.00      0.00         4
       12.0       0.00      0.00      0.00         2
       13.0       0.00      0.00      0.00         1
       14.0       0.00      0.00      0.00         1

avg / total       0.26      0.34      0.29      4337

             precision    recall  f1-score   support

        0.0       0.47      0.71      0.57       446
        1.0       0.28      0.27      0.27       401
        2.0       0.25      0.29      0.26       347
        3.0       0.21      0.34      0.26       247
        4.0       0.00      0.00      0.00       194
        5.0       0.00      0.00      0.00       101
        6.0       0.00      0.00      0.00        61
        7.0       0.00      0.00      0.00        31
        8.0       0.00      0.00      0.00        18
        9.0       0.00      0.00      0.00         8
       10.0       0.00      0.00      0.00         4
       11.0       0.00      0.00      0.00         1
       13.0       0.00      0.00      0.00         1

avg / total       0.25      0.33      0.28      1860

Feature Importance
(0.10217336393637823, 'L_speed')
(0.093088329644439222, '2nd_o_speed')
(0.092635194989601835, '2nd_o_speed_20R')
(0.075360227964301738, 'F10.7')
(0.075162723733324707, 'hour')
(0.060248208204694925, 'LymanAlpha')
(0.056812832819784748, 'Central_PA')
(0.052773346909765755, 'Width')
(0.042827630101142759, 'CME')
(0.04272818297068276, 'ssn')
```
![alt text](https://github.com/dinob0t/ga_project_final/blob/master/prediction_cmes_vs_ssn_daily.png)<br>
(Fig 8: Sunspots vs CME numbers per day (blue - data, red - predicted))


2) Ridge Regression (2nd degree polynomial)- predicting for a month
```
R^2 train: 0.653400997278
R^2 test: 0.507888792648
MAPE train: 129.544504755
MAPE test: 173.152010099
```
![alt text](https://github.com/dinob0t/ga_project_final/blob/master/prediction_cmes_vs_ssn_monthly.png) </br> <br>
(Fig 9: Sunspots vs CME numbers per month (blue - data, red - predicted))

### Discussion
The ability of the Random Forest to predict the number of CMEs the next day is poor. It is only able to classify a small percentage of between 0-3 CMEs total per day and misses the classification of the 4-14 CMEs total per day category. What is interesting is the feature importance of the classifier which indicates that many of the predictive features are associated with the CMEs occurring the day before (i.e. CME linear speeds, number of CMEs, hour of CME events, etc.) The other important features are the irradiance and radio flux, but due to the high correlation of these features with sunspot area and sunspot number, it is likely that these could be equally substituted.

The ability of the Ridge Regressor to predict the monthly CME number is better than the day to day prediction above. Unfortunately the R^2 and MAPE values of the test and training sets indicates an over-fitting problem which can not be easily rectified. Because the data is now grouped by month, there is only 203 rows representing the data from 1997-2013. This makes the regularisation of the Ridge Regressor essential, but there simply isn't enough data yet to do much better.

## Part 2 - Vector magnetogram analysis
### Summary
Vector magnetograms directly image the magnetic field at the sun's chromosphere. The hope is that the magnetic configurations recorded by these images contains some information that can be used to predict the onset of a CME.  The HMI instrument as part of the launch of the Solar Dynamic Observatory records these magnetograms every 15 minutes. They are available in a variety of resolutions, the smallest being 256x256 pixels at this site: http://jsoc.stanford.edu/data/hmi/images/. 

![alt text](https://github.com/dinob0t/ga_project_final/blob/master/slides.key/Data/jsoc_dirs-400.png) <br>
(Fig 10: Directory structure of the hosted HMI vector magnetogram data.)

### Data scraping
To recursively parse the directory structure and download images, `wget` can be a useful tool, i.e.
`wget -r  -nd -N -np --tries=75 --no-clobber  --accept '*_M_256.jpg' http://jsoc.stanford.edu/data/hmi/images`
While this works fine, it is very slow for a large number of files (order of 10 hours). One ugly strategy is to run the above command multiple times separated by the `&` symbol. Since each call won't overwrite an existing file, we now have multiple 'threads' each working on different parts of different directories. This now reduces the time to a few hours, but somehow some of the files are missed.

A better solution is to use the `asyncio` library and Python 3. A list is generated which contains all the urls that we are interested in, and this list is chopped up into (200) multiple chunks. Each chunk is then processed asynchronously yielding all the files we expect in approximately 1 hour.

### Data preparation
Initially we have 256x256=65536 pixels in our image that could potentially be features. The first thing to do is to create a mask that selects only pixels in the solar disc and then flatten the array. This reduces the number of pixels to 42981. Each pixels is greyscale and therefore can take one of 256 values.

The CME catalogue information in `cme_catalogue.md` is used to classify each image as either a '1' if the image timestamp indicates that it occurs directly before a CME, or a '0' otherwise. The images are only available from 2010-2013 and so we have 126144 images total, of which 2794 are 'pre-CME' frames with the target set to '1'.

An additional array the same dimension as the number of images 'time_since' is also calculated which records the number of frames since the last CME. In Part 1 of the project, information relating to previous CME activity turned out to be strong predictors of future CMEs.

### Dimensionality reduction 1 - PCA
Clearly the dimensionality is very large and we have limited target information. Either regularisation needs to be imposed or dimensionality reduction should be employed. Using something like Logistic Regression with a 'L1', or 'L2' regularisation penalty is not possible on a 126144x42981 array due to computational constraints.

![alt text](https://github.com/dinob0t/ga_project_final/blob/master/pca_cum_var_explained_1000.png) <br>
(Fig 11: Cumulative explained variance ratios for up to 1000 PCA components.)

Figure 11 shows the total explained variance if we were to choose up to 1000 principal components. The strategy is normally to choose the number of components that corresponds to just after the steepest gradient on this curve i.e., at around 100 components. However, the values on the y-axis show that we are only capturing a very small amount of the variance.

Nonetheless, if we push on and reduce our dimensionality to now 126144x100 we can now at least attempt classification. The classifier used will be the Multinomial Naiive Bayes and the metric we are primarily interested in is the Area Under the Curve (AUC). The initial results are below:
```
R^2 train: 0.976806342016
R^2 test: 0.976508825706
             precision    recall  f1-score   support

          0       0.98      1.00      0.99     86350
          1       0.02      0.00      0.00      1950

avg / total       0.96      0.98      0.97     88300

             precision    recall  f1-score   support

          0       0.98      1.00      0.99     37000
          1       0.08      0.00      0.01       844

avg / total       0.96      0.98      0.97     37844

Area Under Curve train: 0.499933781717
Area Under Curve test: 0.501707506084
**Area Under Curve average:** 0.500820643901
```
An average AUC of 0.5 indicates that the classifier is no better than random chance. If we now add in the additional feature 'time_since' which records the number of frames since the last CME the classifier performs better than before, but still poorly:
```
R^2 train: 0.659943374858
R^2 test: 0.663169855195
             precision    recall  f1-score   support

          0       0.98      0.66      0.79     86353
          1       0.03      0.48      0.06      1947

avg / total       0.96      0.66      0.78     88300

             precision    recall  f1-score   support

          0       0.98      0.67      0.79     36997
          1       0.03      0.46      0.06       847

avg / total       0.96      0.66      0.78     37844

Area Under Curve train: 0.569851588204
Area Under Curve test: 0.561822556275
**Area Under Curve average:** 0.56583707224
```

### PCA Discussion
The improvement in the classifier is almost entirely due to the 'time_since' feature. Whilst there is no guarantee that we would have done better if the original dimensionality was somehow all retained, it appears that the PCA was ineffective. This is due to the small amount of variance that is explained by our 100 components as in Fig 11. It appears that there is just too much variance in the base features of the sun over several years of data in order to reduce it to a subset of principal components. For example, if we attempted to reduce only one solar rotation (~27 days) to 100 components, we capture much more of the variance (Fig 12).

![alt text](https://github.com/dinob0t/ga_project_final/blob/master/pca_cum_var_explained_1000_single_month.png) <br>
(Fig 12: Cumulative explained variance ratios for up to 1000 PCA components for one solar rotation only.)

Figures 13-14 show how different the 'Eigensuns' appears for two different solar rotations - one in May 2010 closer to solar minimum, and one in December 2013 closer to solar maximum. There is far more structure in the magnetic fields as the sun approaches solar maximum.

![alt text](https://github.com/dinob0t/ga_project_final/blob/master/may_2010.png) <br>
(Fig 13: First 16 Eigensuns from PCA computed for the solar rotation in May 2010.)

![alt text](https://github.com/dinob0t/ga_project_final/blob/master/december_2013.png) <br>
(Fig 14: First 16 Eigensuns from PCA computed for the solar rotation in December 2013.)

### Dimensionality reduction 2 - Image similarity
One may ask if there is anything measurably different about the state of the sun before a CME compared to a quiet state of the sun. A quiet state here is defined as an image frame that occurs at the half-way point between 2 consecuative pre-CMEs images - this separation may be anywhere from 15 mins to many hours. PCA is applied to all the images forming the 'quiet' sun state and the 'pre-CME' state, and the Eigensuns are plotted in Figures 15-16.

![alt text](https://github.com/dinob0t/ga_project_final/blob/master/quiet.png) <br>
(Fig 15: First 16 Eigensuns from PCA computed for the sun in a 'quiet' state.)

![alt text](https://github.com/dinob0t/ga_project_final/blob/master/cmes.png) <br>
(Fig 16: First 16 Eigensuns from PCA computed for the sun in a 'pre-CME' state.)

For each image in our dataset we can compute the correlation between the image and each of these Eigensuns representing the 'pre-CME' state and the 'quiet' state. This gives us additional features that indicate whether the present image is more like a 'quiet' sun or a 'pre-CME' sun. Due to computational constrains only 10 Eigensuns are used and the use of a correlation measure is chosen over the Self-Similarity Image Index. After adding in these additional features the Multinomial Naiive Bayes classifier is attempted again.
```
R^2 train: 0.688482446206
R^2 test: 0.688748546665
             precision    recall  f1-score   support

          0       0.98      0.69      0.81     86321
          1       0.03      0.47      0.06      1979

avg / total       0.96      0.69      0.80     88300

             precision    recall  f1-score   support

          0       0.98      0.69      0.81     37029
          1       0.03      0.48      0.06       815

avg / total       0.96      0.69      0.80     37844

Area Under Curve train: 0.581466741011
Area Under Curve test: 0.587151522923
**Area Under Curve average:** 0.584309131967
```
This shows another improvement in the classifiers accuracy. Future work should include expanding the number of Eigensuns for which this similarity index is calculated, revisiting the classification of the 'quiet' sun state by ensuring a longer time period between CME events', and the use of the Self-Similarity Image Index which is slower to compute but possibly more accurate.

### Dimensionality reduction 3 - Greyscale count
The last lower dimensional feature implemented is a simple binning count of the number of different grey-level pixels per image. This adds another 256 features to our dataset which with the previous features now gives us dimensions of 126144x523. The results are now further improved to:
```
R^2 train: 0.672718006795
R^2 test: 0.671757742311
             precision    recall  f1-score   support

          0       0.98      0.68      0.80     86368
          1       0.04      0.53      0.07      1932

avg / total       0.96      0.67      0.79     88300

             precision    recall  f1-score   support

          0       0.98      0.67      0.80     36982
          1       0.04      0.54      0.07       862

avg / total       0.96      0.67      0.78     37844

Area Under Curve train: 0.601700331654
Area Under Curve test: 0.60827553782
Area Under Curve average: 0.604987934737
```

## Conclusion
This project to predict CMEs was ambitious, especially to the accuracy of within 15-30 mins as in the vector magnetograms in Part 2. In Part 1, much better results were obtained when trying to predict on the time scale of a month, but this was at the expense of reducing the dataset considerable. With the right image analysis in Part 2, the classification does start to show some promise. Overall, however, the ability to accurately predict the onset of a CME from the 256x256 vector magnetogram was poor. This might be expected from simply considering the resolution of the data used. Figures 17-18 shows the difference between the magnetograms used in this project and the high resolution data available. 

![alt text](https://github.com/dinob0t/ga_project_final/blob/master/slides.key/Data/20100501_000000_M_256-385.jpg) <br>
(Fig 17: 256x256 vector magnetogram image.)

![alt text](https://github.com/dinob0t/ga_project_final/blob/master/slides.key/Data/20100501_000000_M_4k-682.jpg) <br>
(Fig 18: 4096x4096 vector magnetogram image.)

It is likely that required magnetic field configuration information exists on length scales that only the high resolution magnetograms would capture. However, using this data would mean even higher dimensional data, of the order of 16 million features per image. There are only 2800 or so CMEs to study between 2010-2013, and so this system becomes severely underdetermined. With clever image analysis and the further addition of the data up to the present day in 2014 (an active year for CMEs so far), hopefully our predictive capabilities will further improve.
