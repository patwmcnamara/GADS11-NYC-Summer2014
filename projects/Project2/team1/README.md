#Code Execution

##To Run the Best Fitting Algorithm

	import analyze
	
	#assume baseball.csv is located at ~/home/baseball.csv
	
	pth = '~/home/baseball.csv'
	file_path = analyze.year_based_significance_log_regression(pth)
	
	res_df = analyze.year_based_log_regression(pth)
	

The columns `['is-r2', 'os-r2', 'mae']` stand for in-sample r-squared, out-sample r-squared, and 
mean absolute error.

