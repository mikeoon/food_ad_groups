import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')



def get_averages(df, mask, col):
	'''
	Calculates the average of the column given

	Parameters: dataframe df, filter mask, column col

	Returns: Average of col

	'''
	return df[mask][col].sum() / len(df[mask])


def quick_ad_look(df, ad_groups):
	for ad in ad_groups:
		mask = (df['ad'] == ad)
		avg_revenue = get_averages(df, mask, 'total_revenue')
		avg_ctr = get_averages(df, mask, 'ctr')
		avg_convert = get_averages(df, mask, 'converted')
		avg_cost = get_averages(df, mask, 'ad_click_costs')
		avg_profit = get_averages(df, mask, 'ad_profit')
		avg_click_cost = get_averages(df, mask, 'avg_cost_per_click')
		print(ad)
		print(f'Average Revenue {avg_revenue}')
		print(f'Average CTR {avg_ctr}')
		print(f'Average Convertion {avg_convert}')
		print(f'Average Cost(per click) {avg_cost}')
		print(f'Average Profit {avg_profit}')
		print(f'Average click cost {avg_click_cost}')
		print()
		print()


def clean_ad_name(df):
	func = lambda x: int(x.split('_')[-1])
	df['ad'] = df['ad'].apply(func)
	return df


def find_top_avg(df, field, top, asc=False):
	ad_mean_field = []
	ad_labels = []
	result = []
	for ad in df['ad'].unique():
		mask = (df['ad'] == ad)
		ad_mean_field.append(df[mask][field].mean())
		ad_labels.append(ad)

	for i in np.array(ad_mean_field).argsort():
	    result.append(ad_labels[i])

	if asc:
		return result[:top]
	else:
		result=result[::-1]
		return result[:top]



def plot_positives(df, field, figs=(20, 10)):
	fig, ax = plt.subplots(figsize=figs)

	for ad in df['ad'].unique():
	    mask = (df['ad'] == ad)
	    if get_averages(df, mask, 'ad_profit') >= 0:
	        ax.plot(df[mask]['date'], df[mask][field],'--.' ,label=ad)

	ax.axhline(df[field].mean(), df['date'].min(), df['date'].max(), c='k', linestyle='--', alpha=0.3)
	ax.set_xlabel('Day')
	ax.set_ylabel(field)
	ax.set_title(f'ad_groups and {field}')
	plt.xticks(rotation=45)
	ax.legend()


def plot_multi_ads(df, field, ads, figs=(20, 10)):
	fig, ax = plt.subplots(figsize=figs)

	for ad in ads:
	    mask = (df['ad'] == ad)
	    ax.plot(df[mask]['date'], df[mask][field],'--.' ,label=ad)

	ax.axhline(df[field].mean(), df['date'].min(), df['date'].max(), c='k', linestyle='--', alpha=0.3)
	ax.set_xlabel('Day')
	ax.set_ylabel(field)
	ax.set_title(f'ad_groups and {field}')
	plt.xticks(rotation=45)
	ax.legend()


def plot_cumsums(df, field, ads, figs=(20,10)):
	fig, ax = plt.subplots(figsize=figs)

	for ad in ads:
	    mask = (df['ad'] == ad)
	    ax.plot(df[mask]['date'], df[mask][field].cumsum(),label=ad)

	ax.set_xlabel('Day')
	ax.set_ylabel(field)
	ax.set_title(f'ad_groups and {field}')
	plt.xticks(rotation=45)
	ax.legend()


def bar_totals(df, field, tier_ads, figs=(20, 10)):
	fig, ax = plt.subplots(figsize=figs)

	avg_sum = 0	
	total_ads = 0
	for t, ads in enumerate(tier_ads):
		ad_heights = [df[df['ad'] == ad][field].sum() for ad in ads]
		ax.bar(ads, ad_heights, label=f'tier_{t}')
		avg_sum+=sum(ad_heights)
		total_ads+=len(ads)

	ax.axhline(avg_sum/total_ads, c='k', linestyle='--', alpha=0.3, label='sum_average')

	ax.set_xlabel('ad_group')
	ax.set_ylabel(field)
	ax.set_title(f'ad_groups and {field}')
	ax.legend()



# For finding trends in data

def make_design_matrix(arr):
	"""Construct a design matrix from a numpy array, converting to a 2-d array
	and including an intercept term."""
    return sm.add_constant(arr.reshape(-1, 1), prepend=False)

def fit_linear_trend(series):
	"""Fit a linear trend to a time series.  Return the fit trend as a numpy array."""
	X = make_design_matrix(np.arange(len(series)) + 1)
	linear_trend_ols = sm.OLS(series.values, X).fit()
	linear_trend = linear_trend_ols.predict(X)
	return linear_trend, linear_trend_ols.params


def plot_linear_trend(ax, name, series):
	linear_trend = fit_linear_trend(series)
	plot_trend_data(ax, name, series)
	ax.plot(series.index.date, linear_trend)


def find_trends(df, field):
	ad_trends = {}
	for ad in df['ad'].unique():
		mask = (df['ad'] == ad)
		temp_series = df[mask][['date', field]].set_index('date')
		preds, params = fit_linear_trend(temp_series)
		ad_trends[ad] = params[0]
	return ad_trends

def create_ts_df(df, field):
	return ad_table_df[['ad', 'date', field]].copy()


