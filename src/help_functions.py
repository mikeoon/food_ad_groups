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


def plot_positives(df, field, figs=(20, 10)):
	fig, ax = plt.subplots(figsize=figs)

	for ad in df['ad'].unique():
	    mask = (df['ad'] == ad)
	    if get_averages(df, mask, 'ad_profit') >= 0:
	        ax.plot(df[mask]['date'], df[mask][field],'--.' ,label=ad)

	ax.set_xlabel('Day')
	ax.set_ylabel(field)
	ax.set_title(f'ad_groups and {field}')
	plt.xticks(rotation=45)
	ax.legend()