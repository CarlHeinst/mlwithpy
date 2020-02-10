
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

def prepare_country_stats(oecd_bli, gdp_per_capita, year):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={year: "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

# Load the data
oecd_bli = pd.read_csv("oecd_blie_2019.csv", thousands=',')
gdp_per_capita = pd.read_csv("WEO_Data.tsv", thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")

## Prepare the data and run the missing dumb asses function
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita, "2019")
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')


model = sklearn.linear_model.LinearRegression()

model.fit(X, y)

X_new = [[28334]]
print("Value for Cyprus", model.predict(X_new))
plt.show(), country_stats