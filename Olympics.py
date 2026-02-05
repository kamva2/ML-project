import pandas as pd

teams = pd.read_csv('teams.csv')
#print(teams)

# minimize the dataset to only include relevant columns
teams = teams[['team', 'country','year','athletes', 'age', 'prev_medals', 'medals']]
print(teams)

print(teams.corr(numeric_only=True)["medals"])

import seaborn as sns
import matplotlib.pyplot as plt

sns.lmplot(x='athletes', y='medals', data=teams,ci=None)
plt.show()

print(teams.plot.hist('medals'))




