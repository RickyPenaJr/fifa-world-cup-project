
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
matches = pd.read_csv("data/WorldCupMatches.csv")
tournaments = pd.read_csv("data/WorldCups.csv")

# Clean column names
matches.columns = matches.columns.str.strip()
tournaments.columns = tournaments.columns.str.strip()

# Feature Engineering
matches['Home Goals'] = matches['Home Team Goals']
matches['Away Goals'] = matches['Away Team Goals']
matches['Total Goals'] = matches['Home Goals'] + matches['Away Goals']

# Team performance summary
team_goals = matches.groupby('Home Team Name')['Home Goals'].sum() +              matches.groupby('Away Team Name')['Away Goals'].sum()

top_10_teams = team_goals.sort_values(ascending=False).head(10)
top_10_teams.plot(kind='bar', figsize=(10, 6), title='Top 10 Teams by Goals Scored')
plt.ylabel("Total Goals")
plt.tight_layout()
plt.show()

# Goal trends over the years
tournaments['Year'] = tournaments['Year'].astype(int)
tournaments['Avg Goals per Match'] = tournaments['GoalsScored'] / tournaments['MatchesPlayed']

plt.figure(figsize=(10, 6))
sns.lineplot(data=tournaments, x="Year", y="Avg Goals per Match", marker="o")
plt.title("Average Goals per Match Over World Cups")
plt.ylabel("Goals")
plt.grid(True)
plt.tight_layout()
plt.show()

# Hypothesis Test: Did goals per match change pre vs post 2000?
pre_2000 = tournaments[tournaments['Year'] < 2000]['Avg Goals per Match']
post_2000 = tournaments[tournaments['Year'] >= 2000]['Avg Goals per Match']

t_stat, p_val = stats.ttest_ind(pre_2000, post_2000, equal_var=False)

print("T-test Results:")
print("T-statistic =", round(t_stat, 3))
print("P-value =", round(p_val, 3))

if p_val < 0.05:
    print("✅ Statistically significant change in goals per match (pre vs. post 2000).")
else:
    print("❌ No significant change in goals per match.")
