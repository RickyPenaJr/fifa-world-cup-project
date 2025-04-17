
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load all yearly tournament data
data_path = "data/FIFA World Cup Datasets"
all_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and "Summary" not in f]

df_list = []
for file in all_files:
    df = pd.read_csv(os.path.join(data_path, file))
    df['Year'] = file.split('-')[1].strip().split('.')[0]
    df_list.append(df)

matches = pd.concat(df_list, ignore_index=True)

# Load summary file
summary = pd.read_csv(os.path.join(data_path, "FIFA - World Cup Summary.csv"))

# Data cleaning
matches.columns = matches.columns.str.strip()
summary.columns = summary.columns.str.strip()

matches['Year'] = matches['Year'].astype(int)
matches['Total Goals'] = matches['Home Team Goals'] + matches['Away Team Goals']

avg_goals_per_year = matches.groupby('Year')['Total Goals'].mean()

# Plot average goals per match over time
plt.figure(figsize=(10, 6))
sns.lineplot(x=avg_goals_per_year.index, y=avg_goals_per_year.values, marker="o")
plt.title("Average Goals per Match Over World Cups")
plt.xlabel("Year")
plt.ylabel("Goals")
plt.grid(True)
plt.tight_layout()
plt.savefig("visuals/avg_goals_per_year.png")
plt.show()

# Team performance
home_goals = matches.groupby('Home Team')['Home Team Goals'].sum()
away_goals = matches.groupby('Away Team')['Away Team Goals'].sum()
team_goals = home_goals.add(away_goals, fill_value=0)
top_10_teams = team_goals.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
top_10_teams.plot(kind='bar', title='Top 10 Teams by Goals Scored')
plt.ylabel("Total Goals")
plt.tight_layout()
plt.savefig("visuals/top_10_teams.png")
plt.show()

# Hypothesis test
pre_2000 = avg_goals_per_year[avg_goals_per_year.index < 2000]
post_2000 = avg_goals_per_year[avg_goals_per_year.index >= 2000]

t_stat, p_val = stats.ttest_ind(pre_2000.values, post_2000.values, equal_var=False)

print("T-test Results:")
print("T-statistic =", round(t_stat, 3))
print("P-value =", round(p_val, 3))

if p_val < 0.05:
    print("✅ Statistically significant change in goals per match (pre vs. post 2000).")
else:
    print("❌ No significant change in goals per match.")
