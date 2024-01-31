### Data Visualisation

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('data/london-borough-profiles-jan2018.csv', encoding='latin1')

features = data.columns.tolist()
for feature in features:
    print(feature)

# Filtering relevant columns for Population Density Visualization
pop_density_data = data[['Area name', 'Population density (per hectare) 2017']].sort_values(by='Population density (per hectare) 2017', ascending=False)

# Creating a bar plot for Population Density
plt.figure(figsize=(12, 8))
sns.barplot(x='Population density (per hectare) 2017', y='Area name', hue='Area name', data=pop_density_data, palette='viridis')
# Create a bar plot
plt.figure(figsize=(12, 8))
barplot = sns.barplot(
    x='Population density (per hectare) 2017', 
    y='Area name', 
    hue = 'Area name',
    data=pop_density_data
)
# Get the number of bars in the barplot
num_bars = len(barplot.patches)
# Create a color map
cmap = plt.get_cmap('viridis')

# Assign a color from the color map to each bar
for i, bar in enumerate(barplot.patches):
    bar.set_color(cmap(i / num_bars))

plt.title('Population Density per Hectare in London Boroughs (2017)')
plt.xlabel('Population Density (per hectare)')
plt.ylabel('London Boroughs')
plt.savefig(f'visualisations/Population Density per Hectare in London Boroughs (2017).png', dpi=300)
plt.show()

# Extracting the relevant data for the political landscape visualization
political_data = data['Political control in council'].value_counts().reset_index()
political_data.columns = ['Political Control', 'Count']

# Creating a pie chart for the political landscape analysis
plt.figure(figsize=(10, 6))
plt.pie(political_data['Count'], labels=political_data['Political Control'], autopct='%1.1f%%')
plt.title('Political Landscape in London Boroughs')

# Displaying the plot
plt.show()

# Selecting relevant columns for Age Distribution Visualization
age_data = data[['Area name', 'Average Age, 2017', 'Proportion of population aged 0-15, 2015',
                        'Proportion of population of working-age, 2015', 'Proportion of population aged 65 and over, 2015']]
age_data_sorted = age_data.sort_values(by='Average Age, 2017')

# Creating bar plots for Age Distribution
fig, axes = plt.subplots(4, 1, figsize=(12, 18))

# Average Age
sns.barplot(ax=axes[0], x='Average Age, 2017', y='Area name', hue='Area name', data=age_data_sorted, palette='coolwarm', legend=False)
axes[0].set_title('Average Age in London Boroughs (2017)')
axes[0].set_xlabel('Average Age')
axes[0].set_ylabel('')

# Proportion of population aged 0-15
sns.barplot(ax=axes[1], x='Proportion of population aged 0-15, 2015', y='Area name', hue='Area name', data=age_data_sorted, palette='coolwarm', legend=False)
axes[1].set_title('Proportion of Population Aged 0-15 in London Boroughs (2015)')
axes[1].set_xlabel('Proportion of Population Aged 0-15')
axes[1].set_ylabel('')

# Proportion of population of working-age
sns.barplot(ax=axes[2], x='Proportion of population of working-age, 2015', y='Area name', hue='Area name', data=age_data_sorted, palette='coolwarm', legend=False)
axes[2].set_title('Proportion of Working-Age Population in London Boroughs (2015)')
axes[2].set_xlabel('Proportion of Working-Age Population')
axes[2].set_ylabel('')

# Proportion of population aged 65 and over
sns.barplot(ax=axes[3], x='Proportion of population aged 65 and over, 2015', y='Area name', hue='Area name', data=age_data_sorted, palette='coolwarm', legend=False)
axes[3].set_title('Proportion of Population Aged 65 and Over in London Boroughs (2015)')
axes[3].set_xlabel('Proportion of Population Aged 65 and Over')
axes[3].set_ylabel('')

plt.tight_layout()
plt.show()

# Selecting a subset of features for initial visualizations
# Demographics vs. Economic Indicators
feature_set_1 = ['Population density (per hectare) 2017', 'Average Age, 2017', 'Gross Annual Pay, (2016)']

# Health and Wellbeing vs. Environmental Factors
feature_set_2 = ['People aged 17+ with diabetes (%)', 'Total carbon emissions (2014)']

# Education vs. Demographic Information
feature_set_3 = ['% of pupils whose first language is not English (2015)', 'Achievement of 5 or more A*- C grades at GCSE or equivalent including English and Maths, 2013/14']

# Political Landscape and Public Services
feature_set_4 = ['Political control in council', 'Average Public Transport Accessibility score, 2014']

# Preparing for visualization
sns.set(style="whitegrid")

# Creating a figure with multiple subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Plotting each feature set
sns.scatterplot(data=data, x=feature_set_1[0], y=feature_set_1[1], size=feature_set_1[2], ax=axs[0, 0], legend='full')
sns.scatterplot(data=data, x=feature_set_2[0], y=feature_set_2[1], ax=axs[0, 1])
sns.scatterplot(data=data, x=feature_set_3[0], y=feature_set_3[1], ax=axs[1, 0])
sns.barplot(data=data, x=feature_set_4[0], y=feature_set_4[1], ax=axs[1, 1])

# Setting titles for each subplot
axs[0, 0].set_title('Population Density vs. Average Age and Gross Pay')
axs[0, 1].set_title('Diabetes Prevalence vs. Carbon Emissions')
axs[1, 0].set_title('First Language Not English vs. GCSE Achievement')
axs[1, 1].set_title('Political Control vs. Public Transport Accessibility')

plt.tight_layout()
plt.show()

x = data['Male life expectancy, (2012-14)'] = pd.to_numeric(data['Male life expectancy, (2012-14)'], errors='coerce')
y = data['Female life expectancy, (2012-14)'] = pd.to_numeric(data['Female life expectancy, (2012-14)'], errors='coerce')

plt.scatter(x, y)
plt.xlabel('Male life expectancy, (2012-14)')
plt.ylabel('Female life expectancy, (2012-14)')
plt.show()