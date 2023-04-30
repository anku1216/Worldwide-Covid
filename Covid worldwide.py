#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pyforest


# In[2]:


import pandas as pd
from pyforest import *
active_imports


# In[4]:


df = pd.read_csv('covid_worldwide.csv')


# In[5]:


df.head()


# In[6]:


df.tail()


# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


df


# In[11]:


df.describe()


# In[12]:


df.isna().any()


# In[13]:


df.isna().sum()


# In[14]:


df.dropna(axis=0, inplace = True)


# In[15]:


df.isna().sum()


# In[17]:


df['Total Cases'] = df['Total Cases'].astype(str).str.replace(',', '').astype(int)
df['Total Deaths'] = df['Total Deaths'].astype(str).str.replace(',', '').astype(int)
df['Total Recovered'] = df['Total Recovered'].astype(str).str.replace(',', '').astype(int)
df['Active Cases'] = df['Active Cases'].astype(str).str.replace(',', '').astype(int)
df['Total Test'] = df['Total Test'].astype(str).str.replace(',', '').astype(int)
df['Population'] = df['Population'].astype(str).str.replace(',', '').astype(int)


# In[18]:


df.info()


# In[19]:


df.describe()


# In[20]:


df.columns


# In[21]:


plt.figure(figsize=(10,6))
plt.plot(df['Country'].head(10), df['Total Cases'].head(10), label='Total Cases')
plt.plot(df['Country'].head(10), df['Total Deaths'].head(10), label='Total Deaths')
plt.plot(df['Country'].head(10), df['Total Recovered'].head(10), label='Total Recovered')
plt.title('COVID-19 Cases, Deaths, and Recoveries')
plt.xlabel('Country')
plt.ylabel('Number of Cases')
plt.legend()
plt.show()


# In[22]:


plt.figure(figsize=(10,6))
plt.bar(df['Country'].head(20), df['Total Cases'].head(20), label='Total Cases')
plt.bar(df['Country'].head(20), df['Total Deaths'].head(20), label='Total Deaths')
plt.bar(df['Country'].head(20), df['Total Recovered'].head(20), label='Total Recovered')
plt.title('COVID-19 Cases, Deaths, and Recoveries by Country')
plt.xlabel('Country')
plt.ylabel('Number of Cases')
plt.xticks(rotation=90)
plt.legend()
plt.show()


# In[24]:


pip install geopandas


# In[25]:


import geopandas as gpd;


# In[26]:


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Merge data with map
merged = world.merge(df, left_on='name', right_on='Country')

# Create map
fig, ax = plt.subplots(figsize=(50, 20))
ax.set_title('COVID-19 Cases, Deaths, and Recoveries by Country')
merged.plot(column='Total Cases', cmap='OrRd', legend=True, ax=ax)
plt.show()


# Stacked bar chart of total cases and deaths by country

# In[27]:


by_country = df.groupby('Country')[['Total Cases', 'Total Deaths']].sum().head(30)

# Create stacked bar chart
by_country.plot(kind='bar', stacked=True)

# Set axis labels and title
plt.xlabel('Country')
plt.ylabel('Number of Cases')
plt.title('Total Cases and Deaths by Country')

plt.show()


# In[28]:


plt.scatter(df['Population'], df['Total Cases'])

# Set axis labels and title
plt.xlabel('Population')
plt.ylabel('Total Cases')
plt.title('Total Cases vs Population')

plt.show()


# In[29]:


by_country = df.groupby('Country')['Active Cases'].sum().head(4)

# Create pie chart
plt.pie(by_country.values, labels=by_country.index)

# Set title
plt.title('Active Cases by Country')

plt.show()


# In[30]:


by_country = df.groupby('Country')['Active Cases'].sum().tail(4)

# Create pie chart
plt.pie(by_country.values, labels=by_country.index)

# Set title
plt.title('Active Cases by Country')

plt.show()


# In[31]:


by_country_month = df.pivot_table(index='Country', columns='Population', values='Total Cases').head(5)

# Create heatmap
sns.heatmap(by_country_month, cmap='Blues')

# Set axis labels and title
plt.xlabel('Population')
plt.ylabel('Country')
plt.title('Total Cases by Country and Population')

plt.show()


# In[32]:


sns.boxplot(x='Country', y='Total Cases', data=df.head(10))

# Set axis labels and title
plt.xlabel('Country')
plt.ylabel('Total Cases')
plt.title('Total Cases by Continent')

plt.show()


# In[33]:


df['Cases per Million'] = df['Total Cases'] / (df['Population'] / 1e6)
df['Deaths per Million'] = df['Total Deaths'] / (df['Population'] / 1e6)

# Group by country and get the maximum value of cases and deaths per million
by_country = df.groupby('Country')[['Cases per Million', 'Deaths per Million']].max().head(20)

# Create bar chart
by_country.plot(kind='bar')

# Set axis labels and title
plt.xlabel('Country')
plt.ylabel('Cases/Deaths per Million People')
plt.title('Total Cases and Deaths per Million People by Country')

plt.show()


# In[34]:


plt.figure(figsize=(12, 6))
sns.histplot(data=df, x="Total Cases", bins=30, kde=True)
plt.title("Distribution of Total Cases by Country")
plt.xlabel("Total Cases")
plt.ylabel("Count")
plt.show()


# In[36]:


plt.figure(figsize=(12, 6))
sns.histplot(data=df, x="Total Recovered", bins=30, kde=True)
plt.title("Distribution of Total Recovered by Country")
plt.xlabel("Total Recovered")
plt.ylabel("Count")
plt.show()


# In[37]:


plt.figure(figsize=(12, 6))
sns.histplot(data=df, x="Active Cases", bins=30, kde=True)
plt.title("Distribution of Active Cases by Country")
plt.xlabel("Active Cases")
plt.ylabel("Count")
plt.show()


# In[38]:


plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x="Total Cases", y="Total Deaths")
plt.title("Total Cases vs. Total Deaths by Country")
plt.xlabel("Total Cases")
plt.ylabel("Total Deaths")
plt.show()


# In[39]:


plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x="Total Cases", y="Total Recovered")
plt.title("Total Cases vs. Total Recovered by Country")
plt.xlabel("Total Cases")
plt.ylabel("Total Recovered")
plt.show()


# In[40]:


plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x="Total Cases", y="Active Cases")
plt.title("Total Cases vs. Active Cases by Country")
plt.xlabel("Total Cases")
plt.ylabel("Active Cases")
plt.show()


# In[ ]:





# In[ ]:




