#!/usr/bin/env python
# coding: utf-8

# # **Fundamentals of Data Analysis visualization in Banking**
# 

# The purpose of this lab is to master visual data analysis in banking for machine learning models.
# 
# After completing this lab you will be able to:
# 
# 1. Visualize a banking dataset with MatplotLib, Seaborn and Plotly libraries.
# 2. Visually analyze single features and feature interaction.
# 3. Do a comprehensive visual data analysis for the source dataset.
# 

# ## Outline
# 

# * Materials and Methods
# * General Part
#   * Import Libraries
#   * Load the Dataset
#   * Overview of Python libraries for visual data analysis
#   * Visual analysis of single features
#   * Visual analysis of feature interaction
#   * Comprehensive visual analysis of the source banking dataset
# * Tasks
# * Authors
# 
# 

# ----
# 

# ## Materials and Methods
# 

# The data that we are going to use for this is a subset of an open source Bank Marketing Data Set from the UCI ML repository: https://archive.ics.uci.edu/ml/citation_policy.html.
# 
# > This dataset is public available for research. The details are described in [Moro et al., 2014].
# Please include this citation if you plan to use this database:
# [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
# 
# During the work, the task of a preliminary analysis of a positive response (term deposit) to direct calls from the bank is solved. In essence, the task is the matter of bank scoring, i.e. according to the characteristics of clients (potential clients), their behavior is predicted (loan default, a wish to open a deposit, etc.).
# 
# In this lesson, we will try to give answers to a set of questions that may be relevant when analyzing banking data:
# 
# 1. What are the most useful Python libraries for visual analysis?
# 2. How to build interactive plots?
# 3. How to visualize single features?
# 4. How to do a visual analysis for the feature interaction?
# 5. How to provide a comprehensive visual analysis for numerical and categorical features?
# 
# In addition, we will make the conclusions for the obtained results of our visual analysis to plan marketing banking campaigns more effectively.
# 

# [Matplotlib](https://matplotlib.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsVDA_Banking_L2474-2022-01-01) is a plotting library for the Python programming language and its numerical mathematics extension NumPy. Matplotlib uses an object oriented API to embed plots in Python applications.
# 
# 

# [Seaborn](https://seaborn.pydata.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsVDA_Banking_L2474-2022-01-01) is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics. Seaborn provides an API on top of Matplotlib that offers sane choices for plot style and color defaults, defines simple high-level functions for common statistical plot types, and integrates with the functionality provided by Pandas DataFrames.
# 

# [Plotly](https://plotly.com/python/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsVDA_Banking_L2474-2022-01-01) is an interactive, open-source plotting library that supports over 40 unique chart types covering a wide range of statistical, financial, geographic, scientific, and 3-dimensional use-cases. Built on top of the Plotly JavaScript library (plotly.js), plotly enables Python users to create beautiful interactive web-based visualizations that can be displayed in Jupyter notebooks and saved to standalone HTML files.
# 

# ## Import Libraries
# 

# Download data using a URL.
# 

# Import the libraries necessary to use in this lab. We can add some aliases to make the libraries easier to use in our code and set a default figure size for further plots. Ignore the warnings.
# 

# In[60]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (8, 6)

import warnings
warnings.filterwarnings('ignore')

import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)


# Further specify the value of the `precision` parameter equal to 2 to display two decimal signs (instead of 6 as default).
# 

# ## Load the Dataset
# 

# In this section you will load the source dataset.
# 

# In[61]:


df = pd.read_csv('bank-additional.csv', sep = ';')
df.head(5)


# The target feature shows a positive behavior of a phone call during the marketing campaign. Mark the positive outcome as 1 and negative one as 0.
# 

# Use the `map` function for **replacement the values ​​in column** by passing it as an argument dictionary in form of `{old_value: new_value} `. Replace ##YOUR CODE GOES HERE## with your Python code.
# 

# ### Attribute Information
# 

#  Output the column (feature) names:
# 

# In[7]:


df.columns


# <details>
# <summary><b>Click to see attribute information</b></summary>
# Input features (column names):
# 
# 1. `age` - client age in years (numeric)
# 2. `job` - type of job (categorical: `admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown`)
# 3. `marital` - marital status (categorical: `divorced`, `married`, `single`, `unknown`)
# 4. `education` - client education (categorical: `basic.4y`, `basic.6y`, `basic.9y`, `high.school`, `illiterate`, `professional.course`, `university.degree`, `unknown`)
# 5. `default` - has credit in default? (categorical: `no`, `yes`, `unknown`)
# 6. `housing` - has housing loan? (categorical: `no`, `yes`, `unknown`)
# 7. `loan` - has personal loan? (categorical: `no`, `yes`, `unknown`)
# 8. `contact` - contact communication type (categorical: `cellular`, `telephone`)
# 9. `month` - last contact month of the year (categorical: `jan`, `feb`, `mar`, ..., `nov`, `dec`) 
# 10. `day_of_week` - last contact day of the week (categorical: `mon`, `tue`, `wed`, `thu`, `fri`) 
# 11. `duration` - last contact duration, in seconds (numeric).
# 12. `campaign` - number of contacts performed for this client during this campaign (numeric, includes last contact) 
# 13. `pdays` - number of days that have passed after the client was last contacted from the previous campaign (numeric; 999 means the client has not been previously contacted) 
# 14. `previous` - number of contacts performed for this client before this campaign (numeric) 
# 15. `poutcome` - outcome of the previous marketing campaign (categorical: `failure`, `nonexistent`, `success`)
# 16. `emp.var.rate` - employment variation rate, quarterly indicator (numeric) 
# 17. `cons.price.idx` - consumer price index, monthly indicator (numeric) 
# 18. `cons.conf.idx` - consumer confidence index, monthly indicator (numeric) 
# 19. `euribor3m` - euribor 3 month rate, daily indicator (numeric) 
# 20. `nr.employed` - number of employees, quarterly indicator (numeric)
# 
# Output feature (desired target):
# 
# 21. `y` - has the client subscribed a term deposit? (binary: `yes`,`no`)
# </details>
# 

# Let's look at the dataset size.
# 

# In[8]:


df.shape


# The dataset contains 41188 objects (rows), for each of which 21 features are set (columns), including 1 target feature (y).
# 

# ## Overview of Python libraries for visual data analysis
# 

# ### Matplotlib
# 

# Let's start our overview of Python libraries for visual data analysis with the simplest and fastest way to visualize data from Pandas DataFrame - to use the functions `plot` and` hist`. The implementation of these functions in Pandas is based on the matplotlib library.
# 

# For each feature, you can build a separate histogram with `hist` function:
# 

# In[9]:


df["age"].hist()


# The histogram shows that most of our clients are between the ages of 25 and 50, which corresponds to the actively working part of the population.
# 

# We will build a graph of the average client age depending on the marital status. To begin with, we only specify the columns we need, then calc the average values ​​and for the received DataFrame call the `plot` function without parameters.
# 
# Replace ##YOUR CODE GOES HERE## with your Python code.
# 

# In[10]:


df[["age", "marital"]].groupby(
    "marital"
).mean().plot();


# Double-click **here** for the solution.
# 
# <!-- 
# df[["age", "marital"]].groupby(
#     "marital"
# ).mean().plot();
# -->
# 

# The plot shows that the average age of unmarried clients is significantly lower than that of the other clients.
# 

# With the `kind` parameter you can change the plot type, for example, to a bar chart. MATPLOTLIB allows you to configure graphics very flexibly. You can change almost anything on the chart, but you will need to look up the necessary parameters in the [documentation](https://matplotlib.org/stable/contents.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsVDA_Banking_L2474-2022-01-01). For example, the `rot` parameter is responsible for the angle of tilt signatures to the x axis.
# 

# In[11]:


df[["age", "marital"]].groupby(
    "marital"
).mean().plot(kind="bar", rot=45);


# ### Seaborn
# 

# Now let's go to the seaborn library. Seaborn is a higher-level API based on the matplotlib library. Seaborn contains more adequate default graphics settings. Also there are quite complex types of visualization in the library, which would require a large amount of code in matplotlib.
# 
# We will get acquainted with the first "complex" type of pair plot graphics (Scatter Plot Matrix). This visualization will help us to look at one picture as at interconnection of various features.
# 

# In[12]:


sns.pairplot(
    df[["age", "duration", "campaign"]]
);


# This visualization allows us to identify an interesting inverse relationship between a campaign and duration, which indicates a decrease in the duration of contact with the client with an increase in their contact quantity during the campaign.
# 

# Also with the help of `seaborn` you can build a distribution, for example, look at the distribution of the client age. To do this, build `distplot`. By default, the graph shows a histogram and [Kernel Density Estimation](https://en.wikipedia.org/wiki/kernel_density_estimation?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsVDA_Banking_L2474-2022-01-01).
# 

# In[13]:


sns.distplot(df.age);


# In order to look more for the relationship between two numerical features, there is also `joint_plot` - this is a hybrid Scatter Plot and Histogram (there are also histograms of feature distributions). Let's look at the relationship between the number of contacts in a campaign and the last contact duration.
# 

# In[14]:


sns.jointplot(x="age", y="duration", data=df, kind="scatter")


# Another useful seaborn plot type is [**Box Plot** ("Box and whisker plot")](https://en.wikipedia.org/wiki/box_plot?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsVDA_Banking_L2474-2022-01-01). Let's compare the age of customers for the top 5 of the most common employment forms.
# 

# In[15]:


top_jobs = (
    df.job.value_counts().sort_values(ascending=False).head(5).index.values
)
sns.boxplot(
    y="job", x="age", data=df[df.job.isin(top_jobs)], orient="h"
)


# The plot shows that among the top-5 client categories by the type of employment, the most senior customers represent the management, and the largest number of outliers is among the categories of admin. and technician.
# 

# And one more plot type (the last of those we consider in this chapter) is a `heat map`. A Heat Map allows you to look at the distribution of some numerical feature in two categories. We visualize the distribution of clients on family status and the type of employment.
# 

# The plot shows that the largest number of attracted clients among administrative workers is married (652), and there is the smallest number of attracted clients among customers with an unknown family status.
# 

# ### Plotly
# 

# We looked at the visualization based on the Library `Matplotlib` and `Seaborn`. However, this is not the only option to build charts with `Python`. We will also get acquainted with the library `plotly`. `Plotly` is an open-source library that allows you to build interactive graphics in a jupyter notebook without having to break into JavaScript code.
# 
# The beauty of interactive graphs is that you can see the exact numerical value on mouse hover, hide the uninteresting rows in the visualization, zoom in a certain area of ​​graphics, etc.
# 

# To begin with, we build __Line Plot__ with the distribution of the total number and the number of attracted clients by age.
# 

# In[19]:


age_df = (
    df.groupby("age")[["y"]]
    .sum()
    .join(df.groupby("age")[["y"]].count(), rsuffix='_count')
)
age_df.columns = ["Attracted", "Total Number"]


# In `Plotly`, we create the `Figure` object, which consists of data (list of lines that are called `traces`) and design/style, for which the object `Layout` was created. In simple cases, you can call the function `iplot` just for the `traces` list.
# 

# In[20]:


trace0 = go.Scatter(x=age_df.index, y=age_df["Attracted"], name="Attracted")
trace1 = go.Scatter(x=age_df.index, y=age_df["Total Number"], name="Total Number")

data = [trace0, trace1]
layout = {"title": "Statistics by client age"}

fig = go.Figure(data=data, layout=layout)

iplot(fig, show_link=False)


# Let us also see the distribution of customers by months, designed by the number of attracted clients and on the total number of clients. To do this, build __Bar Chart__.
# 

# In[21]:


month_index = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
month_df = (
    df.groupby("month")[["y"]]
    .sum()
    .join(df.groupby("month")[["y"]].count(), rsuffix='_count')
).reindex(month_index)
month_df.columns = ["Attracted", "Total Number"]


# In[22]:


trace0 = go.Bar(x=month_df.index, y=month_df["Attracted"], name="Attracted")
trace1 = go.Bar(x=month_df.index, y=month_df["Total Number"], name="Total Number")

data = [trace0, trace1]
layout = {"title": "Share of months"}

fig = go.Figure(data=data, layout=layout)

iplot(fig, show_link=False)


# `plotly` can build the __Box plot__. Consider the differences in the client age depending on the family status.
# 

# In[23]:


data = []

for status in df.marital.unique():
    data.append(go.Box(y=df[df.marital == status].age, name=status))
iplot(data, show_link=False)


# The plot clearly shows the distribution of clients by age, the presence of outliers for all categories of the family status, except for `unknown`. Moreover, the plot is interactive - hovering the mouse pointer to its elements allows you to obtain additional statistical characteristics of the series. Discover the characteristics.
# 

# ## Visual analysis of single features
# 

# Let us give the most commonly used plot types to analyze single features of data sets.
# 

# #### Numerical features
# 

# For the analysis of numerical features, a histogram and a box plot are most often used.
# 

# In[24]:


df["age"].hist();


# Build a box plot for the `cons.price.idx` feature with `sns.boxplot` function.
# 
# Replace ##YOUR CODE GOES HERE## with your Python code.
# 

# In[25]:


sns.boxplot(df["cons.price.idx"])


# Double-click **here** for the solution.
# 
# <!-- 
# sns.boxplot(df["cons.price.idx"])
# -->
# 

# ### Categorical features
# 

# Use the `countplot` graphics for effective analysis of categorical features.
# It's effective to use the graphics of the type `CountPlot` for analyzing categorical features.
# 
# Calculate the client distribution of marital status.
# 

# In[26]:


df["marital"].value_counts().head()


# Let's calculate the client distribution on the fact of their involvement for signing a deposit as well.
# 

# In[27]:


df["y"].value_counts()


# Present this information graphically.
# 

# In[28]:


sns.countplot(df["y"]);


# Build the count plot for the `marital` feature with `sns.countplot` function.
# 
# Replace ##YOUR CODE GOES HERE## with your Python code.
# 

# In[31]:


# Ensure that the 'marital' column is treated as a categorical variable
df['marital'] = df['marital'].astype('category')

# Create the countplot
sns.countplot(x='marital', data=df)


# Plot the graphical client distribution by the 5 most common types of employment.
# 

# In[33]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Ensure that the 'job' column is treated as a categorical variable
df['job'] = df['job'].astype('category')

# Create the countplot
plot = sns.countplot(x='job', data=df[df["job"].isin(df["job"].value_counts().head(5).index)])
plt.setp(plot.get_xticklabels(), rotation=90)

# Show the plot
plt.show()


# ## Visual analysis of the feature interaction
# 

# ### Numerical features
# 

# To analyze the interaction of numerical features, use `hist` (histogram), `pairplot` and `heatmap` plot functions.
# 

# We visualize the values ​​of the economy macro indicators from the dataset.
# 

# In[34]:


feat = ["cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

df[feat].hist()


# Build a pair plot set for the `feat` list with `sns.pairplot` function.
# 
# Replace ##YOUR CODE GOES HERE## with your Python code.
# 

# In[35]:


sns.pairplot(df[feat])


# Build a Heat Map for the economy macro indicators [correlation matrix](https://en.wikipedia.org/wiki/Correlation_and_dependence?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsVDA_Banking_L2474-2022-01-01#Correlation_matrices).
# 

# In[36]:


sns.heatmap(df[feat].corr());


# We see a strong interaction between the `euribor3m` and `nr.employed` features.
# 

# ### Numerical and categorical features
# 

# `boxplot` and `violinplot` are used for visual analysis of the numerical and categorical features.
# 
# Let's look at the `age` feature box plot by the target feature.
# 

# In[38]:


# Ensure that the 'job' column is treated as a categorical variable
df['job'] = df['job'].astype('category')

# Create the countplot
plot = sns.countplot(x='job', data=df[df["job"].isin(df["job"].value_counts().head(5).index)])
plt.setp(plot.get_xticklabels(), rotation=90)

# Show the plot
plt.show()


# Build the box plot for the `marital` feature with `sns.boxplot` function.
# 
# Replace ##YOUR CODE GOES HERE## with your Python code.
# 

# In[39]:


sns.boxplot(x="marital", y="age", data=df)


# You can draw a combination of boxplot and kernel density estimate with a `violinplot` function. A [violin plot](https://en.wikipedia.org/wiki/Violin_plot?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkGuidedProjectsVDA_Banking_L2474-2022-01-01) plays a similar role as a box and whisker plot. It shows the distribution of quantitative data across several levels of one (or more) categorical variables such that those distributions can be compared. 
# 
# Plot the client age distribution across the target feature.
# 

# In[41]:


# Ensure that the 'job' column is treated as a categorical variable
df['job'] = df['job'].astype('category')

# Create the countplot
plot = sns.countplot(x='job', data=df[df["job"].isin(df["job"].value_counts().head(5).index)])
plt.setp(plot.get_xticklabels(), rotation=90)

# Show the plot
plt.show()


# It is useful to combine grouping with a `boxplot`. Calculate the mean client for the grouping by the `housing` feature values.
# 

# In[42]:


df.groupby("housing")["age"].mean()


# Build a box plot for the `age` feature by the `housing` values with `sns.boxplot` function.
# 
# Replace ##YOUR CODE GOES HERE## with your Python code.
# 

# In[43]:


sns.boxplot(x="housing", y="age", data=df)


# ### Categorical features
# 

# Use `countplot` for a visual interaction analysis between categorical features.
# 
# Calculate and visualize the interaction between target and client marital status features.
# 

# In[44]:


pd.crosstab(df["y"], df["marital"])


# Build the count plot for the `month` feature by the `y` feature target values with `sns.countplot` function.
# 
# Replace ##YOUR CODE GOES HERE## with your Python code.
# 

# Double-click **here** for the solution.
# 
# <!-- 
# sns.countplot(x="month", hue="y", data=df)
# -->
# 

# ## Comprehensive visual analysis of the source banking dataset
# 

# Create the `categorical` and `numerical` lists for the correspondent dataset features.
# 
# Let's look at the distribution of numerical features with `hist` function.
# 

# In[49]:


categorical = []
numerical = []
for feature in df.columns:
    if df[feature].dtype == object:
        categorical.append(feature)
    else:
        numerical.append(feature)

df[numerical].hist(figsize=(20,12), bins=100, color='lightgreen')


# From the histograms, we see that **for each numerical feature there is one or more dominant segments of values​**, that is why we got pronounced peaks.
# 
# In addition, we see that the target feature is unbalanced. **The number of positive outcomes is significantly lower than negative**, which is quite natural for telephone marketing. As a result, the problem arises with the fact that many methods are sensitive to the imbalance of features. We will try to solve this problem later.
# 
# Next, let's look at the categorical features.
# 

# In[50]:


df.describe(include = ['object'])


# Visualize the categorical features with bar plots.
# 

# As we see, for many features, some of the groups stand out, for example, in the dataset more than half of the clients are married.
# 

# Let's look at the correlation matrix (for the numerical features).
# 

# In[52]:


correlation_table = df.corr()
correlation_table


# We visualize the correlation matrix.
# 

# In[53]:


sns.heatmap(correlation_table)


# Let's look at the visualized dependences of numerical features from the target feature with scatter plots.
# 

# In[55]:


import matplotlib.pyplot as plt
import numpy as np

numerical = ['age', 'duration', 'campaign']

# Create a subplot grid with specified dimensions
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(24, 18))
plt.subplots_adjust(wspace=None, hspace=0.4)

# Iterate through numerical features and create scatter plots
for i in range(len(numerical)):
    row = i // 4
    col = i % 4
    df.plot(x=numerical[i], y='y', label=numerical[i], ax=axes[row, col], kind='scatter', color='green')
    axes[row, col].set_title(numerical[i])

# Adjust layout and show the plots
plt.tight_layout()
plt.show()


# As you can see, there are points that can be interpreted as outliers, however, we will not hurry to delete them because they don't seem to be true outliers. These points are too strong so we will leave them. In addition, we will use some models that are resistant to outliers.
# 

# We visualize **the distribution of positive target responses** by groups:
# 

# In such a form, plots are already more interesting. So we see, for many features, the chance of a positive response is significantly higher.  
# 
# We also see that `housing`, `loan` and `day_of_week` features will hardly help us, because judging by the plots, the share of positive target responses hardly depends on them. 
# 

# ### Conclusions
# 

# There are neither any data missing, nor explicit outliers that should be cut. But we can omit `housing`, `loan` and `day_of_week` features in the next steps. 
#    
# The `euribor3m` and `nr.employed` features strongly correlate with `emp.var.rate`. Let me remind you that `emp.var.rate` - Employment Variation Rate is a quarterly indicator, `euribor3m` - euribor 3 month rate is a day indicator, and `nr.employed` - number of employees is a quarterly indicator. The correlation of the employment change with the number of employed issues itself is obvious, but its correlation with EURIBOR (Euro Interbank Offered Rate, the European interbank offer rate) is interesting. This indicator is based on the average interbank interest rates in Eurozone. It also has a positive effect since the higher the interest rate is, the more willingly customers will spend their money on financial tools.
# 
# Therefore, if banks want to improve their lead generation, what they should do is to improve the quality of phone conversations and run their campaigns when interest rates are high and the macroeconomic environment is stable. 
# 

# ## Tasks
# 

# In this section you will solve an additional task with the source bank dataset.
# 

# Let's compare the age of customers for the top 3 of the most common levels of education using a box plot.
# 

# |  Date (YYYY-MM-DD) |  Version | Changed By  |  Change Description |
# |---|---|---|---|
# | 2021-06-12  | 0.2  | Yatsenko, Roman  |  Translate to english |
# | 2021-06-04  | 0.1  | Yatsenko, Roman  |  Created Lab |
# 
