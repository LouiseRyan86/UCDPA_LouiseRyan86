### Section 1 - Real world scenario Data Source - Amazon Top 50 Bestselling Books 2009 - 2019
### Importing all basic libraries needed for this project
import pandas as pd  # linear algebra
import numpy as np  # data processing
import matplotlib.pyplot as plt  # library to visualise graphs
import seaborn as sns
import difflib
import plotly.express as px

### Section 2 - Loading the dataset from Kaggle
# Loading the data from csv into Pandas library and making dataframe called book_data
book_data = pd.read_csv('bestsellers_with_categories.csv')

### Section 3 - Analysing data
# Sort by Year and User Rating
book_data_sort = book_data.sort_values(["Year", "User Rating"], ascending=True, inplace=True)
# View snapshot of first five rows using head(). this is to see the data is correct without viewing all the rows
print(book_data.head())
# Using shape() attribute to study data
print(book_data.shape)
# Using info() to get summary
print(book_data.info())
# Check for correct format
print(book_data.columns)
# Getting a summary about the data with all the basic summary statistics using the describe() method:
print(book_data.describe())
print(book_data.describe(include='all')) # This shows more detail

### Sanitising the data
print(book_data.isnull().any())
print(book_data.isnull().sum())

diff =[]
for c in ['Name','Author'] :
    for v in book_data[c].unique():
        for vv in book_data[c].unique():
            if v != vv and {v,vv} not in diff and len(difflib.get_close_matches(v,[vv],cutoff=0.9))!=0  :
                diff.append({v,vv})
print(diff)


### Section 4 - Python, regex,numpy,dictionary
# Using regex search in pandas to get fiction and non fiction
genre_sort = book_data.loc[book_data['Genre'].str.contains("Fiction", case=False)]
genre = genre_sort['Genre'].value_counts(dropna=False)
print("Splitting up Genre using regex word search in Pandas")
print(genre)

# Showing 2014 trents
data_2014 = book_data.loc[(book_data['Year']==2014)&(book_data['Genre']=='Fiction')]
print(data_2014)

# Showing 2015 trents
data_2015 = book_data.loc[(book_data['Year']==2015)&(book_data['Genre']=='Non Fiction')]
print(data_2015)

# Drop 'Year' column from file dataframe, then add to numpy array and print
panda = book_data.drop(columns='Year')
panda.drop_duplicates(subset='Name', inplace=True)
np_array = panda.to_numpy()
print(np_array)
# convert to dictionary
dict1 = dict(enumerate(np_array.flatten(), 1))
print("This is dict1",dict1)

# using function and iteration to get number of books with highest marks
new_dataset = book_data[['User Rating']].copy()
np_list = new_dataset.to_numpy().flatten()

## Itterate and print out totals of each user rating
data = {}
for i in range(len(np_list) - 1):
    x = np_list[i]
    c = 0
    for j in range(i, len(np_list)):
        if np_list[j] == np_list[i]:
            c = c + 1
    count = dict({x: c})
    if x not in data.keys():
        data.update(count)
        sorted_data = sorted(data.items(), key=lambda kv: kv[0])
print("Showing user rating and count of user rating (Rating, Count)", sorted_data)

### Section 6 and 7 - visualise graphs and generate valuable insights
## Plotting graphs
# Non Fiction vs Fiction over the years
####### Among bestselling books, from 2009 to 2019,
####### Non Fiction books are usually more than Fictions, except 2014
sns.countplot(data=book_data, x='Year', hue='Genre')
plt.show()

# Price boxplot
####### Most bestselling book's price range from 5 to 20, and didn't change much over the years.
sns.catplot(kind='box', data=book_data, x='Year', y='Price')
plt.show()

# Count each book's winning times as "Amazon bestseller".
# plot the result as histogram
####### Among 351 books, about 250 won "Amazon bestseller" once, about 50 won twice.
winning_time = book_data['Name'].value_counts()
winning_time.hist()
plt.grid(False)
plt.show()

# Show books that appeared more than once as "amazon bestseller book".
# Create a dataframe that show book name and winning_times.
winning_times = pd.DataFrame(winning_time)
winning_times.reset_index(inplace=True)
winning_times.columns = ['Name', 'Winning_times']
print(winning_times)
winning_times[winning_times['Winning_times'] > 1].plot(kind='bar', x='Name', y='Winning_times', figsize=(20, 10))
plt.show()

# Drop 'Year' column from file dataframe, then drop duplicated books.
file_ny = book_data.drop(columns='Year')
file_ny.drop_duplicates(subset='Name', inplace=True)
print(file_ny)
print(file_ny.shape)
plt.show()

# Join file and multiwinner dataframe
file_times = winning_times.join(file_ny.set_index('Name'), on='Name', how='left')
print(file_times)

### Now we get "Amazon bestseller book" information about their winning times.
# Count how many books won this prize more than once.
multi_winner = len(file_times[file_times['Winning_times'] > 1])
print("{} books won 'Amazon bestseller books' prize more than once.".format(multi_winner))

### Find authors who has more than one book won "Amazon bestseller book" prize.
author_counts = file_times['Author'].value_counts()
print("{} authors wrote 'Amazon bestseller books'.\n".format(len(author_counts)))
print(author_counts)
author_counts.plot(kind='bar', figsize=(40, 10))
plt.show()
print("Among {} authors, {} authors has more than one book won 'Amazon bestseller books'.".format(len(author_counts),
                                                                                                  len(author_counts[
                                                                                                          author_counts > 1])))

author_list = author_counts[author_counts > 1].index
file_author = file_times[file_times['Author'].isin(author_list)].sort_values(by='Author')
file_author.reset_index(drop=True)
print(file_author)

scatter = px.scatter(book_data,x='Reviews',y='User Rating',color='Genre',color_discrete_map={'Fiction':'#1f77b4','Non Fiction':'#ff7f0e'})
scatter.update_layout(template='simple_white')
scatter.show()