Chicken Text Analyzer

This program filters through the text of the "Chicken" Wikipedia page. It repeats this process to the first 20 other pages that are linked within the article then uses word frequency and similarity analysis to determine the similarity
between the original article and the sub-articles that were linked from the original. CSV files and a heatmap are created to demonstrate the similarities between these files visually.

Text-Analyzer.py is the main python script that performs the text processing, word frequency analysis, and creates the visualizations of the data. 

data.csv contains the text processed through in each of the 20 linked pages. 

similarity_heatmap.png is a heat map created by the python script that demonstrates the individual similarities between each of the articles. 

top5Words.csv contains the top 5 most frequently used words in each of the articles to be used for comparison. 
