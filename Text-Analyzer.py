#import appropriate libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

#collect the text from a given url connected to a web page
def getPageText(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; student-assignment/1.0)'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        included = soup.find('div', {'id': 'mw-content-text'})
        if included:
            for infobox in included.find_all('table', {'class': 'infobox'}):
                infobox.decompose()
            for navbox in included.find_all('table', {'class': 'navbox'}):
                navbox.decompose()
            paragraphs = included.find_all('p')
        else:
            return ""
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        print(f"Error occurred while fetching page text from {url}: {e}")
        return ""

#collect the first 20 unique links from the web page 
def getLinks(url, limit=20):
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; student-assignment/1.0)'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    included = soup.find('div', {'id': 'mw-content-text'})
    if not included:
        return []
    for infobox in included.find_all('table', {'class': 'infobox'}):
        infobox.decompose()
    for ref in included.find_all('span', {'class': 'reflist'}):
        ref.decompose()
    for refer in included.find_all('div', {'class': 'references'}):
        refer.decompose()
    links = included.find_all('a', href=True)
    valid_links = []

    for link in links:
        href = link['href']

        if href.startswith('/wiki/') and ":" not in href and "#" not in href:
            full_url = 'https://en.wikipedia.org' + href
            
            if full_url not in valid_links:
                valid_links.append(full_url)

        if len(valid_links) == limit:
            break
    return valid_links

#calls the getLinks and getPageText functions to gather all the text to process
def scrape():
    urlStart = "https://en.wikipedia.org/wiki/Chicken"
    urls = []
    urls.append({"id": 0, "url": urlStart, "text": getPageText(urlStart)})
    links = getLinks(urlStart)
    for i, link in enumerate(links):
        urls.append({"id": i+1, "url": link, "text": getPageText(link)})
    return urls

#preprocess the text before use
def preprocess(text):

    # Convert to lowercase
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words 
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

#performs the preprocess function on each of the web pages
def analyze(urls):
    cleaned = []
    for url in urls:
        cleaned.append({"id": url["id"], "url": url["url"], "tokens": preprocess(url["text"])})
    return cleaned

#makes a dataframe of the information from each of the web pages
def makeDataset(urls, cleaned):
    data = []
    for url, clean in zip(urls, cleaned):
        data.append({"id": url["id"], "url": url["url"], "raw_text": url["text"], "new_text": ' '.join(clean["tokens"])})
    df = pd.DataFrame(data)
    df.to_csv('data.csv', index=False)
    return df

#determines the top 5 most commonly used words from each of the web pages and creates a csv file containing them
def topWords(cleaned):
    rows = []
    for doc in cleaned:
        word_counts = Counter(doc["tokens"])
        top_5 = word_counts.most_common(5)
        row = [doc["id"]]
        for word, count in top_5:
            row.append(f"{word} ({count})")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv('top5Words.csv',index=False, header=False)

#calculates the document similarity between each of the documents
def similarity(cleaned):
    documents = [' '.join(doc["tokens"]) for doc in cleaned if doc["tokens"]]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

#creates a heatmap to visualize the document similarity
def visualize(similarity_matrix,labels):
    plt.figure(figsize=(15,15))
    sns.heatmap(similarity_matrix, xticklabels=labels, yticklabels=labels, cmap='coolwarm',annot=False)
    plt.title('Heatmap of Document Similarity')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.savefig('similarity_heatmap.png')
    plt.close()

#main method including function calls to run the program
def main():
    urls = scrape()
    for u in urls:
        print(u["id"], u["url"], len(u["text"]))
    cleaned = analyze(urls)
    df = makeDataset(urls, cleaned)
    topWords(cleaned)
    similarity_matrix = similarity(cleaned)
    labels = [u["url"].split("/wiki/")[-1].replace("_", " ") for u, c in zip(urls, cleaned) if c["tokens"]]
    visualize(similarity_matrix, labels)
if __name__ == "__main__":
    main()
