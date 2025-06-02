import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from data_loading import load_amazon_reviews

def plot_rating_distribution(df):
    """Plot the distribution of ratings"""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='rating')
    plt.title('Distribution of Ratings')
    plt.savefig('../outputs/rating_distribution.png')
    plt.close()

def plot_review_length_distribution(df):
    """Plot the distribution of review lengths"""
    df['review_length'] = df['review_text'].str.len()
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='review_length', bins=50)
    plt.title('Distribution of Review Lengths')
    plt.xlabel('Review Length (characters)')
    plt.savefig('../outputs/review_length_distribution.png')
    plt.close()

def create_wordcloud(df):
    """Create and save wordcloud from review texts"""
    text = ' '.join(df['review_text'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Reviews')
    plt.savefig('../outputs/wordcloud.png')
    plt.close()

def perform_eda():
    """Perform exploratory data analysis"""
    df = load_amazon_reviews()
    if df is None:
        return
    
    # Basic statistics
    print("Dataset Shape:", df.shape)
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Create visualizations
    plot_rating_distribution(df)
    plot_review_length_distribution(df)
    create_wordcloud(df)
    
    # Save basic statistics to file
    with open('../outputs/basic_statistics.txt', 'w') as f:
        f.write("Dataset Statistics\n")
        f.write("="*20 + "\n")
        f.write(f"Total number of reviews: {len(df)}\n")
        f.write(f"Average rating: {df['rating'].mean():.2f}\n")
        f.write(f"Median rating: {df['rating'].median()}\n")
        
if __name__ == "__main__":
    perform_eda() 