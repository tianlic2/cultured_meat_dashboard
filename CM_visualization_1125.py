import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import re
import ipywidgets as widgets
from IPython.display import display, clear_output

# Load data
def load_data():
    # Google Drive file ID
    file_id = "1Dt6rTzM24OYxxujeSDW5pBS-H5lgAPXZ"
    
    # Build download link
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Read data
    df = pd.read_csv(url)
    df['Day'] = pd.to_datetime(df['Day'])
    
    # Add sentiment mapping
    sentiment_mapping = {
        1: 'Positive',
        2: 'Negative', 
        3: 'Neutral',
        4: 'Combination',
        1.0: 'Positive',
        2.0: 'Negative',
        3.0: 'Neutral',
        4.0: 'Combination'
    }
    
    # Convert numbers to sentiment labels
    df['Sentiment'] = df['Sentiment'].map(sentiment_mapping)
    
    return df

def create_overview(df):
    # Create widgets
    metric_widget = widgets.RadioButtons(
        options=['Count', 'Percentage'],
        description='Select Metric:'
    )
    
    date_range_widget = widgets.DateRangeSlider(
        value=(df['Day'].min(), df['Day'].max()),
        min=df['Day'].min(),
        max=df['Day'].max(),
        description='Date Range:'
    )
    
    df_valid = df.dropna(subset=['Country']).copy()
    country_counts = df_valid['Country'].value_counts()
    countries = ['All'] + [f"{country} ({count:,} tweets)" 
                         for country, count in country_counts.items()]
    
    country_widget = widgets.SelectMultiple(
        options=countries,
        value=['All'],
        description='Countries:'
    )
    
    sentiments = ['Positive', 'Negative', 'Neutral', 'Combination']
    sentiment_widget = widgets.SelectMultiple(
        options=sentiments,
        value=sentiments,
        description='Sentiments:'
    )
    
    def update_plot(metric, date_range, countries, sentiments):
        # Data processing
        df_filtered = df.copy()
        
        # Date filtering
        mask = (df_filtered['Day'].dt.date >= date_range[0]) & \
               (df_filtered['Day'].dt.date <= date_range[1])
        df_filtered = df_filtered[mask]
        
        # Country filtering
        if 'All' not in countries:
            country_values = [c.split(' (')[0] for c in countries]
            df_filtered = df_filtered[df_filtered['Country'].isin(country_values)]
        
        # Sentiment filtering
        df_filtered = df_filtered[df_filtered['Sentiment'].isin(sentiments)]
        
        # Create chart
        monthly_sentiment = df_filtered.groupby([pd.Grouper(key='Day', freq='M'), 'Sentiment']).size().unstack(fill_value=0)
        
        if metric == 'Percentage':
            monthly_sentiment = monthly_sentiment.div(monthly_sentiment.sum(axis=1), axis=0) * 100
        
        # Color scheme
        colors = {
            'Positive': '#ef5675',
            'Negative': '#7a5195',
            'Neutral': '#ffa600',
            'Combination': '#003f5c'
        }
        
        # Create chart
        fig = go.Figure()
        
        for sentiment in sentiments:
            if sentiment in monthly_sentiment.columns:
                fig.add_trace(
                    go.Scatter(
                        x=monthly_sentiment.index,
                        y=monthly_sentiment[sentiment],
                        name=sentiment,
                        line=dict(color=colors[sentiment]),
                        mode='lines+markers'
                    )
                )
        
        # Update layout
        title_text = "Sentiment Trends Over Time"
        if 'All' not in countries:
            title_text += f" ({', '.join(country_values)})"
            
        fig.update_layout(
            title=title_text,
            xaxis_title="Date",
            yaxis_title="Count" if metric == 'Count' else "Percentage (%)",
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        fig.show()
        
        # Display statistics
        print("\nData Statistics")
        total_tweets = len(df_filtered)
        print(f"Total Tweets: {total_tweets:,}")
        
        sentiment_stats = df_filtered['Sentiment'].value_counts()
        sentiment_percentages = (sentiment_stats / len(df_filtered) * 100).round(2)
        
        stats_df = pd.DataFrame({
            'Count': sentiment_stats,
            'Percentage (%)': sentiment_percentages
        })
        display(stats_df)
    
    # Create interactive controls
    widgets.interactive(
        update_plot,
        metric=metric_widget,
        date_range=date_range_widget,
        countries=country_widget,
        sentiments=sentiment_widget
    )

def create_event_analysis(df):
    # Define important events
    events = {
        'Event 1: First Cultured Beef Burger (2013-08-05)': '2013-08-05',
        'Event 2: Singapore Approval (2020-12-02)': '2020-12-02', 
        'Event 3: FDA Clearance (2022-11-16)': '2022-11-16'
    }
    
    # Create widgets
    event_widget = widgets.Dropdown(
        options=list(events.keys()),
        description='Event:'
    )
    
    df_valid = df.dropna(subset=['Country']).copy()
    country_counts = df_valid['Country'].value_counts()
    countries = ['All'] + [f"{country} ({count:,} tweets)" 
                         for country, count in country_counts.items()]
    
    country_widget = widgets.SelectMultiple(
        options=countries,
        value=['All'],
        description='Countries:'
    )
    
    def update_plot(event, countries):
        # Data processing
        event_dt = pd.to_datetime(events[event])
        before_start = event_dt - pd.Timedelta('31D')
        after_end = event_dt + pd.Timedelta('31D')
        
        df_filtered = df_valid.copy()
        
        if 'All' not in countries:
            country_values = [c.split(' (')[0] for c in countries]
            df_filtered = df_filtered[df_filtered['Country'].isin(country_values)]
        
        # Create period labels
        df_filtered['Period'] = None
        df_filtered.loc[(df_filtered['Day'] >= before_start) & 
                       (df_filtered['Day'] < event_dt), 'Period'] = 'Before Event (31 days)'
        df_filtered.loc[(df_filtered['Day'] >= event_dt) & 
                       (df_filtered['Day'] <= after_end), 'Period'] = 'After Event (31 days)'
        
        df_filtered = df_filtered[df_filtered['Period'].notna()]
        
        if len(df_filtered) == 0:
            print("No data available for the selected period and region(s)")
            return
        
        # Create subplots
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'bar'}, {'type':'pie'}]])
        
        # Calculate sentiment distribution
        period_sentiment = df_filtered.groupby(['Period', 'Sentiment']).size().unstack(fill_value=0)
        period_percentages = period_sentiment.div(period_sentiment.sum(axis=1), axis=0) * 100
        
        colors = {
            'Positive': '#ef5675',
            'Negative': '#7a5195',
            'Neutral': '#ffa600',
            'Combination': '#003f5c'
        }
        
        # Add bar chart
        for sentiment in df_filtered['Sentiment'].unique():
            fig.add_trace(
                go.Bar(
                    name=sentiment,
                    x=['Before Event (31 days)', 'After Event (31 days)'],
                    y=period_percentages[sentiment],
                    marker_color=colors[sentiment],
                    text=[f'{period_percentages[sentiment][period]:.1f}%<br>({period_sentiment[sentiment][period]:,})' 
                          for period in period_percentages.index],
                    textposition='auto',
                ),
                row=1, col=1
            )
        
        # Add pie chart
        total_sentiment = df_filtered['Sentiment'].value_counts()
        total_percentages = (total_sentiment / len(df_filtered) * 100).round(1)
        
        fig.add_trace(
            go.Pie(
                labels=total_percentages.index,
                values=total_percentages.values,
                marker=dict(colors=[colors[sent] for sent in total_percentages.index]),
                textinfo='label+percent',
                texttemplate="%{label}<br>%{value:.1f}%"
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Sentiment Distribution Around {event}",
            height=600,
            showlegend=True
        )
        
        fig.show()
        
        # Display country distribution if All is selected
        if 'All' in countries:
            print("\nTop 10 Countries/Regions Distribution")
            print("\nBefore Event (31 days):")
            before_countries = df_filtered[df_filtered['Period'] == 'Before Event (31 days)']['Country'].value_counts().head(10)
            display(pd.DataFrame({
                'Country': before_countries.index,
                'Tweet Count': before_countries.values
            }))
            
            print("\nAfter Event (31 days):")
            after_countries = df_filtered[df_filtered['Period'] == 'After Event (31 days)']['Country'].value_counts().head(10)
            display(pd.DataFrame({
                'Country': after_countries.index,
                'Tweet Count': after_countries.values
            }))
    
    # Create interactive controls
    widgets.interactive(
        update_plot,
        event=event_widget,
        countries=country_widget
    )

def create_geo_comparison(df):
    # Define event options
    event_options = {
        'Event 1 (UK vs Non-UK)': {
            'date': '2013-08-05',
            'country': 'United Kingdom',
            'name': 'UK'
        },
        'Event 3 (US vs Non-US)': {
            'date': '2022-11-16',
            'country': 'United States of America',
            'name': 'US'
        }
    }
    
    # Create widgets
    event_widget = widgets.Dropdown(
        options=list(event_options.keys()),
        description='Event:'
    )
    
    time_widget = widgets.Dropdown(
        options=['1 week (7 days)', '2 weeks (14 days)', '3 weeks (21 days)', '4 weeks (28 days)'],
        description='Time Range:'
    )
    
    def update_plot(event, time_range):
        # Get event information
        event_info = event_options[event]
        event_date = pd.to_datetime(event_info['date'])
        country = event_info['country']
        country_name = event_info['name']
        
        # Get number of days
        days = int(time_range.split()[0]) * 7
        
        # Filter data
        end_date = event_date + pd.Timedelta(days=days)
        mask = (df['Day'] >= event_date) & (df['Day'] < end_date)
        df_filtered = df[mask].copy()
        
        # Separate user groups
        local_users = df_filtered[df_filtered['Country'] == country]
        non_local_users = df_filtered[
            (df_filtered['Country'].notna()) & 
            (df_filtered['Country'] != country)
        ]
        
        # Calculate sentiment distribution
        def calculate_sentiment_dist(data):
            sentiment_counts = data['Sentiment'].value_counts()
            total = len(data)
            return (sentiment_counts / total * 100).round(2)
        
        local_sentiment = calculate_sentiment_dist(local_users)
        non_local_sentiment = calculate_sentiment_dist(non_local_users)
        
        # Create chart
        fig = go.Figure()
        
        # Set colors
        colors = {
            'Positive': '#ef5675',
            'Negative': '#7a5195',
            'Neutral': '#ffa600',
            'Combination': '#003f5c'
        }
        
        # Add bars
        for sentiment in colors.keys():
            if sentiment in local_sentiment.index:
                fig.add_trace(go.Bar(
                    name=sentiment,
                    x=[f'{country_name} Users'],
                    y=[local_sentiment.get(sentiment, 0)],
                    marker_color=colors[sentiment],
                    text=[f'{local_sentiment.get(sentiment, 0):.1f}%<br>({int(local_users[local_users["Sentiment"]==sentiment].shape[0]):,})'],
                    textposition='auto',
                ))
            
            if sentiment in non_local_sentiment.index:
                fig.add_trace(go.Bar(
                    name=sentiment,
                    x=[f'Non-{country_name} Users'],
                    y=[non_local_sentiment.get(sentiment, 0)],
                    marker_color=colors[sentiment],
                    text=[f'{non_local_sentiment.get(sentiment, 0):.1f}%<br>({int(non_local_users[non_local_users["Sentiment"]==sentiment].shape[0]):,})'],
                    textposition='auto',
                    showlegend=False,
                ))
        
        # Update layout
        fig.update_layout(
            title=f"Sentiment Distribution: {country_name} vs Non-{country_name} Users<br>({days} days after {event.split('(')[0].strip()})",
            yaxis_title="Percentage (%)",
            barmode='group',
            showlegend=True,
            height=600,
            legend=dict(
                title="Sentiment",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.show()
        
        # Display statistics
        print(f"\n{country_name} Users Statistics")
        print(f"Total tweets: {len(local_users):,}")
        
        print(f"\nNon-{country_name} Users Statistics")
        print(f"Total tweets: {len(non_local_users):,}")
    
    # Create interactive controls
    widgets.interactive(
        update_plot,
        event=event_widget,
        time_range=time_widget
    )

def create_word_cloud(df):
    # Download NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print('Downloading required NLTK data...')
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
    
    # Create widgets
    sentiment_widget = widgets.Dropdown(
        options=["All", "Positive", "Negative", "Neutral", "Combination"],
        description='Sentiment:'
    )
    
    max_words_widget = widgets.IntSlider(
        value=100,
        min=50,
        max=200,
        step=10,
        description='Max Words:'
    )
    
    date_range_widget = widgets.DateRangeSlider(
        value=(df['Day'].min(), df['Day'].max()),
        min=df['Day'].min(),
        max=df['Day'].max(),
        description='Date Range:'
    )
    
    def update_plot(sentiment, max_words, date_range):
        # Filter data
        mask = (df['Day'].dt.date >= date_range[0]) & \
               (df['Day'].dt.date <= date_range[1])
        df_filtered = df[mask].copy()
        
        # Apply sentiment filter
        if sentiment != "All":
            df_filtered = df_filtered[df_filtered['Sentiment'] == sentiment]
        
        if len(df_filtered) == 0:
            print("No data available for selected criteria")
            return
        
        # Process text
        print('Processing text...(It might take up to 1 minute)')
        
        df_filtered['Tweet'] = df_filtered['Tweet'].fillna('')
        all_text = " ".join(df_filtered['Tweet'].astype(str).apply(preprocess_text))
        
        if not all_text.strip():
            print("No meaningful text found after preprocessing")
            return
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            contour_width=3,
            contour_color='steelblue',
            colormap='viridis',
            random_state=42
        ).generate(all_text)
        
        # Calculate word frequencies
        words = all_text.split()
        word_freq_dict = {}
        for word in words:
            if word:
                word_freq_dict[word] = word_freq_dict.get(word, 0) + 1
        
        # Sort by frequency and get top 10
        sorted_words = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)
        top_words = dict(sorted_words[:10])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), 
                                     gridspec_kw={'height_ratios': [3, 1]})
        
        # Display word cloud
        ax1.imshow(wordcloud, interpolation='bilinear')
        ax1.axis('off')
        title = f"Word Cloud - {sentiment} Sentiment" if sentiment != "All" else "Word Cloud - All Sentiments"
        ax1.set_title(title, fontsize=16, pad=20)
        
        # Create bar chart
        bars = ax2.bar(range(len(top_words)), list(top_words.values()), 
                     color='steelblue')
        ax2.set_xticks(range(len(top_words)))
        ax2.set_xticklabels(list(top_words.keys()), rotation=45, ha='right')
        ax2.set_title("Top 10 Most Frequent Words", pad=20)
        ax2.set_ylabel("Frequency (Count)")
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Display statistics
        print("\nWord Frequency Details")
        freq_df = pd.DataFrame({
            'Word': list(top_words.keys()),
            'Count': list(top_words.values()),
            'Percentage': [f"{(count/sum(top_words.values()))*100:.1f}%" 
                         for count in top_words.values()]
        })
        display(freq_df)
        
        print("\nAnalysis Statistics")
        print(f"Total Tweets Analyzed: {len(df_filtered):,}")
        unique_words = len(set(all_text.split()))
        print(f"Unique Words: {unique_words:,}")
        avg_words = len(all_text.split()) / len(df_filtered)
        print(f"Avg Words per Tweet: {avg_words:.1f}")
    
    # Create interactive controls
    widgets.interactive(
        update_plot,
        sentiment=sentiment_widget,
        max_words=max_words_widget,
        date_range=date_range_widget
    )

def main():
    print("Cultured Meat Sentiment Analysis Dashboard")
    
    # Load data
    df = load_data()
    
    # Create view selection widget
    view_widget = widgets.RadioButtons(
        options=["Overview", "Event Analysis", "Psychological Distance Analysis", "Word Cloud Analysis"],
        description='Select View:'
    )
    
    def update_view(view):
        clear_output(wait=True)
        print("Cultured Meat Sentiment Analysis Dashboard")
        display(view_widget)
        
        if view == "Overview":
            create_overview(df)
        elif view == "Event Analysis":
            create_event_analysis(df)
        elif view == "Psychological Distance Analysis":
            create_geo_comparison(df)
        elif view == "Word Cloud Analysis":
            create_word_cloud(df)
    
    # Create interactive view selection
    widgets.interactive(update_view, view=view_widget)

if __name__ == "__main__":
    main()