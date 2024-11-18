import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Cultured Meat Sentiment Analysis",
    page_icon="🧬", 
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    # Google Drive file ID
    file_id = "1Dt6rTzM24OYxxujeSDW5pBS-H5lgAPXZ"
    
    # Build download link
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        # First read raw data without date parsing
        df = pd.read_csv(
            url,
            sep='\t',           # Use tab as separator
            encoding='utf-8'     # Specify encoding
        )
        
        # Ensure all required columns exist
        expected_columns = ['Country', 'Sentiment', 'Tweet', 'Day']
        if not all(col in df.columns for col in expected_columns):
            missing_cols = [col for col in expected_columns if col not in df.columns]
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Process date column separately
        df['Day'] = pd.to_datetime(df['Day'], format='%m/%d/%y', errors='coerce')
        
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
        
        # Convert numeric values to sentiment labels
        df['Sentiment'] = df['Sentiment'].map(sentiment_mapping)
        
        # Remove missing values
        df = df.dropna(subset=['Day', 'Sentiment'])
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        if 'df' in locals():
            st.error("Data columns found: " + ", ".join(df.columns.tolist()))
            st.error("First few rows of Day column: ")
            st.write(df['Day'].head() if 'Day' in df.columns else "Day column not found")
        return pd.DataFrame()  # Return empty DataFrame if loading fails

def create_overview(df):
    st.header("Overview of Sentiment Distribution")
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Controls")
        
        # Metric selection
        metric = st.radio(
            "Select Metric:",
            options=['Count', 'Percentage']
        )
        
        # Date range selection
        date_range = st.date_input(
            "Select Date Range",
            value=(df['Day'].min(), df['Day'].max()),
            min_value=df['Day'].min().date(),
            max_value=df['Day'].max().date()
        )
        
        # Country/Region selection
        df_valid = df.dropna(subset=['Country']).copy()
        country_counts = df_valid['Country'].value_counts()
        countries = ['All'] + [f"{country} ({count:,} tweets)" 
                             for country, count in country_counts.items()]
        selected_countries = st.multiselect(
            "Select Countries/Regions:",
            options=countries,
            default=['All']
        )
        
        # Sentiment selection - using predefined sentiment list
        sentiments = ['Positive', 'Negative', 'Neutral', 'Combination']
        selected_sentiments = st.multiselect(
            "Select Sentiments:",
            options=sentiments,
            default=sentiments
        )
    
    # Data processing
    df_filtered = df.copy()
    
    # Date filtering
    mask = (df_filtered['Day'].dt.date >= date_range[0]) & \
           (df_filtered['Day'].dt.date <= date_range[1])
    df_filtered = df_filtered[mask]
    
    # Country filtering
    if 'All' not in selected_countries:
        country_values = [c.split(' (')[0] for c in selected_countries]
        df_filtered = df_filtered[df_filtered['Country'].isin(country_values)]
    
    # Sentiment filtering
    df_filtered = df_filtered[df_filtered['Sentiment'].isin(selected_sentiments)]
    
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
    
    for sentiment in selected_sentiments:
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
    if 'All' not in selected_countries:
        title_text += f" ({', '.join(country_values)})"
        
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Count" if metric == 'Count' else "Percentage (%)",
        hovermode='x unified',
        template='plotly_white',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
   
    
    # Display data statistics
    st.subheader("Data Statistics")
    total_tweets = len(df_filtered)
    st.write(f"Total Tweets: {total_tweets:,}")
    
    sentiment_stats = df_filtered['Sentiment'].value_counts()
    sentiment_percentages = (sentiment_stats / len(df_filtered) * 100).round(2)
    
    stats_df = pd.DataFrame({
        'Count': sentiment_stats,
        'Percentage (%)': sentiment_percentages
    })
    st.dataframe(stats_df)

def create_event_analysis(df):
    st.header("Event Analysis")
    
    # Define important events
    events = {
        'Event 1: First Cultured Beef Burger (2013-08-05)': '2013-08-05',
        'Event 2: Singapore Approval (2020-12-02)': '2020-12-02',
        'Event 3: FDA Clearance (2022-11-16)': '2022-11-16'
    }
    
    # Controls
    selected_event = st.selectbox("Select Event:", list(events.keys()))
    
    # Country/Region selection
    df_valid = df.dropna(subset=['Country']).copy()
    country_counts = df_valid['Country'].value_counts()
    countries = ['All'] + [f"{country} ({count:,} tweets)" 
                         for country, count in country_counts.items()]
    selected_countries = st.multiselect(
        "Select Countries/Regions:",
        options=countries,
        default=['All']
    )
    
    # Data processing
    event_dt = pd.to_datetime(events[selected_event])
    before_start = event_dt - pd.Timedelta('31D')
    after_end = event_dt + pd.Timedelta('31D')
    
    df_filtered = df_valid.copy()
    
    if 'All' not in selected_countries:
        country_values = [c.split(' (')[0] for c in selected_countries]
        df_filtered = df_filtered[df_filtered['Country'].isin(country_values)]
    
    # Create period labels
    df_filtered['Period'] = None
    df_filtered.loc[(df_filtered['Day'] >= before_start) & 
                   (df_filtered['Day'] < event_dt), 'Period'] = 'Before Event (31 days)'
    df_filtered.loc[(df_filtered['Day'] >= event_dt) & 
                   (df_filtered['Day'] <= after_end), 'Period'] = 'After Event (31 days)'
    
    df_filtered = df_filtered[df_filtered['Period'].notna()]
    
    if len(df_filtered) == 0:
        st.warning("No data available for the selected period and region(s)")
        return
    
    # Create charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Calculate sentiment distribution
        period_sentiment = df_filtered.groupby(['Period', 'Sentiment']).size().unstack(fill_value=0)
        period_percentages = period_sentiment.div(period_sentiment.sum(axis=1), axis=0) * 100
        
        colors = {
            'Positive': '#ef5675',
            'Negative': '#7a5195',
            'Neutral': '#ffa600',
            'Combination': '#003f5c'
        }
        
        fig = go.Figure()
        
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
                )
            )
        
        fig.update_layout(
            title=f"Sentiment Distribution Around {selected_event}",
            barmode='group',
            xaxis_title="Period",
            yaxis_title="Percentage (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Display overall distribution pie chart
        total_sentiment = df_filtered['Sentiment'].value_counts()
        total_percentages = (total_sentiment / len(df_filtered) * 100).round(1)
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=total_percentages.index,
            values=total_percentages.values,
            marker=dict(colors=[colors[sent] for sent in total_percentages.index]),
            textinfo='label+percent',
            texttemplate="%{label}<br>%{value:.1f}%"
        )])
        
        fig_pie.update_layout(
            title="Overall Sentiment Distribution",
            height=500
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # If All is selected, display country distribution
    if 'All' in selected_countries:
        st.subheader("Top 10 Countries/Regions Distribution")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("Before Event (31 days):")
            before_countries = df_filtered[df_filtered['Period'] == 'Before Event (31 days)']['Country'].value_counts().head(10)
            before_df = pd.DataFrame({
                'Country': before_countries.index,
                'Tweet Count': before_countries.values
            })
            st.dataframe(before_df)
        
        with col4:
            st.write("After Event (31 days):")
            after_countries = df_filtered[df_filtered['Period'] == 'After Event (31 days)']['Country'].value_counts().head(10)
            after_df = pd.DataFrame({
                'Country': after_countries.index,
                'Tweet Count': after_countries.values
            })
            st.dataframe(after_df)

def create_geo_comparison(df):
    st.header("Psychological Distance Analysis")
    
    # Select event
    event_options = {
        'Event 1 (UK vs Non-UK)': {
            'date': '2013-08-05',
            'country': 'United Kingdom',
            'name': 'UK'
        },
        'Event 3 (US vs Non-US)': {
            'date': '2022-11-16',
            'country': 'United States',
            'name': 'US'
        }
    }
    
    selected_event = st.selectbox(
        "Select Event:",
        options=list(event_options.keys())
    )
    
    # Select time range
    time_range = st.selectbox(
        "Select Time Range:",
        options=['1 week (7 days)', '2 weeks (14 days)', '3 weeks (21 days)', '4 weeks (28 days)'],
        index=0
    )
    
    # Get number of days
    days = int(time_range.split()[0]) * 7
    
    # Get event information
    event_info = event_options[selected_event]
    event_date = pd.to_datetime(event_info['date'])
    country = event_info['country']
    country_name = event_info['name']
    
    # Filter data
    end_date = event_date + pd.Timedelta(days=days)
    mask = (df['Day'] >= event_date) & (df['Day'] < end_date)
    df_filtered = df[mask].copy()
    
    # Separate user groups
    local_users = df_filtered[df_filtered['Country'] == country]
    non_local_users = df_filtered[
        (df_filtered['Country'].notna()) & # Ensure Country is not NaN
        (df_filtered['Country'] != country)  # Ensure Country is not the selected country
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
    
    # Add local users bars
    for sentiment in colors.keys():
        if sentiment in local_sentiment.index:
            fig.add_trace(go.Bar(
                name=f'{sentiment} ({country_name})',
                x=[f'{country_name} Users'],
                y=[local_sentiment.get(sentiment, 0)],
                marker_color=colors[sentiment],
                text=[f'{local_sentiment.get(sentiment, 0):.1f}%<br>({int(local_users[local_users["Sentiment"]==sentiment].shape[0]):,})'],
                textposition='auto',
            ))
    
    # Add non-local users bars
    for sentiment in colors.keys():
        if sentiment in non_local_sentiment.index:
            fig.add_trace(go.Bar(
                name=f'{sentiment} (Non-{country_name})',
                x=[f'Non-{country_name} Users'],
                y=[non_local_sentiment.get(sentiment, 0)],
                marker_color=colors[sentiment],
                text=[f'{non_local_sentiment.get(sentiment, 0):.1f}%<br>({int(non_local_users[non_local_users["Sentiment"]==sentiment].shape[0]):,})'],
                textposition='auto',
            ))
    
    # Update layout
    fig.update_layout(
        title=f"Sentiment Distribution: {country_name} vs Non-{country_name} Users<br>({days} days after {selected_event.split('(')[0].strip()})",
        yaxis_title="Percentage (%)",
        barmode='group',
        showlegend=True,
        width=1000,
        height=600
    )
    
    # Display chart
    st.plotly_chart(fig)
    
    # Display statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"### {country_name} Users Statistics")
        st.write(f"Total tweets: {len(local_users):,}")
        
    with col2:
        st.write(f"### Non-{country_name} Users Statistics")
        st.write(f"Total tweets: {len(non_local_users):,}")

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import re

# Download required nltk data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    # Add custom stopwords
    custom_stop_words = {'rt', 'amp', 'cultured', 'meat', 'lab', 'grown'}
    stop_words.update(custom_stop_words)
    
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)

def create_word_cloud(df):
    st.header("Word Cloud Analysis")
    
    # Controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Sentiment selection
        sentiment = st.selectbox(
            "Select Sentiment:",
            options=['All'] + list(df['Sentiment'].unique())
        )
        
        # Date range selection
        date_range = st.date_input(
            "Select Date Range",
            value=(df['Day'].min(), df['Day'].max()),
            min_value=df['Day'].min().date(),
            max_value=df['Day'].max().date()
        )
        
        # Generate button
        generate = st.button("Generate Word Cloud")
    
    if generate:
        # Filter data
        mask = (df['Day'].dt.date >= date_range[0]) & \
               (df['Day'].dt.date <= date_range[1])
        df_filtered = df[mask]
        
        if sentiment != 'All':
            df_filtered = df_filtered[df_filtered['Sentiment'] == sentiment]
        
        # Preprocess text
        with st.spinner('Processing text...'):
            processed_texts = df_filtered['Tweet'].apply(preprocess_text)
            all_text = ' '.join(processed_texts)
            
            if not all_text.strip():
                st.warning("No valid text data found")
                return
            
            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis',
                collocations=False
            ).generate(all_text)
            
            # Display word cloud
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                
                title = f"Word Cloud for {sentiment} Sentiment" if sentiment != 'All' else "Word Cloud for All Sentiments"
                plt.title(title)
                
                st.pyplot(fig)
            
            # Display word frequency statistics
            st.subheader("Top Words")
            word_freq = {}
            for text in processed_texts:
                for word in text.split():
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Convert to DataFrame and display top 20 most common words
            word_freq_df = pd.DataFrame(
                sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20],
                columns=['Word', 'Frequency']
            )
            st.dataframe(word_freq_df)

# Add new view option in main function
def main():
    st.title("Cultured Meat Sentiment Analysis Dashboard")
    
    # Load data
    df = load_data()
    
    # Create view selection
    view = st.sidebar.radio(
        "Select View:",
        ["Overview", "Event Analysis", "Psychological Distance Analysis", "Word Cloud Analysis"]
    )
    
    if view == "Overview":
        create_overview(df)
    elif view == "Event Analysis":
        create_event_analysis(df)
    elif view == "Psychological Distance Analysis":  # 修改这里的条件判断
        create_geo_comparison(df)
    elif view == "Word Cloud Analysis":
        create_word_cloud(df)

if __name__ == "__main__":
    main()