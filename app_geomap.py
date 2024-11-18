import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Cultured Meat Sentiment Analysis",
    page_icon="🧬",
    layout="wide"
)

# Country code mapping dictionary
country_code_to_name = {
    'ABW': 'Aruba', 'AFG': 'Afghanistan', 'AGO': 'Angola', 'AIA': 'Anguilla',
    'ALA': 'Åland Islands', 'ALB': 'Albania', 'AND': 'Andorra', 'ARE': 'United Arab Emirates',
    'ARG': 'Argentina', 'ARM': 'Armenia', 'ASM': 'American Samoa', 'ATA': 'Antarctica',
    'ATF': 'French Southern Territories', 'ATG': 'Antigua and Barbuda', 'AUS': 'Australia',
    'AUT': 'Austria', 'AZE': 'Azerbaijan', 'BDI': 'Burundi', 'BEL': 'Belgium',
    'BEN': 'Benin', 'BES': 'Bonaire', 'BFA': 'Burkina Faso', 'BGD': 'Bangladesh',
    'BGR': 'Bulgaria', 'BHR': 'Bahrain', 'BHS': 'Bahamas', 'BIH': 'Bosnia and Herzegovina',
    'BLM': 'Saint Barthélemy', 'BLR': 'Belarus', 'BLZ': 'Belize', 'BMU': 'Bermuda',
    'BOL': 'Bolivia', 'BRA': 'Brazil', 'BRB': 'Barbados', 'BRN': 'Brunei',
    'BTN': 'Bhutan', 'BWA': 'Botswana', 'CAF': 'Central African Republic', 'CAN': 'Canada',
    'CHE': 'Switzerland', 'CHL': 'Chile', 'CHN': 'China', 'CIV': 'Côte d\'Ivoire',
    'CMR': 'Cameroon', 'COD': 'DR Congo', 'COG': 'Congo', 'COL': 'Colombia',
    'CRI': 'Costa Rica', 'CUB': 'Cuba', 'CYP': 'Cyprus', 'CZE': 'Czech Republic',
    'DEU': 'Germany', 'DNK': 'Denmark', 'DOM': 'Dominican Republic', 'DZA': 'Algeria',
    'ECU': 'Ecuador', 'EGY': 'Egypt', 'ESP': 'Spain', 'EST': 'Estonia',
    'ETH': 'Ethiopia', 'FIN': 'Finland', 'FJI': 'Fiji', 'FRA': 'France',
    'GBR': 'United Kingdom', 'GEO': 'Georgia', 'GHA': 'Ghana', 'GRC': 'Greece',
    'HKG': 'Hong Kong', 'HRV': 'Croatia', 'HUN': 'Hungary', 'IDN': 'Indonesia',
    'IND': 'India', 'IRL': 'Ireland', 'IRN': 'Iran', 'IRQ': 'Iraq',
    'ISL': 'Iceland', 'ISR': 'Israel', 'ITA': 'Italy', 'JAM': 'Jamaica',
    'JOR': 'Jordan', 'JPN': 'Japan', 'KAZ': 'Kazakhstan', 'KEN': 'Kenya',
    'KOR': 'South Korea', 'KWT': 'Kuwait', 'LBN': 'Lebanon', 'LKA': 'Sri Lanka',
    'LTU': 'Lithuania', 'LUX': 'Luxembourg', 'LVA': 'Latvia', 'MAC': 'Macao',
    'MAR': 'Morocco', 'MEX': 'Mexico', 'MYS': 'Malaysia', 'NGA': 'Nigeria',
    'NLD': 'Netherlands', 'NOR': 'Norway', 'NPL': 'Nepal', 'NZL': 'New Zealand',
    'PAK': 'Pakistan', 'PER': 'Peru', 'PHL': 'Philippines', 'POL': 'Poland',
    'PRT': 'Portugal', 'ROU': 'Romania', 'RUS': 'Russia', 'SAU': 'Saudi Arabia',
    'SGP': 'Singapore', 'SRB': 'Serbia', 'SVK': 'Slovakia', 'SVN': 'Slovenia',
    'SWE': 'Sweden', 'THA': 'Thailand', 'TUN': 'Tunisia', 'TUR': 'Turkey',
    'TWN': 'Taiwan', 'UKR': 'Ukraine', 'USA': 'United States', 'VNM': 'Vietnam',
    'ZAF': 'South Africa'
}

# Load data
@st.cache_data
def load_data():
    # Google Drive file ID
    file_id = "1yNyhvlaYuf92XIkz858mtqZfrkAHCMv_"
    
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

# 保持原有的create_overview, create_event_analysis, create_geo_comparison函数
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
        df_valid = df.dropna(subset=['Country Code']).copy()
        df_valid['Country Name'] = df_valid['Country Code'].map(country_code_to_name)
        country_counts = df_valid['Country Name'].value_counts()
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
        df_filtered['Country Name'] = df_filtered['Country Code'].map(country_code_to_name)
        df_filtered = df_filtered[df_filtered['Country Name'].isin(country_values)]
    
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
    df_valid = df.dropna(subset=['Country Code']).copy()
    df_valid['Country Name'] = df_valid['Country Code'].map(country_code_to_name)
    country_counts = df_valid['Country Name'].value_counts()
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
        df_filtered = df_filtered[df_filtered['Country Name'].isin(country_values)]
    
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
            before_countries = df_filtered[df_filtered['Period'] == 'Before Event (31 days)']['Country Name'].value_counts().head(10)
            before_df = pd.DataFrame({
                'Country': before_countries.index,
                'Tweet Count': before_countries.values
            })
            st.dataframe(before_df)
        
        with col4:
            st.write("After Event (31 days):")
            after_countries = df_filtered[df_filtered['Period'] == 'After Event (31 days)']['Country Name'].value_counts().head(10)
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
            'country': 'GBR',
            'name': 'UK'
        },
        'Event 3 (US vs Non-US)': {
            'date': '2022-11-16',
            'country': 'USA',
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
    country_code = event_info['country']
    country_name = event_info['name']
    
    # Filter data
    end_date = event_date + pd.Timedelta(days=days)
    mask = (df['Day'] >= event_date) & (df['Day'] < end_date)
    df_filtered = df[mask].copy()
    
    # Separate user groups
    local_users = df_filtered[df_filtered['Country Code'] == country_code]
    non_local_users = df_filtered[
        (df_filtered['Country Code'].notna()) & # Ensure Country Code is not NaN
        (df_filtered['Country Code'] != country_code)  # Ensure Country Code is not the selected country
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


# 新增地理热图分析
def create_geo_heatmap(df):
    st.header("Geographic Distribution Analysis")
    
    # Date range filter
    date_range = st.date_input(
        "Select Date Range",
        value=(df['Day'].min(), df['Day'].max()),
        min_value=df['Day'].min().date(),
        max_value=df['Day'].max().date()
    )
    
    # Filter data by date
    mask = (df['Day'].dt.date >= date_range[0]) & (df['Day'].dt.date <= date_range[1])
    df_filtered = df[mask]
    
    # Calculate tweet counts by country
    country_counts = df_filtered['Country Code'].value_counts()
    
    # Create choropleth map
    fig = go.Figure(data=go.Choropleth(
        locations=country_counts.index,
        z=country_counts.values,
        text=[f"{country_code_to_name.get(code, code)}: {count:,} tweets" 
              for code, count in country_counts.items()],
        colorscale='Viridis',
        marker_line_color='white',
        marker_line_width=0.5,
        colorbar_title="Tweet Count"
    ))
    
    fig.update_layout(
        title="Global Distribution of Tweets",
        geo=dict(showframe=False, showcoastlines=True),
        width=800,
        height=500
    )
    
    st.plotly_chart(fig)
    
    # Show top 10 countries table
    st.subheader("Top 10 Countries by Tweet Count")
    top_10_df = pd.DataFrame({
        'Country': [country_code_to_name.get(code, code) for code in country_counts.head(10).index],
        'Tweet Count': country_counts.head(10).values,
        'Percentage': (country_counts.head(10).values / len(df_filtered) * 100).round(2)
    })
    st.dataframe(top_10_df)

# 新增国家对比分析
def create_country_comparison(df):
    st.header("Country Comparison Over Time")
    
    # Country selection
    countries = df['Country Code'].dropna().unique()
    selected_countries = st.multiselect(
        "Select Countries to Compare:",
        options=[country_code_to_name.get(code, code) for code in countries],
        default=[country_code_to_name.get(countries[0], countries[0])]
    )
    
    # Convert country names back to codes
    country_name_to_code = {v: k for k, v in country_code_to_name.items()}
    selected_codes = [country_name_to_code[name] for name in selected_countries]
    
    # Time frequency selection
    freq = st.selectbox(
        "Select Time Frequency:",
        options=['Daily', 'Weekly', 'Monthly'],
        index=1
    )
    
    freq_map = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}
    
    # Create time series for each country
    fig = go.Figure()
    
    for country in selected_codes:
        country_data = df[df['Country Code'] == country]
        time_series = country_data.groupby([pd.Grouper(key='Day', freq=freq_map[freq]), 'Sentiment']).size().unstack(fill_value=0)
        
        for sentiment in time_series.columns:
            fig.add_trace(go.Scatter(
                x=time_series.index,
                y=time_series[sentiment],
                name=f"{country_code_to_name.get(country, country)} - {sentiment}",
                mode='lines+markers'
            ))
    
    fig.update_layout(
        title=f"{freq} Sentiment Distribution by Country",
        xaxis_title="Date",
        yaxis_title="Tweet Count",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig)

# 新增情感矩阵分析
def create_sentiment_matrix(df):
    st.header("Sentiment Distribution Matrix")
    
    # Time period selection
    period = st.selectbox(
        "Select Time Period:",
        options=['Daily', 'Weekly', 'Monthly'],
        index=1
    )
    
    # Group data by country and sentiment
    df_grouped = df.groupby(['Country Code', 'Sentiment']).size().unstack(fill_value=0)
    
    # Calculate percentages
    df_percentages = df_grouped.div(df_grouped.sum(axis=1), axis=0) * 100
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=df_percentages.values,
        x=df_percentages.columns,
        y=[country_code_to_name.get(code, code) for code in df_percentages.index],
        colorscale='RdBu',
        text=np.round(df_percentages.values, 1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Sentiment Distribution by Country",
        xaxis_title="Sentiment",
        yaxis_title="Country",
        height=max(400, len(df_percentages) * 20)
    )
    
    st.plotly_chart(fig)

# 新增时间趋势分析
def create_time_analysis(df):
    st.header("Temporal Pattern Analysis")
    
    # Add controls
    metric = st.selectbox(
        "Select Time Unit:",
        ["Hour of Day", "Day of Week", "Month"]
    )
    
    # Process data based on selection
    if metric == "Hour of Day":
        df['TimeUnit'] = df['Day'].dt.hour
        x_title = "Hour"
    elif metric == "Day of Week":
        df['TimeUnit'] = df['Day'].dt.day_name()
        x_title = "Day of Week"
    else:
        df['TimeUnit'] = df['Day'].dt.month_name()
        x_title = "Month"
    
    # Create visualization
    fig = go.Figure()
    
    for sentiment in df['Sentiment'].unique():
        sentiment_data = df[df['Sentiment'] == sentiment]
        counts = sentiment_data['TimeUnit'].value_counts()
        
        fig.add_trace(go.Bar(
            name=sentiment,
            x=counts.index,
            y=counts.values,
            text=counts.values,
            textposition='auto',
        ))
    
    fig.update_layout(
        title=f"Tweet Distribution by {metric}",
        xaxis_title=x_title,
        yaxis_title="Number of Tweets",
        barmode='group'
    )
    
    st.plotly_chart(fig)

# 新增情感变化率分析
def create_sentiment_change(df):
    st.header("Sentiment Change Analysis")
    
    # Calculate daily sentiment percentages
    daily_sentiment = df.groupby([pd.Grouper(key='Day', freq='D'), 'Sentiment']).size().unstack()
    daily_percentages = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0) * 100
    
    # Calculate day-over-day changes
    sentiment_changes = daily_percentages.diff()
    
    # Create visualization
    fig = go.Figure()
    
    for sentiment in daily_percentages.columns:
        fig.add_trace(go.Scatter(
            x=sentiment_changes.index,
            y=sentiment_changes[sentiment],
            name=sentiment,
            mode='lines+markers',
            hovertemplate="Date: %{x}<br>" +
                         "Change: %{y:.2f}%<br>"
        ))
    
    fig.update_layout(
        title="Day-over-Day Sentiment Distribution Changes",
        xaxis_title="Date",
        yaxis_title="Percentage Point Change",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig)

# 更新主函数
def main():
    st.title("Cultured Meat Sentiment Analysis Dashboard")
    
    # Load data
    df = load_data()
    
    # Create view selection
    view = st.sidebar.radio(
        "Select View:",
        ["Overview", 
         "Event Analysis", 
         "Psychological Distance Analysis",
         "Geographic Heatmap",
         "Country Comparison",
         "Sentiment Matrix",
         "Temporal Patterns",
         "Sentiment Changes"]
    )
    
    if view == "Overview":
        create_overview(df)
    elif view == "Event Analysis":
        create_event_analysis(df)
    elif view == "Psychological Distance Analysis":
        create_geo_comparison(df)
    elif view == "Geographic Heatmap":
        create_geo_heatmap(df)
    elif view == "Country Comparison":
        create_country_comparison(df)
    elif view == "Sentiment Matrix":
        create_sentiment_matrix(df)
    elif view == "Temporal Patterns":
        create_time_analysis(df)
    else:
        create_sentiment_change(df)

if __name__ == "__main__":
    main()