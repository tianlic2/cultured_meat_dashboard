import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# 设置页面配置
st.set_page_config(
    page_title="Cultured Meat Sentiment Analysis",
    page_icon="🧬",
    layout="wide"
)

# 国家代码映射字典
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

# 加载数据
@st.cache_data
def load_data():
    # Google Drive 文件 ID
    file_id = "1yNyhvlaYuf92XIkz858mtqZfrkAHCMv_"
    
    # 构建下载链接
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # 读取数据
    df = pd.read_csv(url)
    df['Day'] = pd.to_datetime(df['Day'])
    
    # 添加情感映射
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
    
    # 将数字转换为情感标签
    df['Sentiment'] = df['Sentiment'].map(sentiment_mapping)
    
    return df

def create_overview(df):
    st.header("Overview of Sentiment Distribution")
    
    # 侧边栏控件
    with st.sidebar:
        st.subheader("Controls")
        
        # 指标选择
        metric = st.radio(
            "Select Metric:",
            options=['Count', 'Percentage']
        )
        
        # 日期范围选择
        date_range = st.date_input(
            "Select Date Range",
            value=(df['Day'].min(), df['Day'].max()),
            min_value=df['Day'].min().date(),
            max_value=df['Day'].max().date()
        )
        
        # 国家/地区选择
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
        
        # 情感选择 - 使用预定义的情感列表
        sentiments = ['Positive', 'Negative', 'Neutral', 'Combination']
        selected_sentiments = st.multiselect(
            "Select Sentiments:",
            options=sentiments,
            default=sentiments
        )
    
    # 数据处理
    df_filtered = df.copy()
    
    # 日期过滤
    mask = (df_filtered['Day'].dt.date >= date_range[0]) & \
           (df_filtered['Day'].dt.date <= date_range[1])
    df_filtered = df_filtered[mask]
    
    # 国家过滤
    if 'All' not in selected_countries:
        country_values = [c.split(' (')[0] for c in selected_countries]
        df_filtered['Country Name'] = df_filtered['Country Code'].map(country_code_to_name)
        df_filtered = df_filtered[df_filtered['Country Name'].isin(country_values)]
    
    # 情感过滤
    df_filtered = df_filtered[df_filtered['Sentiment'].isin(selected_sentiments)]
    
    # 创建图表
    monthly_sentiment = df_filtered.groupby([pd.Grouper(key='Day', freq='M'), 'Sentiment']).size().unstack(fill_value=0)
    
    if metric == 'Percentage':
        monthly_sentiment = monthly_sentiment.div(monthly_sentiment.sum(axis=1), axis=0) * 100
    
    # 颜色方案
    colors = {
        'Positive': '#ef5675',
        'Negative': '#7a5195',
        'Neutral': '#ffa600',
        'Combination': '#003f5c'
    }
    
    # 创建图表
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
    
    # 更新布局
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
   
    
    # 显示数据统计
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
    
    # 定义重要事件
    events = {
        'Event 1: First Cultured Beef Burger (2013-08-05)': '2013-08-05',
        'Event 2: Singapore Approval (2020-12-02)': '2020-12-02',
        'Event 3: FDA Clearance (2022-11-16)': '2022-11-16'
    }
    
    # 控件
    selected_event = st.selectbox("Select Event:", list(events.keys()))
    
    # 国家/地区选择
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
    
    # 数据处理
    event_dt = pd.to_datetime(events[selected_event])
    before_start = event_dt - pd.Timedelta('31D')
    after_end = event_dt + pd.Timedelta('31D')
    
    df_filtered = df_valid.copy()
    
    if 'All' not in selected_countries:
        country_values = [c.split(' (')[0] for c in selected_countries]
        df_filtered = df_filtered[df_filtered['Country Name'].isin(country_values)]
    
    # 创建时期标签
    df_filtered['Period'] = None
    df_filtered.loc[(df_filtered['Day'] >= before_start) & 
                   (df_filtered['Day'] < event_dt), 'Period'] = 'Before Event (31 days)'
    df_filtered.loc[(df_filtered['Day'] >= event_dt) & 
                   (df_filtered['Day'] <= after_end), 'Period'] = 'After Event (31 days)'
    
    df_filtered = df_filtered[df_filtered['Period'].notna()]
    
    if len(df_filtered) == 0:
        st.warning("No data available for the selected period and region(s)")
        return
    
    # 创建图表
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 计算情感分布
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
        # 显示总体分布饼图
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
    
    # 如果选择All，显示国家分布
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
    
    # 选择事件
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
    
    # 选择时间范围
    time_range = st.selectbox(
        "Select Time Range:",
        options=['1 week (7 days)', '2 weeks (14 days)', '3 weeks (21 days)', '4 weeks (28 days)'],
        index=0
    )
    
    # 获取天数
    days = int(time_range.split()[0]) * 7
    
    # 获取事件信息
    event_info = event_options[selected_event]
    event_date = pd.to_datetime(event_info['date'])
    country_code = event_info['country']
    country_name = event_info['name']
    
    # 过滤数据
    end_date = event_date + pd.Timedelta(days=days)
    mask = (df['Day'] >= event_date) & (df['Day'] < end_date)
    df_filtered = df[mask].copy()
    
    # 分离用户组
    local_users = df_filtered[df_filtered['Country Code'] == country_code]
    non_local_users = df_filtered[df_filtered['Country Code'] != country_code]
    
    # 计算情感分布
    def calculate_sentiment_dist(data):
        sentiment_counts = data['Sentiment'].value_counts()
        total = len(data)
        return (sentiment_counts / total * 100).round(2)
    
    local_sentiment = calculate_sentiment_dist(local_users)
    non_local_sentiment = calculate_sentiment_dist(non_local_users)
    
    # 创建图表
    fig = go.Figure()
    
    # 设置颜色
    colors = {
        'Positive': '#ef5675',
        'Negative': '#7a5195',
        'Neutral': '#ffa600',
        'Combination': '#003f5c'
    }
    
    # 添加本地用户的柱状图
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
    
    # 添加非本地用户的柱状图
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
    
    # 更新布局
    fig.update_layout(
        title=f"Sentiment Distribution: {country_name} vs Non-{country_name} Users<br>({days} days after {selected_event.split('(')[0].strip()})",
        yaxis_title="Percentage (%)",
        barmode='group',
        showlegend=True,
        width=1000,
        height=600
    )
    
    # 显示图表
    st.plotly_chart(fig)
    
    # 显示统计信息
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"### {country_name} Users Statistics")
        st.write(f"Total tweets: {len(local_users):,}")
        
    with col2:
        st.write(f"### Non-{country_name} Users Statistics")
        st.write(f"Total tweets: {len(non_local_users):,}")

# 在主函数中添加新的视图选项
def main():
    st.title("Cultured Meat Sentiment Analysis Dashboard")
    
    # 加载数据
    df = load_data()
    
    # 创建视图选择
    view = st.sidebar.radio(
        "Select View:",
        ["Overview", "Event Analysis", "Psychological Distance Analysis"]
    )
    
    if view == "Overview":
        create_overview(df)
    elif view == "Event Analysis":
        create_event_analysis(df)
    else:
        create_geo_comparison(df)

if __name__ == "__main__":
    main()