import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import requests
import pydeck as pdk
from bs4 import BeautifulSoup
from plotly.subplots import make_subplots
from geopy.geocoders import Nominatim


# Page configuration
st.set_page_config(
    layout="wide",
    page_title="Stock Market Insights",
    page_icon="📈")


# Read ticker symbols and titles from file
def read_tickers_and_titles_from_file(filename):
    data = pd.read_csv(filename)
    tickers = data['ticker'].tolist()
    titles = data['title'].tolist()
    return tickers, titles

# Fetch list of ticker symbols and titles from file
ticker_file = 'ticker_list.txt'
tickers, titles = read_tickers_and_titles_from_file(ticker_file)



# Page title
st.markdown('''
# Real-Time Stock Market Insights 
        
### *By Zakaria B.* 👨‍💼
''')

#st.write('---') # same as 'st.divider()'

col1, col2, col3 = st.columns(3)

with col1:
    # It maps the indices in the dropdown to the corresponding ticker titles
    ticker_index = st.selectbox('Select Ticker Symbol', list(range(len(tickers))), format_func=lambda x: tickers[x])

# Period filters
with col2:
    start_date = st.date_input('Start Date', pd.to_datetime('2020-01-01'))

with col3:
    end_date = st.date_input('End Date', pd.to_datetime('today'))



# Select ticker based on index
selected_ticker = tickers[ticker_index]

# Fetch stock data
# stock_data = yf.download(selected_ticker, start=start_date, end=end_date, period=period)
stock_data = yf.download(selected_ticker, start=start_date, end=end_date)

# Function to fetch metrics
stock = yf.Ticker(selected_ticker)
metrics_data = stock.info



################################################ Import Country Flags ##################################################

def get_flag_image(country_name):
    # Construct the URL for the Wikipedia page
    wikipedia_url = f"https://en.wikipedia.org/wiki/Gallery_of_sovereign_state_flags"
    # Fetch the HTML content of the Wikipedia page
    response = requests.get(wikipedia_url)
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")
        # Find all images on the page
        images = soup.find_all("img")
        # Search for the flag image URL for the specified country
        for img in images:
            if country_name.lower() in img.get("alt", "").lower():
                flag_image_url = "https:" + img["src"]
                return flag_image_url
        else:
            st.error("Flag not found")
            return None
    else:
        st.error("Failed to fetch Wikipedia page")
        return None


selected_country = metrics_data['country']

flag_image_url = get_flag_image(selected_country)




############################################ Company Info ###########################################

col1, col2 = st.columns(2)

with col1:
    # Get coordinates of the selected city
    geolocator = Nominatim(user_agent="city_locator")
    location = geolocator.geocode(metrics_data['city'])

    city_latitude, city_longitude = location.latitude, location.longitude

    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=city_latitude,
            longitude=city_longitude,
            zoom=11,
            pitch=200,
        )
    ))



with col2:
    container_info = st.container(border=True, height=500)
    container_info.header('{} ({})'.format(titles[ticker_index], selected_ticker))
    container_info.image(flag_image_url, width=50)
    container_info.write(f"Country: {metrics_data['country']}")
    container_info.write(f"City: {metrics_data['city']}")
    container_info.write(f"Sector: {metrics_data['sector']}")
    container_info.write(f"Website: {metrics_data['website']}")
    container_info.divider()

    col1, col2, col3, col4 = container_info.columns(4)  

    with col1:
        st.metric("Closing Price", f"{stock_data['Close'][-1]:.2f}", f"{((stock_data['Close'][-1] - stock_data['Close'][-2])/stock_data['Close'][-2])*100:.2%}")

    with col2:
        st.metric("Volume (Million)", f"{stock_data['Volume'][-1]/1000000:.2f}", f"{((stock_data['Volume'][-1] - stock_data['Volume'][-2])/stock_data['Volume'][-2])*100:.2%}")

    with col3:
        st.metric("Market Cap. (Bn $)", f"{metrics_data['marketCap']/1000000000:.2f}")

    with col4:
        stock_data['Daily Perf %'] = stock_data['Close'].pct_change() * 100
        average_daily_return = stock_data['Daily Perf %'].mean()
        st.metric("Avg. Daily Perf.", f"{average_daily_return:.2%}")        




st.write(' ')
st.write(' ')




########################################## Stock price & volume ##################################

st.subheader('Stock Price & Volume')

# Create a bar chart for volume
fig_volume = px.bar(stock_data, x=stock_data.index, y='Volume', labels={'x': 'Date', 'y': 'Volume'})


# Create a line chart for Close with a secondary y-axis
fig_volume.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Closing', yaxis='y2', line=dict(color='yellow')))

# Update layout to include secondary y-axis
fig_volume.update_layout(
    yaxis2=dict(
        title='Closing price',
        overlaying='y',
        side='right'
    )
)


#previous_closing = stock_data['Close'].shift(1)

open_close_variation = (stock_data['Close'] - stock_data['Open'])/stock_data['Open']

# Set colors based on the Closing_Price_Change_Percentage
colors = ['green' if x >= 0 else 'red' for x in open_close_variation]

# Update the color of the bars in the bar chart
fig_volume.update_traces(marker=dict(color=colors))

st.plotly_chart(fig_volume, use_container_width=True)


expander = st.expander("Price/Volume relationship 📋")
expander.write('''
Stock volume transactions refer to the total number of shares traded within a specific time frame (daily basis here).
It shows the level of activity in a stock, with high volume indicating strong interest and low volume suggesting less activity. 
Analyzing volume alongside price movements helps investors gauge market sentiment and identify potential trading opportunities.
''')

expander.image("https://centerpointsecurities.com/wp-content/uploads/2022/04/Relative-Volume-Example.png")


#st.divider()
st.write(' ')
st.write(' ')
st.write(' ')



############################### Daily performance & Stock Data ########################################

col1, col2 = st.columns(2)


with col1:

    st.subheader(f'{titles[ticker_index]} Data')

    stock_data['Range Ratio %'] = ((stock_data['High'] - stock_data['Low']) / stock_data['Close'])*100
    st.write(stock_data)

    st.write(' ')
    st.write(' ')
    st.write(' ')

    expander = st.expander("Columns explanation 📋")
    expander.write('''
    Open: The price of a financial asset at the beginning of a trading period, such as a day, week, or month.
                   
    Close: The price of a financial asset at the end of a trading period. It's the last traded price before the market closes.
                   
    High: The highest price of a financial asset reached during a trading period.
                   
    Low: The lowest price of a financial asset reached during a trading period.
                   
    Adj Close: The adjusted closing price of a financial asset, which accounts for factors such as dividends, stock splits, and other corporate actions that affect the price.
    
    Volume: The total number of shares or contracts traded during a given period of time (in this case daily). It represents the level of activity in the market for that asset.
                   
    Daily Performance: It's the percentage change in a stock's closing price compared to the previous day's closing price.
    
    Range Ratio: It's a measure of stock price volatility, comparing the difference between the high and low prices to the closing price.
    ''')



with col2:
# Calculate daily returns

    st.subheader(f'Daily Performance Distribution')


    # Plot daily return distribution histogram
    fig_hist = px.histogram(stock_data['Daily Perf %'].dropna(), nbins=150)
    fig_hist.update_xaxes(title_text='Daily Performance (%)')
    fig_hist.update_layout(showlegend=False)

    st.plotly_chart(fig_hist, use_container_width=True)


    # Volatility explanation
    expander = st.expander("Performance Distribution explanation 📋")
    expander.write('''
    A daily performance distribution illustrates the range and frequency of price changes for a particular asset within a single trading day. 
    It shows how much the asset's value typically fluctuates over the course of one day, providing insights into the volatility and trading patterns of the asset. 
    This distribution can help investors understand the potential risks and rewards associated with trading or holding the asset on a daily basis.
    A performance below zero indicates a Closing Price lower than the previous one, and vice versa.
    ''')

    #expander.image("https://www.k2analytics.co.in/wp-content/uploads/2020/05/standard-deviation.png")


# Details of ticker data
# st.subheader('Stock Data Metrics Details')
# st.write(metrics_data)


st.write(' ')
st.write(' ')
st.write(' ')




############################### Bollinger bands & Candlestick ########################################

col1, col2 = st.columns(2)

with col1:
    window = 20
    stock_data['MA'] = stock_data['Close'].rolling(window=window).mean()
    stock_data['Std'] = stock_data['Close'].rolling(window=window).std()
    stock_data['Upper_band'] = stock_data['MA'] + (stock_data['Std'] * 2)
    stock_data['Lower_band'] = stock_data['MA'] - (stock_data['Std'] * 2)


    st.subheader(f'Bollinger Bands')



    # Plot Bollinger Bands
    fig_bb = go.Figure(data=[go.Candlestick(x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'])])
    #fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Closing'))
    fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Upper_band'], mode='lines', line=dict(color='grey'), name='Upper'))
    fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Lower_band'], mode='lines', line=dict(color='grey'), name='Lower'))
    fig_bb.update_layout(showlegend=False)

    st.plotly_chart(fig_bb, use_container_width=True)


    expander = st.expander("Bollinger Bands explanation 📋")
    expander.write('''
        Bollinger Bands are a technical analysis tool used to measure volatility and identify potential price reversal points. 
        They consist of three lines: a middle line, typically a simple moving average, and upper and lower bands that are a certain number of standard deviations away from the middle line.
        The distance between the bands widens when volatility increases and narrows when it decreases.  
        Traders often interpret price movements relative to the bands: when prices touch or exceed the outer bands, it may indicate overbought or oversold conditions, potentially signaling a reversal in the price trend.
    ''')

    expander.image("https://scanz.com/wp-content/uploads/2018/12/bollingerbands.jpg")



with col2:

    st.subheader('Price/Volume relationship')

    # Create a scatter plot with Volume on the x-axis and Close on the y-axis
    fig_price_volume = px.scatter(stock_data, x='Volume', y='Close', labels={'Volume' : 'Volume (Million)', 'Close' : 'Closing price'})

    st.plotly_chart(fig_price_volume, use_container_width=True)


    expander = st.expander("Price-Volume explanation 📋")
    expander.write('''
        A Price-Volume Chart is a graphical representation that shows the relationship between a financial asset's price movements and the corresponding trading volume over a specific period. 
        This chart helps investors analyze the relationship between price trends and trading activity, providing insights into market sentiment and potential price movements.
    ''')

    expander.image("https://miro.medium.com/v2/resize:fit:980/1*RJ_w1c7dyLdW_Uvlgoecug.png")




st.write(' ')
st.write(' ')
st.write(' ')


############################################## Moving Averages #######################################################

# Calculate moving averages
short_window = 20
long_window = 50
stock_data['Short_MA'] = stock_data['Close'].rolling(window=short_window, min_periods=1).mean()
stock_data['Long_MA'] = stock_data['Close'].rolling(window=long_window, min_periods=1).mean()


st.subheader('Moving Averages')

# Plot moving averages
fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Short_MA'], mode='lines', name='Short Moving Average'))
fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Long_MA'], mode='lines', name='Long Moving Average'))
st.plotly_chart(fig_ma, use_container_width=True)

expander = st.expander("Moving Averages explanation 📋")
expander.write('''
    Short and long moving averages are tools used in technical analysis to smooth out price data and identify trends. 
    The short moving average (SMA) calculates the average price over a short period, like 20 days, while the long moving average does the same over a longer period, such as 50 days. 
    Traders use these averages to gauge the direction of the trend. 
    When the short-term average crosses above the long-term one, it may signal a bullish trend, and vice versa for a bearish trend.
''')

expander.image("https://scanz.com/wp-content/uploads/2018/12/movingaveragecrossovers.jpg")


