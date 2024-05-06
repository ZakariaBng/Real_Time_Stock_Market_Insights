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




st.set_page_config(layout="wide")

#try:
# Read ticker symbols and titles from file
def read_tickers_and_titles_from_file(filename):
    data = pd.read_csv(filename)
    tickers = data['ticker'].tolist()
    titles = data['title'].tolist()
    return tickers, titles

# Fetch list of ticker symbols and titles from file
ticker_file = 'ticker_list.txt'
tickers, titles = read_tickers_and_titles_from_file(ticker_file)


# Sort tickers and titles alphabetically
#tickers, titles = zip(*sorted(zip(tickers, titles), key=lambda x: x[0]))

# Page title
st.markdown('''
# Real-Time Stock Market Insights 📈
        
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



# # Sidebar filters
# st.sidebar.header('Filters')
# #period = st.sidebar.selectbox('Select Period', ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'])
# start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2020-01-01'))
# end_date = st.sidebar.date_input('End Date', pd.to_datetime('today'))


# # Sort ticker index
# #sorted_ticker_index = sorted(range(len(tickers)), key=lambda k: tickers[k])

# # It maps the indices in the dropdown to the corresponding ticker titles
# ticker_index = st.sidebar.selectbox('Select Ticker Symbol', list(range(len(tickers))), format_func=lambda x: tickers[x])
# #ticker_index = st.sidebar.selectbox('Stock ticker', ticker_list)

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

#selected_country = st.selectbox("Select a country", ["United States", "United Kingdom", "France", "Argentina", "South Africa"])  # Add more countries as needed

selected_country = metrics_data['country']

flag_image_url = get_flag_image(selected_country)



# Display ticker 
#st.header('{} ({})'.format(titles[ticker_index], selected_ticker))
#st.title('{} ({}) {} to {}'.format(titles[ticker_index], selected_ticker, start_date, end_date))

# Company info

# st.image(flag_image_url, width=50)
# st.write(f"Country: {metrics_data['country']}")
# st.write(f"City: {metrics_data['city']}")
# st.write(f"Sector: {metrics_data['sector']}")
# st.write(f"Number of Employees: {metrics_data['fullTimeEmployees']}")
# st.write(f"Website: {metrics_data['website']}")


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
        st.metric("Closing Price (end date)", f"{stock_data['Close'][-1]:.2f}", f"{((stock_data['Close'][-1] - stock_data['Close'][-2])/stock_data['Close'][-2])*100:.2%}")

    with col2:
        st.metric("Volume in Million (end date)", f"{stock_data['Volume'][-1]/1000000:.2f}", f"{((stock_data['Volume'][-1] - stock_data['Volume'][-2])/stock_data['Volume'][-2])*100:.2%}")

    with col3:
        st.metric("Current Market Cap. (Bn $)", f"{metrics_data['marketCap']/1000000000:.2f}")

    with col4:
        stock_data['Daily Perf %'] = stock_data['Close'].pct_change() * 100
        average_daily_return = stock_data['Daily Perf %'].mean()
        st.metric("Average Daily Performance", f"{average_daily_return:.2%}")



# st.write('---')

# # Creating a table with metrics for the selected ticker
# st.subheader('Metrics and Ratios (today)')

# st.write(pd.DataFrame({
#     'Market Cap.' : metrics_data['marketCap'], 
#     'ROA' : metrics_data['returnOnAssets'], 
#     'ROE' : metrics_data['returnOnEquity'],  
#     'Dividend Rate' : metrics_data['dividendRate'],
#     'Dividend Yield' : metrics_data['dividendYield'],
#     'Beta' : metrics_data['beta'],
#     'Trailing P/E' : metrics_data['trailingPE'],
#     'Forward P/E' : metrics_data['forwardPE'],
#     'Price to Book' : metrics_data['priceToBook'],
#     'Trailing EPS' : metrics_data['trailingEps'],
#     'Forward EPS' : metrics_data['forwardEps'],
#     'PEG' : metrics_data['pegRatio'],
#     'EBITDA Margin' : metrics_data['ebitdaMargins']
#     }, index=['']))



# col1, col2, col3, col4, col5 = st.columns(5)
# col1.metric("Closing Price", f"{stock_data['Close'][-1]:.2f}", f"{((stock_data['Close'][-1] - stock_data['Close'][-2])/stock_data['Close'][-2])*100:.2f}%")
# col2.metric("Volume (Million)", f"{stock_data['Volume'][-1]/1000000:.2f}", f"{((stock_data['Volume'][-1] - stock_data['Volume'][-2])/stock_data['Volume'][-2])*100:.2f}%")
# col3.metric("Market Cap. (Billion $)", f"{metrics_data['marketCap']/1000000000:.2f}")
# #col4.metric("ROA", f"{metrics_data['returnOnAssets']*100:.2f} %")
# col4.metric("ROE", f"{metrics_data['returnOnEquity']*100:.2f} %")
# col5.metric("Debt to Equity", f"{metrics_data['debtToEquity']:.2f}")


# col7.metric("Dividend Yield", f"{metrics_data['debtToEquity']*100:.2f} %")
# col8.metric("Debt to Equity", f"{metrics_data['debtToEquity']:.2f}")



# col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
# col1.metric("Beta", f"{metrics_data['beta']:.2f}")
# col2.metric("Trailing P/E", f"{metrics_data['trailingPE']:.2f}")
# col3.metric("Forward P/E", f"{metrics_data['forwardPE']:.2f}")
# col4.metric("Price to Book", f"{metrics_data['priceToBook']:.2f}")
# col5.metric("Trailing EPS", f"{metrics_data['trailingEps']:.2f}")
# col6.metric("Forward EPS", f"{metrics_data['forwardEps']:.2f}")
# col7.metric("Gross Margin", f"{metrics_data['grossMargins']*100:.2f} %")
# col8.metric("Operating Margin", f"{metrics_data['operatingMargins']*100:.2f} %")



# col4.metric("Humidity", "86%", "4%")
# col4.metric("Humidity", "86%", "4%")
# col4.metric("Humidity", "86%", "4%")
# col4.metric("Humidity", "86%", "4%")
# col4.metric("Humidity", "86%", "4%")
# col4.metric("Humidity", "86%", "4%")
# col4.metric("Humidity", "86%", "4%")


#st.write('---')


# # Display stock data
# st.subheader('Stock Data')
# st.write(stock_data)

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
    #stock_data['Daily_Performance'] = stock_data['Close'].pct_change()

    st.subheader(f'Daily Performance Distribution')


    # Plot daily return distribution histogram
    fig_hist = px.histogram(stock_data['Daily Perf %'].dropna(), nbins=150)
    fig_hist.update_xaxes(title_text='Daily Performance (%)')
    fig_hist.update_layout(showlegend=False)

    st.plotly_chart(fig_hist, use_container_width=True)


    # # Key performance indicators
    # average_daily_return = stock_data['Daily_Performance'].mean()
    # volatility = stock_data['Daily_Performance'].std()

    # st.write(f'Average Daily Performance: {average_daily_return:.2%}')
    # st.write(f'Volatility (Standard Deviation of Daily Performance): {volatility:.2%}')


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


#st.write('---')


############################################## Bollinger bands & Candlestick #######################################################

window = 20
stock_data['MA'] = stock_data['Close'].rolling(window=window).mean()
stock_data['Std'] = stock_data['Close'].rolling(window=window).std()
stock_data['Upper_band'] = stock_data['MA'] + (stock_data['Std'] * 2)
stock_data['Lower_band'] = stock_data['MA'] - (stock_data['Std'] * 2)


st.subheader(f'Bollinger Bands')


# fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
#                 open=stock_data['Open'],
#                 high=stock_data['High'],
#                 low=stock_data['Low'],
#                 close=stock_data['Close'])])
# st.plotly_chart(fig, use_container_width=True)


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



############################################## Volume of transactions #######################################################

# # Plot volume
# st.subheader('Volume of transactions (Million) 📈')
# fig_volume = px.bar(stock_data, x=stock_data.index, y='Volume', labels={'x': 'Date', 'y': 'Volume'})

# st.plotly_chart(fig_volume, use_container_width=True)


# expander = st.expander("Volume explanation 📋")
# expander.write('''
#     Stock volume transactions refer to the total number of shares traded within a specific time frame (daily basis here).
#     It shows the level of activity in a stock, with high volume indicating strong interest and low volume suggesting less activity. 
#     Analyzing volume alongside price movements helps investors gauge market sentiment and identify potential trading opportunities.
# ''')

# expander.image("https://centerpointsecurities.com/wp-content/uploads/2022/04/Relative-Volume-Example.png")


# st.write('---')




############################### Daily Return Distribution & Candlestick Chart ########################################

# col1, col2 = st.columns(2)
# with col1:
#     #container_daily_return = st.container(border=True, height=700)

#     # Calculate daily returns
#     stock_data['Daily_Return'] = stock_data['Close'].pct_change()

#     st.subheader(f'Daily Return Distribution')

#     # Volatility explanation
#     expander = st.expander("Standard deviation explanation")
#     expander.write('''
#         The standard deviation (volatility in the financial sector) is a statistical measure that quantifies the degree of variation or dispersion in the returns of a financial asset. 
#         It reflects the extent to which the price of the asset deviates from its average or expected value over a given period.
#         High volatility implies greater uncertainty and risk, as prices can experience rapid and significant fluctuations, while low volatility indicates more stable and predictable price movements.
#         Investors often use volatility as a gauge of risk when making investment decisions and managing portfolios.
#     ''')

#     expander.image("https://www.k2analytics.co.in/wp-content/uploads/2020/05/standard-deviation.png")


#     # Plot daily return distribution histogram
#     fig_hist = px.histogram(stock_data['Daily_Return'].dropna(), nbins=150)

#     st.plotly_chart(fig_hist, use_container_width=True)

#     # Key performance indicators
#     average_daily_return = stock_data['Daily_Return'].mean()
#     volatility = stock_data['Daily_Return'].std()


#     st.write(f'Average Daily Return: {average_daily_return:.2%}')
#     st.write(f'Volatility (Standard Deviation of Daily Returns): {volatility:.2%}')





# with col2:
#     container_candlestick = st.container(border=True, height=700)

#     container_candlestick.subheader('Candlestick Chart')

#     expander = container_candlestick.expander("📋 Candlestick explanation")
#     expander.write('''
#         A candlestick chart shows the price movement of a stock over time.
#         Each candlestick represents the open, high, low, and close prices for a specific period.
#         Green or white candlesticks indicate price gains, while red or black ones show price declines. 
#         It helps traders analyze trends and make decisions based on market sentiment and potential reversals.
#     ''')

#     expander.image("https://www.binaryoptions.com/wp-content/uploads/Candlestick-explained.png")


#     fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
#                     open=stock_data['Open'],
#                     high=stock_data['High'],
#                     low=stock_data['Low'],
#                     close=stock_data['Close'])])
#     container_candlestick.plotly_chart(fig, use_container_width=True)



#st.write('---')


st.write(' ')
st.write(' ')
st.write(' ')


############################### Moving Averages & Price/Volume relationship ########################################


col1, col2 = st.columns(2)

with col1:
    #container_moving_avg = st.container(border=True, height=600)

# Calculate moving averages
    short_window = 20
    long_window = 50
    stock_data['Short_MA'] = stock_data['Close'].rolling(window=short_window, min_periods=1).mean()
    stock_data['Long_MA'] = stock_data['Close'].rolling(window=long_window, min_periods=1).mean()

    
    st.subheader('Moving Averages')

    # Plot moving averages
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Short_MA'], mode='lines', name='Short MA'))
    fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Long_MA'], mode='lines', name='Long MA'))
    st.plotly_chart(fig_ma, use_container_width=True)

    expander = st.expander("Moving Averages explanation 📋")
    expander.write('''
        Short and long moving averages are tools used in technical analysis to smooth out price data and identify trends. 
        The short moving average (SMA) calculates the average price over a short period, like 20 days, while the long moving average does the same over a longer period, such as 50 days. 
        Traders use these averages to gauge the direction of the trend. 
        When the short-term average crosses above the long-term one, it may signal a bullish trend, and vice versa for a bearish trend.
    ''')

    expander.image("https://scanz.com/wp-content/uploads/2018/12/movingaveragecrossovers.jpg")


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






############################################## Price-Volume 2 #######################################################


# st.subheader('Price-Volume Chart 2')

# # Create a scatter plot with Volume on the x-axis and Close on the y-axis
# fig_price_volume = px.scatter(stock_data, x='Volume', y='Close', size='Volume', labels={'x': 'Volume', 'y': 'Price', 'size': 'Volume'})

# st.plotly_chart(fig_price_volume, use_container_width=True)


# expander = st.expander("Price-Volume explanation")
# expander.write('''
#     A Price-Volume Chart is a graphical representation that shows the relationship between a financial asset's price movements and the corresponding trading volume over a specific period. 
#     This chart helps investors analyze the relationship between price trends and trading activity, providing insights into market sentiment and potential price movements.
# ''')

# expander.image("https://miro.medium.com/v2/resize:fit:980/1*RJ_w1c7dyLdW_Uvlgoecug.png")





# ############################################## Stock price/volume relationship  #######################################################

# st.subheader('Stock price/volume relationship')

# # Create a bar chart for volume
# fig_volume = px.bar(stock_data, x=stock_data.index, y='Volume', labels={'x': 'Date', 'y': 'Volume'})


# # Create a line chart for Close with a secondary y-axis
# fig_volume.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close', yaxis='y2', line=dict(color='yellow')))

# # Update layout to include secondary y-axis
# fig_volume.update_layout(
#     yaxis2=dict(
#         title='Close',
#         overlaying='y',
#         side='right'
#     )
# )

# st.plotly_chart(fig_volume, use_container_width=True)

# st.slider("This is a slider", start_date, end_date, (start_date, end_date))


# st.write('---')


# ############################################## Plot candlestick chart #######################################################

# st.subheader('Candlestick Chart 🕯️')
# fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
#                 open=stock_data['Open'],
#                 high=stock_data['High'],
#                 low=stock_data['Low'],
#                 close=stock_data['Close'])])
# st.plotly_chart(fig, use_container_width=True)

# expander = st.expander("📋 Candlestick explanation")
# expander.write('''
#     A candlestick chart shows the price movement of a stock over time.
#     Each candlestick represents the open, high, low, and close prices for a specific period.
#     Green or white candlesticks indicate price gains, while red or black ones show price declines. 
#     It helps traders analyze trends and make decisions based on market sentiment and potential reversals.
# ''')

# expander.image("https://www.binaryoptions.com/wp-content/uploads/Candlestick-explained.png")


# st.write('---')


# ############################################## Short and Long Moving Averages #######################################################

# # Calculate moving averages
# short_window = 20
# long_window = 50
# stock_data['Short_MA'] = stock_data['Close'].rolling(window=short_window, min_periods=1).mean()
# stock_data['Long_MA'] = stock_data['Close'].rolling(window=long_window, min_periods=1).mean()


# # Plot moving averages
# st.subheader('Moving Averages')
# fig_ma = go.Figure()
# fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Short_MA'], mode='lines', name='Short Moving Average'))
# fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Long_MA'], mode='lines', name='Long Moving Average'))
# st.plotly_chart(fig_ma, use_container_width=True)


# expander = st.expander("Moving Averages explanation")
# expander.write('''
#     Short and long moving averages are tools used in technical analysis to smooth out price data and identify trends. 
#     The short moving average (SMA) calculates the average price over a short period, like 20 days, while the long moving average does the same over a longer period, such as 50 days. 
#     Traders use these averages to gauge the direction of the trend. 
#     When the short-term average crosses above the long-term one, it may signal a bullish trend, and vice versa for a bearish trend.
# ''')

# expander.image("https://scanz.com/wp-content/uploads/2018/12/movingaveragecrossovers.jpg")


# st.write('---')


# ############################################## Bollinger Bands #######################################################

# # Calculate Bollinger Bands
# window = 20
# stock_data['MA'] = stock_data['Close'].rolling(window=window).mean()
# stock_data['Std'] = stock_data['Close'].rolling(window=window).std()
# stock_data['Upper_band'] = stock_data['MA'] + (stock_data['Std'] * 2)
# stock_data['Lower_band'] = stock_data['MA'] - (stock_data['Std'] * 2)

# # Plot Bollinger Bands
# st.subheader(f'Bollinger Bands: {titles[ticker_index]} ({selected_ticker})')
# fig_bb = go.Figure()
# fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
# fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Upper_band'], mode='lines', line=dict(color='red'), name='Upper Band'))
# fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Lower_band'], mode='lines', line=dict(color='red'), name='Lower Band'))
# st.plotly_chart(fig_bb, use_container_width=True)


# expander = st.expander("Bollinger Bands explanation")
# expander.write('''
#     Bollinger Bands are a technical analysis tool used to measure volatility and identify potential price reversal points. 
#     They consist of three lines: a middle line, typically a simple moving average, and upper and lower bands that are a certain number of standard deviations away from the middle line.
#     The distance between the bands widens when volatility increases and narrows when it decreases.  
#     Traders often interpret price movements relative to the bands: when prices touch or exceed the outer bands, it may indicate overbought or oversold conditions, potentially signaling a reversal in the price trend.
# ''')

# expander.image("https://scanz.com/wp-content/uploads/2018/12/bollingerbands.jpg")


# st.write('---')


# ############################################## Daily Return #######################################################

# # Calculate daily returns
# stock_data['Daily_Return'] = stock_data['Close'].pct_change()

# st.subheader(f'Daily Return Distribution: {titles[ticker_index]} ({selected_ticker})')

# # Daily return explanation
# expander = st.expander("Daily Return explanation")
# expander.write('''
#     Daily return distribution shows how the returns of a financial asset vary over a single trading day, indicating the likelihood of different levels of gains or losses. 
#     It helps investors assess risk and potential profits.
# ''')

# expander.image("https://advfinangelinvestor.files.wordpress.com/2012/04/untitled2.png")


# # Plot daily return distribution histogram
# fig_hist = px.histogram(stock_data['Daily_Return'].dropna(), nbins=100)
# st.plotly_chart(fig_hist, use_container_width=True)

# # Key performance indicators
# average_daily_return = stock_data['Daily_Return'].mean()
# volatility = stock_data['Daily_Return'].std()


# st.write(f'Average Daily Return: {average_daily_return:.2%}')
# st.write(f'Volatility (Standard Deviation of Daily Returns): {volatility:.2%}')


# # Volatility explanation
# expander = st.expander("Standard deviation explanation")
# expander.write('''
#     The standard deviation (volatility in the financial sector) is a statistical measure that quantifies the degree of variation or dispersion in the returns of a financial asset. 
#     It reflects the extent to which the price of the asset deviates from its average or expected value over a given period.
#     High volatility implies greater uncertainty and risk, as prices can experience rapid and significant fluctuations, while low volatility indicates more stable and predictable price movements.
#     Investors often use volatility as a gauge of risk when making investment decisions and managing portfolios.
# ''')

# expander.image("https://www.k2analytics.co.in/wp-content/uploads/2020/05/standard-deviation.png")




# st.write('---')


# ############################################################################################


# def calculate_obv(stock_data):
#     obv = np.where(stock_data['Close'] > stock_data['Close'].shift(1), stock_data['Volume'], np.where(stock_data['Close'] < stock_data['Close'].shift(1), -stock_data['Volume'], 0)).cumsum()
#     return obv


# # Add OBV Chart
# obv = calculate_obv(stock_data)
# st.subheader('On-Balance Volume (OBV)')
# st.line_chart(obv)



# ############################################## Price-Volume #######################################################

# # Add Price-Volume Chart
# fig_price_volume = px.scatter(stock_data, x=stock_data.index, y='Close', size='Volume', labels={'x': 'Date', 'y': 'Price', 'size': 'Volume'})
# st.subheader('Price-Volume Chart')
# st.plotly_chart(fig_price_volume, use_container_width=True)




#except:
#st.write("No Data available")


