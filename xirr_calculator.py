import pandas as pd
import numpy as np
from scipy import optimize
import datetime
import argparse
import io
from datetime import datetime, timedelta
import warnings
import plotly.express as px
import plotly.graph_objects as go

def xirr(dates, cash_flows):
    """
    Calculate the Extended Internal Rate of Return (XIRR) for a series of cash flows.
    
    Args:
        dates: List of dates for each cash flow
        cash_flows: List of cash flow amounts (negative for outflows, positive for inflows)
    
    Returns:
        XIRR as a decimal (e.g., 0.124 for 12.4%)
    """
    if len(dates) != len(cash_flows):
        raise ValueError("Cash flows and dates must have the same length")
    
    if not any(cf < 0 for cf in cash_flows) or not any(cf > 0 for cf in cash_flows):
        return 0  # No investments or all investments with no returns
    
    # Convert dates to days from first date
    days = [(date - dates[0]).days for date in dates]
    
    def xirr_objective(rate):
        result = 0
        for i in range(len(days)):
            result += cash_flows[i] / (1 + rate) ** (days[i] / 365.0)
        return result
    
    # Use 0.1 (10%) as an initial guess
    try:
        return optimize.newton(xirr_objective, 0.1)
    except RuntimeError:
        # If Newton's method fails, try a more robust but slower method
        try:
            return optimize.brentq(xirr_objective, -0.999, 10)
        except ValueError:
            # If bounds are invalid, return NaN
            return np.nan

def match_trades_fifo(buys, sells):
    """
    Match buy and sell trades using FIFO (First In, First Out) method to track lots.
    
    Args:
        buys: DataFrame containing buy trades sorted by date
        sells: DataFrame containing sell trades sorted by date
        
    Returns:
        Tuple of (matched_lots, remaining_buys) where matched_lots is a list of matched trades
        and remaining_buys is a DataFrame of buys that haven't been matched yet
    """
    # Sort buys and sells by date
    buys = buys.sort_values('trade_date').reset_index(drop=True)
    sells = sells.sort_values('trade_date').reset_index(drop=True)
    
    remaining_buys = buys.copy()
    remaining_buys['remaining_quantity'] = remaining_buys['quantity']
    
    matched_lots = []
    
    for _, sell in sells.iterrows():
        sell_quantity = sell['quantity']
        sell_date = sell['trade_date']
        sell_price = sell['price']
        
        while sell_quantity > 0 and not remaining_buys.empty:
            # Get the oldest buy with remaining quantity
            buy = remaining_buys.iloc[0]
            
            # Determine quantity to match
            match_quantity = min(sell_quantity, buy['remaining_quantity'])
            
            # Calculate holding period for this lot
            holding_period = max(1, (sell_date - buy['trade_date']).days)
            
            # Calculate profit/loss for this lot
            buy_value = match_quantity * buy['price']
            sell_value = match_quantity * sell_price
            
            # Add to matched lots
            matched_lots.append({
                'buy_date': buy['trade_date'],
                'sell_date': sell_date,
                'quantity': match_quantity,
                'buy_price': buy['price'],
                'sell_price': sell_price,
                'buy_value': buy_value,
                'sell_value': sell_value,
                'holding_period': holding_period,
                'profit_loss': sell_value - buy_value
            })
            
            # Update remaining quantities
            sell_quantity -= match_quantity
            remaining_buys.at[0, 'remaining_quantity'] -= match_quantity
            
            # Remove buy if fully matched
            if remaining_buys.at[0, 'remaining_quantity'] <= 0:
                remaining_buys = remaining_buys.iloc[1:].reset_index(drop=True)
    
    # Keep only buys with remaining quantity
    remaining_buys = remaining_buys[remaining_buys['remaining_quantity'] > 0].reset_index(drop=True)
    return matched_lots, remaining_buys

def process_trades(df, currency='₹', short_term_days=7):
    """
    Process trading data to calculate performance metrics
    
    Args:
        df: DataFrame containing trade data
        currency: Currency symbol to use in formatting
        short_term_days: Number of days to consider a trade short-term
        
    Returns:
        Tuple of (results_df, overall_metrics, portfolio_cash_flows)
    """
    # Check for required columns
    required_columns = ['symbol', 'trade_date', 'quantity', 'price', 'trade_type']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Try to convert quantity and price to numeric
    for col in ['quantity', 'price']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check for missing values
    if df['quantity'].isna().any() or df['price'].isna().any():
        raise ValueError("Some quantity or price values couldn't be converted to numbers")
        
    # Try different date formats if the default fails
    try:
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%m/%d/%y')
    except:
        try:
            # Try ISO format (YYYY-MM-DD)
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d')
        except:
            try:
                # Try with automatic detection
                df['trade_date'] = pd.to_datetime(df['trade_date'])
            except:
                raise ValueError("Could not parse trade_date column. Please make sure it contains valid dates.")
    
    # Handle missing trade_id for duplicate removal
    if 'trade_id' not in df.columns:
        # Create a synthetic ID based on other columns
        df['trade_id'] = df['symbol'] + '_' + df['trade_date'].astype(str) + '_' + \
                         df['quantity'].astype(str) + '_' + df['price'].astype(drop=True)
    
    # Remove duplicates if any (same trade appearing in multiple files)
    df = df.drop_duplicates(subset=['trade_id', 'symbol', 'trade_date', 'quantity', 'price']).reset_index(drop=True)
    
    # Calculate transaction values
    df['transaction_value'] = df['quantity'] * df['price']
    
    # Create cash flow column (negative for buys, positive for sells)
    df['cash_flow'] = np.where(df['trade_type'] == 'buy', -df['transaction_value'], df['transaction_value'])
    
    # Group by symbol to process each stock
    results = []
    portfolio_cash_flows = []
    portfolio_dates = []
    matched_lots_all = []
    open_positions_count = 0
    partially_closed_count = 0
    fully_closed_count = 0
    holding_periods = []
    trade_dates = []
    
    for symbol, group in df.groupby('symbol'):
        # Filter to buys and sells
        buys = group[group['trade_type'] == 'buy']
        sells = group[group['trade_type'] == 'sell']
        
        # Skip if no buys
        if len(buys) == 0:
            continue
        
        # Calculate total quantities
        total_buy_qty = buys['quantity'].sum()
        total_sell_qty = sells['quantity'].sum()
        
        # Check if position is fully closed (buy qty = sell qty)
        is_fully_closed = abs(total_buy_qty - total_sell_qty) < 0.001  # Allow for tiny rounding differences
        if is_fully_closed:
            fully_closed_count += 1
        
        # Match trades using FIFO
        matched_lots, remaining_buys = match_trades_fifo(buys, sells)
        
        # Check if this is an open position (still holding some shares)
        is_open_position = not remaining_buys.empty
        if is_open_position:
            open_positions_count += 1
            
        # Process closed positions (have both buys and sells)
        has_closed_positions = len(matched_lots) > 0
        
        if has_closed_positions and is_open_position:
            partially_closed_count += 1
        
        # Calculate metrics based on matched lots and remaining buys
        if has_closed_positions:
            # Create a DataFrame for matched lots
            lots_df = pd.DataFrame(matched_lots)
            
            # Calculate metrics for closed positions
            closed_buy_value = lots_df['buy_value'].sum()
            closed_sell_value = lots_df['sell_value'].sum()
            closed_profit_loss = lots_df['profit_loss'].sum()
            
            # Calculate average buy price for closed positions (weighted by quantity)
            closed_quantity = lots_df['quantity'].sum()
            closed_avg_buy_price = (lots_df['buy_price'] * lots_df['quantity']).sum() / closed_quantity
            
            # Calculate weighted average holding period for closed positions
            avg_holding_period = (lots_df['holding_period'] * lots_df['quantity']).sum() / closed_quantity
            
            # Check if any lots have very short holding periods
            short_term_flag = any(period < short_term_days for period in lots_df['holding_period'])
            
            # Collect holding periods for visualization
            holding_periods.extend(lots_df['holding_period'].tolist())
            
            # Collect trade dates for timeline visualization
            trade_dates.extend(group['trade_date'].tolist())
        else:
            # No closed positions
            closed_buy_value = 0
            closed_sell_value = 0
            closed_profit_loss = 0
            closed_avg_buy_price = 0
            avg_holding_period = np.nan
            short_term_flag = False
        
        # Calculate metrics for open positions
        if is_open_position:
            open_buy_value = (remaining_buys['price'] * remaining_buys['remaining_quantity']).sum()
            open_quantity = remaining_buys['remaining_quantity'].sum()
            avg_open_buy_price = open_buy_value / open_quantity
            
            # Calculate holding period for open positions (using current date)
            current_date = pd.Timestamp.now()
            weighted_open_holding = 0
            for _, row in remaining_buys.iterrows():
                days_held = (current_date - row['trade_date']).days
                weighted_open_holding += days_held * row['remaining_quantity']
            avg_open_holding_period = weighted_open_holding / open_quantity if open_quantity > 0 else 0
        else:
            open_buy_value = 0
            open_quantity = 0
            avg_open_buy_price = 0
            avg_open_holding_period = 0
        
        # Total buy value (closed + open)
        total_buy_value = closed_buy_value + open_buy_value
        total_quantity = (closed_quantity if has_closed_positions else 0) + open_quantity
        
        # Calculate XIRR only for fully closed positions
        stock_xirr_pct = np.nan  # Default to NaN
        simple_annual_return = np.nan  # Default to NaN
        
        if has_closed_positions and is_fully_closed:
            # Only calculate for fully closed positions
            dates = []
            cash_flows = []
            
            # Use transaction data for cash flows
            for _, row in group.iterrows():
                dates.append(row['trade_date'])
                cash_flows.append(row['cash_flow'])
            
            # Calculate XIRR
            try:
                stock_xirr = xirr(dates, cash_flows)
                # Ensure XIRR is reasonable - limit extreme values
                if stock_xirr > 100:  # Cap extremely high values
                    stock_xirr = 100
                if stock_xirr < -0.9:  # Floor extremely negative values
                    stock_xirr = -0.9
                stock_xirr_pct = stock_xirr * 100 if not np.isnan(stock_xirr) else np.nan
                
                # Calculate simple annual return
                if avg_holding_period > 0:
                    annual_factor = 365 / avg_holding_period
                    simple_annual_return = (closed_profit_loss / closed_buy_value) * annual_factor * 100
                
                # Add cash flows to portfolio-level cash flows for overall XIRR
                if is_fully_closed:  # Only include fully closed positions in portfolio XIRR
                    for i in range(len(dates)):
                        portfolio_cash_flows.append({
                            'date': dates[i],
                            'cash_flow': cash_flows[i],
                            'weight': abs(cash_flows[i])
                        })
            except:
                stock_xirr_pct = np.nan
                simple_annual_return = np.nan
        
        # Determine status
        if is_fully_closed:
            status = "Fully Closed"
        elif is_open_position and has_closed_positions:
            status = "Partially Closed"
        elif is_open_position:
            status = "Open"
        else:
            status = "Error"  # More sells than buys
            # Invalidate holding period for error cases
            avg_holding_period = np.nan
        
        # Add to results
        results.append({
            'Symbol': symbol,
            'Status': status,
            'Total Buy Value': total_buy_value,
            'Closed Buy Value': closed_buy_value,
            'Sell Value': closed_sell_value,
            'Realized P&L': closed_profit_loss,
            'Avg Buy Price': closed_avg_buy_price if has_closed_positions else avg_open_buy_price,
            'Avg Holding Period (days)': avg_holding_period if has_closed_positions else avg_open_holding_period,
            'XIRR (%)': stock_xirr_pct,  # Will be NaN for non-fully closed positions
            'Simple Annual Return (%)': simple_annual_return,  # Will be NaN for non-fully closed positions
            'Short Term Flag': short_term_flag,
            'Open Quantity': open_quantity,
            'Total Buy Qty': total_buy_qty,
            'Total Sell Qty': total_sell_qty,
            'Fully Closed': is_fully_closed
        })
        
        # Add collected data to results
        matched_lots_all.extend(matched_lots)
        if has_closed_positions:
            portfolio_dates.append(dates[-1])
    
    # Transform results into a DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate overall weighted XIRR for closed positions only
    if portfolio_cash_flows:
        # Prepare data for XIRR calculation
        pcf_df = pd.DataFrame(portfolio_cash_flows)
        dates = pcf_df['date'].tolist()
        cash_flows = pcf_df['cash_flow'].tolist()
        
        # Calculate portfolio XIRR
        try:
            portfolio_xirr = xirr(dates, cash_flows)
            
            # Apply reasonable limits
            if portfolio_xirr > 100:
                portfolio_xirr = 100
            if portfolio_xirr < -0.9:
                portfolio_xirr = -0.9
                
            portfolio_xirr_pct = portfolio_xirr * 100 if not np.isnan(portfolio_xirr) else np.nan
        except Exception as e:
            print(f"Error calculating portfolio XIRR: {str(e)}")
            portfolio_xirr_pct = np.nan
    else:
        portfolio_xirr_pct = np.nan
    
    # Compile overall metrics with the properly calculated XIRR
    overall_metrics = {
        'Total Buy Value': results_df['Total Buy Value'].sum(),
        'Closed Buy Value': results_df['Closed Buy Value'].sum(),
        'Total Sell Value': results_df['Sell Value'].sum(),
        'Total Realized P&L': results_df['Realized P&L'].sum(),
        'Avg Holding Period (days)': results_df['Avg Holding Period (days)'].mean(),
        'Overall XIRR (%)': portfolio_xirr_pct,  # Add the calculated portfolio XIRR
        'Stocks with Short Term Trades': results_df['Short Term Flag'].sum(),
        'Open Positions': open_positions_count,
        'Partially Closed Positions': partially_closed_count,
        'Fully Closed Positions': fully_closed_count,
        'holding_periods': holding_periods,
        'trade_dates': trade_dates,
        'matched_lots': matched_lots_all,
        'portfolio_dates': portfolio_dates,
        'portfolio_cash_flows': portfolio_cash_flows
    }
    
    return results_df, overall_metrics, portfolio_cash_flows

def create_ui():
    """
    Create Streamlit UI for the XIRR calculator
    """
    # Import streamlit here so it's only needed when create_ui is called
    import streamlit as st
    
    # Set page configuration
    st.set_page_config(
        page_title="XIRR Portfolio Calculator",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Define default values
    currency = "₹"  # Default currency symbol
    short_term_threshold = 7  # Default days threshold for short-term trades
    
    # Add custom CSS with improved aesthetics
    st.markdown("""
    <style>
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            text-align: center;
        }
        .metric-label {
            font-size: 14px;
            font-weight: 500;
            color: #6c757d;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            color: #212529;
            margin: 8px 0;
        }
        .metric-delta {
            font-size: 14px;
            font-weight: 500;
        }
        .positive {
            color: #28a745;
        }
        .negative {
            color: #dc3545;
        }
        .feature-box {
            background-color: #f8f9fa;
            border-left: 4px solid #3949ab;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 4px;
            height: 100%;  /* Make all boxes the same height */
        }
        .feature-title {
            font-weight: 600;
            color: #3949ab;
            margin-bottom: 8px;
        }
        .step {
            display: flex;
            align-items: flex-start;
            margin-bottom: 20px;
        }
        .step-number {
            background-color: #3949ab;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
            flex-shrink: 0;
            font-weight: bold;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e9ecef;
            text-align: center;
            color: #6c757d;
            font-size: 14px;
        }
        /* Colorful buttons */
        .stButton > button {
            background-color: #3949ab !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important;
            padding: 8px 16px !important;
            font-weight: 500 !important;
            transition: all 0.3s !important;
        }
        .stButton > button:hover {
            background-color: #283593 !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
            transform: translateY(-1px) !important;
        }
        .stButton > button:active {
            background-color: #1a237e !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            transform: translateY(0) !important;
        }
        /* Background for upload section */
        .upload-section {
            background: linear-gradient(to bottom right, #f5f7ff, #e8eeff);
            padding: 30px;
            border-radius: 10px;
            margin: 20px 0;
        }
        /* Make download button green */
        .stDownloadButton > button {
            background-color: #4CAF50 !important;
        }
        .stDownloadButton > button:hover {
            background-color: #45a049 !important;
        }
        /* Additional table styling */
        .profit {
            color: #28a745 !important;
            font-weight: 500 !important;
        }
        .loss {
            color: #dc3545 !important;
            font-weight: 500 !important;
        }
        /* Enhanced table styling */
        .dataframe {
            border-collapse: collapse;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .dataframe th {
            background-color: #f8f9fa;
            color: #495057;
            font-weight: 600;
            text-align: left;
            padding: 8px 12px;
            border-bottom: 2px solid #dee2e6;
        }
        .dataframe td {
            padding: 8px 12px;
            border-bottom: 1px solid #e9ecef;
        }
        .dataframe tr:hover {
            background-color: rgba(0, 0, 0, 0.02);
        }
        /* Highlight table values more prominently */
        .profit {
            font-weight: 600 !important;
            color: #04724d !important;
        }
        .loss {
            font-weight: 600 !important;
            color: #c0392b !important;
        }
        /* Improved styling for metrics with consistent heights and better alignment */
        div[data-testid="metric-container"] {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px 10px;
            box-shadow: 0 1px 6px rgba(0, 0, 0, 0.05);
            width: 100%;
            min-height: 120px;
        }
        div[data-testid="stHorizontalBlock"] {
            gap: 10px;
        }
        div[data-testid="metric-container"] label {
            font-weight: 500;
            color: #555;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            font-size: 22px;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # TITLE SECTION
    st.markdown("<h1 style='text-align: center; color: #3949ab;'>XIRR Portfolio Calculator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; margin-bottom: 30px;'>Calculate accurate returns on your stock investments using Extended Internal Rate of Return</p>", unsafe_allow_html=True)
    
    # WHY THIS TOOL SECTION
    st.markdown("## Why This Tool")

    st.markdown("""
    I created this tool because I could never find a way to calculate the XIRR of my individual stock investments. 

    While most brokers provide XIRR calculations for mutual funds, none offered this crucial metric for individual stocks I bought and sold. Without XIRR, it was nearly impossible to compare my stock picks against other investment options like mutual funds or fixed deposits.

    This tool fills that gap by calculating the true time-weighted returns (XIRR) for each stock in your portfolio, giving you an accurate picture of your actual investment performance. Now you can see which of your stock picks truly outperformed the market, accounting for the timing and size of all your buy and sell transactions.
    """)
    
    # KEY FEATURES SECTION
    st.markdown("## Key Features")

    # Use a much simpler approach with native Streamlit components
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Accurate Return Calculation")
        st.markdown("Uses XIRR method to account for the timing and size of your cash flows, giving you a true picture of your investment performance.")
        
        st.markdown("### 🔍 Stock-by-Stock Analysis")
        st.markdown("See detailed metrics for each stock in your portfolio, including XIRR, profit/loss, and holding periods.")

    with col2:
        st.markdown("### 📈 Trading Pattern Insights")
        st.markdown("Visualize your win/loss ratios, trading frequency, and performance patterns over time.")
        
        st.markdown("### 📋 Position Status Tracking")
        st.markdown("Track open, partially closed, and fully closed positions with comprehensive performance metrics.")
    
    # Re-add HOW IT WORKS SECTION as collapsible
    st.markdown("## How It Works")

    # Use Streamlit's expander component to make it collapsible
    with st.expander("View Step-by-Step Guide", expanded=False):
        st.markdown("""
        1. **Prepare Your Trade Data**
           * Organize your stock trades in a CSV file with these columns:
           * `symbol` - Stock ticker symbol
           * `trade_date` - Date of the transaction (MM/DD/YY)
           * `quantity` - Number of shares bought/sold
           * `price` - Price per share
           * `trade_type` - Either 'buy' or 'sell'

        2. **Upload Your Data**
           * Click the "Upload CSV File" button below
           * Select your trade data file
           * The tool automatically processes your trades

        3. **Analyze Performance**
           * Review your portfolio metrics
           * See detailed stock-by-stock analysis
           * Visualize your trading patterns and returns
           * Understand your current allocation

        ### Using Zerodha or Other Broker Data

        If you use Zerodha, you can easily export your trade data:
        1. Log in to your Zerodha Console
        2. Go to Tax P&L Reports section
        3. Download your trades as a CSV
        4. Upload the file directly to this tool

        For other brokers, ensure your data includes the columns mentioned above. Most broker platforms allow exporting trade history in CSV format that can be easily adapted to work with this tool.

        > **Note:** This tool does not store your data. All processing is done in your browser, keeping your financial information private and secure.
        """)
    
    # UPLOAD SECTION WITH BACKGROUND
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    st.markdown("### Upload Your Tradebook")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose CSV file(s) with your trading history", 
        type="csv", 
        accept_multiple_files=True,
        help="Upload multiple tradebook CSV files - they'll be automatically consolidated. All processing happens locally for privacy."
    )
    
    # File info
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded. The calculator will combine all data for analysis.")
    else:
        st.info("Upload multiple tradebook CSV files - they'll be automatically consolidated. All processing happens locally for privacy.")
    
    # Options
    with st.expander("Calculation Options"):
        col1, col2 = st.columns(2)
        with col1:
            currency = st.text_input("Currency Symbol", "₹", help="The currency symbol to display in reports")
        with col2:
            short_term_threshold = st.number_input(
                "Short-term Trade Threshold (days)", 
                min_value=1, 
                value=7,
                help="Trades held for less than this many days will be flagged as short-term trades"
            )
    
    # Calculate button
    st.markdown("<div style='text-align:center; margin-top:30px; margin-bottom:20px;'>", unsafe_allow_html=True)
    calculate_button = st.button("Calculate Portfolio Performance", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)  # Close upload-section
    
    # DATA PROCESSING SECTION
    if uploaded_files and calculate_button:
        try:
            with st.spinner('Processing your trading data...'):
                # Combine data from all uploaded files
                all_data = []
                
                for uploaded_file in uploaded_files:
                    # Read CSV file
                    df = pd.read_csv(uploaded_file)
                    # Add to combined data
                    all_data.append(df)
                
                # Combine all data frames
                if all_data:
                    combined_df = pd.concat(all_data, ignore_index=True)
                    
                    # Process the combined data
                    results_df, overall_metrics, portfolio_cash_flows = process_trades(combined_df, currency, short_term_threshold)
                    
                    # Show success message
                    st.success("Your portfolio data has been processed successfully!")
                    
                    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
                    
                    # Data visualization section
                    st.header("Portfolio Insights")
                    
                    # Prepare chart data for later use
                    if 'results_df' in locals() and not results_df.empty:
                        # Filter to include only rows with valid XIRR values for charts
                        chart_df = results_df.dropna(subset=['XIRR (%)']).copy()
                        
                        # Filter out extreme XIRR values that skew charts (>5000%)
                        if not chart_df.empty:
                            chart_df = chart_df[chart_df['XIRR (%)'] <= 5000]
                    
                    # Create main tabs with new organization
                    tab_performance, tab_stock_details, tab_allocation = st.tabs([
                        "Portfolio Performance", 
                        "Stock Wise Details",
                        "Portfolio Allocation"
                    ])
                    
                    # Tab 1: Portfolio Performance
                    with tab_performance:
                        # Use a cleaner approach with minimal styling
                        st.markdown("""
                        <style>
                        /* Very minimal styling for metrics */
                        div[data-testid="metric-container"] {
                            background-color: #f8f9fa;
                            border-radius: 6px;
                            padding: 10px;
                            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # Use a simple 2x4 grid with Streamlit's native columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Left column metrics
                            st.metric("💰 Total Buy Value", f"{currency}{overall_metrics['Total Buy Value']:,.2f}")
                            
                            profit_loss = overall_metrics['Total Realized P&L']
                            delta_color = "normal" if profit_loss >= 0 else "inverse"
                            icon = "📈" if profit_loss >= 0 else "📉"
                            delta = f"{profit_loss/overall_metrics['Closed Buy Value']*100:.2f}%" if overall_metrics['Closed Buy Value'] > 0 else None
                            st.metric(f"{icon} Total P&L", f"{currency}{profit_loss:,.2f}", delta=delta, delta_color=delta_color)
                            
                            st.metric("🔓 Open Positions", f"{overall_metrics['Open Positions']}")
                            
                            st.metric("⚡ Short-Term Trades", f"{overall_metrics['Stocks with Short Term Trades']}")
                        
                        with col2:
                            # Right column metrics
                            st.metric("💵 Total Sell Value", f"{currency}{overall_metrics['Total Sell Value']:,.2f}")
                            
                            xirr_val = overall_metrics['Overall XIRR (%)']
                            if not np.isnan(xirr_val):
                                st.metric("📊 Portfolio XIRR", f"{xirr_val:.2f}%")
                            else:
                                st.metric("📊 Portfolio XIRR", "Not available")
                            
                            st.metric("🔒 Closed Positions", f"{overall_metrics['Fully Closed Positions']}")
                            
                            st.metric("⏱️ Avg Holding Period", f"{overall_metrics['Avg Holding Period (days)']:.1f} days")
                    
                    # Tab 2: Stock Wise Details - with sub-tabs and pruned columns
                    with tab_stock_details:
                        if not results_df.empty:
                            # Reorder tabs as requested: fully closed, partially closed, open, error
                            subtab_closed, subtab_partial, subtab_open, subtab_error = st.tabs([
                                "Fully Closed", 
                                "Partially Closed",
                                "Open Positions", 
                                "Errors"
                            ])
                            
                            # Enhanced color highlighting for XIRR values with intensity gradients
                            def highlight_pl(val):
                                """Highlight function with color intensity based on value magnitude"""
                                try:
                                    # Convert string values to float if needed
                                    if isinstance(val, str):
                                        # Try to extract numeric value from formatted string
                                        val = float(val.replace('%', '').replace(currency, '').replace(',', ''))
                                    
                                    if isinstance(val, (int, float)):
                                        if val == 0:
                                            return 'color: #808080;'  # Gray for zero
                                        elif val > 0:
                                            # Green with intensity scaling
                                            intensity = min(1.0, abs(val) / 50.0)  # Scale factor - 50% XIRR is full intensity
                                            r = int(4 + (1-intensity) * 250)  # Lower red for higher intensity
                                            g = int(114 + (1-intensity) * 141)  # Adjust green component
                                            b = int(77 + (1-intensity) * 178)  # Lower blue for higher intensity
                                            return f'color: rgb({r},{g},{b}); font-weight: {500 + int(intensity * 300)};'
                                        else:
                                            # Red with intensity scaling
                                            intensity = min(1.0, abs(val) / 50.0)  # Scale factor - 50% XIRR is full intensity
                                            r = int(192 + (1-intensity) * 63)  # Keep red component high
                                            g = int(57 + (1-intensity) * 198)  # Lower green for higher intensity
                                            b = int(43 + (1-intensity) * 212)  # Lower blue for higher intensity
                                            return f'color: rgb({r},{g},{b}); font-weight: {500 + int(intensity * 300)};'
                                except:
                                    pass
                                return ''
                            
                            # Define a unified set of columns for all position types
                            unified_columns = [
                                'Symbol',
                                'Status',
                                'Total Buy Value',
                                'Total Buy Qty',
                                'Total Sell Qty',
                                'Open Quantity',
                                'Realized P&L',
                                'XIRR (%)',
                                'Avg Holding Period (days)'
                            ]
                            
                            # Improved function to format the dataframe with strict 2 decimal places
                            def format_position_table(df, unified_columns):
                                # Make a copy to avoid modifying the original
                                display_df = df.copy()
                                
                                # Only include columns that exist in the source dataframe
                                available_columns = [col for col in unified_columns if col in df.columns]
                                display_df = display_df[available_columns].copy()
                                
                                # Format all numeric columns to exactly 2 decimal places
                                for col in display_df.columns:
                                    if col not in ['Symbol', 'Status']:  # Skip non-numeric columns
                                        # Force format to 2 decimal places
                                        display_df[col] = display_df[col].apply(
                                            lambda x: f"{float(x):.2f}" if isinstance(x, (int, float)) else x
                                        )
                                
                                return display_df
                            
                            # Now update each subtab with the same column set (tab order now changed)
                            
                            # 1. Fully Closed
                            with subtab_closed:
                                closed_df = results_df[results_df['Status'] == 'Fully Closed'].copy()
                                if not closed_df.empty:
                                    closed_display = format_position_table(closed_df, unified_columns)
                                    
                                    # Apply the enhanced styling with color gradients
                                    styled_df = closed_display.style.applymap(
                                        highlight_pl, subset=['Realized P&L', 'XIRR (%)']
                                    )
                                    
                                    # Additional styling for better visual hierarchy
                                    st.dataframe(styled_df, use_container_width=True)
                                else:
                                    st.info("No fully closed positions in your portfolio.")
                            
                            # 2. Partially Closed
                            with subtab_partial:
                                partial_df = results_df[results_df['Status'] == 'Partially Closed'].copy()
                                if not partial_df.empty:
                                    partial_display = format_position_table(partial_df, unified_columns)
                                    st.dataframe(partial_display.style.applymap(
                                        highlight_pl, subset=['Realized P&L', 'XIRR (%)']
                                    ), use_container_width=True)
                                else:
                                    st.info("No partially closed positions in your portfolio.")
                            
                            # 3. Open positions 
                            with subtab_open:
                                open_df = results_df[results_df['Status'] == 'Open'].copy()
                                if not open_df.empty:
                                    open_display = format_position_table(open_df, unified_columns)
                                    st.dataframe(open_display.style.applymap(
                                        highlight_pl, subset=['Realized P&L', 'XIRR (%)']
                                    ), use_container_width=True)
                                else:
                                    st.info("No open positions in your portfolio.")
                            
                            # 4. Errors
                            with subtab_error:
                                error_df = results_df[results_df['Status'] == 'Error'].copy()
                                if not error_df.empty:
                                    error_display = format_position_table(error_df, unified_columns)
                                    st.dataframe(error_display.style.applymap(
                                        highlight_pl, subset=['Realized P&L', 'XIRR (%)']
                                    ), use_container_width=True)
                                else:
                                    st.info("No positions with errors found.")
                            
                            # Export button for full data (below tabs)
                            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "Download Complete Stock Details CSV",
                                csv,
                                "portfolio_performance.csv",
                                "text/csv",
                                key='download-csv'
                            )

                            # Add a visual separator
                            st.markdown("---")

                            # XIRR by stock (bar chart)
                            st.subheader("XIRR by Stock")
                            if not chart_df.empty:
                                # Create a DataFrame with sorted XIRR values
                                sorted_df = chart_df.sort_values('XIRR (%)', ascending=False).copy()
                                
                                # Create bar chart with conditional coloring
                                fig = px.bar(
                                    sorted_df, 
                                    x='Symbol', 
                                    y='XIRR (%)',
                                    color='XIRR (%)',
                                    color_continuous_scale=['#ef476f', '#f8f9fa', '#06d6a0']
                                )
                                
                                fig.update_layout(
                                    xaxis_title="Stock Symbol",
                                    yaxis_title="XIRR (%)",
                                    xaxis={'categoryorder':'total descending'}
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                if len(chart_df) < len(results_df.dropna(subset=['XIRR (%)'])):
                                    st.info("Some stocks with extremely high XIRR (>5000%) have been filtered from charts to prevent visual skewing.")

                            # P&L by stock (bar chart)
                            st.subheader("Realized Profit/Loss")
                            if 'Realized P&L' in results_df.columns:
                                # Create a DataFrame for P&L visualization
                                pl_df = results_df[['Symbol', 'Realized P&L']].copy()
                                
                                # Check if Realized P&L is already numeric or needs conversion
                                if pl_df['Realized P&L'].dtype == 'object':  # It's a string
                                    # Convert from formatted string to numeric
                                    pl_df['Realized P&L'] = pd.to_numeric(pl_df['Realized P&L'].str.replace(currency, '').str.replace(',', ''))
                                
                                # Sort by P&L
                                pl_df = pl_df.sort_values('Realized P&L', ascending=False)
                                
                                # Create bar chart with conditional coloring
                                fig = px.bar(
                                    pl_df, 
                                    x='Symbol', 
                                    y='Realized P&L',
                                    color='Realized P&L',
                                    color_continuous_scale=['#ef476f', '#f8f9fa', '#06d6a0']
                                )
                                
                                fig.update_layout(
                                    xaxis_title="Stock Symbol",
                                    yaxis_title=f"Realized P&L ({currency})",
                                    xaxis={'categoryorder':'total descending'}
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Tab 3: Portfolio Allocation - simplified and fixed
                    with tab_allocation:
                        st.subheader("Current Portfolio Allocation")
                        
                        # Add disclaimer
                        st.info("This section shows the allocation of your current holdings only (open positions and partially closed positions).")
                        
                        if not results_df.empty:
                            # Only include positions that are currently held (Open or Partially Closed)
                            current_positions = results_df[
                                (results_df['Status'] == 'Open') | 
                                (results_df['Status'] == 'Partially Closed')
                            ].copy()
                            
                            # Filter to only include positions with shares still held
                            current_positions = current_positions[current_positions['Open Quantity'] > 0]
                            
                            if not current_positions.empty:
                                # Calculate current value using Total Buy Value / Total Buy Qty as the price
                                # This gives us an approximation of current value based on purchase price
                                current_positions['Estimated Current Value'] = current_positions.apply(
                                    lambda row: row['Open Quantity'] * (row['Total Buy Value'] / row['Total Buy Qty']) 
                                    if row['Total Buy Qty'] > 0 else 0, 
                                    axis=1
                                )
                                
                                # Create improved pie chart
                                fig = px.pie(
                                    current_positions, 
                                    values='Estimated Current Value', 
                                    names='Symbol',
                                    color_discrete_sequence=px.colors.qualitative.Bold,  # More vibrant color scheme
                                    hole=0.3  # Add a hole to make it a donut chart (more modern)
                                )
                                
                                # Improve pie chart formatting
                                fig.update_traces(
                                    textposition='inside', 
                                    textinfo='percent+label',
                                    textfont_size=12,
                                    pull=[0.05] * len(current_positions),  # Slightly pull all slices
                                    marker=dict(line=dict(color='white', width=2))  # Add white borders
                                )
                                
                                fig.update_layout(
                                    legend=dict(orientation="h", yanchor="bottom", y=-0.3),  # Move legend to bottom
                                    margin=dict(t=60, b=60, l=20, r=20)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show allocation table with cleaner layout
                                st.subheader("Current Holdings Details")
                                
                                # Create a clean allocation table with only relevant columns
                                allocation_table = current_positions[['Symbol', 'Open Quantity']].copy()
                                
                                # Add estimated value
                                allocation_table['Estimated Value'] = current_positions['Estimated Current Value']
                                
                                # Calculate percentage allocation
                                total_value = allocation_table['Estimated Value'].sum()
                                allocation_table['Allocation (%)'] = (allocation_table['Estimated Value'] / total_value) * 100
                                
                                # Format the values for display
                                # Convert to 2 decimal places for all numeric columns
                                for col in allocation_table.columns:
                                    if col != 'Symbol':
                                        allocation_table[col] = allocation_table[col].apply(
                                            lambda x: f"{float(x):.2f}" if isinstance(x, (int, float)) else x
                                        )
                                
                                # Format currency values
                                allocation_table['Estimated Value'] = allocation_table['Estimated Value'].apply(
                                    lambda x: f"{currency}{float(x.replace(currency, '').replace(',', '')):,.2f}" 
                                    if isinstance(x, str) else f"{currency}{float(x):,.2f}"
                                )
                                
                                # Format percentages
                                allocation_table['Allocation (%)'] = allocation_table['Allocation (%)'].apply(
                                    lambda x: f"{float(x.replace('%', '')):,.2f}%" 
                                    if isinstance(x, str) else f"{float(x):,.2f}%"
                                )
                                
                                # Sort by allocation percentage (descending)
                                allocation_table = allocation_table.sort_values('Allocation (%)', ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
                                
                                # Display the clean table
                                st.dataframe(allocation_table, use_container_width=True)
                            else:
                                st.warning("No current holdings found in your portfolio data.")
                        else:
                            st.info("No allocation data available.")
                    
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.exception(e)  # This will display the traceback
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>Disclaimer:</strong> This tool is for informational purposes only and should not be considered financial advice. 
        Past performance is not indicative of future results. Use at your own risk.</p>
        <p>Created by: Anshul Khandelwal</p>
        <p>Reach out on LinkedIn for feedback: <a href="https://www.linkedin.com/in/anshulkhandelwal/" target="_blank">linkedin.com/in/anshulkhandelwal</a></p>
        <p>© 2023 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Check if streamlit is available
    try:
        import streamlit
        # If run directly, create the UI
        create_ui()
    except ImportError:
        # If streamlit is not available, use command line version
        import sys
        if len(sys.argv) > 1:
            # Use the run_xirr functionality
            from run_xirr import main
            main()
        else:
            print("Streamlit not found. Please install with: pip install streamlit")
            print("Alternatively, run with a file path: python xirr_calculator.py your_file.csv") 
