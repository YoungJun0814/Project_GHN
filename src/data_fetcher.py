import yfinance as yf
import pandas as pd
import numpy as np
import os

def fetch_ghn_data(start_date="2005-01-01", end_date=None):
    """
    Downloads and preprocesses global financial data for the GHN (Global Hydraulic Network) project.
    
    The dataset covers approx. 20 years (2005-Present) to capture:
    1. The 2008 Financial Crisis (Slow Contagion)
    2. The 2020 COVID-19 Crash (Instant Contagion)
    3. The 2022-2025 High Interest Rate Regime (Hydraulic Pressure)

    The selected assets represent the 'Golden 8' nodes of the hydraulic system:
    - Source (Pressure): US S&P500 (^GSPC)
    - Gravity (Potential): US 10Y Bond Yield (^TNX)
    - Valve (Resistance): Dollar Index (DX-Y.NYB)
    - Viscosity (Friction): Crude Oil (CL=F)
    - Targets (Reservoirs): KOSPI (^KS11), Nikkei (^N225), Hang Seng (^HSI)
    - Bridge (Pipe): DAX (^GDAXI)
    
    Args:
        start_date (str): Start date for data download (YYYY-MM-DD). Default is '2005-01-01'.
        end_date (str): End date (YYYY-MM-DD). Default is None (Today).
        
    Returns:
        pd.DataFrame: A cleaned, time-aligned DataFrame of global asset prices.
    """
    
    # 1. Define Global Tickers (Hydraulic Nodes)
    tickers = {
        'US_SP500': '^GSPC',      # Source of Global Liquidity
        'US_Bond10Y': '^TNX',     # Gravity (Risk-free Rate)
        'Dollar_Idx': 'DX-Y.NYB', # Valve (Currency Strength)
        'Crude_Oil': 'CL=F',      # Viscosity (Energy Cost)
        'KR_KOSPI': '^KS11',      # Target 1: South Korea
        'JP_Nikkei': '^N225',     # Target 2: Japan
        'HK_HangSeng': '^HSI',    # Target 3: Hong Kong
        'DE_DAX': '^GDAXI'        # Bridge: Germany (Europe)
    }

    print(f">>> [GHN] Starting data download from {start_date}...")
    
    data_frames = []

    # 2. Iterative Download
    for name, ticker in tickers.items():
        try:
            # Download daily data
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            # Extract 'Close' price safely (handling MultiIndex issues in new yfinance)
            if isinstance(df.columns, pd.MultiIndex):
                series = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
                if isinstance(series, pd.DataFrame): 
                    series = series.iloc[:, 0] 
            else:
                series = df['Close']

            series.name = name
            data_frames.append(series)
            print(f"    - Fetched: {name} ({ticker}) | Rows: {len(series)}")
            
        except Exception as e:
            print(f"    [!] Error fetching {name}: {e}")

    # 3. Merge & Align (Hydraulic System Construction)
    # Use 'outer' join to preserve all timestamps initially
    global_df = pd.concat(data_frames, axis=1)
    
    # 4. Handle Missing Values (Holidays & Time Zones)
    # Logic: Forward Fill (ffill)
    # Reason: If a market is closed (e.g., holiday), the 'pressure' (price) remains 
    # at the last known level until the market reopens.
    global_df = global_df.ffill()
    
    # Drop initial rows with NaNs (due to different start dates or holidays at the beginning)
    global_df = global_df.dropna()

    # 5. Save Raw Data
    # Saves to 'Thesis/Project_GHN/data/ghn_raw_data.csv'
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    file_path = os.path.join(save_dir, 'ghn_raw_data.csv')
    global_df.to_csv(file_path)
    
    print("-" * 60)
    print(f">>> [GHN] Data processing complete.")
    print(f"    - Shape: {global_df.shape} (Rows, Nodes)")
    print(f"    - Period: {global_df.index.min().date()} ~ {global_df.index.max().date()}")
    print(f"    - Saved to: {file_path}")
    print("-" * 60)
    
    return global_df

if __name__ == "__main__":
    fetch_ghn_data()