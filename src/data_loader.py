import pandas as pd

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    price_cols = ['open', 'high', 'low', 'close', 'next_weeks_open', 'next_weeks_close']
    for col in price_cols:
        df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

    df['volume'] = df['volume'].replace(',', '', regex=True).astype(int)

    percent_cols = ['percent_change_price', 'percent_change_volume_over_last_wk', 'percent_change_next_weeks_price']
    for col in percent_cols:
        df[col] = df[col].astype(float)

    df['date'] = pd.to_datetime(df['date'])

    columns = [
        'quarter', 'stock', 'date', 'percent_change_price', 'percent_change_volume_over_last_wk',
        'days_to_next_dividend', 'percent_return_next_dividend', 'percent_change_next_weeks_price'
    ]
    return df[columns].copy()
