import pandas as pd
from src.data_manager import read_csv, save_csv
from src.tokenizer import tokenize

if __name__ == "__main__":
    review_data_path = '../data/review_data.csv'
    hotel_data_path = '../data/hotel_data.csv'

    # Load dataset
    review_data = read_csv(review_data_path, ['HOTEL_ID', 'COMMENT'])
    hotel_data = read_csv(hotel_data_path, ['HOTEL_ID', 'NAME'])

    # Fill NA data
    review_data['COMMENT'] = review_data['COMMENT'].fillna('')

    # Merge data ( HOTEL_ID | HOTEL_NAME | HOTEL_REVIEW )
    data = pd.merge(hotel_data, review_data, on='HOTEL_ID')

    # Rename columns
    data = data.rename(columns={'HOTEL_ID': 'HOTEL_ID', 'NAME': 'HOTEL_NAME', 'COMMENT': 'HOTEL_REVIEW'})

    # Apply tokenize
    data['HOTEL_REVIEW'] = data['HOTEL_REVIEW'].apply(tokenize)

    # Fix wrong hotel name
    data['HOTEL_NAME'] = data['HOTEL_NAME'].replace('"', '')

    # Drop NA data
    data['HOTEL_REVIEW'] = data['HOTEL_REVIEW'].replace('', None)
    data = data.dropna()

    # Save file
    save_csv(data, '../data/data.csv')
