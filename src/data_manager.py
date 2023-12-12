import pandas as pd
import pickle


def read_csv(file_path, columns=None):
    try:
        if columns is None:
            return pd.read_csv(file_path)
        else:
            return pd.read_csv(file_path, usecols=columns)
    except Exception as e:
        print(f"Fail to load csv: {e}")
        exit(1)


def save_csv(data, file_path):
    try:
        data.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"File saved successfully at {file_path}")
    except Exception as e:
        print(f"Error saving file: {e}")


def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pkl(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    data = read_csv('../data/hotel_data.csv')
    hotel_dict = pd.Series(data.NAME.values, index=data.HOTEL_ID).to_dict()
    print(hotel_dict)

    data = read_csv('../data/review_data.csv')
    review_dict = pd.Series(data.COMMENT.values, index=data.HOTEL_ID).to_dict()
    print(review_dict)
