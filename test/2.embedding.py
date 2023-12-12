from sentence_transformers import SentenceTransformer
from collections import defaultdict

from src.data_manager import read_csv, save_pkl

if __name__ == '__main__':
    # Model ref: https://github.com/snunlp/KR-SBERT
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

    data = read_csv('../data/data.csv').dropna()
    print(data.head())

    # A dictionary storing lists of embedded tensors for each hotel
    embedding_dict = defaultdict(list)

    # embedding and append to list
    for row in data.itertuples():
        try:
            tensor = model.encode(row.HOTEL_REVIEW, convert_to_tensor=True)
            embedding_dict[row.HOTEL_ID].append(tensor)
        except Exception as e:
            print(row, e)
            exit(1)

    print(embedding_dict)

    save_pkl(embedding_dict, '../data/embedding_dict.pkl')
