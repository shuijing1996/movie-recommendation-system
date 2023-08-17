"""Use SentenceTransformer on Overview + Genre."""
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler


def run():
    """Runner function."""
    file_path = 'data/Sr._Data_Scientist_Assessment_Dataset.csv'
    output_path = file_path.replace('.csv', '_embedding.csv')
    min_overview_len = 5

    #check gpu availability
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print('=== reading in data')
    data = pd.read_csv(file_path, lineterminator='\n')

    print(
        f'=== removing records that have less than {min_overview_len} words in Overview'
    )
    data['overview_len'] = data.Overview.apply(lambda v: len(v.split(' ')))
    data = data[data.overview_len >= min_overview_len]
    data.reset_index(inplace=True, drop=True)

    print(
        '=== standardize numeric features: Popularity, Vote_Count, Vote_Average'
    )
    scaler = MinMaxScaler()
    data_scaler = pd.DataFrame(
        scaler.fit_transform(data[['Popularity', 'Vote_Count',
                                   'Vote_Average']]),
        columns=['Popularity_stand', 'Vote_Count_stand', 'Vote_Average_stand'])
    data = data.join(data_scaler)

    print('=== appending Genre after Overview')
    data['overview_w_genre'] = data.apply(
        lambda v: v['Overview'] + ' ' + v['Genre'], axis=1)

    print('=== getting embedding for overview_w_genre')
    embedder = SentenceTransformer('all-mpnet-base-v2', device=device)
    corpus = data.overview_w_genre.tolist()
    corpus_embeddings = embedder.encode(corpus)
    corpus_embeddings_ls = [list(c) for c in corpus_embeddings]
    data['embedding'] = corpus_embeddings_ls

    print('=== appending numberic features after embedding')
    numeric_ls = ['Popularity_stand', 'Vote_Count_stand', 'Vote_Average_stand']
    data['embedding_w_numeric'] = data.apply(
        lambda v: v['embedding'] + list(v[numeric_ls]), axis=1)

    print(f'=== saving embedding data to: {output_path}')
    data.to_csv(output_path, index=False)


if __name__ == '__main__':
    run()
