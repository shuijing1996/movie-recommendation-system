"""Recommend movie based on embedding vectror cosine similarity."""
import argparse
import os
from ast import literal_eval

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def recommend_movie(data, movie, top_n=5):
    """Calculate cosine similarity against target movie vector."""
    idx_ls = data[data.Title == movie].index

    recommend_dict = {}
    for idx in idx_ls:
        target_vector = literal_eval(data.loc[idx, 'embedding_w_numeric'])
        target_year = data.loc[idx, 'Release_Date']

        for data_ix in tqdm(range(len(data))):
            movie_vector_ix = literal_eval(data.loc[data_ix,
                                                    'embedding_w_numeric'])
            data.loc[data_ix, 'cos_similarity'] = cosine_similarity(
                [target_vector], [movie_vector_ix])[0][0]

        data_sort = data.sort_values(by='cos_similarity', ascending=False)
        recommend_ls = data_sort.Title[1:top_n + 1].tolist()

        recommend_dict.update({f'{movie}_{target_year}': recommend_ls})

    return recommend_dict


#--recommend_movie


def run(data_path, movie, top_n):
    """Runner function."""
    data = pd.read_csv(data_path, lineterminator='\n')

    if movie in data.Title.tolist():
        recommend_dict = recommend_movie(data, movie, top_n)

        for k_movie, v_recommend_ls in recommend_dict.items():
            release_date = k_movie.split('_')[1]
            print(
                f'Given provided movie: {k_movie} (release date: {release_date})'
            )
            print(f'{top_n} recommended movies are:')
            print(*v_recommend_ls, sep='\n')
            print()
    else:
        print('Please provide a valid movie name.')


#--run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_movie", required=True, type=str)
    parser.add_argument("--recommend_count",
                        required=False,
                        type=str,
                        default='5')
    args = parser.parse_args()
    target_movie = args.target_movie
    recommend_count = int(args.recommend_count)

    EMBEDDING_PATH = 'data/Sr._Data_Scientist_Assessment_Dataset_embedding.csv'
    if os.path.exists(EMBEDDING_PATH):
        run(EMBEDDING_PATH, target_movie, recommend_count)
    else:
        print('=== please run train.py to create embedding first')
