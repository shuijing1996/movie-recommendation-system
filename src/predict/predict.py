'''
recommend movie based on embedding vectror cosine similarity 
'''
import os
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def recommend_movie(data, target_movie, top_n=5):
	idx_ls = data[data.Title == target_movie].index

	recommend_dict = dict()
	for idx in idx_ls:
		target_vector = eval(data.loc[idx, 'embedding_w_numeric'])
		target_year = data.loc[idx, 'Release_Date']

		for ix in tqdm(range(len(data))):
			movie_vector_ix = eval(data.loc[ix, 'embedding_w_numeric'])
			data.loc[ix, 'cos_similarity'] = cosine_similarity([target_vector], [movie_vector_ix])[0][0]

		data_sort = data.sort_values(by='cos_similarity', ascending=False)
		recommend_ls = data_sort.Title[1:top_n+1].tolist()

		recommend_dict.update({f'{target_movie}_{target_year}': recommend_ls})

	return recommend_dict
#--recommend_movie


def run(EMBEDDING_PATH, target_movie, recommend_count):
	data = pd.read_csv(EMBEDDING_PATH, lineterminator='\n')

	if target_movie in data.Title.tolist():
		recommend_dict = recommend_movie(data, target_movie, recommend_count)

		for k, v in recommend_dict.items():
			release_date = k.split('_')[1]
			print(f'Given provided movie: {target_movie} (release date: {release_date})')
			print(f'{recommend_count} recommended movies are:')
			print(*v, sep='\n')
			print()
	else:
		print('Please provide a valid movie name.')
#--run

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--target_movie", required=True, type=str)
	parser.add_argument("--recommend_count", required=False, type=str, default='5')
	args = parser.parse_args()
	target_movie = args.target_movie
	recommend_count = int(args.recommend_count)
	
	EMBEDDING_PATH = 'data/Sr._Data_Scientist_Assessment_Dataset_embedding.csv'
	if os.path.exists(EMBEDDING_PATH):
		run(EMBEDDING_PATH, target_movie, recommend_count)
	else:
		print('=== please run train.py to create embedding first')

