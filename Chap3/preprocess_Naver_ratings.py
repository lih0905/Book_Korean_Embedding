
def preprocess_naver_ratings(corpus_path, output_fname, with_label):
    with open(corpus_path,'r',encoding='utf-8') as f1,\
            open(output_fname,'w',encoding='utf-8') as f2:
        next(f1)
        for line in f1:
            _, sentence, label = line.strip().split('\t')

            if not sentence: continue
            if with_label:
                f2.writelines(sentence+'\u241E' + label + '\n')
            else:
                f2.writelines(sentence+'\n')

corpus_path = 'data/raw/ratings.txt'
output_fname = 'data/processed/processed_ratings.txt'
with_label = False
preprocess_naver_ratings(corpus_path, output_fname, with_label)

corpus_path = 'data/raw/ratings_test.txt'
output_fname = 'data/processed/processed_ratings_test.txt'
with_label = True
preprocess_naver_ratings(corpus_path, output_fname, with_label)

corpus_path = 'data/raw/ratings_train.txt'
output_fname = 'data/processed/processed_ratings_train.txt'
with_label = True
preprocess_naver_ratings(corpus_path, output_fname, with_label)

