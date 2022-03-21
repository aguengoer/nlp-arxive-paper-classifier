import json
import re
import warnings

import gensim.models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings(action='ignore')


# nltk.download('popular')


def papers_by_single_category(category: str, papers: list) -> list:
    categories = set([el['categories'] for el in papers])
    cat_wanted_only = []
    for el in categories:
        cats = el.split(' ')
        in_el_flg = True
        for subel in cats:
            if not re.search(category, subel, re.IGNORECASE):
                in_el_flg = False
        if in_el_flg:
            cat_wanted_only.append(el)
    # cat_wanted_only = [el for el in categories if re.search(category, el, re.IGNORECASE) and not re.search(' ', el)]
    for el in cat_wanted_only:
        # print(el)
        pass
    papers_wantedonly = []
    for el in papers:
        if el['categories'] in cat_wanted_only:
            papers_wantedonly.append(el)
    return papers_wantedonly


def papers_by_mixed_categories(category: str, papers: list) -> list:
    categories = set([el['categories'] for el in papers])
    cat_wanted_only = [el for el in categories if re.search(category, el, re.IGNORECASE)]
    for el in cat_wanted_only:
        # print(el)
        pass
    papers_wantedonly = []
    for el in papers:
        if el['categories'] in cat_wanted_only:
            papers_wantedonly.append(el)
    return papers_wantedonly


def main() -> None:
    pd.set_option('display.max_columns', None)
    file = r'arxiv-metadata-oai-snapshot.json'
    data = []
    count = 0
    wanted_keys = ['categories', 'title', 'abstract', 'id']
    for line in open(file):
        if count > 10000:
            break
        line_dict = json.loads(line)
        for key in line_dict:
            if key not in wanted_keys:
                tmp = dict(line_dict)
                del tmp[key]
                line_dict = tmp
        data.append(line_dict)
        count += 1
    papers_physonly = papers_by_single_category('ph', data)
    # print(len(papers_physonly))
    papers_mathonly = papers_by_single_category('math', data)
    # print(len(papers_mathonly))
    papers_csonly = papers_by_single_category('^cs', data)
    # print(len(papers_csonly))
    """
    categories = set([el['categories'] for el in data])
    for el in categories:
        print(el)
    """

    data_physonly = pd.DataFrame([x['abstract'] for x in papers_physonly])
    data_physonly['class'] = (['physics' for x in range(len(data_physonly))])
    data_mathonly = pd.DataFrame([x['abstract'] for x in papers_mathonly])
    data_mathonly['class'] = (['math' for x in range(len(papers_mathonly))])
    data_csonly = pd.DataFrame([x['abstract'] for x in papers_csonly])
    data_csonly['class'] = (['cs' for x in range(len(papers_csonly))])

    data = data_physonly.append(data_mathonly)
    # data = data.append(data_csonly)

    tokenized_abstracts = []
    for index, abstract in enumerate(data.iloc[:, 0]):
        abs = str(abstract.replace("\n", " "))
        abs = re.sub("[\.,;\":\(\)-]", "", abs)
        abs = re.sub("\$", "", abs)
        abs = re.sub("\{", "", abs)
        abs = re.sub("\}", "", abs)
        # abs = re.sub("\W","",abs)
        temp = []
        for j in word_tokenize(abs):
            temp.append(j.lower())

        tokenized_abstracts.append(temp)

    data['clean'] = tokenized_abstracts
    data.columns = ['text', 'class', 'clean']

    data['class'] = data['class'].map({'physics': 0, 'math': 1})
    for i in range(1):
        X_train, X_test, Y_train, Y_test = train_test_split(data['clean'], data['class'], test_size=0.2)

        w2v_model = gensim.models.Word2Vec(X_train,
                                           vector_size=100,
                                           window=5,
                                           min_count=2)

        show_words = False
        plottable_data = [(i, np.array(w2v_model.wv[i])) for i in w2v_model.wv.index_to_key]
        # plottable_data.columns = ['word','word-vector']

        # test = plottable_data.loc[:,'word-vector']

        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform([x[1] for x in plottable_data])

        tsne_df = pd.DataFrame()
        tsne_df['tsne0'] = tsne_results[:, 0]
        tsne_df['tsne1'] = tsne_results[:, 1]

        plt.figure()
        sns.scatterplot(x='tsne0', y='tsne1', palette=sns.color_palette('hls', 10), data=tsne_df, legend='full',
                        alpha=0.3)
        if show_words:
            for i in range(len(plottable_data)):
                p1.text(tsne_df.loc[i, 'tsne0'] + 0.01, tsne_df.loc[i, 'tsne1'],
                        plottable_data[i][0], horizontalalignment='left',
                        size='medium', color='black', weight='semibold')

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform([x[1] for x in plottable_data])
        pca_result = pca_result

        pca_df = pd.DataFrame()
        pca_df['pca0'] = pca_result[:, 0]
        pca_df['pca1'] = pca_result[:, 1]

        plt.figure()
        p1 = sns.scatterplot(x='pca0', y='pca1', palette=sns.color_palette('hls', 10), data=pca_df, legend='full',
                             alpha=0.3)
        if show_words:
            for i in range(len(plottable_data)):
                p1.text(pca_df.loc[i, 'pca0'] + 0.01, pca_df.loc[i, 'pca1'],
                        plottable_data[i][0], horizontalalignment='left',
                        size='medium', color='black', weight='semibold')
        plt.show()

        pca = PCA(n_components=3)
        pca_result = pca.fit_transform([x[1] for x in plottable_data])
        pca_result = pca_result

        pca_df = pd.DataFrame()
        pca_df['pca0'] = pca_result[:, 0]
        pca_df['pca1'] = pca_result[:, 1]
        pca_df['pca2'] = pca_result[:, 2]
        ax = plt.figure().gca(projection='3d')
        ax.scatter(xs=pca_df['pca0'],
                   ys=pca_df['pca1'],
                   zs=pca_df['pca2'],
                   cmap='tab10')
        plt.show()

        # print(w2v_model.wv.most_similar('paper'))

        words = set(w2v_model.wv.index_to_key)
        X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                                 for ls in X_train])
        X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                                for ls in X_test])

        X_train_vect_avg = []
        for v in X_train_vect:
            if v.size:
                X_train_vect_avg.append((v.mean(axis=0)))
            else:
                X_train_vect_avg.append(np.zeros(100, dtype=float))

        X_test_vect_avg = []
        for v in X_test_vect:
            if v.size:
                X_test_vect_avg.append((v.mean(axis=0)))
            else:
                X_test_vect_avg.append(np.zeros(100, dtype=float))

        rf = RandomForestClassifier()
        rf_model = rf.fit(X_train_vect_avg, Y_train.values.ravel())

        y_pred = rf_model.predict(X_test_vect_avg)

        precision = precision_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred)
        print(f'Precision: {round(precision, 3)} /'
              f' Recall: {round(recall, 3)} /'
              f' Accuracy: {round((y_pred == Y_test).sum() / len(y_pred), 3)}')


if __name__ == '__main__':
    main()
