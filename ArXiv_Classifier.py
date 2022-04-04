import json
import pickle
import random
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

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


def tsne_plot(w2v_model, show_words=False):
    plottable_data = [(i, np.array(w2v_model.wv[i])) for i in w2v_model.wv.index_to_key]
    # plottable_data.columns = ['word','word-vector']

    # test = plottable_data.loc[:,'word-vector']

    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform([x[1] for x in plottable_data])

    tsne_df = pd.DataFrame()
    tsne_df['tsne0'] = tsne_results[:, 0]
    tsne_df['tsne1'] = tsne_results[:, 1]

    plt.figure()
    p1 = sns.scatterplot(x='tsne0', y='tsne1', palette=sns.color_palette('hls', 10), data=tsne_df, legend='full',
                         alpha=0.3)
    if show_words:
        for i in range(len(plottable_data)):
            p1.text(tsne_df.loc[i, 'tsne0'] + 0.01, tsne_df.loc[i, 'tsne1'],
                    plottable_data[i][0], horizontalalignment='left',
                    size='medium', color='black', weight='semibold')
    plt.show()


def pca_plot(w2v_model, show_words=False):
    plottable_data = [(i, np.array(w2v_model.wv[i])) for i in w2v_model.wv.index_to_key]
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


def class_balancer(dataframes: list, size: int) -> pd.DataFrame:
    """
    Balances classes by random selection with replacement
    :param dataframes: list of pd.Dataframes each one representing 1 class
    :param size: size of output pd.Dataframe
    :return: shuffled pd.Dataframe with equal class representation
    """
    classes = len(dataframes)
    output_data = pd.DataFrame()
    for i in range(size):
        cat = random.randint(0, classes - 1)
        el = random.randint(0, len(dataframes[cat].index) - 1)
        test = dataframes[cat].iloc[el, :]
        output_data = pd.concat([output_data, pd.DataFrame(dataframes[cat].iloc[el, :]).T])
    output_data.reset_index(drop=True, inplace=True)
    return output_data


def main() -> None:
    pd.set_option('display.max_columns', None)
    file = r'arxiv-metadata-oai-snapshot.json'
    data = []
    count = 0
    wanted_keys = ['categories', 'title', 'abstract', 'id']
    write_file = 'daten_bsp.json'
    write_lines = []
    print('read 10 lines')
    for line in open(file):
        if count > 10:
            break
        write_lines.append(line)
        count += 1

    print('write 10 lines')
    with open(write_file, 'w') as outfile:
        for line in write_lines:
            outfile.write(line)

    print('read 10000 lines')
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
    print('sort papers')
    papers_physonly = papers_by_single_category('ph', data)
    # print(len(papers_physonly))
    papers_mathonly = papers_by_single_category('math', data)
    # print(len(papers_mathonly))
    papers_csonly = papers_by_single_category('^cs', data)
    # print(len(papers_csonly))
    papers_econonly = papers_by_single_category('econ', data)
    # print(len(papers_econonly))
    # 0 econ only papers
    papers_nlinonly = papers_by_single_category('nlin', data)
    # print(len(papers_nlinonly))
    papers_qbioonly = papers_by_single_category('q-bio', data)
    # print(len(papers_qbioonly))

    categories = set([el['categories'] for el in data])
    categories = sorted(categories)
    # for el in categories:
    #     #print(el)
    #     pass

    data_physonly = pd.DataFrame([x['abstract'] for x in papers_physonly])
    data_physonly['class'] = (['physics' for x in range(len(data_physonly))])
    data_mathonly = pd.DataFrame([x['abstract'] for x in papers_mathonly])
    data_mathonly['class'] = (['math' for x in range(len(papers_mathonly))])
    data_nlinonly = pd.DataFrame([x['abstract'] for x in papers_nlinonly])
    data_nlinonly['class'] = (['nlin' for x in range(len(papers_nlinonly))])
    data_csonly = pd.DataFrame([x['abstract'] for x in papers_csonly])
    data_csonly['class'] = (['cs' for x in range(len(papers_csonly))])
    data_qbioonly = pd.DataFrame([x['abstract'] for x in papers_qbioonly])
    data_qbioonly['class'] = (['qbio' for x in range(len(papers_qbioonly))])

    data = data_physonly.append(data_mathonly)
    data = data.append(data_csonly)
    data = data.append(data_csonly)
    data = data.append(data_csonly)
    data.reset_index(drop=True, inplace=True)

    print('balancing data')
    # Data balanced for equal class representation
    equal_data = class_balancer([data_physonly, data_mathonly, data_csonly, data_nlinonly, data_qbioonly], 10000)
    equal_data.columns = ['abstract', 'class']
    data = equal_data
    tokenized_abstracts = []
    print('tokenizing data')
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

    data['class'] = data['class'].map({'physics': 0, 'math': 1, 'cs': 2, 'nlin': 3, 'qbio': 4})
    # print(data.describe())
    print('start clustering and training')
    for i in range(1):
        X_train, X_test, Y_train, Y_test = train_test_split(data['clean'], data['class'], test_size=0.2)

        w2v_model = gensim.models.Word2Vec(X_train,
                                           vector_size=100,
                                           window=5,
                                           min_count=2)
        show_words = False

        tsne_plot(w2v_model, show_words)
        pca_plot(w2v_model, show_words)

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

        precision = precision_score(Y_test, y_pred, average='macro')
        recall = recall_score(Y_test, y_pred, average='macro')
        print(f'Random forest: Precision: {round(precision, 3)} /'
              f' Recall: {round(recall, 3)} /'
              f' Accuracy: {round((y_pred == Y_test).sum() / len(y_pred), 3)}')

        kn = KNeighborsClassifier()
        kn_model = kn.fit(X_train_vect_avg, Y_train.values.ravel())

        y_pred = kn_model.predict(X_test_vect_avg)

        precision = precision_score(Y_test, y_pred, average='macro')
        recall = recall_score(Y_test, y_pred, average='macro')
        print(f'KNN: Precision: {round(precision, 3)} /'
              f' Recall: {round(recall, 3)} /'
              f' Accuracy: {round((y_pred == Y_test).sum() / len(y_pred), 3)}')

        nn = MLPClassifier(verbose=False)
        nn_model = nn.fit(X_train_vect_avg, Y_train.values.ravel())

        y_pred = nn_model.predict(X_test_vect_avg)

        precision = precision_score(Y_test, y_pred, average='macro')
        recall = recall_score(Y_test, y_pred, average='macro')
        print(f'Neural Net: Precision: {round(precision, 3)} /'
              f' Recall: {round(recall, 3)} /'
              f' Accuracy: {round((y_pred == Y_test).sum() / len(y_pred), 3)}')
        print('\n')

    rf_file = 'rf_model.sav'
    knn_file = 'knn_model.sav'
    nn_file = 'nn_model.sav'
    pickle.dump(rf_model, open(rf_file, 'wb'))
    pickle.dump(kn_model, open(knn_file, 'wb'))
    pickle.dump(nn_model, open(nn_file, 'wb'))


if __name__ == '__main__':
    main()
