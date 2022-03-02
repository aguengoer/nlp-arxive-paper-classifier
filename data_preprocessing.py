import json
import re
import pandas as pd


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
        print(el)
    papers_wantedonly = []
    for el in papers:
        if el['categories'] in cat_wanted_only:
            papers_wantedonly.append(el)
    return papers_wantedonly


def papers_by_mixed_categories(category: str, papers: list) -> list:
    categories = set([el['categories'] for el in papers])
    cat_wanted_only = [el for el in categories if re.search(category, el, re.IGNORECASE)]
    for el in cat_wanted_only:
        print(el)
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
        if count > 1000:
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
    print(len(papers_physonly))
    papers_mathonly = papers_by_single_category('math', data)
    print(len(papers_mathonly))
    papers_csonly = papers_by_single_category('^cs', data)
    print(len(papers_csonly))
    """
    categories = set([el['categories'] for el in data])
    for el in categories:
        print(el)
    """
    papers = pd.DataFrame(data)
    print(papers)
    papers.to_csv('papers.csv', sep=';')


if __name__ == '__main__':
    main()
