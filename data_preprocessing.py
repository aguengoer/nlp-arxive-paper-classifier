import json
import re


def main()->None:
    file=r'arxiv-metadata-oai-snapshot.json'
    data=[]
    count=0
    for line in open(file):
        if count >10000:
            break
        data.append(json.loads(line))
        count+=1
    categories=set([el['categories'] for el in data])
    print(len(categories))
    cat_physonly=[el for el in categories if re.search('ph',el,re.IGNORECASE) and not re.search(' ',el)]
    for el in cat_physonly:
        print(el)
    papers_physonly=[]
    for el in data:
        if el['categories'] in cat_physonly:
            papers_physonly.append(el)
    print(len(papers_physonly))
    for x in papers_physonly[0]: print(x,papers_physonly[0][x])

if __name__=='__main__':
    main()