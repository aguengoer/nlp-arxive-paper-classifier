import json
import re


def main()->None:
    file=r'arxiv-metadata-oai-snapshot.json'
    data=[]
    count=0
    wanted_keys=['categories','title','abstract','id']
    for line in open(file):
        if count >10000:
            break
        line_dict = json.loads(line)
        for key in line_dict:
            if key not in wanted_keys:
                tmp=dict(line_dict)
                del tmp[key]
                line_dict=tmp
        data.append(line_dict)
        count+=1

    for el in data[0]:
        print(el,data[0][el])
    exit()
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