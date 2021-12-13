from tabulate import tabulate

from common import persistency, relspecs

def main():
    headers = ['rel']
    for t in ('train', 'test'):
        for cr in ('rows', 'cols'):
            k = t + "_npa_" + cr
            headers.append(k)

    table = []
    for rel in relspecs.rels:
        row = []
        doc = persistency.load_tsets_document(rel)
        if len(set(headers) - set(doc.keys())) != 0:
            # incomplete document
            continue
        for k in headers:
            row.append(doc[k])
        table.append(row)

    print(tabulate(table, headers=headers))

if __name__ == '__main__':
    main()