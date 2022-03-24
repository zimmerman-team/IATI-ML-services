from tabulate import tabulate

from common import dataset_persistency, specs_config


def main():
    headers = ['spec']
    for t in ('train', 'test'):
        for cr in ('rows', 'cols'):
            k = t + "_npa_" + cr
            headers.append(k)

    table = []
    for spec in specs_config.specs:
        row = []
        doc = dataset_persistency.load_splits_document(spec)
        if len(set(headers) - set(doc.keys())) != 0:
            # incomplete document
            continue
        for k in headers:
            row.append(doc[k])
        table.append(row)

    print(tabulate(table, headers=headers))


if __name__ == '__main__':
    main()