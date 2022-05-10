from __future__ import annotations
import csv

def export_csv(data: list[dict], filepath: str, fields: list[str]):

    with open(file=filepath, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()

        for d in data:
            # print(d)
            # print(d.values())
            writer.writerow(d)
            

def import_csv(filepath: str) -> list[dict]:
    with open(file=filepath, newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        # for row in rows:
        #     print(row)
        #     acc = eval(row['acc'])
        #     print(acc[1])
    return rows


if __name__ == '__main__':
    data = [
        dict(name="t1", acc=[0.1, 0.2, 0.9]),
        dict(name="t2", acc=[0.3, 0.2, 0.9])
    ]

    export_csv(data, 'tmp.csv', ["name", "acc"])
    import_csv(filepath='tmp.csv')