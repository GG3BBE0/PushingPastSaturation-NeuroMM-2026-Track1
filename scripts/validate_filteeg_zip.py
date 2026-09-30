import zipfile, csv
with zipfile.ZipFile('submissions/submission_test1_filteeg.zip') as z:
    with z.open('submission.csv') as f:
        rows = list(csv.reader([line.decode() for line in f]))
print('header:', rows[0])
print('row count (incl header):', len(rows))
print('first row:', rows[1])
print('last row:', rows[-1])
preds = [float(r[1]) for r in rows[1:]]
print(f'pred range: {min(preds):.4f} ~ {max(preds):.4f}, mean={sum(preds)/len(preds):.4f}')
