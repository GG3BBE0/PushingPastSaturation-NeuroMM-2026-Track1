import zipfile, shutil, os
shutil.copy('submission_test1_filteeg.csv', 'submissions/submission.csv')
with zipfile.ZipFile('submissions/submission_test1_filteeg.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    z.write('submissions/submission.csv', arcname='submission.csv')
os.unlink('submissions/submission.csv')
print('zipped: submissions/submission_test1_filteeg.zip')
