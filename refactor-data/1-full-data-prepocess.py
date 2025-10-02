import pandas as pd
import random
import ast
import random
random.seed(42)


raw_train = pd.read_csv('../dataset/train.csv', sep=';')

data_train = {'sample': {}, 'annotation': {}}
data_validation = {'sample': {}, 'annotation': {}}
data_train_with_validation = {'sample': {}, 'annotation': {}}


def preprocess(tuples):
    changed_tuples = []
    labels = {
        'B-TYPE': 0, 'I-TYPE': 1, 'B-BRAND': 2,
        'I-BRAND': 3, 'B-VOLUME': 4, 'I-VOLUME': 5,
        'B-PERCENT': 6, 'I-PERCENT': 7, 'O': 8, '0': 8}
    for tup in tuples:
        changed_tuples.append(labels[tup[2]])
    return changed_tuples


for ind, sam, ann in raw_train.itertuples():
    annotations = preprocess(ast.literal_eval(ann))
    samples = sam.split()
    data_train_with_validation['sample'][ind] = samples
    data_train_with_validation['annotation'][ind] = annotations
    break

data_train_with_validation_df = pd.DataFrame(data_train_with_validation)
data_train_with_validation_df.to_csv('../dataset/train_with_valid.csv', sep=';', index=False)


used_ind = []
label2id = {
    'B-TYPE': 0, 'I-TYPE': 1, 'B-BRAND': 2,
    'I-BRAND': 3, 'B-VOLUME': 4, 'I-VOLUME': 5,
    'B-PERCENT': 6, 'I-PERCENT': 7, 'O': 8}

nums_valid = random.sample(range(100, 27201), 1400)

for ind, sam, ann in data_train_with_validation_df.itertuples():
    if any(char in ann for char in ('4', '5', '6', '7')):
        data_validation['sample'][ind] = sam
        data_validation['annotation'][ind] = ann
        used_ind.append(sam)



for ind, sam, ann in data_train_with_validation_df.itertuples():
    if (ind not in used_ind) and (ind in nums_valid):
        data_validation['sample'][ind] = sam
        data_validation['annotation'][ind] = ann
    else:
        data_train['sample'][ind] = sam
        data_train['annotation'][ind] = ann


data_validation_df = pd.DataFrame(data_validation)
data_validation_df.to_csv('../dataset/validation.csv', sep=';', index=False)
data_train_df = pd.DataFrame(data_train)
data_train_df.to_csv('../dataset/train_without_valid.csv', sep=';', index=False)
