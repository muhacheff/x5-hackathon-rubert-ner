from ruwordnet import RuWordNet
import pandas as pd
import ast


rwn = RuWordNet()


def get_synonym(word):
    synsets = rwn.get_synsets(word)
    for s in synsets:
        for sense in s.senses:
            sen = sense.name.lower()
            if sen != word.lower() and " " not in sen:
                return sen
    return word



df_raw = pd.read_csv('../dataset/train_with_valid.csv', sep=';')
data_aug = {'sample': {}, 'annotation': {}}


for ind, sam, ann in df_raw.itertuples():
    annotations = ast.literal_eval(ann)
    samples = ast.literal_eval(sam)
    synonym_list = []
    for word in samples:
        syn = get_synonym(word)
        synonym_list.append(syn)
    if synonym_list != samples:
        data_aug['sample'][ind] = synonym_list
        data_aug['annotation'][ind] = annotations
    break


data_csv= pd.DataFrame(data_aug)
df_all = pd.concat([df_raw, data_csv], ignore_index=True)
df_all.to_csv("../dataset/train-augmented_full.csv", sep=";", index=False)