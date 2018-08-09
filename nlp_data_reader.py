# code mainly borrowed from https://github.com/coastalcph/mtl-disparate/blob/master/preproc/data_reader.py
import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from collections import defaultdict


# from data.semeval.data_loader import SemEvalDataLoader

# a = SemEvalDataLoader()

# print(a.get_data("BD",2016,"train"))
# for x in a.get_data("BD",2016,"train"):
#     print(x)



def split_train_data(data_train):
    """Split the train data into train and dev data."""
    train_ids, _ = train_test_split(range(len(data_train['seq1'])),
                                    test_size=0.1, random_state=42)
    data_dev = defaultdict(list)
    new_data_train = defaultdict(list)
    for key, examples in data_train.items():
        if key == 'labels':
            continue
        # no numpy indexing, so we iterate over the examples
        for i, example in enumerate(examples):
            if i in train_ids:
                new_data_train[key].append(example)
            else:
                data_dev[key].append(example)
    new_data_train['labels'] = data_train['labels']
    data_dev['labels'] = data_train['labels']
    return new_data_train, data_dev


def read_absa_laptops(datafolder="./data/", debug=True, num_instances=9999999999):
    return read_absa('Laptops', datafolder, debug, num_instances)


def read_absa_restaurants(datafolder="./data/", debug=True, num_instances=999999999):
    return read_absa('Restaurants', datafolder, debug, num_instances)


def read_absa(domain, datafolder="./data/", debug=True, num_instances=20):
    assert domain in ['Laptops', 'Restaurants'], '%s is not a valid domain.' % domain
    absa_path = os.path.join(datafolder, '')
    train_path = os.path.join(absa_path, 'ABSA16_%s_Train_SB1_v2.xml' % domain)
    test_path = os.path.join(absa_path, 'ABSA16_%s_Test_SB1.xml' % domain)
    for path_ in [absa_path, train_path, test_path]:
        assert os.path.exists(path_), 'Error: %s does not exist.' % path_

    data_train = parse_absa(train_path, debug, num_instances)
    data_test = parse_absa(test_path)

    # trial data is a subset of training data; instead we split the train data
    data_train, data_dev = split_train_data(data_train)
    return data_train, data_dev, data_test


def parse_absa(file_path, debug=False, num_instances=20):
    """
    Extracts all reviews from an XML file and returns them as a list of Review objects.
    Adds a NONE aspect to all sentences with no aspect.
    :param file_path: the path of the XML file
    :return: a list of Review objects each containing a list of Sentence objects and other attributes
    """
    data = {"seq1": [], "seq2": [], "stance": []}
    e = ET.parse(file_path).getroot()
    for i, review_e in enumerate(e):
        if debug and i >= num_instances+1:
            continue
        for sentence_e in review_e.find('sentences'):
            text = sentence_e.find('text').text
            # we do not care about sentences that do not contain an aspect
            if sentence_e.find('Opinions') is not None:
                for op in sentence_e.find('Opinions'):
                    # the category is of the form ENTITY#ATTRIBUTE, e.g. LAPTOP#GENERAL
                    target = ' '.join(op.get('category').split('#'))
                    polarity = op.get('polarity')
                    data['seq1'].append(target)
                    data['seq2'].append(text)
                    data['stance'].append(polarity)
    data["labels"] = sorted(list(set(data["stance"])))
    #assert data["labels"] == ABSA_LABELS
    return data

def read_target_dependent(datafolder="./data/", debug=True, num_instances=999999999):
    target_dependent_path = os.path.join(datafolder, 'target-dependent')
    train_path = os.path.join(target_dependent_path, 'train.raw')
    test_path = os.path.join(target_dependent_path, 'test.raw')
    for path_ in [target_dependent_path, train_path, test_path]:
        assert os.path.exists(path_), 'Error: %s does not exist.' % path_

    data_train = parse_target_dependent(train_path, debug, num_instances)
    data_test = parse_target_dependent(test_path)
    data_train, data_dev = split_train_data(data_train)
    return data_train, data_dev, data_test

def parse_target_dependent(file_path, debug=False, num_instances=20):
    TARGET_LABELS = ['-1', '0', '1']
    data = {"seq1": [], "seq2": [], "stance": []}
    with open(file_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 3 == 0:  # the tweet is always first
                data["seq2"].append(line.strip())
            elif i % 3 == 1:  # followed by the target
                data["seq1"].append(line.strip())
            elif i % 3 == 2:  # followed by the sentiment
                data["stance"].append(line.strip())
            if debug and i >= num_instances+1:
                continue
        assert len(data["seq1"]) == len(data["seq2"]) == len(data["stance"]),\
            'Error: %d != %d != %d.' % (len(data["seq1"]), len(data["seq2"]),
                                        len(data["stance"]))

    # replace the placeholder $T$ in every tweet with the target
    for i in range(len(data["seq1"])):
        target = data["seq1"][i]
        data["seq2"][i] = data["seq2"][i].replace("$T$", target)
    data["labels"] = sorted(list(set(data["stance"])))
    assert data["labels"] == TARGET_LABELS
    return data


def main():
    train_set, dev_set, test_set = read_target_dependent(datafolder='./data/')
    print(train_set.keys())
    print(train_set['seq1'])

if __name__ == "__main__":
    main()
