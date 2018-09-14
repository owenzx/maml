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

def parse_snli(file_path, debug=False, num_instances=999999999999999):
    import pandas
    df = pandas.read_csv(file_path, sep='\t')
    df_needed = df[['sentence1','sentence2','gold_label']]
    df_needed.columns = ['seq1', 'seq2', 'stance']
    df_needed = df_needed.query('stance != \'-\'')
    df_needed = df_needed.dropna()
    data = df_needed.to_dict(orient='list')
    data['labels'] = sorted(list(set(data['stance'])))
    return data

def read_snli(datafolder="", debug=True, num_instances=99999999999999):
    train_path = os.path.join(datafolder, 'snli_1.0_train.txt')
    dev_path = os.path.join(datafolder, 'snli_1.0_dev.txt')
    test_path = os.path.join(datafolder, 'snli_1.0_test.txt')
    data_train = parse_snli(train_path)
    data_dev = parse_snli(dev_path, debug, num_instances)
    data_test = parse_snli(test_path, debug, num_instances)
    return data_train, data_dev, data_test

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
    data_test = parse_absa(test_path, debug, num_instances)

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

def read_sst_5(datafolder = './data/', debug=True, num_instance=999999999):
    train_path = os.path.join(datafolder, 'stsa.fine.train')
    phrase_train_path = os.path.join(datafolder, 'stsa.fine.phrases.train')
    dev_path = os.path.join(datafolder, 'stsa.fine.dev')
    test_path = os.path.join(datafolder, 'stsa.fine.test')
    data_train = parse_sst_5(train_path, debug, num_instance)
    data_dev = parse_sst_5(dev_path, debug, num_instance)
    data_test = parse_sst_5(test_path, debug, num_instance)
    return data_train, data_dev, data_test

def parse_sst_5(file_path, debug=False, num_instance=999999999):
    data = {"seq1": [], "stance": []}
    with open(file_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            data["stance"].append(line[0])
            data["seq1"].append(line[2:-1])
    data['labels'] = sorted(list(set(data['stance'])))
    return data

    

def read_sst(datafolder='./data/', debug=True, num_instances=999999999):
    import pandas
    train_path = os.path.join(datafolder, 'train.tsv')
    dev_path = os.path.join(datafolder, 'dev.tsv')
    test_path = os.path.join(datafolder, 'test.tsv')
    data_train = parse_sst(train_path, debug, num_instances)
    data_dev = parse_sst(dev_path, debug, num_instances)
    #Test data has no label
    #data_test = parse_sst(test_path, debug, num_instances)
    data_test = {"seq1":[], "stance":[]}
    return data_train, data_dev, data_test

def parse_sst(file_path, debug=False, num_instance=99999999999):
    import pandas
    df = pandas.read_csv(file_path, sep='\t')
    df.columns = ['seq1', 'stance']
    data = df.to_dict(orient='list')
    data['labels'] = sorted(list(set(data['stance'])))
    return data

#def parse_sst(file_path, debug=False, num_instance = 9999999999999999):
#    data = {"seq1":[], "stance":[]}
#    with open(file_path) as f:
#        for i, line in enumerate(f):
#            seq1, stance = line.split("\t")
#            data["seq1"].append(seq1)
#            data["stance"].append(stance)
#    data['labels'] = sorted(list(set(data['stance'])))
#    return data
    

def read_target_dependent(datafolder="./data/", debug=True, num_instances=999999999):
    #target_dependent_path = os.path.join(datafolder, 'target-dependent')
    target_dependent_path = datafolder
    train_path = os.path.join(target_dependent_path, 'train.raw')
    test_path = os.path.join(target_dependent_path, 'test.raw')
    for path_ in [target_dependent_path, train_path, test_path]:
        assert os.path.exists(path_), 'Error: %s does not exist.' % path_

    data_train = parse_target_dependent(train_path, debug, num_instances)
    data_test = parse_target_dependent(test_path, debug, num_instances)
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

def parse_topic_based(file_path, debug=False, num_instances=20):
    data = {"seq1": [], "seq2": [], "stance": []}
    with open(file_path) as f:
        for i, line in enumerate(f):
            id_, target, sentiment, tweet = line.split('\t')
            try:
                sentiment = float(sentiment)
            except ValueError:
                pass
            if debug and i >= num_instances+1:
                continue
            if tweet.strip() == 'Not Available':
                continue
            data["seq1"].append(target)
            data["seq2"].append(tweet)
            data["stance"].append(sentiment)

    # we have to sort the labels so that they're in the order
    # -2,-1,0,1,2 and are mapped to 0,1,2,3,4 (for subtask C)
    data["labels"] = sorted(list(set(data["stance"])))
    return data

def parse_topic_test_data(examples_path, labels_path):
    # Note: no debugging for the test data (20k tweets for subtask C)
    data = {"seq1": [], "seq2": [], "stance": []}
    with open(examples_path) as f_examples, open(labels_path) as f_labels:
        for i, (line_examples, line_labels) in enumerate(zip(f_examples, f_labels)):
            _, examples_target, _, *tweet = line_examples.strip().split('\t')
            # two lines contain a tweet, for some reason
            _, labels_target, sentiment, *_ = line_labels.strip().split('\t')
            # one test tweet contains a tab character
            if isinstance(tweet, list):
                tweet = '\t'.join(tweet)
            try:
                sentiment = float(sentiment)
            except ValueError:
                pass

            assert examples_target == labels_target,\
                '%s != %s at line %d in files %s and %s.' % (
                examples_target, labels_target, i, examples_path, labels_path)

            if tweet.strip() == 'Not Available':
                continue
            data["seq1"].append(examples_target)
            data["seq2"].append(tweet)
            data["stance"].append(sentiment)
    data["labels"] = sorted(list(set(data["stance"])))
    return data

def readTopicBased(datafolder="./data/", debug=True, num_instances=20):
    TOPIC_LABELS = ['negative', 'positive']
    topic_based_path = os.path.join(datafolder, 'semeval2016-task4b-topic-based-sentiment')
    train_path = os.path.join(topic_based_path, '100_topics_XXX_tweets.topic-two-point.subtask-BD.train.gold_downloaded.tsv')
    dev1_path = os.path.join(topic_based_path, '100_topics_XXX_tweets.topic-two-point.subtask-BD.dev.gold_downloaded.tsv')
    dev2_path = os.path.join(topic_based_path, '100_topics_XXX_tweets.topic-two-point.subtask-BD.devtest.gold_downloaded.tsv')
    test_data_path = os.path.join(topic_based_path, 'SemEval2016-task4-test.subtask-BD.txt')
    test_labels_path = os.path.join(topic_based_path, 'SemEval2016_task4_subtaskB_test_gold.txt')

    for path_ in [topic_based_path, train_path, dev1_path, dev2_path, test_data_path, test_labels_path]:
        assert os.path.exists(path_), 'Error: %s does not exist.' % path_

    data_train = parse_topic_based(train_path, debug, num_instances)
    data_dev1 = parse_topic_based(dev1_path, debug, num_instances)
    data_dev2 = parse_topic_based(dev2_path, debug, num_instances)
    data_test = parse_topic_test_data(test_data_path, test_labels_path)
    assert data_train["labels"] == TOPIC_LABELS
    data_dev1["labels"] = data_train["labels"]
    data_test["labels"] = data_train["labels"]

    # add the second dev data to the train set
    data_train["seq1"] += data_dev2["seq1"]
    data_train["seq2"] += data_dev2["seq2"]
    data_train["stance"] += data_dev2["stance"]
    return data_train, data_dev1, data_test

def readTopic5Way(datafolder="./data/", debug=True, num_instances=200):
    TOPIC_5WAY_LABELS = [-2.0, -1.0, 0.0, 1.0, 2.0]
    TOPIC_3WAY_LABELS = [-1.0, 0.0, 1.0]


    #topic_based_path = os.path.join(datafolder, 'semeval2016-task4c-topic-based-sentiment')
    topic_based_path = datafolder
    train_path = os.path.join(topic_based_path, '100_topics_100_tweets.topic-five-point.subtask-CE.train.gold_downloaded.tsv')
    dev1_path = os.path.join(topic_based_path, '100_topics_100_tweets.topic-five-point.subtask-CE.dev.gold_downloaded.tsv')
    dev2_path = os.path.join(topic_based_path, '100_topics_100_tweets.topic-five-point.subtask-CE.devtest.gold_downloaded.tsv')
    test_data_path = os.path.join(topic_based_path, 'SemEval2016-task4-test.subtask-CE.txt')
    test_labels_path = os.path.join(topic_based_path, 'SemEval2016_task4_subtaskC_test_gold.txt')

    for path_ in [topic_based_path, train_path, dev1_path, dev2_path,
                  test_data_path, test_labels_path]:
        assert os.path.exists(path_), 'Error: %s does not exist.' % path_

    data_train = parse_topic_based(train_path, debug, num_instances)
    data_dev1 = parse_topic_based(dev1_path, debug, num_instances)
    data_dev2 = parse_topic_based(dev2_path, debug, num_instances)
    data_test = parse_topic_test_data(test_data_path, test_labels_path)
    #print(data_train["labels"])
    assert data_train["labels"] == TOPIC_5WAY_LABELS
    data_dev1["labels"] = data_train["labels"]
    data_test["labels"] = data_train["labels"]

    # add the second dev data to the train set
    data_train["seq1"] += data_dev2["seq1"]
    data_train["seq2"] += data_dev2["seq2"]
    data_train["stance"] += data_dev2["stance"]
    return data_train, data_dev1, data_test

def map5to3(x):
    if x>0:
        return 1.0
    if x<0:
        return -1.0
    return 0.0

def readTopic3Way(datafolder="./data/", debug=True, num_instances=99999999):
    data_train, data_dev1, data_test = readTopic5Way(datafolder=datafolder, debug=debug, num_instances=num_instances)
    data_train["labels"] = data_dev1["labels"] = data_test["labels"] =  [-1.0, 0.0, 1.0]
    data_train["stance"] = [map5to3(x) for x in data_train["stance"]]
    data_dev1["stance"] = [map5to3(x) for x in data_dev1["stance"]]
    data_test["stance"] = [map5to3(x) for x in data_test["stance"]]
    return data_train, data_dev1, data_test



def main():
    train_set, dev_set, test_set = read_sst(datafolder='./data/SST-2')
    l = [len(x.split(' ')) for x in train_set['seq1']]
    print(max(l))
    print(sum(l)/len(l))

if __name__ == "__main__":
    main()
