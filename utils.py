import transformers

data_folder = 'teamdata/training_dataset/'
FILE_train_users = data_folder+'train_users.json'
FOLDER_train_data = data_folder+'train/'


TOKENS_MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 4
EPOCHS = 10

TOKENIZER = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True,truncation=True)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count