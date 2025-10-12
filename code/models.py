import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, BertModel

# Ref: https://www.kaggle.com/code/trnmtin/phobert-classification-for-vietnamese-text


class PhoBERTSentClassifer(torch.nn.Module):
    def __init__(self, n_classes: int):
        super(PhoBERTSentClassifer, self).__init__()
        self.pretrained_layer = AutoModel.from_pretrained("vinai/phobert-base-v2")

        self.dropout = torch.nn.Dropout(p=0.3)
        self.linear = torch.nn.Linear(
            self.pretrained_layer.config.hidden_size, n_classes
        )
        torch.nn.init.normal_(self.linear.weight, std=0.02)
        torch.nn.init.normal_(self.linear.bias, 0)

    def forward(self, input_ids, attention_mask):
        _, output = self.pretrained_layer(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )

        x = self.dropout(output)
        x = self.linear(x)
        return x


class ViSoBERTSentClassifer(torch.nn.Module):
    def __init__(self, n_classes: int):
        super(ViSoBERTSentClassifer, self).__init__()
        self.pretrained_layer = AutoModel.from_pretrained("uitnlp/visobert")

        self.dropout = torch.nn.Dropout(p=0.3)
        self.linear = torch.nn.Linear(
            self.pretrained_layer.config.hidden_size, n_classes
        )
        torch.nn.init.normal_(self.linear.weight, std=0.02)
        torch.nn.init.normal_(self.linear.bias, 0)

    def forward(self, input_ids, attention_mask):
        _, output = self.pretrained_layer(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )

        x = self.dropout(output)
        x = self.linear(x)
        return x


class Vibert4newsSentClassifer(torch.nn.Module):
    def __init__(self, n_classes: int):
        super(Vibert4newsSentClassifer, self).__init__()
        self.pretrained_layer = BertModel.from_pretrained(
            "NlpHUST/vibert4news-base-cased"
        )

        self.dropout = torch.nn.Dropout(p=0.3)
        self.linear = torch.nn.Linear(
            self.pretrained_layer.config.hidden_size, n_classes
        )
        torch.nn.init.normal_(self.linear.weight, std=0.02)
        torch.nn.init.normal_(self.linear.bias, 0)

    def forward(self, input_ids, attention_mask):
        _, output = self.pretrained_layer(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )

        x = self.dropout(output)
        x = self.linear(x)
        return x


class VnSBERTSentClassifer(torch.nn.Module):
    def __init__(self, n_classes: int):
        super(VnSBERTSentClassifer, self).__init__()
        self.pretrained_layer = AutoModel.from_pretrained("keepitreal/vietnamese-sbert")

        self.dropout = torch.nn.Dropout(p=0.3)
        self.linear = torch.nn.Linear(
            self.pretrained_layer.config.hidden_size, n_classes
        )
        torch.nn.init.normal_(self.linear.weight, std=0.02)
        torch.nn.init.normal_(self.linear.bias, 0)

    def forward(self, input_ids, attention_mask):
        # Access the last_hidden_state directly when return_dict=False
        _, output = self.pretrained_layer(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )

        x = self.dropout(output)
        x = self.linear(x)
        return x


class FeedbackDataset(Dataset):
    def __init__(
        self, feedback_df: pd.DataFrame, tokeniser: AutoTokenizer, max_length: int
    ):
        self.feedback_df = feedback_df
        self.tokeniser = tokeniser
        self.max_length = max_length

    def __len__(self):
        return len(self.feedback_df)

    def __getitem__(self, index):
        row = self.feedback_df.iloc[index]
        text = row["sentence"]
        sentiment = row["sentiment"]

        # Encode_plus will:
        # (1) split text into token
        # (2) Add the '[CLS]' and '[SEP]' token to the start and end
        # (3) Truncate/Pad sentence to max length
        # (4) Map token to their IDS
        # (5) Create attention mask
        # (6) Return a dictionary of outputs

        encoding = self.tokeniser.encode_plus(
            text=text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "sentiment": torch.tensor(sentiment, dtype=torch.long),
        }


# Ref: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    """This class implements early stopping"""

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def reset(self):
        self.counter = 0
        self.min_validation_loss = float("inf")


class TrainArgs:
    def __init__(
        self,
        loss_fn: torch.nn.Module,
        optimiser: torch.optim,
        early_stopper: EarlyStopper,
        lr: float,
        batch_size: int,
        max_length: int,
        epochs: int,
        num_gens: int,
    ):
        self.loss_fn = loss_fn
        self.optimiser = optimiser
        self.early_stopper = early_stopper
        self.lr = lr
        self.batch_size = batch_size
        self.max_length = max_length
        self.epochs = epochs
        self.num_gens = num_gens
