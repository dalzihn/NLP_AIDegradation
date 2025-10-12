import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizer

# Ref: https://www.learnpytorch.io/


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimiser: torch.optim,
    train_dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Executes the training step of a model

    Args:
      model: the model to be trained
      loss_fn: the loss function
      optimiser: the optimisation strategy
      train_dataloader: the train data
      device: device used for calculation

    Returns:
      tuple[float, float]: loss and accuracy
    """
    # Put model in train mode
    model.train()

    # Set up train loss and accuracy values
    train_loss, train_acc = 0, 0

    # Loop through dataloader and batches
    for batch, data in enumerate(train_dataloader):
        # Send data to target device
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        y = data["sentiment"].to(device)  # y is sentiment

        # 1. Forward pass
        y_pred = model(input_ids=input_ids, attention_mask=attention_mask)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimiser zero grad
        optimiser.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimiser step
        optimiser.step()

        # Calculate and accumate accuracy metric
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(train_dataloader)
    train_acc = train_acc / len(train_dataloader)

    return train_loss, train_acc


def test(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    """
    Executes the testing step of a model

    Args:
      model: the model to be trained
      loss_fn: the loss function
      test_dataloader: the test data
      device: device used for calculation

    Returns:
      tuple[float, float, float, float, float]: loss, accuracy, precision, recall, f1
    """
    # Put model in eval mode
    model.eval()

    # Set up test loss and accuracy
    test_loss = 0
    all_preds = []
    all_truths = []

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through the test data
        for batch, data in enumerate(test_dataloader):
            # Send data to target device
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            ground_truth = data["sentiment"].to(device)

            # 1. Forward pass
            test_pred = model(input_ids=input_ids, attention_mask=attention_mask)

            # 2. Calculate loss
            loss = loss_fn(test_pred, ground_truth)
            test_loss += loss.item()

            # Get predicted class and store for metrics calculation
            test_pred_class = torch.argmax(torch.softmax(test_pred, dim=1), dim=1)
            all_preds.extend(test_pred_class.cpu().numpy())
            all_truths.extend(ground_truth.cpu().numpy())

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(test_dataloader)

    # Calculate additional metrics
    test_acc = accuracy_score(all_truths, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_truths, all_preds, average="weighted", zero_division=0
    )

    return test_loss, test_acc, precision, recall, f1


def train_and_test(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimiser: torch.optim,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    device: torch.device,
    epochs: int,
    early_stopper: EarlyStopper = None,
):
    """
    Trains and tests a model

    Args:
        model: the model to be trained
        loss_fn: the loss function
        optimiser: the optimisation strategy
        train_dataloader: the train data
        test_dataloader: the test data
        device: device used for calculation
        epochs: number of epochs
        early_stopper: early stopper

    Returns:
        dict: dictionary of results
    """
    # Create empty results dictionary
    log = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "test_precision": [],  # Added for precision
        "test_recall": [],  # Added for recall
        "test_f1": [],  # Added for f1
    }
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train(
            model=model,
            loss_fn=loss_fn,
            optimiser=optimiser,
            train_dataloader=train_dataloader,
            device=device,
        )

        test_loss, test_acc, test_precision, test_recall, test_f1 = (
            test(  # Modified to capture new metrics
                model=model,
                loss_fn=loss_fn,
                test_dataloader=test_dataloader,
                device=device,
            )
        )

        # Print out what's happening
        print(
            f" Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            f"test_precision: {test_precision:.4f} | "  # Added print
            f"test_recall: {test_recall:.4f} | "  # Added print
            f"test_f1: {test_f1:.4f}"  # Added print
        )
        # Update results dictionary
        log["train_loss"].append(train_loss)
        log["train_acc"].append(train_acc)
        log["test_loss"].append(test_loss)
        log["test_acc"].append(test_acc)
        log["test_precision"].append(test_precision)  # Added to log
        log["test_recall"].append(test_recall)  # Added to log
        log["test_f1"].append(test_f1)  # Added to log

        if early_stopper is not None:
            if early_stopper.early_stop(test_loss):
                break
    return log


def prepare_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokeniser: AutoTokenizer,
    max_length: int,
    batch_size: int,
    is_shuffle: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Prepares PyTorch DataLoaders for training and testing.

    Args:
        train_df: DataFrame containing training data.
        test_df: DataFrame containing testing data.
        tokeniser: The tokenizer to use for encoding text.
        max_length: The maximum length of the tokenized sequences.
        batch_size: The batch size for the DataLoaders.
        is_shuffle: Whether to shuffle the training data. Defaults to False.

    Returns:
        A tuple containing the training DataLoader and the testing DataLoader.
    """
    train_dataset = FeedbackDataset(
        feedback_df=train_df, tokeniser=tokeniser, max_length=max_length
    )

    test_dataset = FeedbackDataset(
        feedback_df=test_df, tokeniser=tokeniser, max_length=max_length
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=is_shuffle
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=is_shuffle
    )

    return train_dataloader, test_dataloader


def get_predictions(
    df: pd.DataFrame,
    model: torch.nn.Module,
    tokeniser: AutoTokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
    is_shuffle: bool = False,
) -> pd.DataFrame:
    """
    Gets predictions from a model on a given DataFrame.

    Args:
        df: DataFrame containing the data to get predictions for.
        model: The PyTorch model to use for predictions.
        tokeniser: The tokenizer to use for encoding text.
        max_length: The maximum length of the tokenized sequences.
        batch_size: The batch size for the DataLoader.
        device: The device to use for calculations (e.g., 'cuda' or 'cpu').
        is_shuffle: Whether to shuffle the data. Defaults to False.

    Returns:
        pd.DataFrame: The input DataFrame with 'sentiment' column containing the predicted class and 'confidence' column with predicted probabilities.
    """
    df = df.copy(deep=True)
    df["actual_sentiment"] = df["sentiment"]
    df["sentiment"] = -100
    # Add columns for confidence scores
    for i in range(model.linear.out_features):
        df[f"confidence_{i}"] = 0.0

    df_to_dataset = FeedbackDataset(
        feedback_df=df, tokeniser=tokeniser, max_length=max_length
    )

    dataset_to_dataloader = DataLoader(
        df_to_dataset, batch_size=batch_size, shuffle=is_shuffle
    )
    labels = []
    confidence_scores = []
    # Put model in eval mode
    model.to(device)
    model.eval()

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through the test data
        for batch, data in enumerate(dataset_to_dataloader):
            # Send data to target device
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            # 1. Forward pass
            pred = model(input_ids=input_ids, attention_mask=attention_mask)
            # Get predicted class
            pred_class = torch.argmax(torch.softmax(pred, dim=1), dim=1)
            labels.extend(pred_class.tolist())

            # Get confidence scores (probabilities)
            probabilities = torch.softmax(pred, dim=1).cpu().numpy()
            confidence_scores.extend(probabilities)

    df["sentiment"] = labels
    # Assign confidence scores to respective columns
    for i in range(model.linear.out_features):
        df[f"confidence_{i}"] = [score[i] for score in confidence_scores]

    return df


def train_with_true_labels(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model: torch.nn.Module,
    tokeniser: AutoTokenizer,
    args: TrainArgs,
    device: torch.device,
    is_shuffle: bool = True,
    optimiser: torch.optim = None,
) -> tuple[dict, torch.optim]:
    """Trains the model using the true labels in the training data.

    Args:
        train_df: DataFrame containing the training data with true labels.
        test_df: DataFrame containing the testing data.
        model: The PyTorch model to train.
        tokeniser: The tokenizer to use for encoding text.
        args: Training arguments (loss function, optimizer, etc.).
        device: The device to use for calculations (e.g., 'cuda' or 'cpu').
        is_shuffle: Whether to shuffle the training data. Defaults to True.
        optimiser: The optimizer to use. If None, a new AdamW optimizer will be created.

    Returns:
        tuple: A tuple containing a dictionary of training and testing results and the optimizer.
    """
    train_dataloader, test_dataloader = prepare_dataset(
        train_df=train_df,
        test_df=test_df,
        tokeniser=tokeniser,
        max_length=args.max_length,
        batch_size=args.batch_size,
        is_shuffle=is_shuffle,
    )

    if optimiser is None:
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

    results = train_and_test(
        model=model,
        loss_fn=args.loss_fn,
        optimiser=optimiser,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        epochs=args.epochs,
        early_stopper=args.early_stopper,
    )

    return results, optimiser


def train_with_pred_labels(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model: torch.nn.Module,
    tokeniser: AutoTokenizer,
    args: TrainArgs,
    device: torch.device,
    is_shuffle: bool = False,
    optimiser: torch.optim = None,
) -> tuple[pd.DataFrame, dict, torch.optim]:
    """Trains the model using predicted labels on the training data.

    Args:
        train_df: DataFrame containing the training data (labels will be predicted).
        test_df: DataFrame containing the testing data.
        model: The PyTorch model to use for prediction and training.
        tokeniser: The tokenizer to use for encoding text.
        args: Training arguments (loss function, optimizer, etc.).
        device: The device to use for calculations (e.g., 'cuda' or 'cpu').
        is_shuffle: Whether to shuffle the training data. Defaults to False.
        optimiser: The optimizer to use. If None, a new AdamW optimizer will be created.

    Returns:
        tuple: A tuple containing a dictionary of training and testing results and the optimizer.
    """
    train_df = get_predictions(
        df=train_df,
        model=model,
        tokeniser=tokeniser,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        is_shuffle=is_shuffle,
    )

    train_dataloader, test_dataloader = prepare_dataset(
        train_df=train_df,
        test_df=test_df,
        tokeniser=tokeniser,
        max_length=args.max_length,
        batch_size=args.batch_size,
        is_shuffle=is_shuffle,
    )

    if optimiser is None:
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

    results = train_and_test(
        model=model,
        loss_fn=args.loss_fn,
        optimiser=optimiser,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        epochs=args.epochs,
        early_stopper=args.early_stopper,
    )

    return train_df, results, optimiser


def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k_folds: list[pd.DataFrame],
    model_name: str,
    tokeniser: AutoTokenizer,
    args: TrainArgs,
    device: torch.device,
    use_predicted_labels: bool = False,
    is_shuffle: bool = True,
) -> tuple[
    list[pd.DataFrame], list[dict], torch.nn.Module
]:  # Modified to return the trained model
    """
    Wraps the training functions to train the model with either true or predicted labels.

    Args:
        train_df: DataFrame containing the training data.
        test_df: DataFrame containing the testing data.
        k_folds: List of DataFrames containing subsets of data for predictions.
        model_name: Name of the model.
        tokeniser: The tokenizer to use for encoding text.
        args: Training arguments (loss function, optimizer, etc.).
        device: The device to use for calculations (e.g., 'cuda' or 'cpu').
        use_predicted_labels: If True, use predicted labels for training data.
                              If False, use true labels (default).
        is_shuffle: Whether to shuffle the training data. Defaults to True.

    Returns:
        tuple: A tuple containing a list of dictionaries with training and testing results and the trained model.
    """
    results = []
    predictions = []
    model = initialise_model(model_name=model_name)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if use_predicted_labels:
        print("Starting training models with predicted labels")
        for i in range(args.num_gens):
            print(f"Loading model from: {model_name}/{model_name}_{i+1}")

            # Load model state
            model_path = model_log_dir / f"{model_name}/{model_name}_{i+1}"
            if model_path.exists():
                model.load_state_dict(torch.load(model_path, map_location=device))
                print("Model state loaded.")
            else:
                print(f"Model state not found at {model_path}, initializing new model.")
                model = initialise_model(model_name=model_name)

            # Load optimizer state if it exists
            optimizer_path = (
                model_log_dir / f"{model_name}/{model_name}_{i+1}_optimizer"
            )
            if optimizer_path.exists():
                optimiser.load_state_dict(
                    torch.load(optimizer_path, map_location=device)
                )
                print("Optimizer state loaded.")
            else:
                print("No optimizer state found, initializing new optimizer.")
                optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

            args.early_stopper.reset()  # Reset early stopper

            for param in model.pretrained_layer.parameters():
                param.requires_grad = False

            print("Starting train")
            df_preds, step_result, optimiser = train_with_pred_labels(
                train_df=k_folds[i],
                test_df=test_df,
                model=model,
                tokeniser=tokeniser,
                args=args,
                device=device,
                is_shuffle=False,  # Override is_shuffle to False for predicted labels
                optimiser=optimiser,  # Pass the existing optimizer
            )

            results.append(step_result)
            predictions.append(df_preds)
            # Save model and optimizer state
            torch.save(
                model.state_dict(), model_log_dir / f"{model_name}/{model_name}_{i+2}"
            )
            torch.save(
                optimiser.state_dict(),
                model_log_dir / f"{model_name}/{model_name}_{i+2}_optimizer",
            )
            print(f"Model and optimizer saved at: {model_name}, index {i+2}")
            time.sleep(3)

    else:
        print("Starting training models with true labels")

        args.early_stopper.reset()  # Reset early stopper

        step_result, optimiser = train_with_true_labels(
            train_df=train_df,
            test_df=test_df,
            model=model,
            tokeniser=tokeniser,
            args=args,
            device=device,
            is_shuffle=is_shuffle,
            optimiser=optimiser,  # Pass the existing optimizer
        )
        results.append(step_result)

        # Save model and optimizer state
        torch.save(model.state_dict(), model_log_dir / f"{model_name}/{model_name}_1")
        torch.save(
            optimiser.state_dict(),
            model_log_dir / f"{model_name}/{model_name}_1_optimizer",
        )
        print(f"Model and optimizer saved at: {model_name}/{model_name}_1")

    return predictions, results, model  # Return the trained model


def initialise_model(
    model_name: str,
) -> tuple[torch.nn.Module, AutoTokenizer | BertTokenizer]:
    """Takes a name of a model and returns its model initialisation

    Args:
      mode_name: name of the model

    Returns:
      A tuple contains the model"""

    if model_name == "phobert":
        model = PhoBERTSentClassifer(n_classes=3).to(device)

    elif model_name == "visobert":
        model = ViSoBERTSentClassifer(n_classes=3).to(device)

    elif model_name == "vibert4news":
        model = Vibert4newsSentClassifer(n_classes=3).to(device)

    elif model_name == "vnsbert":
        model = VnSBERTSentClassifer(n_classes=3).to(device)

    else:
        raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")

    return model


def initialise_tokeniser(
    model_name: str,
) -> tuple[torch.nn.Module, AutoTokenizer | BertTokenizer]:
    """Takes a name of a model and returns its tokeniser

    Args:
      mode_name: name of the model

    Returns:
      A tuple contains the tokeniser"""

    if model_name == "phobert":
        tokeniser = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

    elif model_name == "visobert":
        tokeniser = AutoTokenizer.from_pretrained("uitnlp/visobert")

    elif model_name == "vibert4news":
        tokeniser = BertTokenizer.from_pretrained("NlpHUST/vibert4news-base-cased")

    elif model_name == "vnsbert":
        tokeniser = AutoTokenizer.from_pretrained("keepitreal/vietnamese-sbert")

    else:
        raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")

    return tokeniser
