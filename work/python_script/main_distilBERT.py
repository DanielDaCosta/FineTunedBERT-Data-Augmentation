import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from utils import *
import os
from datasets import concatenate_datasets


##################
# ETA: 26:30 min #
##################

# out_distilbert_original.txt: Score:  {'accuracy': 0.923}

# out_distilbert_transformed.txt Score:  {'accuracy': 0.87332}

# out_distilbert_augmented_transformed.txt Score:  {'accuracy': 0.88216}

# out_distilbert_augmented_original.txt Score: Score:  {'accuracy': 0.9178}

# Set seed
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Tokenize the input
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Core training function
def do_train(args, model, train_dataloader, save_dir="./out_distilbert"):

    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    model.train()
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad() # zero gradients
            progress_bar.update(1)   
    
    print("Training completed...")
    print("Saving Model....")
    model.save_pretrained(save_dir)
    
    return
    
    
# Core evaluation function
def do_eval(eval_dataloader, output_dir, out_file):
    
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    model.eval()

    metric = evaluate.load("accuracy")

    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        
        # write to output file
        for i in range(predictions.shape[0]):
            out_file.write(str(predictions[i].item()) + "\n")
            #out_file.write("\n")
            out_file.write(str(batch["labels"][i].item()) + "\n\n")
            #out_file.write("\n\n")

    score = metric.compute()
    
    return score

# Created a dataladoer for the augmented training dataset
def create_augmented_dataloader(dataset):

    # 5000 randomly transformed examples
    train_augmented_size = 5000
    train_transformed_sample = dataset["train"].shuffle(seed=42).select(range(train_augmented_size))
    train_transformed_sample = train_transformed_sample.map(custom_transform, load_from_cache_file=False) 

    # Augment the training data with 5000 randomly transformed examples to create the new augmented training dataset
    # Final dataset train size: "25,000" + "5,000" = "30,000" 
    train_transformed_dataset = concatenate_datasets([dataset["train"], train_transformed_sample])                                                
    
    train_dataloader = None

    tokenized_dataset = train_transformed_dataset.map(tokenize_function, batched=True)

    # Prepare dataset for use by model
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    # Create dataloaders for iterating over the dataset
    train_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=8)
    
    ##### YOUR CODE ENDS HERE ######
    
    return train_dataloader

# Create a dataloader for the transformed test set
def create_transformed_dataloader(dataset, debug_transformation):
    
    # Print 5 random transformed examples
    if debug_transformation:
        small_dataset = dataset["test"].shuffle(seed=42).select(range(5))
        small_transformed_dataset = small_dataset.map(custom_transform, load_from_cache_file=False)
        for k in range(5):
            print("Original Example ", str(k))
            print(small_dataset[k])
            print("\n")
            print("Transformed Example ", str(k))
            print(small_transformed_dataset[k])
            print('='*30)

        exit()
      
    
    transformed_dataset = dataset["test"].map(custom_transform, load_from_cache_file=False)                                                    
    transformed_tokenized_dataset = transformed_dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_tokenized_dataset.remove_columns(["text"])
    transformed_tokenized_dataset = transformed_tokenized_dataset.rename_column("label", "labels")
    transformed_tokenized_dataset.set_format("torch")

    transformed_val_dataset = transformed_tokenized_dataset    
    eval_dataloader = DataLoader(transformed_val_dataset, batch_size=8)
    
    return eval_dataloader



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    # Arguments
    parser.add_argument("--train", action="store_true", help="train a model on the training data")
    parser.add_argument("--train_augmented", action="store_true", help="train a model on the augmented training data")
    parser.add_argument("--eval", action="store_true", help="evaluate model on the test set")
    parser.add_argument("--eval_transformed", action="store_true", help="evaluate model on the transformed test set")
    parser.add_argument("--model_dir", type=str, default="./out_distilbert")
    parser.add_argument("--debug_transformation", action="store_true", help="print a few transformed examples for debugging")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--small", action="store_true", help="use small dataset") 
    
    args = parser.parse_args()
    
    global device
    global tokenizer
    
    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    
    # Tokenize the dataset
    dataset = load_dataset("imdb")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Prepare dataset for use by model
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(4000))
    small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

    # Create dataloaders for iterating over the dataset
    train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(tokenized_dataset["test"], batch_size=8)

    if args.small:
      print("Using small dataloader")
      train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
      eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

    
    # Train model on the original training dataset
    if args.train:
        
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=2)
        model.to(device)
        do_train(args, model, train_dataloader, save_dir="./out_distilbert")
        
        # Change eval dir
        args.model_dir = "./out_distilbert"
        

    # Train model on the augmented training dataset
    if args.train_augmented:
        
        train_dataloader = create_augmented_dataloader(dataset)
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=2)
        model.to(device)
        do_train(args, model, train_dataloader, save_dir="./out_distilbert_augmented")
        
        # Change eval dir
        args.model_dir = "./out_distilbert_augmented"

        
    # Evaluate the trained model on the original test dataset
    if args.eval:
        
        out_file = os.path.basename(os.path.normpath(args.model_dir))
        out_file = out_file + "_original.txt"
        out_file = open(out_file, "w")
        
        score = do_eval(eval_dataloader, args.model_dir, out_file)
        print("Score: ", score)
        
        out_file.close()
        
        
    # Evaluate the trained model on the transformed test dataset
    if args.eval_transformed:
        
        out_file = os.path.basename(os.path.normpath(args.model_dir))
        out_file = out_file + "_transformed.txt"
        out_file = open(out_file, "w")
        
        eval_transformed_dataloader = create_transformed_dataloader(dataset, args.debug_transformation)
        score = do_eval(eval_transformed_dataloader, args.model_dir, out_file)
        print("Score: ", score)
        
        out_file.close()

