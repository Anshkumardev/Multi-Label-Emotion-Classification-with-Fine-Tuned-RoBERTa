import argparse
import datasets
import pandas as pd
import transformers
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from transformers import TFAutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize(examples):
    """Tokenize and encode the text for RoBERTa."""
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)

def create_custom_model(num_labels, dropout_rate=0.4):
    """Create a custom model with RoBERTa and additional pooling and dropout layers."""

    roberta = TFAutoModel.from_pretrained("roberta-base")
    input_ids = tf.keras.Input(shape=(64,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(64,), dtype=tf.int32, name="attention_mask")
    outputs = roberta(input_ids, attention_mask=attention_mask)
    last_hidden_state = outputs.last_hidden_state  
    pooled_output = tf.reduce_mean(last_hidden_state, axis=1)
    dropout = tf.keras.layers.Dropout(dropout_rate)(pooled_output)
    logits = tf.keras.layers.Dense(num_labels, activation="sigmoid")(dropout)
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=logits)
    return model

def train(train_path="/content/drive/MyDrive/Colab Notebooks/final-project-Anshkumardev/train.csv", dev_path="/content/drive/MyDrive/Colab Notebooks/final-project-Anshkumardev/dev.csv"):

    """Train the model using the mentioned Hyper parameters: LR: 0.00002, Batch size: 16, Epoc: 3, Optimizer: Adam. Using F1 Score as the evaluation matrix. Also, not saving the model to the directory and using it directly as it was creating some error while running in the cloud environment."""
    data = datasets.load_dataset("csv", data_files={"train": train_path, "validation": dev_path})
    labels = data["train"].column_names[1:] 
    num_labels = len(labels)
    model = create_custom_model(num_labels, dropout_rate=0.1)
    data = data.map(lambda example: {"labels": [float(example[l]) for l in labels]})
    data = data.map(tokenize, batched=True)
    train_dataset = data["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask"], label_cols="labels", batch_size=16, shuffle=True
    )
    dev_dataset = data["validation"].to_tf_dataset(
        columns=["input_ids", "attention_mask"], label_cols="labels", batch_size=16
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00002)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False) 
    metrics = [tf.keras.metrics.F1Score(average="micro", threshold=0.5)]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.fit(
        train_dataset,
        validation_data=dev_dataset,
        epochs=3,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_binary_accuracy", patience=3, mode='max', restore_best_weights=True)
        ]
    )
    
    return model

def predict(model, input_path="/content/drive/MyDrive/Colab Notebooks/final-project-Anshkumardev/dev.csv"):
    """Prdicting the labels on the validation set. The path here can be changed to test data when the test phase starts."""
    df = pd.read_csv(input_path)
    data = datasets.Dataset.from_pandas(df)
    data = data.map(tokenize, batched=True)
    tf_dataset = data.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        batch_size=16
    )
    predictions = model.predict(tf_dataset)
    df.iloc[:, 1:] = (predictions > 0.5).astype(int)
    df.to_csv("submission.zip", index=False, compression=dict(method='zip', archive_name='submission.csv'))
    
model = train()

predict(model)
