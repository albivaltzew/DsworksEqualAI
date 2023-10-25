import pandas as pd
import os
import random
import json
import shutil


# --------------------------
random.seed(42)

# Load the JSON file with class mappings
with open("slovo_annotations/classes.json", "r", encoding="utf-8") as file:
    class_mappings = json.load(file)


# Путь к TSV-файлу с данными
tsv_file = "cropped_slovo_annotations/SLOVO_DATAFRAME.tsv"

# Function to map text to class based on the JSON file
def map_text_to_class(text):
    return class_mappings.get(text, -1)  # -1 if text is not found in the JSON file

# Загрузка данных из TSV-файла в DataFrame
data = pd.read_csv(tsv_file, delimiter='\t')

data["class"] = data['text'].apply(map_text_to_class)
data["attachment_id"] = data["attachment_id"].apply(lambda x: x + ".mp4")

print(data)

# Создание списка уникальных пользователей
unique_users = data['user_id'].unique()


# Случайное перемешивание пользователей
random.shuffle(unique_users)

# Задайте процентное соотношение для обучающего набора (например, 80%)
train_ratio = 0.8

# Определение размеров тренировочного и валидационного наборов
train_size = int(train_ratio * len(unique_users))
print(train_size)
val_size = len(unique_users) - train_size

# Разделение пользователей на тренировочный и валидационный наборы
train_users = unique_users[:train_size]
val_users = unique_users[train_size:]

train_data = data[data['user_id'].isin(train_users)]
val_data = data[data['user_id'].isin(val_users)]


train_data = train_data[["attachment_id", "class"]]
train_data.to_csv("slovo_train_video.txt", sep=" ", index=False, header=False)
val_data = val_data[["attachment_id", "class"]]
val_data.to_csv("slovo_val_video.txt", sep=" ", index=False, header=False)


print(train_data)
print(val_data)
print(len(train_data), len(val_data))


# Define the source and destination folders
source_folder = "cropped_slovo/"  # Replace with the path to your source folder
train_folder = "slovo_split/train/"
val_folder = "slovo_split/val/"

# Iterate over the first DataFrame (train data)
for _, row in train_data.iterrows():
    attachment_id = row['attachment_id']
    attachment_class = row['class']
    
    # Construct source and destination file paths
    source_file = source_folder + attachment_id
    destination_file = train_folder + attachment_id

    # Move the file to the train folder
    shutil.copy(source_file, destination_file)

# Iterate over the second DataFrame (validation data)
for _, row in val_data.iterrows():
    attachment_id = row['attachment_id']
    attachment_class = row['class']
    
    # Construct source and destination file paths
    source_file = source_folder + attachment_id
    destination_file = val_folder + attachment_id
    
    # Move the file to the val folder
    shutil.copy(source_file, destination_file)





#------------------------------------



# random.seed(42)

# # Путь к TSV-файлу с данными
# tsv_file = "cropped_slovo_annotations/SLOVO_DATAFRAME.tsv"

# # Загрузка данных из TSV-файла в DataFrame
# data = pd.read_csv(tsv_file, delimiter='\t')

# # print(data)
# data_cropped = data[["attachment_id","user_id" ,"text"]]
# # print(data_cropped)

# text_column = data["text"]
# print(text_column)




# # Function to map text to class based on the JSON file
# def map_text_to_class(text):
#     return class_mappings.get(text, -1)  # -1 if text is not found in the JSON file

# # Add a new column "class" to the DataFrame
# data['class'] = data['text'].apply(map_text_to_class)

# # # Display the updated DataFrame
# print(data)

# # mapped_df = data[["text", "class"]]
# # mapped_df.to_csv("video-class.txt", sep="\t", index=False, header=True)
# # print(mapped_df)


# dataset = data[["attachment_id", "class"]]
# dataset["attachment_id"] = dataset["attachment_id"].apply(lambda x: x + ".mp4")
# print(dataset)

# read classes.json

# # Load the JSON data from your file
# with open('slovo_annotations/classes.json', 'r', encoding='utf-8') as json_file:
#     data = json.load(json_file)

# # Create a DataFrame with a single column
# df = pd.DataFrame(data.items(), columns=['Text', "class"])

# # Save the DataFrame to a text file (tab-separated)
# df.to_csv('output.txt', sep='\t', index=False, header=True)
# # Display the DataFrame
# print(df)
