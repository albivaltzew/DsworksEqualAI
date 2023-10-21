import cv2
import pandas as pd

# Путь к TSV-файлу
tsv_file = "slovo_annotations/SLOVO_DATAFRAME.tsv"

# Загрузка TSV-файла в DataFrame
data = pd.read_csv(tsv_file, delimiter='\t')

# Папка, где находятся видео
video_folder = "slovo/"

# Проход по каждой строке TSV-файла
for index, row in data.iterrows():
    attachment_id = row["attachment_id"]
    begin_frame = int(row["begin"])
    end_frame = int(row["end"])
    
    # Создание объекта VideoCapture
    video_path = video_folder + attachment_id + ".mp4"
    cap = cv2.VideoCapture(video_path)

    # Перемещение кадров к начальному кадру
    cap.set(cv2.CAP_PROP_POS_FRAMES, begin_frame)

    frames = []
    frame_num = 0

    # Чтение и сохранение обрезанных кадров
    while cap.isOpened() and frame_num <= (end_frame - begin_frame):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

        frame_num += 1

    # Закрыть видео
    cap.release()

    # Обновить значения кадра в DataFrame
    data.at[index, "begin"] = 0
    data.at[index, "end"] = frame_num - 1

    # Сохранить обрезанные кадры в новом видео
    out_video_path = "cropped_slovo/" + attachment_id + ".mp4"
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (row["width"], row["height"]))

    for frame in frames:
        out.write(frame)

    out.release()

# Сохранить обновленный TSV-файл
data.to_csv("cropped_slovo_annotations/SLOVO_DATAFRAME.tsv", sep='\t', index=False)
