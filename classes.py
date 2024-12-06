import os
import shutil
import logging


def find_image(annotation_file, input_dir, image_extension=".jpg"):
    """
    Ищет изображение, соответствующее файлу аннотации, в указанной папке и всех вложенных.
    :param annotation_file: Имя файла аннотации (без пути).
    :param input_dir: Путь к корневой папке для поиска.
    :param image_extension: Расширение изображения (по умолчанию .jpg).
    :return: Путь к изображению или None, если изображение не найдено.
    """
    base_name = os.path.splitext(annotation_file)[0]  # Убираем расширение
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file == base_name + image_extension:
                return os.path.join(root, file)
    return None


def process_yolo_dataset(
        input_dir,
        output_dir,
        target_classes,
        remove_classes,
        new_class_id=0
):
    """
    Обрабатывает датасет YOLO: фильтрует аннотации, оставляет только указанные классы и меняет их id.

    :param input_dir: Путь к исходной папке с датасетом.
    :param output_dir: Путь к новой папке для сохранения обработанного датасета.
    :param target_classes: Список ID классов, которые нужно оставить.
    :param remove_classes: Список ID классов, которые нужно удалить.
    :param new_class_id: Новый ID для целевых классов. По умолчанию 0.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("process_yolo.log"), logging.StreamHandler()],
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                annotation_path = os.path.join(root, file)
                rel_dir = os.path.relpath(root, input_dir)  # Относительный путь папки аннотаций
                output_annotation_dir = os.path.join(output_dir, rel_dir)
                os.makedirs(output_annotation_dir, exist_ok=True)

                output_annotation_path = os.path.join(output_annotation_dir, file)

                # Ищем соответствующее изображение
                image_path = find_image(file, input_dir)
                if not image_path:
                    logging.warning(f"Для аннотации {annotation_path} не найдено изображение. Пропуск.")
                    continue

                # Сохраняем относительный путь папки изображения
                image_rel_dir = os.path.relpath(os.path.dirname(image_path), input_dir)
                output_image_dir = os.path.join(output_dir, image_rel_dir)
                os.makedirs(output_image_dir, exist_ok=True)

                output_image_path = os.path.join(output_image_dir, os.path.basename(image_path))

                with open(annotation_path, 'r') as f:
                    lines = f.readlines()

                # Фильтруем строки с целевыми классами и удаляем ненужные
                filtered_lines = [
                    line for line in lines
                    if int(line.split()[0]) in target_classes
                ]

                # Если файл аннотации содержит целевой класс
                if filtered_lines:
                    # Обновляем ID целевых классов
                    updated_lines = []
                    for line in filtered_lines:
                        parts = line.split()
                        class_id = int(parts[0])
                        if class_id in target_classes:
                            parts[0] = str(new_class_id)
                        updated_lines.append(' '.join(parts) + '\n')

                    # Копируем аннотации и изображения в новую папку
                    shutil.copy2(image_path, output_image_path)
                    with open(output_annotation_path, 'w') as f:
                        f.writelines(updated_lines)

                    logging.info(
                        f"Обработаны: {image_path} и {annotation_path}, "
                        f"целевые классы: {target_classes}, новый ID: {new_class_id}"
                    )
                else:
                    # Если целевой класс отсутствует, не копируем данные
                    logging.info(f"Удалены: {image_path} и {annotation_path} (нет целевого класса)")
    logging.info("Обработка завершена.")


process_yolo_dataset(
    input_dir="D:\Детекция\Футбольный мяч\Football Analyzer.v1i.yolov8",
    output_dir="D:\Детекция\Футбольный мяч\ofutbool_dataset",
    target_classes=[0],  # ID классов, которые нужно оставить
    remove_classes=[1,2],  # ID классов, которые нужно удалить
    new_class_id=0  # Новый ID для целевых классов
)
