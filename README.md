# movs_classification_2023
## **Команда: Коротков Иван, Шуляк Данила, Силаев Николай**

## **Куратор: Пирогов Вячеслав**

### **Описание проекта:**
Решаем задачу классификации изображений животных по их типу.

Задача классификации изображений - одна из классических задач компьютерного зрения,
которая не теряет своей актуальности, поскольку в различных приложениях и сервисах 
требуется классифицировать объекты, изображенные на фотографии (определение пользователя, марки машины и т.д.).

### **План работы:**

1. Первый этап
- Разведочный анализ и первичиная аналитика данных
- Очистка данных - укрупнение категорий, удаление неподходящих элементов и т.д. (при необходимости)

2. Второй этап (ML)
- Создание веб-сервиса, который по запросу выводит картинки из заданной категории (вид животного) по запросу пользователя
- ML задача - классификация заданной фотографии пользователя (определение вида животного)
- DL задача - применение нейронных сетей для решения задачи классификации (определение вида животного)

3. Третий этап (DL)
- Углубление DL задачи, использование новых моделей
- Создание генератора новых изображений из одной категории (Cat) и гибрида двух (Cat и Dog)


### **Данные:**
Данные собраны из двух готовых датасетов с животными
1. Датасет 1

Ссылка: https://www.kaggle.com/datasets/alessiocorrado99/animals10

Кол-во изображений: 26810

Кол-во категорий: 10

Ср. кол-во изображений на категорию: ~2680


2. Датасет 2
   
Ссылка: https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset
Кол-во изображений: 29071
Кол-во категорий: 80
Ср. кол-во изображений на категорию: ~360


3. Объединенный датасет (Data)

Ссылка: https://disk.yandex.ru/d/Mep8flE-QMaIuw
Кол-во категорий: 81
Кол-во изображений всего: 55219
Ср. кол-во изображений на категорию: ~680

### **Модели:**
Было обучено 7 варианов моделей:

1. SIFT SVC
2. HOG SVC
3. ResNet - ретренинг
4. VIT - ретренинг (лучший результат)
5. SVM и XGB на фичах ResNet
6. SVM и XGB на фичах VIT
7. СLIP (zero-shot)

### **Генерация:**
Была обучена модель StyleGAN3 на классах Cat и Cat+Dog.
Модель и механизм обучения были взяты из: https://github.com/NVlabs/stylegan3

### **Телеграм бот:**
Был сделан телеграм бот, определяющий класс пользовательского изображения животного с помощью моделей ResNet и VIT, а также генерирующий изображения

Бот задеплоен удаленно на DigitalOcean, работает здесь: @animal_guesser_bot

### **REPRODUCE CODE:**
1. Clone repo and install packages

```python
git clone https://github.com/hserocks/movs_classification_2023.git
cd movs_classification_2023
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # Install pytorch separately for CUDA to work. If non-Windows, see appropriate command here: https://pytorch.org/get-started/locally/

```

2. Get data

```python
python scripts/downloader.py --selection all 
```

3. Train ResNet / or VIT on our data (can skip and go to 4. to use downloaded weights)

```python
python scripts/train.py --model resnet --path Data_small
python scripts/train.py --model vit --path Data_small
```

4. Evaluation of ResNet / VIT / CLIP

```python
python scripts/train.py --model resnet --path Data_small --eval True
python scripts/train.py --model vit --path Data_small --eval True
python scripts/train.py --model clip --path Data_small --eval True

```

5. Extract features from ResNet / VIT (can skip and go to 6. to use downloaded features)

```python
python scripts/get_features.py --model resnet
python scripts/get_features.py --model vit
```

6. Train SVM and XGB on extracted features (can skip and go to 7. to use downloaded models)

```python
python scripts/train_svm_xgb.py --features resnet --model svm
python scripts/train_svm_xgb.py --features resnet --model xgb
python scripts/train_svm_xgb.py --features vit --model svm
python scripts/train_svm_xgb.py --features vit --model xgb
```

7. Evaluate above 4 models

```python
python scripts/train_svm_xgb.py --features resnet --model svm --eval True
python scripts/train_svm_xgb.py --features resnet --model xgb --eval True
python scripts/train_svm_xgb.py --features vit --model svm --eval True
python scripts/train_svm_xgb.py --features vit --model xgb --eval True
```

8. Inference
```python
# python inference.py --model {model_name} --path  {image_path}
python scripts/inference.py --model resnet_svm --path  Deer.jpg
python scripts/inference.py --model vit --path  Deer.jpg
python scripts/inference.py --model resnet --path https://cdn.britannica.com/71/234471-050-093F4211/shiba-inu-dog-in-the-snow.jpg
python scripts/inference.py --model vit_xgb --path https://cdn.britannica.com/71/234471-050-093F4211/shiba-inu-dog-in-the-snow.jpg

```