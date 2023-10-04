import pandas as pd
from catboost import CatBoostRegressor
from scipy.spatial import cKDTree
import os
import numpy as np


def import_dataset_from_file(path_to_file: str) -> pd.DataFrame:
    """
    Функция импортирования исходных данных.
    :param path_to_file: путь к загружаемому файлу;
    :return: структура данных.
    """
    dataset = pd.read_table(path_to_file, delim_whitespace=True, names=['x', 'y', 'z'])

    return dataset


def export_dataset_to_file(dataset: pd.DataFrame, gen_path):
    """
    Функция экспортирования результата в файл result.txt.
    :param dataset: входная структура данных.
    """
    n, c = dataset.shape

    assert c == 3, 'Количество столбцов должно быть 3'
    assert n == 1196590, 'Количество строк должно быть 1196590'

    with open(os.path.join(gen_path, 'data', 'Result.txt'), 'w') as f:
        for i in range(n):
            f.write('%.2f %.2f %.5f\n' % (dataset.x[i], dataset.y[i], dataset.z[i]))


def calc(point_grid, maps, point_dataset):

    # РАБОТА С POINT GRID
    # обединение point_grid с maps
    a = []
    for i, map in enumerate(maps):
        a.append(point_grid.merge(map, on=['x', 'y'], how='left'))
    for n, i in enumerate(a):
        point_grid[f'map_{n}'] = i['z_y']
    point_grid = point_grid.drop('z', axis=1)
    # осталось всего 500 строчек с неизвестными данными

    # поиск ближайших 4 точек к неизвестным и нахождение значения для неизвестной точки с учетом расстояния от этих 4 ближайших
    point_data_all = point_grid
    for i, map in enumerate(maps):
        point_data = point_data_all[point_data_all[f'map_{i}'].isna()]
        grid_data = map
        grid_points = grid_data[['x', 'y']].values
        grid_values = grid_data['z'].values

        # Создайте cKDTree для ускоренного поиска ближайших соседей
        tree = cKDTree(grid_points)

        # Функция для нахождения z1 для одной точки
        def find_z1(row):
            distances, indices = tree.query([row['x'], row['y']], k=4)  # Находим 4 ближайшие точки
            weights = 1 / distances
            weighted_z = np.sum(grid_values[indices] * weights) / np.sum(weights)

            return weighted_z

        # Примените функцию find_z1 к каждой строке в point_data и создайте новый столбец 'z1' с результатами

        point_data[f'map_{i}'] = point_data.apply(find_z1, axis=1)
        point_data_all.loc[point_data_all[f'map_{i}'].isna(), f'map_{i}'] = point_data[f'map_{i}']

    point_grid = point_data_all



    # РАБОТА С POINT DATA
    # в point data нет пересечений с сеткой maps, поэтому сразу поиск ближайших точек
    # поиск ближайших 4 точек к неизвестным и нахождение значения для неизвестной точки с учетом расстояния от этих 4 ближайших
    point_data_all = point_dataset
    for i, map in enumerate(maps):
        point_data = point_data_all
        grid_data = map
        grid_points = grid_data[['x', 'y']].values
        grid_values = grid_data['z'].values

        # Создайте cKDTree для ускоренного поиска ближайших соседей
        tree = cKDTree(grid_points)

        # Функция для нахождения z1 для одной точки
        def find_z1(row):
            distances, indices = tree.query([row['x'], row['y']], k=4)  # Находим 4 ближайшие точки
            weights = 1 / distances
            weighted_z = np.sum(grid_values[indices] * weights) / np.sum(weights)

            return weighted_z

        # Примените функцию find_z1 к каждой строке в point_data и создайте новый столбец 'z1' с результатами

        point_data[f'map_{i}'] = point_data.apply(find_z1, axis=1)
        point_data_all.loc[point_data_all[f'map_{i}'].isna(), f'map_{i}'] = point_data[f'map_{i}']

    point_dataset = point_data_all

    print(point_dataset.columns)
    print(point_grid.columns)


    # обучение модели
    X = point_grid
    X_t = point_dataset.drop('z', axis = 1)
    y = point_dataset['z']
    model = CatBoostRegressor(verbose=10)
    model.fit(X_t, y=y)
    y_pred = model.predict(X)

    out = X[['x', 'y']]
    out['z'] = y_pred



    # УЛУЧШЕНИЕ РЕЗУЛЬТАТА
    # применяю ту же технику уже к результату. Нахожу ближайшую точку из point_data к
    grid_data = out  # где искать -

    map = point_dataset  # данные в которых искать -

    grid_points = grid_data[['x', 'y']].values
    grid_values = grid_data['z'].values

    # Создайте cKDTree для ускоренного поиска ближайших соседей
    tree = cKDTree(grid_points)
    indexes = []

    # Функция для нахождения z1 для одной точки
    def find_z1(row):

        distances, indices = tree.query([row['x'], row['y']], k=1)

        grid_data['z'].iloc[indices] = row['z']
        indexes.append(indices)
        return row['z']

    # Примените функцию find_z1 к каждой строке в point_data и создайте новый столбец 'z1' с результатами

    a = point_dataset.apply(find_z1, axis=1)

    return out



if __name__ == "__main__":
    os.chdir('..')
    gen_path = os.getcwd()
    # Вспомогательные данные, по которым производится моделирование
    map_1_dataset = import_dataset_from_file(os.path.join(gen_path, 'data', "Map_1.txt"))
    map_2_dataset = import_dataset_from_file(os.path.join(gen_path, 'data', "Map_2.txt"))
    map_3_dataset = import_dataset_from_file(os.path.join(gen_path, 'data', "Map_3.txt"))
    map_4_dataset = import_dataset_from_file(os.path.join(gen_path, 'data', "Map_4.txt"))
    map_5_dataset = import_dataset_from_file(os.path.join(gen_path, 'data', "Map_5.txt"))
    maps = [map_1_dataset,  map_2_dataset, map_3_dataset, map_4_dataset,map_5_dataset]

    # Данные, по которым необходимо смоделировать
    point_dataset = import_dataset_from_file(os.path.join(gen_path, 'data', "Point_dataset.txt"))

    # Точки данных, в которые необходимо провести моделирование (сетка данных)
    point_grid = import_dataset_from_file(os.path.join(gen_path, 'data', "Result_schedule.txt"))

    # Блок вычислений
    dataset = calc(point_grid, maps, point_dataset)

    # Экспорт данных в файл (смотри Readme.txt)
    export_dataset_to_file(dataset=dataset, gen_path= gen_path)
