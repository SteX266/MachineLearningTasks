import numpy as np
import pandas as pd


def find_knn(target: np.ndarray, x_train: np.ndarray, y_train: np.ndarray, h: float):
    neighbours = []
    for i in range(len(x_train)):
        if distance(target, x_train[i]) < h:
            neighbours.append(y_train[i])
    return neighbours


def distance(x: np.ndarray, y: np.ndarray):
    return np.linalg.norm(x - y)


def rmse(y_predicted: np.ndarray, y_actual: np.ndarray):
    err = y_actual - y_predicted
    err = np.square(err).sum() / len(y_actual)
    err = np.sqrt(err)
    return err


def knn(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, h=5.0):
    y_predicted = []
    for x in x_test:
        neighbours = np.array(find_knn(x, x_train, y_train, h))
        y_predicted.append(neighbours.mean())

    y_predicted = np.array(y_predicted)

    return y_predicted


def normalization(x, mean, std):
    return (x - mean) / std


def denormalize(x, mean, std):
    return (x * std) + mean


def cross_validation(df: pd.DataFrame,train_percentage=0.8,):
    shuffled_data = df.sample(frac=1, random_state=69)

    n = len(shuffled_data)

    train_size = int(train_percentage * n)

    train_data = shuffled_data[:train_size]
    test_data = shuffled_data[train_size:]

    return train_data, test_data

def year_deviation(year):
    return abs(year - 2010)


def mileage_deviation(miles):
    return abs(miles - 208_000)


if __name__ == '__main__':
    df = pd.read_csv('zadatak2/train.csv')
    # Perform one-hot encoding on the categorical columns
    grouped_df = df.groupby("make")['price'].mean().reset_index()

    df = df.drop(columns=['engine_size', 'color'])
    df = pd.get_dummies(df, columns=['make', 'category', 'fuel', 'transmission'], drop_first=True)

    index = 0
    min_price = min(grouped_df['price'])
    print(min_price)
    coefficients = {}
    for group in grouped_df['make']:
        coefficients[group] = grouped_df['price'][index] / min_price
        index += 1
    print(coefficients)
    for d in df:
        if 'make' in d:
            print(d[5:])
            index = 0
            for value in df[d]:
                if not value:
                    df[d][index] = 0
                else:
                    df[d][index] = coefficients[d[5:]]
                index += 1
    df['mileage'] = df['mileage'].apply(mileage_deviation)
    mean = df['mileage'].mean()
    std = df['mileage'].std()
    df['mileage'] = df['mileage'].apply(normalization, args=(mean, std))

    df['year'] = df['year'].apply(year_deviation)
    mean = df['year'].mean()
    std = df['year'].std()
    df['year'] = df['year'].apply(normalization, args=(mean, std))


    mean = df['price'].mean()
    std = df['price'].std()
    df['price'] = df['price'].apply(normalization, args=(mean, std))

    train_data, test_data = cross_validation(df)

    x_train = train_data.drop(columns=['price']).values
    y_train = train_data['price'].values



    x_test = test_data.drop(columns=['price']).values
    y_test = test_data['price'].values
    y_test = denormalize(y_test, mean, std)
    h = 1.55
    while h < 20:
        y_predicted = denormalize(knn(x_train, y_train, x_test, h), mean, std)
        print(mean)
        print(f"RMSE for h = {h}: {rmse(y_predicted, y_test)}")

        h *= 1.1
