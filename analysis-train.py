import os
import json
import pathlib
import tensorflow as tf
from train_lib.security.homomorphic_addition import secure_addition


RESULT_PATH = '/opt/pht_results/'
DATA_PATH = '/opt/train_data/'

HE_RESULT_FILE = RESULT_PATH + 'results.he.json'
RESULT_FILE = RESULT_PATH + 'results.txt'


def get_user_pk():
    try:
        with open('/opt/train_config.json', 'r') as train_conf:
            conf = json.load(train_conf)
            bytes_key = bytes.fromhex(conf['creator']['paillier_public_key'])
            json_key = json.loads(bytes_key.decode('utf-8'))
            return json_key['n'], json_key['g']
    except Exception:
        default_key = dict({"n": "211888633188985050337719185074307935631",
                            "g": "29500842607978265371298695392488196009468683100322399408083516130147975452616"})
        return default_key['n'], default_key['g']


def paillier_addition(prev_result, local_result, number_to_add):
    try:
        curr_result = prev_result['analysis'][number_to_add]
        print("Previous secure addition value from {} {}".format(number_to_add, curr_result))
    except KeyError:
        print("Previous secure addition from {} empty".format(number_to_add))
        curr_result = None
    user_he_key = get_user_pk()

    return secure_addition(local_result, curr_result, int(user_he_key[0]), int(user_he_key[1]))


def load_if_exists(model_path: str):
    """
    Load previous computed results, if available
    :param model_path: Path of models or results to load
    :return: model
    """
    p = pathlib.Path(model_path)
    if pathlib.Path.is_file(p):
        print('Loading previous results')
        with open(p, "r") as model_file:
            model = json.load(model_file)
        return model
    else:
        return None


def save_results(results, result_path, file_to_save):
    """
    Create (if doesnt exist) a result directory and store the analysis results within
    :param results: Result content
    :param result_path:  Path of results file
    :return: store results as pickle file
    """
    dirPath = result_path
    try:
        # Create target Directory
        os.mkdir(dirPath)
        print('Directory {} created'.format(dirPath))
    except Exception as e:
        pass

    p = pathlib.Path(file_to_save)
    with open(p, 'w+') as results_file:
        json.dump(results, results_file)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True,
                              shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model


def training_pipeline(ds_train):
    # Build a training pipeline
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    return ds_train


def eval_pipeline(ds_test):
    # Build an evaluation pipeline
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    return ds_test


def train_model(model, ds_test):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )
    return model


if __name__ == '__main__':
    """
    Main analysis function of the train - a MNIST minimal demo, requires only result files and no models
    :return:
    """
    # Try to load previous results, if no exist create dictionary and print results before execution of analysis
    try:
        results = load_if_exists(RESULT_FILE)
        he_results = load_if_exists(HE_RESULT_FILE)
    except FileNotFoundError:
        print("No file available")
    if results is None:
        results = {'analysis': {}, 'discovery': {}}
        he_results = {'analysis': {}, 'discovery': {}}
    print("Results: {}".format(results))

    # Write analysis code here
    station = str(len(results["analysis"]) + 1)

    path = DATA_PATH
    try:
        data_dir = path + 'ds'
        print(data_dir)
        ds_train_demo = tf.data.Dataset.load(data_dir)

    except Exception as e:
        print('No data available {}'.format(e))
        exit(1)

    ds_train, ds_test, _ = get_dataset_partitions_tf(ds_train_demo, 20000)
    size_train = len(list(ds_train))  # for HE
    size_test = len(list(ds_test))  # for HE
    ds_train = training_pipeline(ds_train)
    ds_test = eval_pipeline(ds_test)

    if station == str(1):
        print("New Model created")
        model = create_model()
    else:
        print("Previous model loaded")
        model = tf.keras.models.load_model(RESULT_PATH + 'my_model')
        _, acc = model.evaluate(ds_test, verbose=2)
        print("Previous model accuracy before training: {:5.2f}%".format(100 * acc))

    model = train_model(model, ds_train)

    _, acc_local = model.evaluate(ds_test, verbose=2)

    analysis = {'accuracy_model': acc_local}

    results['analysis']['analysis_exec_' + str(len(results['analysis']) + 1)] = analysis

    # HE Encryption of sample and test size
    secure_sample_size = paillier_addition(he_results, size_train, 'training_samples')
    secure_test_size = paillier_addition(he_results, size_test, 'test_samples')

    he_results['analysis']['training_samples'] = secure_sample_size
    he_results['analysis']['test_samples'] = secure_test_size

    # print updated results and save models and results
    print("Updated results: {}".format(results))
    print("Updated HE results: {}".format(he_results))
    try:
        model.save(RESULT_PATH + '/my_model')
        save_results(results, RESULT_PATH, RESULT_FILE)
        save_results(he_results, RESULT_PATH, HE_RESULT_FILE)
        print("Model and results saved")
    except Exception as e:
        print('Error saving model or results: {}'.format(e))
