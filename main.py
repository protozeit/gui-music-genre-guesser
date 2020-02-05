from backend.utils import load_resources, extract_features, predict_class


def main():
    print('Loading...')
    pca, filler_track, model = load_resources()
    print('input filename of wav: ')
    s = input()

    X = extract_features(s, filler_track, pca)
    print(predict_class(X, model))

if __name__ == '__main__':
    main()
