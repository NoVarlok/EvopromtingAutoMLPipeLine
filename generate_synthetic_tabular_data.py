from be_great import GReaT
import pandas as pd


if __name__ == '__main__':
    dataset_path = '/home/lyakhtin/repos/hse/krylov2/PrepareDatasets/raw_datasets/WineQT.csv'
    save_dataset_path = '/home/lyakhtin/repos/hse/krylov2/PrepareDatasets/raw_datasets/WineQT_synthetic.csv'
    data = pd.read_csv(dataset_path)
    model = GReaT(llm='distilgpt2', batch_size=8, epochs=25)
    model.fit(data)
    print(f'Original {len(data)} samples')
    synthesized_size = 10 * len(data)
    print(f'Synthesized {synthesized_size} samples')
    synthetic_data = model.sample(n_samples=synthesized_size)
    synthetic_data.to_csv(save_dataset_path)
