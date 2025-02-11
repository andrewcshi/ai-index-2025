import pandas as pd

def combine_csv_files(file_paths, output_file):
    dataframes = []

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    combined_df.to_csv(output_file, index=False)

aaai_files = [
    "data/aaai_papers1.csv", 
    "data/aaai_papers2.csv", 
    "data/aaai_papers3.csv", 
    "data/aaai_papers4.csv", 
    "data/aaai_papers5.csv", 
    "data/aaai_papers6.csv", 
    "data/aaai_papers7.csv", 
    "data/aaai_papers8.csv", 
    "data/aaai_papers9.csv", 
    "data/aaai_papers10.csv",
    "data/aaai_papers11.csv",
    "data/aaai_papers12.csv",
    "data/aaai_papers13.csv",
    "data/aaai_papers14.csv",
    "data/aaai_papers15.csv",
    "data/aaai_papers16.csv",
    "data/aaai_papers17.csv",
    "data/aaai_papers18.csv",
    "data/aaai_papers19.csv",
    "data/aaai_papers20.csv",
    "data/aaai_papers21.csv"
]

out_file = "data/aaai_papers.csv"
combine_csv_files(aaai_files, out_file)