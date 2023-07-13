from tabulate import tabulate

def visualize(df, small=False):
    print(f"Data Frame has {df.shape[0]} rows and {df.shape[1]} rows\n")
    print("These are the column names:")

    information = []
    counter = 0

    for column in df.columns:
        information.append([f"{counter}: {column}", f"{df[column].nunique()} unique values", f"{df[column].isna().sum()} are NaN", f"Type: {df[column].dtype}"])
        counter += 1

    print(tabulate(information, headers=['Column', 'Values', 'NaN', 'Type']))

    print("\n")

    if small == False:
      counter = 0
      for column in df.columns:
          print(f"{counter}: {column} has {df[column].nunique()} unique values. {df[column].isna().sum()} are NaN.")
          print(f"It's {df[column].dtype}. These are the unique values:")
          print(f"{df[column].unique()}\n")
          counter += 1