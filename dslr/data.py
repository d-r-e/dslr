import dslr.math as mth


def describe_(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = df.select_dtypes(include=numerics)
    df = df.dropna(axis=1, how='all')
    df = df.drop(columns='Index')
    sep = '| '
    print('{:18}'.format(''), end=sep)
    for col in df.columns:
        print('{:18.18}'.format(str(col)), end=sep)
    print()
    print(f'{"Count":18}', end=sep)
    for col in df.columns:
        m = mth.count_(df[col].values)
        print('{:>18.6f}'.format(m), end=sep)
    print()
    print(f'{"Mean":18}', end=sep)
    for col in df.columns:
        m = mth.mean_(df[col].values)
        print('{:>18.6f}'.format(m), end=sep)
    print()
    print(f'{"STD":18}', end=sep)
    for col in df.columns:
        m = mth.std_(df[col].values)
        print('{:>18.6f}'.format(m), end=sep)
    print()
