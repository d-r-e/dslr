import dslr.math as mth


def describe_(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = df.select_dtypes(include=numerics)
    df = df.dropna(axis=1, how='all')
    sep = ' '
    print('{:9}'.format(''), end=sep)
    for col in df.columns:
        print('{:>14.14}'.format(str(col)), end=sep)
    print()
    print(f'{"Count":9}', end=sep)
    for col in df.columns:
        m = mth.count_(df[col].values)
        print('{:>14.6f}'.format(m), end=sep)
    print()
    print(f'{"Mean":9}', end=sep)
    for col in df.columns:
        m = mth.mean_(df[col].values)
        print('{:>14.6f}'.format(m), end=sep)
    print()
    print(f'{"STD":9}', end=sep)
    for col in df.columns:
        m = mth.std_(df[col].values)
        print('{:>14.6f}'.format(m), end=sep)
    print()
    print(f'{"Min":9}', end=sep)
    for col in df.columns:
        m = mth.min_(df[col].values)
        print('{:>14.6f}'.format(m), end=sep)
    print()
    print(f'{"25%":9}', end=sep)
    for col in df.columns:
        m = mth.percentile_(25, df[col].values)
        print('{:>14.6f}'.format(m), end=sep)
    print()
    print(f'{"50%":9}', end=sep)
    for col in df.columns:
        m = mth.percentile_(50, df[col].values)
        print('{:>14.6f}'.format(m), end=sep)
    print()
    print(f'{"75%":9}', end=sep)
    for col in df.columns:
        m = mth.percentile_(75, df[col].values)
        print('{:>14.6f}'.format(m), end=sep)
    print()
    print(f'{"Max":9}', end=sep)
    for col in df.columns:
        m = mth.max_(df[col].values)
        print('{:>14.6f}'.format(m), end=sep)
    print()
