def temporal_train_test_split(cases, train_ratio = 0.8):
    """ Splits cases temporally assuming cases are already sorted by completion time (see preprocessing). """
    if not cases:
        raise ValueError("Empty case list")
    
    # Assume cases are sorted by completion time
    n_train = max(1, int(len(cases) * train_ratio))
    
    return cases[:n_train], cases[n_train:]

