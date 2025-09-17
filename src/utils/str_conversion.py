def str_to_float(input: str, is_ger_format=True) -> float:
    """ Converts string to float, ignoring units and eventually converting from GER format (with comma as decimal point and dot magnitude indicator). """
    # convert format:
    if is_ger_format:
        input = input.replace('.', '').replace(',', '.')

    # drop letters:
    input = ''.join([char for char in input if not char.isalpha()])

    # convert to float and return:
    if input == '': return 0.0
    else:
        return float(input.strip())