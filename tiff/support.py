def generate_supports(config, parcels, logger):
    """
    generate_supports generates support material for all parcels below the bottom surface such that the output
    shape is self-supporting and a minimal amount of support material is used
    :param config:
    :param parcels:
    :param logger:
    """
    for index, parcel in parcels:
        generate_support(config, index, parcel, logger)


def generate_support(config, index, parcel, logger):
    return
