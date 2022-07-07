import geopandas as gpd


def read_shapes(name):
    return gpd.read_file(f'data/{name}.shp')


if __name__ == '__main__':
    h = read_shapes('hexagons')

