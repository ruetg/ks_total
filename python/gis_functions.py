import numpy as np
import rasterio as rio
from shapely.geometry import shape
from rasterio.features import shapes
import lem
import geopandas as gpd


def snap_pourpoint(acc, basin_y, basin_x, pt_y, pt_x, target_area, dA, search_area):
    """
    Parameters
    acc: Accumulation grid
    basin_y: Lat vector for DEM
    basin_x: LOn vector for DEM
    pt_y: Lat of pour point
    pt_x: Lon of pour point
    target_area: Minimum drainage area for snap
    dA: cell size in m2
    search_area: Search area for snap (fraction of DEM

    Returns
    ysnap and xsnap - LAT , LON of new snapped pour point

    """
    x = np.argmin((basin_x - pt_x) ** 2)
    y = np.argmin((basin_y - pt_y) ** 2)
    print(pt_y)
    print(min(basin_y))
    A1 = 0
    w = 1
    m, n = np.shape(acc)
    print(A1)
    print(target_area)
    while A1 < target_area:
        if (x-w<0) | (x+w+1>=n) | (y-w<0) | (y+w+1>=m):

            break

        ny, nx = np.where(acc[y - w:y + w + 1, x - w:x + w + 1] >= np.max(
            acc[y - w:y + w + 1, x - w:x + w + 1]) / 1.25)  # We find the closest point within the min
        print('here')
        xysnap = np.argmin((ny - w) ** 2 + (nx - w) ** 2)
        ysnap = ny[xysnap] + y - w
        xsnap = nx[xysnap] + x - w
        A1 = acc[ysnap, xsnap] * dA
        print(A1)
        w += 1
        if w > search_area * (m / 2 + n / 2):
            break
    return ysnap, xsnap


def generate_pour_point_catchment(filenm, outnm, pt, fillsink=True, target_area=1e5, search_area=.05, flowdir=None):
    """
    Parameters
    filenm: Location/Name of input DEM
    outnm: Base Location/name of outputs
    pt: Pour Point Lat,Lon
    fillsink: Fill DEM sinks?
    target_area: Minimum area to snap to
    search_area: Area to search as a fraction of the whole DEM

    Returns
    df: GeoDataFrame shape of your basin
    acc: Acumulation grid

    """
    DEM = lem.simple_model()  # Initiate the grid object
    DEM.turn_on_off_dynamic_bc(
        False)  # This means that outlets are assigned, not dynamic i.e. everywhere below sea level.
    f = rio.open(filenm)  # File containing DEM
    Z = np.float64(np.squeeze(f.read()))
    #
    #
    # Z[Z<=0]=np.nan
    bc = np.zeros_like(Z)
    bc[Z <= 0] = 1
    m, n = np.shape(Z)
    # Z+=np.random.rand(m,n)*1000

    DEM.set_z(Z)
    # DEM.set_bc(bc)

    lat = np.array([f.xy(i, 0)[1] for i in range(m)])
    lon = np.array([f.xy(0, i)[0] for i in range(n)])
    dL = 111000 * np.cos(np.mean(pt[1]) * np.pi / 180)  # m per degree
    dx = np.mean(np.diff(lon)) * dL  # m per pixel, avg x
    print('dx=' + str(dx))
    dy = np.mean(np.diff(lat)) * 111000  # m per pixel, avg y
    dA = dy * dx
    DEM.dx = dx
    DEM.dy = dy
    if fillsink:
        Z = DEM.get_z().copy()
        Z[np.isnan(Z)] = -9999
        DEM.set_z(Z)
        print('Filling DEM')
        DEM.sinkfill()
        Z = DEM.get_z()
        Z[Z<=0] = np.nan
        DEM.set_z(Z)
    Z = DEM.get_z()
    Z[Z <= 1] = np.nan
    DEM.set_z(Z.copy())
    DEM.set_bc(np.zeros_like(Z))
    DEM.slp()
    if flowdir is not None:
        DEM.receiver = flowdir

    DEM.stack()
    DEM.acc()
    acc = DEM.A.copy()

    ys, xs = snap_pourpoint(acc, lat, lon, pt[0], pt[1], target_area, dA, search_area)
    DEM.pour_point = np.int64([ys, xs])
    DEM.stack()
    DEM.acc()
    Z = np.zeros(np.shape(DEM.get_z())).ravel(order='F')
    Zi = DEM.get_z().ravel(order='F')
    for i in range(len(DEM.stackij)):
        Z[DEM.stackij[i]] = Zi[DEM.stackij[i]]

    Z = Z.reshape(m, n, order='F')
    Z[0, 0] = 0
    profile = f.profile

    Z[Z == 0] = -9999

    with rio.open(outnm + '.tif', 'w', **profile) as dst:
        dst.write(Z.astype(rio.float64), 1)

    with rio.open(outnm + '.tif') as src:
        data = src.read(1)
        data[data > 0] = 1
        mask = data != src.nodata
        transform = src.transform

    shapes_gen = shapes(data.astype(np.uint8), mask=mask, transform=transform)
    geoms = [shape(geom) for geom, value in shapes_gen if value == 1]
    df = gpd.GeoDataFrame(geometry=geoms).dissolve()
    # df.to_file(outnm)
    return df, acc