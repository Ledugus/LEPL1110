import gmsh
import geopandas as gpd
#from numpy.char import center
#WARNING
#J'ai supprimé cette ligne parce que mon ordi ne veux pas importer from numpy.char
#Par contre numpy.char.center() fonctionne
import numpy as np
from plot_utils import plot_mesh_2d 
from gmsh_utils import *
import pyproj

def build_country_mesh(country_name="Australia", mesh_size=100, order=1):
    

    world = gpd.read_file("src/ne_10m_admin_0_countries.zip") 
    countries = country_name.split(",")
    
    country = world[world['NAME'].isin(countries)].geometry.unary_union
    
    country = country.simplify(0.05)

    # Si MultiPolygon (= pays avec îles), garder uniquement la partie continentale
    if hasattr(country, 'geoms'):  
        country = max(country.geoms, key=lambda p: p.area)
    
    coords_lonlat = np.array(country.exterior.coords)
    if np.allclose(coords_lonlat[0], coords_lonlat[-1]):
        coords_lonlat = coords_lonlat[:-1]
    
    # Projeter en mètres puis normaliser en km
    proj = pyproj.Proj("EPSG:3857") #https://epsg.io/3857
    coords = np.array([proj(lon, lat) for lon, lat in coords_lonlat]) / 1000.0
    
    # Recentrer autour de (0,0)
    # Recentrer autour de (0,0)
    center = coords.mean(axis=0)
    coords -= center
    
    
    point_tags = []
    for x, y in coords:
        tag = gmsh.model.occ.addPoint(x, y, 0.0, mesh_size)
        point_tags.append(tag)
    
    n = len(point_tags)
    line_tags = [gmsh.model.occ.addLine(point_tags[i], point_tags[(i+1) % n]) for i in range(n)]
    
    wire = gmsh.model.occ.addCurveLoop(line_tags)
    surface = gmsh.model.occ.addPlaneSurface([wire])
    gmsh.model.occ.synchronize()
    
    gmsh.model.addPhysicalGroup(1, line_tags, tag=1)
    gmsh.model.setPhysicalName(1, 1, "Border")
    gmsh.model.addPhysicalGroup(2, [surface], tag=2)
    gmsh.model.setPhysicalName(2, 2, "Domain")
    
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    
    gmsh.model.mesh.generate(2) #2D MESH
    gmsh.model.mesh.setOrder(order)
    
    elemType = gmsh.model.mesh.getElementType("triangle", order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)
    
    bnds = [("Border", 1)]
    bnds_tags = []
    for name, dim in bnds:
        tag = next(g[1] for g in gmsh.model.getPhysicalGroups(dim)
                   if gmsh.model.getPhysicalName(dim, g[1]) == name)
        bnds_tags.append(gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)[0])
    
    # Retourner aussi les bounds du domaine
    x_coords = nodeCoords.reshape(-1, 3)[:, 0]
    y_coords = nodeCoords.reshape(-1, 3)[:, 1]
    bounds = (x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max())
    
    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags, bounds, center




def main():
    gmsh_init("mesh")
    mesh = build_country_mesh(country_name="Russia", mesh_size=100, order=1)
    plot_mesh_2d(*mesh[:7])


if __name__ == "__main__":
    main()