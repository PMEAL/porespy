# ***********************************************************************************
# * Copyright 2010 - 2016 Paulo A. Herrera. All rights reserved.                    * 
# *                                                                                 *
# * Redistribution and use in source and binary forms, with or without              *
# * modification, are permitted provided that the following conditions are met:     *
# *                                                                                 *
# *  1. Redistributions of source code must retain the above copyright notice,      *
# *  this list of conditions and the following disclaimer.                          *
# *                                                                                 *
# *  2. Redistributions in binary form must reproduce the above copyright notice,   *
# *  this list of conditions and the following disclaimer in the documentation      *
# *  and/or other materials provided with the distribution.                         *
# *                                                                                 *
# * THIS SOFTWARE IS PROVIDED BY PAULO A. HERRERA ``AS IS'' AND ANY EXPRESS OR      *
# * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    *
# * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO      *
# * EVENT SHALL <COPYRIGHT HOLDER> OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,        *
# * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,  *
# * BUT NOT LIMITED TO, PROCUREMEN OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    *
# * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY           *
# * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING  *
# * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS              *
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                    *
# ***********************************************************************************

# **************************************
# *  High level Python library to      *
# *  export data to binary VTK file.   *
# **************************************

from .vtk import * # VtkFile, VtkUnstructuredGrid, etc.
import numpy as np

# =================================
#       Helper functions      
# =================================
def _addDataToFile(vtkFile, cellData, pointData):
    # Point data
    if pointData != None:
        keys = list(pointData.keys())
        vtkFile.openData("Point", scalars = keys[0])
        for key in keys:
            data = pointData[key]
            vtkFile.addData(key, data)
        vtkFile.closeData("Point")

    # Cell data
    if cellData != None:
        keys = list(cellData.keys())
        vtkFile.openData("Cell", scalars = keys[0])
        for key in keys:
            data = cellData[key]
            vtkFile.addData(key, data)
        vtkFile.closeData("Cell")

def _appendDataToFile(vtkFile, cellData, pointData):
    # Append data to binary section
    if pointData != None:
        keys = list(pointData.keys())
        for key in keys:
            data = pointData[key]
            vtkFile.appendData(data)

    if cellData != None:
        keys = list(cellData.keys())
        for key in keys:
            data = cellData[key]
            vtkFile.appendData(data)

# =================================
#       High level functions      
# =================================
def imageToVTK(path, origin = (0.0,0.0,0.0), spacing = (1.0,1.0,1.0), cellData = None, pointData = None ):
    """ Exports data values as a rectangular image.
        
        PARAMETERS:
            path: name of the file without extension where data should be saved.
            origin: grid origin (default = (0,0,0))
            spacing: grid spacing (default = (1,1,1))
            cellData: dictionary containing arrays with cell centered data.
                      Keys should be the names of the data arrays.
                      Arrays must have the same dimensions in all directions and must contain 
                      only scalar data.
            nodeData: dictionary containing arrays with node centered data.
                      Keys should be the names of the data arrays.
                      Arrays must have same dimension in each direction and 
                      they should be equal to the dimensions of the cell data plus one and
                      must contain only scalar data.
         
         RETURNS:
            Full path to saved file.

        NOTE: At least, cellData or pointData must be present to infer the dimensions of the image.
    """
    assert (cellData != None or pointData != None)
    
    # Extract dimensions
    start = (0,0,0)
    end = None
    if cellData != None:
        keys = list(cellData.keys())
        data = cellData[keys[0]]
        end = data.shape
    elif pointData != None:
        keys = list(pointData.keys())
        data = pointData[keys[0]]
        end = data.shape
        end = (end[0] - 1, end[1] - 1, end[2] - 1)

    # Write data to file
    w = VtkFile(path, VtkImageData)
    w.openGrid(start = start, end = end, origin = origin, spacing = spacing)
    w.openPiece(start = start, end = end)
    _addDataToFile(w, cellData, pointData)
    w.closePiece()
    w.closeGrid()
    _appendDataToFile(w, cellData, pointData)
    w.save()
#    return w.getFileName()
    print(w.getFileName())

# ==============================================================================
def gridToVTK(path, x, y, z, cellData = None, pointData = None):
    """
        Writes data values as a rectilinear or rectangular grid.

        PARAMETERS:
            path: name of the file without extension where data should be saved.
            x, y, z: coordinates of the nodes of the grid. They can be 1D or 3D depending if
                     the grid should be saved as a rectilinear or logically structured grid, respectively.
                     Arrays should contain coordinates of the nodes of the grid.
                     If arrays are 1D, then the grid should be Cartesian, i.e. faces in all cells are orthogonal.
                     If arrays are 3D, then the grid should be logically structured with hexahedral cells.
                     In both cases the arrays dimensions should be equal to the number of nodes of the grid.
            cellData: dictionary containing arrays with cell centered data.
                      Keys should be the names of the data arrays.
                      Arrays must have the same dimensions in all directions and must contain 
                      only scalar data.
            pointData: dictionary containing arrays with node centered data.
                       Keys should be the names of the data arrays.
                       Arrays must have same dimension in each direction and 
                       they should be equal to the dimensions of the cell data plus one and
                       must contain only scalar data.

        RETURNS:
            Full path to saved file.

    """
    # Extract dimensions
    start = (0,0,0)
    nx = ny = nz = 0

    if (x.ndim == 1 and y.ndim == 1 and z.ndim == 1):
        nx, ny, nz = x.size - 1, y.size - 1, z.size - 1
        isRect = True
        ftype = VtkRectilinearGrid
    elif (x.ndim == 3 and y.ndim == 3 and z.ndim == 3):
        s = x.shape
        nx, ny, nz = s[0] - 1, s[1] - 1, s[2] - 1
        isRect = False
        ftype = VtkStructuredGrid
    else:
        assert(False)
    end = (nx, ny, nz)


    w =  VtkFile(path, ftype)
    w.openGrid(start = start, end = end)
    w.openPiece(start = start, end = end)

    if isRect:
        w.openElement("Coordinates")
        w.addData("x_coordinates", x)
        w.addData("y_coordinates", y)
        w.addData("z_coordinates", z)
        w.closeElement("Coordinates")
    else:
        w.openElement("Points")
        w.addData("points", (x,y,z))
        w.closeElement("Points")

    _addDataToFile(w, cellData, pointData)
    w.closePiece()
    w.closeGrid()
    # Write coordinates
    if isRect:
        w.appendData(x).appendData(y).appendData(z)
    else:
        w.appendData( (x,y,z) )
    # Write data
    _appendDataToFile(w, cellData, pointData)
    w.save()
    return w.getFileName()


# ==============================================================================
def pointsToVTK(path, x, y, z, data):
    """
        Export points and associated data as an unstructured grid.

        PARAMETERS:
            path: name of the file without extension where data should be saved.
            x, y, z: 1D arrays with coordinates of the points.
            data: dictionary with variables associated to each point.
                  Keys should be the names of the variable stored in each array.
                  All arrays must have the same number of elements.

        RETURNS:
            Full path to saved file.

    """
    assert (x.size == y.size == z.size)
    npoints = x.size
    
    # create some temporary arrays to write grid topology
    offsets = np.arange(start = 1, stop = npoints + 1, dtype = 'int32')   # index of last node in each cell
    connectivity = np.arange(npoints, dtype = 'int32')                    # each point is only connected to itself
    cell_types = np.empty(npoints, dtype = 'uint8') 
   
    cell_types[:] = VtkVertex.tid

    w = VtkFile(path, VtkUnstructuredGrid)
    w.openGrid()
    w.openPiece(ncells = npoints, npoints = npoints)
    
    w.openElement("Points")
    w.addData("points", (x,y,z))
    w.closeElement("Points")
    w.openElement("Cells")
    w.addData("connectivity", connectivity)
    w.addData("offsets", offsets)
    w.addData("types", cell_types)
    w.closeElement("Cells")
    
    _addDataToFile(w, cellData = None, pointData = data)

    w.closePiece()
    w.closeGrid()
    w.appendData( (x,y,z) )
    w.appendData(connectivity).appendData(offsets).appendData(cell_types)

    _appendDataToFile(w, cellData = None, pointData = data)

    w.save()
    return w.getFileName()

# ==============================================================================
def linesToVTK(path, x, y, z, cellData = None, pointData = None):
    """
        Export line segments that joint 2 points and associated data.

        PARAMETERS:
            path: name of the file without extension where data should be saved.
            x, y, z: 1D arrays with coordinates of the vertex of the lines. It is assumed that each line.
                     is defined by two points, then the lenght of the arrays should be equal to 2 * number of lines.
            cellData: dictionary with variables associated to each line.
                  Keys should be the names of the variable stored in each array.
                  All arrays must have the same number of elements.         
            pointData: dictionary with variables associated to each vertex.
                  Keys should be the names of the variable stored in each array.
                  All arrays must have the same number of elements.

        RETURNS:
            Full path to saved file.

    """
    assert (x.size == y.size == z.size)
    assert (x.size % 2 == 0)
    npoints = x.size
    ncells = x.size / 2
    
    # Check cellData has the same size that the number of cells
    
    # create some temporary arrays to write grid topology
    offsets = np.arange(start = 2, step = 2, stop = npoints + 1, dtype = 'int32')   # index of last node in each cell
    connectivity = np.arange(npoints, dtype = 'int32')                              # each point is only connected to itself
    cell_types = np.empty(npoints, dtype = 'uint8') 
   
    cell_types[:] = VtkLine.tid

    w = VtkFile(path, VtkUnstructuredGrid)
    w.openGrid()
    w.openPiece(ncells = ncells, npoints = npoints)
    
    w.openElement("Points")
    w.addData("points", (x,y,z))
    w.closeElement("Points")
    w.openElement("Cells")
    w.addData("connectivity", connectivity)
    w.addData("offsets", offsets)
    w.addData("types", cell_types)
    w.closeElement("Cells")
    
    _addDataToFile(w, cellData = cellData, pointData = pointData)

    w.closePiece()
    w.closeGrid()
    w.appendData( (x,y,z) )
    w.appendData(connectivity).appendData(offsets).appendData(cell_types)

    _appendDataToFile(w, cellData = cellData, pointData = pointData)

    w.save()
    return w.getFileName()

# ==============================================================================
def polyLinesToVTK(path, x, y, z, pointsPerLine, cellData = None, pointData = None):
    """
        Export line segments that joint 2 points and associated data.

        PARAMETERS:
            path: name of the file without extension where data should be saved.
            x, y, z: 1D arrays with coordinates of the vertices of the lines. It is assumed that each line.
                     has diffent number of points.
            pointsPerLine: 1D array that defines the number of points associated to each line. Thus, 
                           the length of this array define the number of lines. It also implicitly 
                           defines the connectivity or topology of the set of lines. It is assumed 
                           that points that define a line are consecutive in the x, y and z arrays.
            cellData: Dictionary with variables associated to each line.
                      Keys should be the names of the variable stored in each array.
                      All arrays must have the same number of elements.         
            pointData: Dictionary with variables associated to each vertex.
                       Keys should be the names of the variable stored in each array.
                       All arrays must have the same number of elements.

        RETURNS:
            Full path to saved file.

    """
    assert (x.size == y.size == z.size)
    npoints = x.size
    ncells = pointsPerLine.size
    
    # create some temporary arrays to write grid topology
    offsets = np.zeros(ncells, dtype = 'int32')         # index of last node in each cell
    ii = 0
    for i in range(ncells):
        ii += pointsPerLine[i]
        offsets[i] = ii
    
    connectivity = np.arange(npoints, dtype = 'int32')      # each line connects points that are consecutive
   
    cell_types = np.empty(npoints, dtype = 'uint8') 
    cell_types[:] = VtkPolyLine.tid

    w = VtkFile(path, VtkUnstructuredGrid)
    w.openGrid()
    w.openPiece(ncells = ncells, npoints = npoints)
    
    w.openElement("Points")
    w.addData("points", (x,y,z))
    w.closeElement("Points")
    w.openElement("Cells")
    w.addData("connectivity", connectivity)
    w.addData("offsets", offsets)
    w.addData("types", cell_types)
    w.closeElement("Cells")
    
    _addDataToFile(w, cellData = cellData, pointData = pointData)

    w.closePiece()
    w.closeGrid()
    w.appendData( (x,y,z) )
    w.appendData(connectivity).appendData(offsets).appendData(cell_types)

    _appendDataToFile(w, cellData = cellData, pointData = pointData)

    w.save()
    return w.getFileName()

# ==============================================================================
def unstructuredGridToVTK(path, x, y, z, connectivity, offsets, cell_types, cellData = None, pointData = None):
    """
        Export unstructured grid and associated data.

        PARAMETERS:
            path: name of the file without extension where data should be saved.
            x, y, z: 1D arrays with coordinates of the vertices of cells. It is assumed that each element
                     has diffent number of vertices.
            connectivity: 1D array that defines the vertices associated to each element. 
                          Together with offset define the connectivity or topology of the grid. 
                          It is assumed that vertices in an element are listed consecutively.
            offsets: 1D array with the index of the last vertex of each element in the connectivity array.
                     It should have length nelem, where nelem is the number of cells or elements in the grid.
            cell_types: 1D array with an integer that defines the cell type of each element in the grid.
                        It should have size nelem. This should be assigned from evtk.vtk.VtkXXXX.tid, where XXXX represent
                        the type of cell. Please check the VTK file format specification for allowed cell types.                       
            cellData: Dictionary with variables associated to each line.
                      Keys should be the names of the variable stored in each array.
                      All arrays must have the same number of elements.        
            pointData: Dictionary with variables associated to each vertex.
                       Keys should be the names of the variable stored in each array.
                       All arrays must have the same number of elements.

        RETURNS:
            Full path to saved file.

    """
    assert (x.size == y.size == z.size)
    npoints = x.size
    ncells = cell_types.size
    assert (offsets.size == ncells)
    
    w = VtkFile(path, VtkUnstructuredGrid)
    w.openGrid()
    w.openPiece(ncells = ncells, npoints = npoints)
    
    w.openElement("Points")
    w.addData("points", (x,y,z))
    w.closeElement("Points")
    w.openElement("Cells")
    w.addData("connectivity", connectivity)
    w.addData("offsets", offsets)
    w.addData("types", cell_types)
    w.closeElement("Cells")
    
    _addDataToFile(w, cellData = cellData, pointData = pointData)

    w.closePiece()
    w.closeGrid()
    w.appendData( (x,y,z) )
    w.appendData(connectivity).appendData(offsets).appendData(cell_types)

    _appendDataToFile(w, cellData = cellData, pointData = pointData)

    w.save()
    return w.getFileName()
    
# ==============================================================================
def cylindricalToVTK(path, x, y, z, sh, cellData):
    """
        Export points and associated data as an unstructured grid.
        
        A cylindrical mesh connectivity is assumed. That is, the mesh is a 
        function
        
        f: D --> R^3
        (x,y,z)=f(i,j,k)
        
        where D is the cartesian product of graphs between a cycle (C_j)
        and two path graphs (P_i and P_k).
        
        D= P_i x C_j x P_k
        
        for further explanation see:
        https://en.wikipedia.org/wiki/Cartesian_product_of_graphs
        https://en.wikipedia.org/wiki/Path_graph
        https://en.wikipedia.org/wiki/Cycle_graph
        

        PARAMETERS:
            path: name of the file without extension where data should be saved.
            x, y, z: 1D arrays with coordinates of the points.
            sh: number of cells in each direction
            cellData: dictionary with variables associated to each cell.
                  Keys should be the names of the variable stored in each array.
                  All arrays must have the same number of elements.

        RETURNS:
            Full path to saved file.

    """
    assert(x.size==y.size==z.size)
    s=sh+(1,0,1)
    npoints = np.prod(s)
    ncells = np.prod(sh)
    
    
    assert(npoints==x.size)
    
    # create some temporary arrays to write grid topology
    offsets = np.arange(start = 8, stop = 8*(ncells + 1), step=8, dtype = 'int32')   # index of last node in each cell
    cell_types = np.empty(ncells, dtype = 'uint8') 
    cell_types[:] = VtkHexahedron.tid
    
    # create connectivity
    connectivity = np.empty(8*ncells, dtype = 'int32') 
    i=0

    for zeta in range(0,sh[2]):
        for tita in range(0,sh[1]):
            for r in range(0,sh[0]):
                for d in ((0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)):
                    connectivity[i]=r+d[0]+s[0]*((tita+d[1])%s[1])+s[0]*s[1]*(zeta+d[2])
                    i+=1

    w = VtkFile(path, VtkUnstructuredGrid)
    w.openGrid()
    w.openPiece(ncells = ncells, npoints = npoints)
    
    w.openElement("Points")
    w.addData("points", (x,y,z))
    w.closeElement("Points")
    w.openElement("Cells")
    w.addData("connectivity", connectivity)
    w.addData("offsets", offsets)
    w.addData("types", cell_types)
    w.closeElement("Cells")
    
    # adaptar cellData segun formato!!! 
    
    _addDataToFile(w, cellData = cellData, pointData = None)

    w.closePiece()
    w.closeGrid()
    w.appendData( (x,y,z) )
    w.appendData(connectivity).appendData(offsets).appendData(cell_types)

    _appendDataToFile(w, cellData = cellData, pointData = None)

    w.save()
    return w.getFileName()


