from paraview.simple import *

from paraview.catalyst import get_args, get_execute_params
from paraview import catalyst

import argparse

parser = argparse.ArgumentParser(description="Plot Isosurfaces of Curl")
parser.add_argument("--output", type=str, help="Name of output files. Use %d for timestep")
parser.add_argument("--isosurfaces", type=float, nargs="+", help="Values of the isosurfaces")
args = parser.parse_args(get_args())


# registrationName must match the channel name used in the
# 'CatalystAdaptor'.
producer = TrivialProducer(registrationName="grid")


view = CreateRenderView()

computeDerivatives1 = ComputeDerivatives(registrationName='ComputeDerivatives1', Input=producer)
computeDerivatives1.Scalars = ['POINTS', 'u']
computeDerivatives1.Vectors = ['POINTS', 'velocity']
computeDerivatives1.OutputVectorType = 'Vorticity'
computeDerivatives1.OutputTensorType = 'Nothing'

resultLUT = GetColorTransferFunction('Vorticity')

display = Show(computeDerivatives1, view, 'GeometryRepresentation')
display.Representation = 'Volume'
display.ColorArrayName = ['POINTS', 'Vorticity']
display.LookupTable = resultLUT
display.SelectTCoordArray = 'None'
display.SelectNormalArray = 'Normals'
display.SelectTangentArray = 'None'
display.OSPRayScaleArray = 'Result'
display.OSPRayScaleFunction = 'PiecewiseFunction'
display.SelectOrientationVectors = 'Vorticity'
display.ScaleFactor = 11.741285705566407
display.SelectScaleArray = 'Result'
display.GlyphType = 'Arrow'
display.GlyphTableIndexArray = 'Result'
display.GaussianRadius = 0.5870642852783203
display.SetScaleArray = ['POINTS', 'Result']
display.ScaleTransferFunction = 'PiecewiseFunction'
display.OpacityArray = ['POINTS', 'Result']
display.OpacityTransferFunction = 'PiecewiseFunction'
display.DataAxesGrid = 'GridAxesRepresentation'
display.PolarAxes = 'PolarAxesRepresentation'
display.SelectInputVectors = ['POINTS', 'Vorticity']
display.WriteLog = ''

display.ScaleTransferFunction.Points = [0, 0.0, 0.5, 0.0, 0.75, 1.0, 0.5, 0.0]
display.OpacityTransferFunction.Points = [0, 0.0, 0.5, 0.0, 0.75, 0.25, 0.5, 0.0]
display.SetScalarBarVisibility(view, False)
resultPWF = GetOpacityTransferFunction('Vorticity')
resultTF2D = GetTransferFunction2D('Vorticity')
ColorBy(display, None)
HideScalarBarIfNotNeeded(resultLUT, view)
display.AmbientColor = [0.3333333333333333, 0.0, 1.0]
display.DiffuseColor = [0.3333333333333333, 0.0, 1.0]
Hide(cellDatatoPointData1, view)
view.OrientationAxesVisibility = 0


view.Update()

view.CameraPosition = [445.8245175004483, 310.69353937581025, 416.6850538377703]
view.CameraFocalPoint = [63.500000000000014, 63.5, 63.50000000000003]
view.CameraViewUp = [-0.28038806220484247, 0.9021477885353584, -0.32789007641856993]
view.CameraParallelScale = 151.5559135019442


def catalyst_execute(info):
    global producer

    contour1.UpdatePipeline()
    view.Update()

    fname = args.output % info.timestep
    ResetCamera()
    SaveScreenshot(fname, ImageResolution=(1024, 1024))
